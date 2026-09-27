"""View tickets are created in the main process, never in worker-local state."""
import math
import random

from .plan import digest_json, stable_seed


class RotatingViewBatchSampler:
    """Wrap the existing balanced SOURCE sampler; emit (source, bank, band).

    An epoch plan is immutable and repeatable, so DataLoader prefetch and worker
    scheduling cannot change the assigned views. Counts advance only when the
    trainer commits a completed epoch. No four-view forward pass is introduced.
    """

    def __init__(self, source_sampler, sources, seed=1234, banks=1, bank_policy='cycle'):
        if type(banks) is not int or banks < 1:
            raise ValueError("banks must be a positive integer")
        self.source_sampler, self.seed, self.banks = source_sampler, int(seed), banks
        if bank_policy not in ('cycle', 'mixed'):
            raise ValueError('Unknown bank policy')
        self.bank_policy = bank_policy
        self.source_ids = [row["offline"] for row in sources]
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Offline source IDs must be unique and nonempty")
        self.source_digest = digest_json(self.source_ids)
        self.visits = [0] * len(self.source_ids)
        self.completed_epoch = 0
        self._plan = self._end_visits = self._epoch = None
        self.mixture = None
        self.bank_visits = self._end_bank_visits = None

    def configure_mixture(self, extra_fraction, warmup_steps):
        """Low-fraction, class-symmetric bank mixing without replacing any source.

        Configure before the first epoch/resume. A whole balanced pair batch
        uses one bank, so real and fake receive exactly the same bank fractions.
        The cumulative quota ramps linearly and differs from its target by <1 batch.
        """
        if (self.completed_epoch or self._plan is not None or self.banks < 2
                or self.bank_policy != 'cycle' or not math.isfinite(extra_fraction)
                or not 0 < extra_fraction <= 1 or type(warmup_steps) is not int or warmup_steps < 0):
            raise ValueError('Invalid scheduled mixture configuration')
        self.mixture = {'extra_fraction': float(extra_fraction), 'warmup_steps': warmup_steps}
        self.bank_visits = [[0] * self.banks for _ in self.source_ids]

    def _extra_batches(self, steps):
        p, warm = self.mixture['extra_fraction'], self.mixture['warmup_steps']
        ramp = min(steps, warm)
        quota = p * (ramp * (ramp + 1) / (2 * warm) + max(0, steps - warm)) if warm else p * steps
        return math.floor(quota + 1e-12)

    def plan_summary(self):
        if self._plan is None:
            raise RuntimeError('No active epoch plan')
        counts = [0] * self.banks
        bands = [[0] * 4 for _ in range(self.banks)]
        for batch in self._plan:
            for _, bank, band in batch:
                counts[bank] += 1
                bands[bank][band] += 1
        total = sum(counts)
        return {'source_views_by_bank': counts, 'source_views_by_bank_band': bands,
                'extra_fraction_actual': sum(counts[1:]) / total if total else 0.,
                'mixture': self.mixture}

    def __len__(self):
        return len(self.source_sampler)

    def set_epoch(self, epoch):
        if self._plan is not None:
            raise RuntimeError("Commit/discard the previous epoch plan first")
        if epoch != self.completed_epoch + 1:
            raise ValueError("Rotation epochs must be consecutive; restore state when continuing")
        self.source_sampler.set_epoch(epoch)
        visits, plan = self.visits.copy(), []
        bank_visits = [row.copy() for row in self.bank_visits] if self.mixture else None
        for step, indices in enumerate(self.source_sampler):
            if len(set(indices)) != len(indices):
                raise ValueError("Each batch needs distinct Offline sources; reduce noisy_pairs_per_batch")
            tickets = []
            if self.mixture:
                previous = self._extra_batches((epoch - 1) * len(self) + step)
                current = self._extra_batches((epoch - 1) * len(self) + step + 1)
                selected_bank = 1 + (current - 1) % (self.banks - 1) if current > previous else 0
            for index in indices:
                if type(index) is not int or not 0 <= index < len(visits):
                    raise ValueError("Invalid source index")
                if self.mixture:
                    bank = selected_bank
                    cycle, slot = divmod(bank_visits[index][bank], 4)
                    order = list(range(4))
                    random.Random(stable_seed(self.seed, 'mixture_views', self.source_ids[index], bank, cycle)).shuffle(order)
                    tickets.append((index, bank, order[slot]))
                    bank_visits[index][bank] += 1
                elif self.bank_policy == 'mixed':
                    cycle, slot = divmod(visits[index], 4 * self.banks)
                    order = [(bank, band) for bank in range(self.banks) for band in range(4)]
                    random.Random(stable_seed(self.seed, 'mixed_views', self.source_ids[index], cycle)).shuffle(order)
                    bank, band = order[slot]
                    tickets.append((index, bank, band))
                else:
                    cycle, slot = divmod(visits[index], 4)
                    order = list(range(4))
                    random.Random(stable_seed(self.seed, "views", self.source_ids[index], cycle)).shuffle(order)
                    tickets.append((index, cycle % self.banks, order[slot]))
                visits[index] += 1
            plan.append(tuple(tickets))
        if len(plan) != len(self):
            raise ValueError("Source sampler length differs from its actual batches")
        self._plan, self._end_visits, self._epoch = tuple(plan), visits, epoch
        self._end_bank_visits = bank_visits

    def __iter__(self):
        if self._plan is None:
            raise RuntimeError("Call set_epoch before iterating the DataLoader")
        yield from self._plan

    def commit_epoch(self, completed_steps):
        if self._plan is None or completed_steps != len(self._plan):
            raise ValueError("Only a fully consumed training epoch can be committed")
        self.visits, self.completed_epoch = self._end_visits, self._epoch
        if self.mixture:
            self.bank_visits = self._end_bank_visits
        self._plan = self._end_visits = self._epoch = None
        self._end_bank_visits = None

    def discard_epoch(self):
        self._plan = self._end_visits = self._epoch = None
        self._end_bank_visits = None

    def state_dict(self):
        # Export only COMMITTED visits. Prefetched tickets are not training history.
        return {"format": "rtc_view_rotation_v2", "source_digest": self.source_digest,
                "seed": self.seed, "banks": self.banks, "steps_per_epoch": len(self),
                "completed_epoch": self.completed_epoch, "visits": self.visits.copy(),
                **({'bank_policy': self.bank_policy} if self.bank_policy != 'cycle' else {}),
                **({'mixture': self.mixture, 'bank_visits': self.bank_visits} if self.mixture else {})}

    def load_state_dict(self, state):
        expected = {"format": "rtc_view_rotation_v2", "source_digest": self.source_digest,
                    "seed": self.seed, "banks": self.banks, "steps_per_epoch": len(self)}
        if state.get('bank_policy', 'cycle') != self.bank_policy:
            raise ValueError('Rotation bank policy changed')
        if state.get('mixture') != self.mixture:
            raise ValueError('Scheduled mixture changed across resume')
        if self._plan is not None or any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Rotation state does not match this sampler")
        visits, epoch = state.get("visits"), state.get("completed_epoch")
        if (not isinstance(visits, list) or len(visits) != len(self.visits)
                or any(type(v) is not int or v < 0 for v in visits)
                or type(epoch) is not int or epoch < 0):
            raise ValueError("Invalid rotation history")
        self.visits, self.completed_epoch = visits.copy(), epoch
        if self.mixture:
            bank_visits = state.get('bank_visits')
            if (not isinstance(bank_visits, list) or len(bank_visits) != len(visits)
                    or any(not isinstance(row, list) or len(row) != self.banks
                           or any(type(v) is not int or v < 0 for v in row)
                           or sum(row) != visits[i] for i, row in enumerate(bank_visits))):
                raise ValueError('Invalid per-bank rotation history')
            self.bank_visits = [row.copy() for row in bank_visits]
