"""View tickets are created in the main process, never in worker-local state."""
import random

from .plan import digest_json, stable_seed


class RotatingViewBatchSampler:
    """Wrap the existing balanced SOURCE sampler; emit (source, bank, band).

    An epoch plan is immutable and repeatable, so DataLoader prefetch and worker
    scheduling cannot change the assigned views. Counts advance only when the
    trainer commits a completed epoch. No four-view forward pass is introduced.
    """

    def __init__(self, source_sampler, sources, seed=1234, banks=1):
        if type(banks) is not int or banks < 1:
            raise ValueError("banks must be a positive integer")
        self.source_sampler, self.seed, self.banks = source_sampler, int(seed), banks
        self.source_ids = [row["offline"] for row in sources]
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Offline source IDs must be unique and nonempty")
        self.source_digest = digest_json(self.source_ids)
        self.visits = [0] * len(self.source_ids)
        self.completed_epoch = 0
        self._plan = self._end_visits = self._epoch = None

    def __len__(self):
        return len(self.source_sampler)

    def set_epoch(self, epoch):
        if self._plan is not None:
            raise RuntimeError("Commit/discard the previous epoch plan first")
        if epoch != self.completed_epoch + 1:
            raise ValueError("Rotation epochs must be consecutive; restore state when continuing")
        self.source_sampler.set_epoch(epoch)
        visits, plan = self.visits.copy(), []
        for indices in self.source_sampler:
            if len(set(indices)) != len(indices):
                raise ValueError("Each batch needs distinct Offline sources; reduce noisy_pairs_per_batch")
            tickets = []
            for index in indices:
                if type(index) is not int or not 0 <= index < len(visits):
                    raise ValueError("Invalid source index")
                cycle, slot = divmod(visits[index], 4)
                order = list(range(4))
                random.Random(stable_seed(self.seed, "views", self.source_ids[index], cycle)).shuffle(order)
                # Optional extra cache banks change only at a SOURCE cycle boundary.
                tickets.append((index, cycle % self.banks, order[slot]))
                visits[index] += 1
            plan.append(tuple(tickets))
        if len(plan) != len(self):
            raise ValueError("Source sampler length differs from its actual batches")
        self._plan, self._end_visits, self._epoch = tuple(plan), visits, epoch

    def __iter__(self):
        if self._plan is None:
            raise RuntimeError("Call set_epoch before iterating the DataLoader")
        yield from self._plan

    def commit_epoch(self, completed_steps):
        if self._plan is None or completed_steps != len(self._plan):
            raise ValueError("Only a fully consumed training epoch can be committed")
        self.visits, self.completed_epoch = self._end_visits, self._epoch
        self._plan = self._end_visits = self._epoch = None

    def discard_epoch(self):
        self._plan = self._end_visits = self._epoch = None

    def state_dict(self):
        # Export only COMMITTED visits. Prefetched tickets are not training history.
        return {"format": "rtc_view_rotation_v2", "source_digest": self.source_digest,
                "seed": self.seed, "banks": self.banks, "steps_per_epoch": len(self),
                "completed_epoch": self.completed_epoch, "visits": self.visits.copy()}

    def load_state_dict(self, state):
        expected = {"format": "rtc_view_rotation_v2", "source_digest": self.source_digest,
                    "seed": self.seed, "banks": self.banks, "steps_per_epoch": len(self)}
        if self._plan is not None or any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Rotation state does not match this sampler")
        visits, epoch = state.get("visits"), state.get("completed_epoch")
        if (not isinstance(visits, list) or len(visits) != len(self.visits)
                or any(type(v) is not int or v < 0 for v in visits)
                or type(epoch) is not int or epoch < 0):
            raise ValueError("Invalid rotation history")
        self.visits, self.completed_epoch = visits.copy(), epoch
