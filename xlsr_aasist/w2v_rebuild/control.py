"""Save promising candidates independently of promotion; stop on actual progress."""
import math
from .selection import noisy_metrics


class AdaptationControl:
    def __init__(self, baseline):
        self.candidate_key = (baseline['robust_f1'], -baseline['robust_ce'])
        self.candidate_epoch = 0
        self.best_ce = baseline['robust_ce']
        self.best_noisy_f1 = noisy_metrics(baseline)['noisy_f1']
        self.stale = 0
        self.last_reduction_epoch = 0

    def observe(self, dev, epoch, reduced):
        key = (dev['robust_f1'], -dev['robust_ce'])
        noisy = noisy_metrics(dev)['noisy_f1']
        if not all(math.isfinite(x) for x in (*key, noisy)):
            raise FloatingPointError('Non-finite adaptation progress')
        candidate = key > self.candidate_key
        progress = []
        if candidate:
            self.candidate_key, self.candidate_epoch = key, epoch
            progress.append('candidate_score')
        if dev['robust_ce'] < self.best_ce * (1 - 1e-3):
            self.best_ce = dev['robust_ce']
            progress.append('balanced_ce')
        if noisy > self.best_noisy_f1 + 1e-12:
            self.best_noisy_f1 = noisy
            progress.append('noisy_f1')
        self.stale = 0 if progress else self.stale + 1
        if reduced:
            self.last_reduction_epoch = epoch
        return candidate, {'improved': progress, 'stale': self.stale,
                           'last_reduction_epoch': self.last_reduction_epoch}

    def should_stop(self, epoch, patience):
        # The reduction applies to the NEXT epoch. At least one full epoch at
        # the reduced rate must finish before early stopping can terminate it.
        return self.stale >= patience and epoch > self.last_reduction_epoch

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        if set(state) != set(vars(self)):
            raise ValueError('Adaptation control state is incomplete')
        for key, value in state.items():
            setattr(self, key, tuple(value) if key == 'candidate_key' else value)
