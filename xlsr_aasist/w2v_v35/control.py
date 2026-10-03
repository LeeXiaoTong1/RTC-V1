"""Online objective selection, independent progress tracking, whole-epoch patience."""
import copy
import math


def quality(dev):
    values = {key: float(dev[key]) for key in ('clean_f1', 'noisy_f1', 'weighted_f1')}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Finite probability-scale Online Dev metrics required')
    if abs(values['weighted_f1'] - (.3 * values['clean_f1'] + .7 * values['noisy_f1'])) > 1e-8:
        raise ValueError('Weighted Dev must be .3 Clean Online + .7 Noisy Online')
    return values


class Controller:
    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.state = copy.deepcopy(state) if state is not None else dict(
            reference=None, best_train=None, best_selected=None, joint_stale_epochs=0,
            joint_epochs_seen=0, completed=False)

    def initialize(self, dev, tag='reference'):
        if self.state['reference'] is not None:
            raise ValueError('Reference has already been established')
        anchor = dict(tag=tag, **quality(dev))
        self.state['reference'] = anchor
        self.state['best_selected'] = copy.deepcopy(anchor)

    def observe(self, dev, tag, phase, full_epoch=True):
        if phase not in ('head', 'joint') or not full_epoch:
            raise ValueError('V3.5 patience is evaluated only at completed epochs')
        observed = dict(tag=tag, **quality(dev))
        minimum = self.cfg.get('selection_min_delta', 1e-5)
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError('selection_min_delta must be finite and nonnegative')
        previous = self.state['best_train']
        improved = previous is None or observed['weighted_f1'] > previous['weighted_f1'] + minimum
        save = []
        if improved:
            self.state['best_train'] = observed
            save.append('best_train')
        selected = self.state['best_selected']
        promoted = observed['weighted_f1'] > selected['weighted_f1'] + minimum
        if promoted:
            self.state['best_selected'] = observed
            save.append('best_model')
        if phase == 'joint':
            self.state['joint_epochs_seen'] += 1
            self.state['joint_stale_epochs'] = 0 if improved else self.state['joint_stale_epochs'] + 1
            self.state['completed'] = (self.state['joint_stale_epochs'] >= self.cfg.get('patience', 2)
                or self.state['joint_epochs_seen'] >= self.cfg.get('joint_epochs', 5))
        return dict(save=save, promoted=promoted, improved=improved,
                    action='phase_complete' if self.state['completed'] else 'continue',
                    warnings=[], quality=quality(dev),
                    remaining_evaluations=max(0, self.cfg.get('joint_epochs', 5) - self.state['joint_epochs_seen']))

    def dump(self):
        return copy.deepcopy(self.state)
