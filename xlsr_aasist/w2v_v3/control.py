"""Bounded validation-driven control; recall protection is separate from ranking."""
import copy
import math


def quality(dev):
    """Use the same Dev protocol and fixed threshold for every candidate."""
    groups = dev['groups']
    real = [groups[k]['recall'][1] for k in
            ('offline/en', 'online/en', 'seen/en', 'heldout/en')]
    fake = [groups[k]['recall'][0] for k in ('seen/en', 'heldout/en')]
    result = {'weighted': float(dev['weighted_f1']), 'noisy': float(dev['noisy_f1']),
              'clean': float(dev['clean_f1']), 'en_real': sum(real) / len(real),
              'noisy_en_fake': sum(fake) / len(fake)}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError('Selection requires finite probabilities and all four EN Dev groups')
    return result


class Controller:
    """Pure state machine, serializable in each training checkpoint.

    All metric winners remain available. Head training chooses the best weighted
    point first. Joint adaptation uses its recalls as floors; accepted models
    can tighten these floors, but can never lower them cumulatively.
    """
    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.state = copy.deepcopy(state) if state is not None else {
            'phase': 'head', 'phase_evals': 0, 'phase_steps': 0,
            'lr_scale': 1., 'reductions': 0, 'rescues': 0,
            'since_reduction': 0, 'stale': 0, 'anchor': None,
            'best_weighted': None, 'best_noisy': None, 'best_safe': None,
            'phase_best': None, 'evaluations': 0,
        }

    def phase_budget(self):
        return self.cfg[self.state['phase'] + '_epochs'] * self.cfg['evals_per_epoch']

    def observe(self, dev, tag):
        q, s = quality(dev), self.state
        s['phase_evals'] += 1
        s['evaluations'] += 1
        s['since_reduction'] += 1
        candidate = {'tag': tag, **q}
        anchor = s['anchor']
        warnings = []
        if anchor and q['en_real'] < anchor['en_real'] - self.cfg.get('en_real_tolerance', .01):
            warnings.append('en_real_below_anchor')
        if anchor and q['noisy_en_fake'] < anchor['noisy_en_fake'] - self.cfg.get('noisy_fake_tolerance', .005):
            warnings.append('noisy_en_fake_below_anchor')
        saved = []
        for name, keys in [('best_weighted', ('weighted', 'noisy')),
                           ('best_noisy', ('noisy', 'weighted')),
                           ('best_safe', ('weighted', 'noisy'))]:
            if name == 'best_safe' and warnings:
                continue
            old = s[name]
            if old is None or tuple(q[k] for k in keys) > tuple(old[k] for k in keys):
                s[name] = copy.deepcopy(candidate)
                saved.append(name)
        if anchor and 'best_safe' in saved:
            for key in ('en_real', 'noisy_en_fake'):
                anchor[key] = max(anchor[key], q[key])
        minimum = self.cfg.get('selection_min_delta', .0001)
        old_phase = s['phase_best']
        meaningful = not warnings and (old_phase is None or
                                      q['weighted'] > old_phase['weighted'] + minimum)
        if meaningful:
            s['phase_best'] = copy.deepcopy(candidate)
            s['stale'] = 0
        else:
            s['stale'] += 1
        remaining = self.phase_budget() - s['phase_evals']
        trial = self.cfg.get('reduced_lr_evals', 2)
        ready = s['reductions'] == 0 or s['since_reduction'] >= trial
        drift = bool(warnings) or (s['best_safe'] is not None and
                   q['weighted'] < s['best_safe']['weighted'] - self.cfg.get('drift_score_tolerance', .003))
        can_reduce = (remaining >= trial and ready and
                      s['reductions'] < self.cfg.get('max_lr_reductions', 2))
        rescue = drift and s['rescues'] < self.cfg.get('drift_rescues', 1)
        plateau = s['stale'] >= self.cfg.get('plateau_evals', 2)
        action = 'continue'
        if can_reduce and (rescue or plateau):
            s['lr_scale'] *= self.cfg.get('lr_factor', .5)
            s['reductions'] += 1
            s['since_reduction'] = 0
            s['stale'] = 0
            if rescue:
                s['rescues'] += 1
            action = 'restore_reduce'
        elif remaining <= 0:
            action = 'phase_complete'
        elif plateau and ready and s['reductions'] >= self.cfg.get('max_lr_reductions', 2):
            action = 'phase_complete'
        return {'action': action, 'save': saved, 'warnings': warnings,
                'quality': q, 'meaningful_improvement': meaningful,
                'remaining_evaluations': max(0, remaining),
                'restore_tag': s['best_safe']['tag'], 'lr_scale': s['lr_scale']}

    def begin_joint(self):
        if self.state['phase'] != 'head':
            raise ValueError('Joint phase may only follow head adaptation')
        if self.state['best_safe'] is None:
            raise ValueError('Joint phase requires a validated head checkpoint')
        self.state.update(phase='joint', phase_evals=0, phase_steps=0,
                          lr_scale=1., reductions=0, rescues=0,
                          since_reduction=0, stale=0,
                          anchor=copy.deepcopy(self.state['best_safe']),
                          phase_best=copy.deepcopy(self.state['best_safe']))

    def dump(self):
        return copy.deepcopy(self.state)


def lr_scale(cfg, phase_step, phase_steps, controller_scale=1.):
    """Linear warmup then cosine decay, multiplied by adaptive reductions."""
    warm = min(cfg.get('lr_warmup_steps', 100), max(1, phase_steps // 10))
    minimum = cfg.get('min_lr_scale', .1)
    if phase_step < warm:
        scale = .1 + .9 * (phase_step + 1) / warm
    else:
        fraction = min(1., (phase_step - warm) / max(1, phase_steps - warm - 1))
        scale = minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * fraction))
    return scale * controller_scale
