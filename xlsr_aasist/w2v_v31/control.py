"""Fixed-baseline selection and bounded recovery for warm V3.1 adaptation."""
import copy
import math


REAL_KEYS = ('offline_en_real', 'online_en_real', 'seen_en_real', 'heldout_en_real')
FAKE_KEYS = ('seen_en_fake', 'heldout_en_fake')


def quality(dev):
    """Keep condition-specific protection separate from aggregate reporting."""
    groups = dev['groups']
    result = {'weighted': float(dev['weighted_f1']),
              'noisy': float(dev['noisy_f1']), 'clean': float(dev['clean_f1'])}
    for condition in ('offline', 'online', 'seen', 'heldout'):
        recall = groups[condition + '/en']['recall']
        result[condition + '_en_real'] = float(recall[1])
        if condition in ('seen', 'heldout'):
            result[condition + '_en_fake'] = float(recall[0])
    result['en_real'] = sum(result[k] for k in REAL_KEYS) / len(REAL_KEYS)
    result['noisy_en_real'] = (result['seen_en_real'] + result['heldout_en_real']) / 2
    result['noisy_en_fake'] = sum(result[k] for k in FAKE_KEYS) / len(FAKE_KEYS)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError('Selection requires finite metrics and all four English Dev groups')
    return result


class Controller:
    """A validated starting checkpoint remains the fallback throughout this run.

    Floors never move: neither a sequence of small declines nor improving clean
    recall can conceal a noisy recall regression. Unconditional metric winners
    remain diagnostic candidates and cannot replace the protected fallback.
    """
    def __init__(self, cfg, state=None):
        self.cfg = cfg
        if cfg.get('joint_epochs', 2) not in (1, 2) or cfg.get('evals_per_epoch', 2) != 2:
            raise ValueError('V3.1 allows one or two joint epochs, evaluated twice per epoch')
        for key, default in (('en_real_tolerance', .005), ('noisy_fake_tolerance', .003),
                             ('clean_tolerance', .001), ('selection_min_delta', .0001),
                             ('metric_epsilon', 1e-12), ('drift_score_tolerance', .003),
                             ('severe_real_drop', .02), ('severe_fake_drop', .01)):
            value = cfg.get(key, default)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError('Invalid selection setting: ' + key)
        if cfg.get('metric_epsilon', 1e-12) > 1e-8:
            raise ValueError('Metric epsilon is only for numerical equality')
        if not 0 < cfg.get('lr_factor', .5) < 1:
            raise ValueError('LR recovery factor must be in (0,1)')
        self.state = copy.deepcopy(state) if state is not None else {
            'controller_schema': 'v31_fixed_baseline_v1',
            'phase': 'joint', 'phase_evals': 0, 'phase_steps': 0,
            'lr_scale': 1., 'reductions': 0, 'rescues': 0,
            'since_reduction': 0, 'stale': 0, 'anchor': None,
            'best_weighted': None, 'best_noisy': None, 'best_safe': None,
            'phase_best': None, 'evaluations': 0,
        }
        if (self.state.get('controller_schema') != 'v31_fixed_baseline_v1'
                or self.state.get('phase') != 'joint'):
            raise ValueError('Resume requires the V3.1 fixed-baseline controller')

    def phase_budget(self):
        return self.cfg.get('joint_epochs', 2) * self.cfg.get('evals_per_epoch', 2)

    def _result(self, q, saved, warnings, action, meaningful=False):
        s = self.state
        return {'action': action, 'save': saved, 'warnings': warnings,
                'quality': q, 'meaningful_improvement': meaningful,
                'remaining_evaluations': max(0, self.phase_budget() - s['phase_evals']),
                'restore_tag': s['best_safe']['tag'], 'lr_scale': s['lr_scale']}

    def initialize(self, dev, tag='baseline'):
        """Register a fresh, fixed-condition baseline without spending training budget."""
        if self.state['anchor'] is not None or self.state['evaluations']:
            raise RuntimeError('The starting baseline may only be registered once')
        q = quality(dev)
        candidate = {'tag': tag, **q}
        for name in ('anchor', 'best_weighted', 'best_noisy', 'best_safe', 'phase_best'):
            self.state[name] = copy.deepcopy(candidate)
        return self._result(q, ['best_weighted', 'best_noisy', 'best_safe'], [], 'continue')

    def observe(self, dev, tag):
        q, s = quality(dev), self.state
        anchor = s['anchor']
        if anchor is None:
            raise RuntimeError('Evaluate and initialize the starting model before training')
        if s['phase_evals'] >= self.phase_budget():
            raise RuntimeError('The V3.1 evaluation budget is exhausted')
        if tag == anchor['tag']:
            raise ValueError('Training candidates cannot reuse the baseline tag')
        s['phase_evals'] += 1
        s['evaluations'] += 1
        s['since_reduction'] += 1
        candidate = {'tag': tag, **q}
        epsilon = self.cfg.get('metric_epsilon', 1e-12)
        warnings = []
        for key in REAL_KEYS:
            if q[key] + epsilon < anchor[key] - self.cfg.get('en_real_tolerance', .005):
                warnings.append(key + '_below_baseline')
        for key in FAKE_KEYS:
            if q[key] + epsilon < anchor[key] - self.cfg.get('noisy_fake_tolerance', .003):
                warnings.append(key + '_below_baseline')
        if q['clean'] + epsilon < anchor['clean'] - self.cfg.get('clean_tolerance', .001):
            warnings.append('clean_f1_below_baseline')
        if q['noisy'] + epsilon < anchor['noisy']:
            warnings.append('noisy_f1_below_baseline')

        saved = []
        for name, keys in (('best_weighted', ('weighted', 'noisy')),
                           ('best_noisy', ('noisy', 'weighted'))):
            old = s[name]
            if tuple(q[k] for k in keys) > tuple(old[k] for k in keys):
                s[name] = copy.deepcopy(candidate)
                saved.append(name)
        minimum = self.cfg.get('selection_min_delta', .0001)
        meaningful = not warnings and q['weighted'] > s['best_safe']['weighted'] + minimum
        if meaningful:
            s['best_safe'] = copy.deepcopy(candidate)
            s['phase_best'] = copy.deepcopy(candidate)
            saved.append('best_safe')
            s['stale'] = 0
        else:
            s['stale'] += 1

        remaining = self.phase_budget() - s['phase_evals']
        severe = (
            any(q[k] < anchor[k] - self.cfg.get('severe_real_drop', .02) for k in REAL_KEYS)
            or any(q[k] < anchor[k] - self.cfg.get('severe_fake_drop', .01) for k in FAKE_KEYS)
            or q['weighted'] < s['best_safe']['weighted'] - self.cfg.get('drift_score_tolerance', .003)
            or q['noisy'] < anchor['noisy'] - self.cfg.get('drift_score_tolerance', .003))
        action = 'continue'
        # A reduction must have two complete future validation intervals. There
        # is at most one recovery; never halve LR immediately before stopping.
        if remaining <= 0:
            action = 'phase_complete'
        elif s['reductions'] == 0 and remaining >= 2 and (severe or s['stale'] >= 2):
            s['lr_scale'] *= self.cfg.get('lr_factor', .5)
            s['reductions'] = 1
            s['rescues'] = int(severe)
            s['since_reduction'] = 0
            s['stale'] = 0
            action = 'restore_reduce'
        return self._result(q, saved, warnings, action, meaningful)

    def dump(self):
        return copy.deepcopy(self.state)


def lr_scale(cfg, phase_step, phase_steps, controller_scale=1.):
    """Warmup/cosine schedule with persistent, multiplicative recovery scaling."""
    warm = min(cfg.get('lr_warmup_steps', 100), max(1, phase_steps // 10))
    minimum = cfg.get('min_lr_scale', .1)
    if phase_step < warm:
        scale = .1 + .9 * (phase_step + 1) / warm
    else:
        fraction = min(1., (phase_step - warm) / max(1, phase_steps - warm - 1))
        scale = minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * fraction))
    return scale * controller_scale
