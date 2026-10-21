"""Unchanged CE exposure; source-verified Online/Noisy positives on side A."""
from collections import Counter
from w2v_v5.sampling import Plan as ParentPlan
from w2v_v3186.sampling import canonical_pairs
from w2v_v3185.common import GROUPS, seed_for
from .contrastive import configuration


class Plan(ParentPlan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.continuation_cfg
        self.contrast_pairs = canonical_pairs(self.rows) if configuration(cfg)[0] else {}
        for left, right in self.contrast_pairs.items():
            if self.canonical[left] != self.canonical[right]:
                raise ValueError('Official contrastive pair crosses canonical sources')

    def _references(self, tickets):
        tickets = super()._references(tickets)
        cfg = self.continuation_cfg
        if not configuration(cfg)[0]:
            return tickets
        for group in GROUPS:
            members = [t for t in tickets if t['meta_side'] == 'A' and t['kind'] == 'offline'
                       and (self.rows[t['index']]['language'], self.rows[t['index']]['label']) == group]
            budget = min(len(members), cfg.get('contrastive_sources_per_group', 3))
            for ticket in members:
                ticket['contrastive_status'] = ('not_selected' if ticket['index'] in self.contrast_pairs
                                                else 'no_official_pair')
            # Reuse official references already needed by PCL before adding any.
            candidates = sorted((t for t in members if t['index'] in self.contrast_pairs),
                key=lambda t: ('pcl_online_index' not in t,
                               seed_for(self.seed, 'v52-contrast', t['occurrence'])))
            for ticket in candidates[:budget]:
                online = self.contrast_pairs[ticket['index']]
                if self.canonical[online] != ticket['canonical_source']:
                    raise ValueError('Contrastive reference is outside the scheduled canonical source')
                ticket.update(contrastive_online_index=online, contrastive_group_budget=budget,
                              contrastive_status='selected')
        return tickets

    def coverage(self, epoch):
        value = super().coverage(epoch)
        groups = Counter(f'{self.rows[i]["language"]}/{self.rows[i]["label"]}' for i in self.contrast_pairs)
        value['contrastive'] = dict(enabled=bool(configuration(self.continuation_cfg)[0]),
            side='A', official_pairs=len(self.contrast_pairs), eligible_by_group=dict(groups),
            target_per_group=self.continuation_cfg.get('contrastive_sources_per_group', 3),
            B_classification_only=True, missing_pair_policy='retain all classification and available PCL',
            positives='same canonical source official Online and simulated Noisy',
            negatives='different canonical source, same language, opposite true label')
        return value
