"""Retain every V5.1 exposure ticket, annotate small six-view metric groups."""
from collections import Counter
from w2v_v5.sampling import Plan as ParentPlan
from w2v_v3184.common import seed_for


def annotate(tickets, rows, seed):
    tickets = [dict(t) for t in tickets]
    for side in ('A', 'B'):
        for language in ('en', 'zh'):
            members = [t for t in tickets if t['meta_side'] == side and rows[t['index']]['language'] == language]
            ordered = sorted(members, key=lambda t: seed_for(seed, 'v55-anchor',
                t['epoch'], t['step'], t['canonical_source']))
            real = [t for t in ordered if t['kind'] == 'offline' and rows[t['index']]['label'] == 1]
            fake = [t for t in ordered if rows[t['index']]['label'] == 0]
            noisy_fake = [t for t in fake if t['kind'] == 'offline']
            if not real:
                raise ValueError('V5.1 batch has no processed Real in a language/side')
            # The minimum eight-source final partial batch can have only two
            # Fake sources per cell. Keep CE/PCL and explicitly record the skip.
            if len(fake) < 3 or not noisy_fake:
                real[0]['hp_skip'] = 'fewer_than_three_distinct_fake_sources'
                continue
            anchor, noisy = real[0], noisy_fake[0]
            others = [t for t in fake if t is not noisy]
            # Prefer matched Clean-RTC Fake; official Online is a clean-domain
            # fallback when a side contains only two Offline Fake sources.
            others.sort(key=lambda t: (t['kind'] != 'offline',
                seed_for(seed, 'v55-negative', t['occurrence'])))
            name = f"{side}/{language}/{anchor['epoch']}/{anchor['step']}"
            for t, role in [(anchor, 'anchor'), (noisy, 'negative_noisy')] + [
                    (t, 'negative_clean' if t['kind'] == 'offline' else 'negative_online') for t in others[:2]]:
                t.update(hp_group=name, hp_member_role=role)
    return tickets


class Plan(ParentPlan):
    def batches(self, epoch):
        for tickets in super().batches(epoch):
            yield annotate(tickets, self.rows, self.seed)

    def coverage(self, epoch):
        value = super().coverage(epoch)
        counts = Counter()
        for tickets in self.batches(epoch):
            for t in tickets:
                if t.get('hp_member_role') == 'anchor':
                    counts[t['meta_side'] + '/' + self.rows[t['index']]['language']] += 1
                if t.get('hp_skip'):
                    counts['skipped/' + t['hp_skip']] += 1
        value['hard_positive'] = dict(groups=dict(counts),
            policy='one rotating Real anchor/language/side/update; 2 processed positives; 3 distinct Fake sources',
            ce_exposure_tickets_unchanged=True, negatives='two Clean-RTC/Online Fake + one Noisy-RTC Fake')
        return value
