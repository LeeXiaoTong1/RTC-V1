"""Resume-safe aggregate evidence for losses, errors and margin activation."""


def accumulate(totals, stats):
    hp = stats['hard_positive']
    for side in hp['sides'].values():
        for record in side['records']:
            for name in ('guidance_violations', 'triplet_violations', 'guidance_raw', 'triplet_raw'):
                totals['hp/' + name] += record[name]
            totals['hp/hardest_' + str(record['hardest'])] += 1
            totals['hp/feature_norm_sum'] += sum(record['feature_norms'])
            totals['hp/anchor_positive_sum'] += sum(record['anchor_positive'])
            totals['hp/anchor_negative_sum'] += sum(record['anchor_negative'])
            totals['hp/hard_negative_sum'] += sum(record['hard_negative'])
            correct = record['correct']
            totals['hp/anchor_correct'] += int(correct[0])
            totals['hp/positive_errors'] += sum(not flag for flag in correct[1:3])
            totals['hp/anchor_correct_positive_wrong'] += sum(correct[0] and not flag for flag in correct[1:3])
            totals['hp/negative_errors'] += sum(not flag for flag in correct[3:])
        for reason, count in side['skips'].items():
            totals['hp/skipped/' + reason] += count
    for cell, values in stats['cells'].items():
        for name in ('views', 'correct', 'raw_ce_sum'):
            totals['cell/' + cell + '/' + name] += values[name]


def summary(totals):
    groups = int(totals.get('hp_groups', 0))
    div = max(1, groups)
    cells = {}
    for key, value in totals.items():
        if key.startswith('cell/'):
            cell, field = key[5:].rsplit('/', 1)
            cells.setdefault(cell, {})[field] = value
    for item in cells.values():
        item['error_rate'] = 1 - item['correct'] / max(1, item['views'])
        item['mean_raw_ce'] = item['raw_ce_sum'] / max(1, item['views'])
    return dict(groups=groups,
        guidance_violation_fraction=totals.get('hp/guidance_violations', 0) / (3*div),
        triplet_violation_fraction=totals.get('hp/triplet_violations', 0) / (6*div),
        mean_raw_guidance=totals.get('hp/guidance_raw', 0) / div,
        mean_raw_triplet=totals.get('hp/triplet_raw', 0) / div,
        mean_feature_norm=totals.get('hp/feature_norm_sum', 0) / (6*div),
        mean_anchor_positive_distance=totals.get('hp/anchor_positive_sum', 0) / (2*div),
        mean_anchor_negative_distance=totals.get('hp/anchor_negative_sum', 0) / (3*div),
        mean_hard_negative_distance=totals.get('hp/hard_negative_sum', 0) / (3*div),
        hardest_counts=[int(totals.get('hp/hardest_' + str(i), 0)) for i in (0,1)],
        anchor_correct=int(totals.get('hp/anchor_correct', 0)),
        processed_real_errors=int(totals.get('hp/positive_errors', 0)),
        clean_correct_processed_wrong=int(totals.get('hp/anchor_correct_positive_wrong', 0)),
        fake_errors=int(totals.get('hp/negative_errors', 0)),
        skips={k[len('hp/skipped/'):]: v for k,v in totals.items() if k.startswith('hp/skipped/')},
        classification_cells=cells,
        scope='Training only; selected metric groups are not the full population or an unseen test')
