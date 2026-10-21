"""Summarize observed training errors; never load audio or run a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
from w2v_v53.common import ROOT, atomic_json
from . import VERSION


def events(path, maximum=None):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer may not have finished the last line.
            if maximum is None or row['update'] <= maximum:
                yield row


def summarize(stage, maximum=None, save=True):
    stage = Path(stage)
    groups, sources, worst = {}, {}, []
    reasons, outcomes = Counter(), Counter()
    count = invalid = omitted = recorded = errors = views = 0
    first = last = None
    total_ce = total_excess = 0.
    for event in events(stage/'hard_examples.jsonl', maximum):
        if not event.get('validated'):
            invalid += 1
            continue
        count += 1
        first = event['update'] if first is None else min(first, event['update'])
        last = event['update'] if last is None else max(last, event['update'])
        total_ce += event['classification_loss']
        total_excess += event['weighted_excess']
        errors += event['errors']
        views += event['views']
        omitted += event.get('omitted_candidates', 0)
        worst.append(dict(update=event['update'], ce=event['classification_loss'],
                          errors=event['errors'], max_raw_ce=event['max_raw_ce'],
                          recorded_excess_fraction=event.get('recorded_excess_fraction', 0.)))
        worst.sort(key=lambda x: x['ce'], reverse=True)
        del worst[50:]
        for key, cell in event['cells'].items():
            target = groups.setdefault(key, dict(views=0, correct=0, objective=0., raw_ce_sum=0.))
            for field in target:
                target[field] += cell[field]
        for example in event.get('examples', []):
            recorded += 1
            reasons[example['reason']] += 1
            if 'pair_outcome' in example:
                outcomes[example['pair_outcome']] += 1
            identity = str(example.get('canonical_source') or example.get('source_id') or example.get('id'))
            key = (identity, example['language'], example['true_class'], example['view'])
            target = sources.setdefault(key, dict(recorded_visits=0, recorded_errors=0,
                first_update=event['update'], last_update=event['update'], maximum_weighted_excess=-1.))
            target['recorded_visits'] += 1
            target['recorded_errors'] += int(not example['correct'])
            target['last_update'] = event['update']
            if example['weighted_excess'] > target['maximum_weighted_excess']:
                target['maximum_weighted_excess'] = example['weighted_excess']
                target['worst'] = dict(update=event['update'], **example)
    for cell in groups.values():
        cell['errors'] = cell['views'] - cell['correct']
        cell['accuracy'] = cell['correct']/max(1,cell['views'])
        cell['raw_ce_mean'] = cell['raw_ce_sum']/max(1,cell['views'])
        cell['mean_step_ce_contribution'] = cell['objective']/max(1,count)
    ranked = sorted(sources.values(), key=lambda x: x['maximum_weighted_excess'], reverse=True)
    result = dict(schema=VERSION, scope='observed training updates only; not held-out evaluation',
        first_observed_update=first, last_observed_update=last, observed_updates=count,
        invalid_observations=invalid, classification_views=views, classification_errors=errors,
        mean_classification_loss=total_ce/max(1,count), mean_ce_excess=total_excess/max(1,count),
        recorded_hard_views=recorded, omitted_hard_candidates=omitted,
        recorded_unique_source_view_groups=len(sources), groups=groups,
        recorded_reasons=dict(reasons), recorded_pair_outcomes=dict(outcomes),
        worst_updates=worst, top_sources=ranked[:100],
        notes=['Observation starts after installation/resume; earlier sample identities are unavailable.',
               'Each update records at most 8 qualifying views, ranked by weighted CE excess.',
               'Group counts include all classified views; source/pair counts include recorded candidates only.',
               'A pair may appear twice if both of its views qualify; pair outcomes are view counts.',
               'Training-time predictions include the existing dropout; no separate evaluation pass.',
               'Full recorded examples and recipes remain in hard_examples.jsonl.'])
    # Keep this summary under the existing 2 MiB report limit. Truncation affects
    # displayed top examples only; all aggregate counts and the local JSONL remain.
    while len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > 1500000 and result['top_sources']:
        result['top_sources'].pop()
    result['included_top_sources'] = len(result['top_sources'])
    if save:
        atomic_json(stage/'hard_examples_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='V5.3 run root or final/screen stage; defaults to latest V5.3')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    run = Path(args.run) if args.run else Path((ROOT/'exp'/'.latest_v53_run').read_text().strip())
    stage = run if (run/'hard_examples.jsonl').exists() else run/'final'
    if not (stage/'hard_examples.jsonl').exists():
        stage = run/'screen'
    if not (stage/'hard_examples.jsonl').exists():
        raise SystemExit('No observations yet. Resume through run_w2v_v53.sh first.')
    # Live inspection never competes with the training writer for the summary file.
    result = summarize(stage, save=False)
    print(f"Observed updates {result['first_observed_update']}..{result['last_observed_update']}; "
          f"errors={result['classification_errors']}/{result['classification_views']}; "
          f"mean CE={result['mean_classification_loss']:.6f}")
    for item in result['top_sources'][:args.top]:
        row = item['worst']
        print(f"update={row['update']} {row['view']}/{row['language']}/{row['true_class']} "
              f"p_true={row['p_true']:.5f} raw_CE={row['raw_ce']:.4f} "
              f"weighted_excess={row['weighted_excess']:.5f} {row['reason']} "
              f"pair={row.get('pair_outcome', 'unpaired')} id={row.get('id', row.get('source_id'))}")
        print('  source=' + str(row.get('path', 'not provided')))


if __name__ == '__main__':
    main()
