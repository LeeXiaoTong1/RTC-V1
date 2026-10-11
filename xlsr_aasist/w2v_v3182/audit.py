"""Input coverage and realized objective budgets, without loading the encoder."""
from collections import Counter
from .sampling import Plan


def describe(rows, seed=31801, stream_sources=16, power=.5):
    plan = Plan(rows, stream_sources, seed, power)
    counts = Counter(f'{r["condition"]}/{r["language"]}/{r["label"]}' for r in rows)
    return dict(version='3.18.2', official_counts=dict(counts), labels={'0':'fake','1':'real'},
                v3182=plan.coverage(0),
                note='Paired views share one verified source. Real/fake view counts differ; their CE and anchor masses are equal.')
