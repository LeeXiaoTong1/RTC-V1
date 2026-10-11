"""Actual exposure and loss budgets; Offline is only a noisy waveform source."""
from collections import Counter
from .sampling import Plan


def describe(rows,seed=31801,stream_sources=16,quotas=(4,2,7,3)):
    counts=Counter((r['condition'],r['language'],r['label']) for r in rows)
    plan=Plan(rows,stream_sources,seed,quotas)
    return dict(official_counts={'/'.join(map(str,k)):v for k,v in sorted(counts.items())},
        v3183=plan.coverage(0),raw_offline_classification=False,contrastive=False,
        note='Independent Online and noisy-source streams; equal loss mass does not imply equal draw counts.')
