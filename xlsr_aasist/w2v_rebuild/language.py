"""Class-conditional language budgets; no sampling or augmentation changes.

Lookup axes are [authenticity, language]: fake=0, real=1; en=0, zh=1.
Each branch must construct its budget from its own actual sampling pool.
"""
import math
import numbers
from pathlib import PurePosixPath

import torch


def language_id(source: str) -> int:
    """Parse an exact en/zh directory component, never a filename substring."""
    if not isinstance(source, str) or not source or '\x00' in source:
        raise ValueError('Expected a nonempty source file ID')
    normalized = source.replace('\\', '/')
    if normalized.endswith('/'):
        raise ValueError('Expected a source file ID, not a directory')
    parts = PurePosixPath(normalized).parts
    if '..' in parts:
        raise ValueError('Source file ID must not contain parent traversal')
    found = {p.lower() for p in parts[:-1]} & {'en', 'zh'}
    if len(found) != 1:
        raise ValueError(f'Unknown or ambiguous en/zh source directory: {source!r}')
    return 0 if 'en' in found else 1


def _binary_cpu(values, name):
    if not isinstance(values, torch.Tensor) or values.device.type != 'cpu':
        raise ValueError(f'{name} must be a CPU tensor')
    if values.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f'{name} must contain integer IDs')
    if not bool(((values == 0) | (values == 1)).all()):
        raise ValueError(f'{name} must contain only 0 and 1')
    return values.long()


class LanguageBudget:
    """Allocate an expected coefficient share within each authenticity class.

    If the pool's English fraction is p and the requested share is q, English
    receives q/p and Chinese (1-q)/(1-p). Thus each class's expected multiplier
    remains one. These are coefficient shares, not realized loss/gradient shares.
    """
    def __init__(self, ids, labels, en_real=.35, en_fake=.40):
        sources = list(ids)
        if not sources or len(sources) != len(labels):
            raise ValueError('Language budget requires nonempty matching IDs and labels')
        canonical = [str(PurePosixPath(x.replace('\\', '/'))) if isinstance(x, str) else x
                     for x in sources]
        if any(not isinstance(x, str) for x in canonical):
            raise ValueError('Language budget source IDs must be strings')
        if len(canonical) != len(set(canonical)):
            raise ValueError('Duplicate source IDs in language budget')
        targets = [en_fake, en_real]
        if any(not isinstance(q, numbers.Real) or isinstance(q, bool)
               or not math.isfinite(q) or not 0 < q < 1 for q in targets):
            raise ValueError('English budget targets must be finite and strictly between 0 and 1')
        try:
            y = torch.as_tensor(labels)
        except (TypeError, ValueError) as exc:
            raise ValueError('Language budget labels must be integer IDs') from exc
        y = _binary_cpu(y, 'Labels')
        if y.ndim != 1 or y.shape != (len(sources),):
            raise ValueError('Language budget labels must be a one-dimensional sequence')
        languages = torch.tensor([language_id(x) for x in sources], dtype=torch.long)
        count = torch.bincount(2*y + languages, minlength=4).reshape(2, 2)
        if not bool((count > 0).all()):
            raise ValueError('Each sampling pool requires nonempty en-fake, zh-fake, en-real and zh-real groups')
        fraction = count.double()[:, 0] / count.sum(1).double()
        q = torch.tensor(targets, dtype=torch.float64)
        self.lookup = torch.stack((q/fraction, (1-q)/(1-fraction)), dim=1).float()
        if not bool((torch.isfinite(self.lookup) & (self.lookup > 0)).all()):
            raise ValueError('Language budget produced invalid coefficients')
        self.report = {
            'source_count': len(sources),
            'axes': {'labels': ['fake', 'real'], 'languages': ['en', 'zh']},
            'counts': count.tolist(),
            'en_targets': [float(x) for x in targets],
            'en_fractions': fraction.tolist(),
            'coefficients': self.lookup.tolist(),
            'normalization': 'class-conditional expected multiplier 1; original CE denominator retained',
        }

    def coefficients(self, labels, languages):
        y = _binary_cpu(labels, 'Labels')
        lang = _binary_cpu(languages, 'Languages')
        if y.shape != lang.shape:
            raise ValueError('Labels and languages must have matching shapes')
        return self.lookup[y, lang]
