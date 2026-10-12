"""C3 preserved; source count and supervision mass are separate contracts."""
from w2v_v318.objective import risk, unit_loss as legacy_loss
from .common import GROUPS
from torch.nn import functional as F
import torch


def unit_loss(logits, rows, quotas=(4,2,7,3), coefficient=.25, smoothing=.02):
    group=(rows[0]['language'],rows[0]['label'])
    quota=quotas[GROUPS.index(group)]
    # Legacy scale is .5 / stream_sources. Passing 4*q yields .125/q,
    # i.e. 25% per group within each 50%-mass stream.
    return legacy_loss(logits,rows,4*quota,coefficient,smoothing)


def micro_losses(logits,units,quotas,coefficient,smoothing):
    rows=[r for unit in units for r in unit]
    labels=torch.tensor([r['label'] for r in rows],device=logits.device)
    logp=F.log_softmax(logits.float(),dim=-1)
    raw=-logp.gather(1,labels[:,None]).squeeze(1)
    ce=(1-smoothing)*raw-smoothing*logp.mean(-1)
    result=[];offset=0
    for unit in units:
        n=len(unit);part=ce[offset:offset+n]
        value=part[0] if n==1 else risk(part[0],part[1],coefficient)
        g=(unit[0]['language'],unit[0]['label'])
        result.append((value*(.125/quotas[GROUPS.index(g)]),raw[offset:offset+n]))
        offset+=n
    return result
