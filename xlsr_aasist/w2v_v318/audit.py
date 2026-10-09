"""Audit official input counts, old V3.17 exposure, and new stream budgets."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .common import GROUPS,atomic_json
from .records import data_inputs
from .sampling import Plan


def describe(rows,seed=31801):
    counts=Counter((r['condition'],r['language'],r['label']) for r in rows)
    pools={g:[r for r in rows if r['condition']=='offline' and (r['language'],r['label'])==g] for g in GROUPS}
    sizes=np.array([len(pools[g]) for g in GROUPS],float)**.5;target=sizes/sizes.sum()*16
    q=np.maximum(1,np.floor(target).astype(int))
    while q.sum()<16:q[int(np.argmax(target-q))]+=1
    while q.sum()>16:q[int(np.argmax(np.where(q>1,q-target,-np.inf))) ]-=1
    online={r['source_id'] for r in rows if r['condition']=='online'}
    old={};condition_mass=Counter()
    for group,quota in zip(GROUPS,q):
        items=pools[group];missing=sum(r['source_id'] not in online for r in items)
        f=missing/len(items)
        old[f'{group[0]}/{group[1]}']=dict(available=len(items),quota=int(quota),ce_mass=.25,
            missing_online=missing,expected_views_per_update=float(quota*(3-f)))
        condition_mass.update(offline=.25*(.1+.1*f),online=.25*.5*(1-f),noisy=.25*(.4+.4*f))
    new=Plan(rows,16,seed)
    return dict(official_counts={'/'.join(map(str,k)):v for k,v in sorted(counts.items())},
        labels={'0':'fake','1':'real'},v317=dict(groups=old,
            expected_condition_ce_mass=dict(condition_mass),class_ce_mass={'fake':.5,'real':.5},
            note='Expected budget from canonical source inventory; exact realized draw counts require that run coverage/steps logs'),
        v318=new.coverage(0),
        interpretation='Metadata en/zh is not a guarantee of monolingual content. Balanced objective does not guarantee equal difficulty or generalization.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-run',required=True);p.add_argument('--out')
    args=p.parse_args();_,rows,_,_=data_inputs(args.data_run);value=describe(rows)
    out=Path(args.out) if args.out else Path(args.data_run).parent/'v318_input_audit.json'
    atomic_json(out,value)
    for key,count in value['official_counts'].items():print(f'{key}: {count}')
    print('V317: fake/real CE budget already 50/50; number of source draws differs. See exact group quotas in audit.')
    print('V318: Online 16 + Offline-source Noisy pairs 16x2; class counts 50/50; stream loss 50/50; no raw Offline classification.')
    print('INPUT_AUDIT='+str(out.resolve()))


if __name__=='__main__':main()
