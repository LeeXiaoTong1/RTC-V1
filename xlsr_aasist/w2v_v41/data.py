"""Same augmentation waveforms; official and simulated PCL preserve identity/time."""
import numpy as np
from .wave_source import Waves as BaseWaves
from .features import PreparedFeatures
from w2v_v3189.data import prepared
from w2v_v3186.augment import estimate_delay, reliable_spans
from w2v_v3189.activity import mask_pairs
from .pcl import valid_matches


def nonsilent(matches,left,right):
    return [m for m in matches if all(b>a and np.mean(w[a:b].astype(np.float64)**2)>=1e-10
        for w,a,b in ((left,int(m[0]),int(m[1])),(right,int(m[2]),int(m[3]))))]


def simulated_matches(matches,original,base,noisy,time_map,metadata):
    delay=estimate_delay(base,noisy)
    coverage=dict(candidates=len(matches),delay=delay,activity_affected=0,valid=0)
    if not delay['valid'] or not metadata.get('pair_eligible',False):return [],coverage
    scale=float(time_map['scale']);offset=float(time_map['offset']);shift=float(delay['offset'])
    candidates=[];base_spans=[]
    for a,b,_,_ in nonsilent(matches,original,original):
        c,d=a*scale+offset,b*scale+offset
        if 0<=c<d<=len(base) and 0<=c+shift<d+shift<=len(noisy):
            candidates.append([a,b,c+shift,d+shift]);base_spans.append([c,d])
    kept=reliable_spans(base,noisy,base_spans,0.,shift)
    candidates=[m for m,ok in zip(candidates,kept) if ok]
    # Activity metadata is before the codec. Guard both original and delayed
    # positions; uncertainty must never invent speech support in a closed gate.
    erased=metadata.get('erased_spans',metadata.get('rtc_activity',{}).get('suppressed_spans',[]))
    spans=list(erased)+[[a+shift,b+shift] for a,b in erased]
    safe,affected=mask_pairs(candidates,[],spans,guard=280)
    safe=valid_matches(safe,len(original),len(noisy))
    coverage.update(activity_affected=affected,valid=len(safe),reliability_rejected=len(kept)-sum(kept))
    return safe,coverage


class Waves(PreparedFeatures, BaseWaves):
    def __init__(self,rows,cfg,split='train'):
        super().__init__(rows,cfg,split);self.by_id={r['id']:r for r in rows}

    def __getitem__(self,ticket):
        return self.attach_features(self._item(ticket))

    def _item(self,ticket):
        out=super().__getitem__(ticket)
        if isinstance(ticket,int):return out
        row=out[0]
        branch='official' if 'pcl_official' in ticket else 'noise' if 'pcl_noise' in ticket else None
        if branch is None:return out
        if self.split!='train':raise ValueError('PCL is Train-only')
        p=ticket['pcl_'+branch];off=self.by_id[p['offline']]
        if p['source']!=row['canonical_source'] or any(off[k]!=row[k] for k in ('canonical_source','language','label','split')):
            raise ValueError('PCL reference has a different source/class')
        original=self.original(off);reference=prepared(off,original,role='offline_ref')
        if branch=='official':
            on=self.by_id[p['online']];received=self.original(on)
            if on['id']!=row['id'] or any(on[k]!=off[k] for k in ('canonical_source','language','label','split')):
                raise ValueError('Invalid official PCL reference')
            matches=valid_matches(nonsilent(p['matches'],original,received),len(original),len(received))
            target=prepared(on,received,role='online_ref')
            reuse=np.array_equal(row['wave'],target['wave'])
            coverage=dict(candidates=len(p['matches']),valid=len(matches))
        else:
            if row['role']!='noisy' or off['id']!=row['id']:raise ValueError('Simulated PCL requires the classified Offline source')
            meta=row.get('base_augmentation')
            if meta:
                base=row.pop('_base_wave');mapping=meta['time_map']
            else:base=original;mapping=dict(scale=1.,offset=0.)
            received=row['wave'][:row['original_samples']]
            matches,coverage=simulated_matches(p['matches'],original,base,received,mapping,row['augmentation'])
            target=None;reuse=True
        row['pcl_coverage']=dict(branch=branch,**coverage)
        row.pop('_base_wave',None)
        if matches:row['pcl']=dict(kind=branch,reference=reference,received=target,reuse_received=reuse,matches=matches)
        return out
