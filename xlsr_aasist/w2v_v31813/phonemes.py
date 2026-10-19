"""Verified original Train references; no synthetic or cross-label alignment."""
from pathlib import Path
from collections import Counter
import numpy as np
from w2v_v3187.phonemes import PhonemeCache, bind_manifest, ALGORITHM
from w2v_v3186.phonemes import match_segments
from w2v_v31810.common import atomic_json, digest, fingerprint, read_json
from .objectives import frame_span


def build_pairs(rows, manifest, output):
    binding=bind_manifest(manifest,rows)
    owner=dict(schema='v31813_original_phone_pairs_v1',manifest=binding,
               rows=fingerprint(rows),geometry=dict(receptive=400,stride=320,center=200))
    output=Path(output)
    if output.is_file():
        result=read_json(output)
        if result['owner']!=owner or result['sha256']!=fingerprint(result['pairs']):
            raise ValueError('Phoneme pair cache changed')
        return result
    cache=PhonemeCache(manifest,rows,binding=binding);by_id={r['id']:r for r in rows}
    pairs={};counts=Counter()
    for on in sorted(rows,key=lambda r:r['id']):
        if on['condition']!='online':continue
        off=by_id.get(on['source_id'])
        if off is None:counts['unpaired']+=1;continue
        if off['condition']!='offline' or any(off[k]!=on[k] for k in ('canonical_source','language','label','split')) or on['split']!='train':
            raise ValueError('PCL needs verified same-source, same-label Train originals')
        a=cache.get(off)['segments'];b=cache.get(on)['segments'];matches=[]
        for i,j in match_segments(a,b):
            phones=(a[i],b[j]);span=[]
            for phone,row in zip(phones,(off,on)):
                x,y=phone['start_sample'],phone['end_sample']
                l,r=frame_span(x,y,row['output_samples'])
                if r<=l or phone.get('posterior',0)<ALGORITHM['min_token_posterior']:break
                span.extend((x,y))
            if len(span)==4:matches.append(span)
        if matches:
            pairs[on['id']]=dict(offline=off['id'],online=on['id'],source=on['canonical_source'],matches=matches)
            counts[on['language']+'/'+str(on['label'])]+=1
        else:counts['no_valid_phones']+=1
    if not pairs:raise ValueError('No reliable original Offline/Online phoneme pairs')
    result=dict(owner=owner,pairs=pairs,sha256=fingerprint(pairs),coverage=dict(counts))
    atomic_json(output,result);return result


def nonsilent(matches, offline, online):
    return [m for m in matches if all(float(np.mean(np.asarray(w[a:b],np.float64)**2))>=1e-10
        for w,a,b in ((offline,m[0],m[1]),(online,m[2],m[3])))]
