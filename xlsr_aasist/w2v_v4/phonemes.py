"""Bind original CTC intervals to the final source registry and BERT time grid."""
from collections import Counter
from pathlib import Path
from w2v_v3187.phonemes import PhonemeCache, bind_manifest, ALGORITHM
from w2v_v3186.phonemes import match_segments
from .pcl import valid_matches
from .common import read_json, atomic_json, fingerprint


def build_pairs(rows, manifest, output):
    binding=bind_manifest(manifest,rows)
    owner=dict(schema='v4_dual_phone_pairs_v2',manifest=binding,rows=fingerprint(rows),
               geometry=dict(receptive=560,stride=320,center=280))
    output=Path(output)
    if output.is_file():
        result=read_json(output)
        if result['owner']!=owner or result['sha256']!=fingerprint((result['official'],result['noise'])):
            raise ValueError('PCL pair cache changed')
        return result
    cache=PhonemeCache(manifest,rows,binding=binding);by_id={r['id']:r for r in rows}
    official={};noise={};counts=Counter()
    for row in sorted(rows,key=lambda r:r['id']):
        if row['split']!='train':raise ValueError('Only Train originals may supply PCL')
        segments=cache.get(row)['segments']
        if row['condition']=='offline':
            spans=[[p['start_sample'],p['end_sample']]*2 for p in segments if p.get('posterior',0)>=ALGORITHM['min_token_posterior']]
            spans=valid_matches(spans,row['output_samples'],row['output_samples'])
            if spans:noise[row['id']]=dict(offline=row['id'],source=row['canonical_source'],matches=spans)
            counts['noise/'+row['language']+'/'+str(row['label'])]+=bool(spans)
        elif row['condition']=='online':
            off=by_id.get(row['source_id'])
            if off is None:counts['unpaired_online']+=1;continue
            if off['condition']!='offline' or any(off[k]!=row[k] for k in ('canonical_source','language','label','split')):
                raise ValueError('Official PCL source identity differs')
            a=cache.get(off)['segments'];spans=[]
            for i,j in match_segments(a,segments):
                if min(a[i].get('posterior',0),segments[j].get('posterior',0))<ALGORITHM['min_token_posterior']:continue
                spans.append([a[i]['start_sample'],a[i]['end_sample'],segments[j]['start_sample'],segments[j]['end_sample']])
            spans=valid_matches(spans,off['output_samples'],row['output_samples'])
            if spans:official[row['id']]=dict(offline=off['id'],online=row['id'],source=row['canonical_source'],matches=spans)
            counts['official/'+row['language']+'/'+str(row['label'])]+=bool(spans)
    if not official or not noise:raise ValueError('No usable original phonemes for one of the requested PCL branches')
    result=dict(owner=owner,official=official,noise=noise,coverage=dict(counts),sha256=fingerprint((official,noise)))
    atomic_json(output,result);return result
