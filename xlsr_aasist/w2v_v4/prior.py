"""Train-only acoustic-unit histograms and cross-source opposite-label neighbors."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import os
import numpy as np
from w2v_v31810.common import read_json,atomic_json,fingerprint,digest
from w2v_v31810.identity import verify_registry
from w2v_v3187.prepare_phonemes import _read_verified_wave
from .source import resolve_data_run as resolve_parent


def originals(rows):
    groups=defaultdict(list)
    for row in rows:
        if row.get('split')!='train' or not row.get('canonical_source'):
            raise ValueError('XEUS accepts only registered original Train sources')
        groups[row['canonical_source']].append(row)
    out=[]
    for source,members in sorted(groups.items()):
        if len({(r['language'],r['label']) for r in members})!=1:raise ValueError('Conflicting source labels')
        offline=sorted((r for r in members if r['condition']=='offline'),key=lambda r:r['id'])
        if not offline:raise ValueError('Source lacks its original Offline reference: '+source)
        out.append(offline[0])
    return out


def normalize(features):
    return features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)


def random_neighbors(rows,seed):
    """Optional negative control: same language/duration/source rules, no XEUS."""
    rows=originals(rows);rng=np.random.default_rng(seed);result={}
    groups={lang:[r for r in rows if r['language']==lang and r['label']==0] for lang in ('en','zh')}
    lengths={lang:np.array([r['output_samples'] for r in group]) for lang,group in groups.items()}
    for r in rows:
        if r['label']!=1:continue
        lang=r['language'];n=r['output_samples'];indices=np.flatnonzero((lengths[lang]>=n/2)&(lengths[lang]<=n*2))
        eligible=[groups[lang][i]['canonical_source'] for i in indices]
        result[r['canonical_source']]=[dict(source=str(s),similarity=None) for s in rng.choice(eligible,min(16,len(eligible)),replace=False)]
    return result


def histogram(features,centers):
    counts=np.zeros(len(centers),np.float64)
    for start in range(0,len(features),2048):
        x=normalize(features[start:start+2048])
        dist=(x*x).sum(1)[:,None]+(centers*centers).sum(1)[None]-2*x@centers.T
        counts+=np.bincount(dist.argmin(1),minlength=len(centers))
    return np.sqrt(counts/max(1,counts.sum())).astype(np.float32)


def neighbors(rows,hist,topk=16,minimum=.8,ratio=2.):
    result={};summary={}
    for lang in ('en','zh'):
        real=[i for i,r in enumerate(rows) if r['language']==lang and r['label']==1 and hist[i].any()]
        fake=[i for i,r in enumerate(rows) if r['language']==lang and r['label']==0 and hist[i].any()]
        if not real or not fake:raise ValueError('Both Train classes need valid XEUS representations in '+lang)
        duration=np.array([rows[i]['output_samples'] for i in fake])
        chosen=[];random_sim=[]
        rng=np.random.default_rng(3181301)
        for begin in range(0,len(real),256):
            ids=real[begin:begin+256];scores=hist[ids]@hist[fake].T
            for i,score in zip(ids,scores):
                n=rows[i]['output_samples'];valid=(duration>=n/ratio)&(duration<=n*ratio)&(score>=minimum)
                order=np.flatnonzero(valid);order=order[np.argsort(-score[order],kind='stable')[:topk]]
                matches=[dict(source=rows[fake[j]]['canonical_source'],similarity=float(score[j])) for j in order
                         if rows[fake[j]]['canonical_source']!=rows[i]['canonical_source']]
                result[rows[i]['canonical_source']]=matches
                if matches:chosen.append(matches[0]['similarity'])
                random_sim.append(float(score[rng.integers(len(score))]))
        summary[lang]=dict(real_sources=len(real),fake_sources=len(fake),matched_real=len(chosen),
            coverage=len(chosen)/len(real),mean_nearest=float(np.mean(chosen)) if chosen else None,
            mean_random=float(np.mean(random_sim)))
        if not chosen:raise ValueError('No qualifying XEUS neighbors in '+lang+'; inspect Train histograms, do not force unrelated pairs')
    return result,summary


def quality_probe(rows,hist,seed=3181301):
    """Train-only warning signals, not evidence of speaker invariance."""
    rng=np.random.default_rng(seed);result={}
    for lang in ('en','zh'):
        pools=[[i for i,r in enumerate(rows) if r['language']==lang and r['label']==label and hist[i].any()] for label in (0,1)]
        if not all(pools):continue
        if min(map(len,pools))<2:
            result[lang]=dict(status='insufficient_sources_for_leave_one_out',fake_sources=len(pools[0]),real_sources=len(pools[1]))
            continue
        n=min(1024,*map(len,pools));candidates=np.concatenate([rng.choice(p,n,replace=False) for p in pools])
        anchors=np.concatenate([rng.choice(p,min(128,len(p)),replace=False) for p in pools])
        scores=hist[anchors]@hist[candidates].T
        same=[];opposite=[];correct=[]
        for i,values in zip(anchors,scores):
            values[candidates==i]=-np.inf
            labels=np.array([rows[j]['label'] for j in candidates]);mask=labels==rows[i]['label']
            same.append(float(values[mask].max()));opposite.append(float(values[~mask].max()))
            correct.append(bool(labels[values.argmax()]==rows[i]['label']))
        result[lang]=dict(balanced_candidate_count=len(candidates),anchors=len(anchors),
            nearest_same_label_fraction=float(np.mean(correct)),mean_same_class_nearest=float(np.mean(same)),
            mean_opposite_class_nearest=float(np.mean(opposite)),
            interpretation='High same-label purity can indicate a class/recording shortcut; speaker/device identity is unknown, not proven invariant')
    return result


def save_array(path,**arrays):
    path=Path(path);tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def prepare(parent,output,teacher,proof,seed=3181301,reuse_run=None,reuse_manifest=None):
    parent=Path(parent);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=read_json(parent/'train_rows.json');cfg=read_json(parent/'config.json')
    verify_registry(rows,read_json(parent/'source_registry.json'),cfg,read_json(parent/'validation_protocol.json'))
    sources=originals(rows)
    owner=dict(schema='v31813_xeus_train_units_v1',rows=fingerprint(rows),registry=cfg['source_registry'],
        teacher=proof,seed=seed,clusters=128,sources_per_cell=256,frames_per_source=32,
        topk=16,min_similarity=.8,max_duration_ratio=2.,chunk_samples=320000,context_samples=16000,
        reuse_manifest_sha256=digest(reuse_manifest) if reuse_manifest else None)
    if (output/'owner.json').exists() and read_json(output/'owner.json')!=owner:raise ValueError('XEUS cache inventory/config changed')
    atomic_json(output/'owner.json',owner)
    if (output/'manifest.json').exists():
        return bind(output/'manifest.json',rows)
    from .reuse import teacher_cache
    reused=teacher_cache(rows,proof,reuse_run,reuse_manifest)
    centers_path=output/'centers.npz'
    if centers_path.exists():
        receipt=read_json(output/'centers.json')
        if receipt['sha256']!=digest(centers_path) or receipt['owner']!=fingerprint(owner):raise ValueError('XEUS centers changed')
        centers=np.load(centers_path,allow_pickle=False)['centers']
    elif reused is not None:
        centers=reused['centers'];save_array(centers_path,centers=centers)
        atomic_json(output/'centers.json',dict(sha256=digest(centers_path),owner=fingerprint(owner),reused_from=reused['manifest_sha256']))
    else:
        from sklearn.cluster import MiniBatchKMeans
        rng=np.random.default_rng(seed);pool=[];counts={}
        for lang in ('en','zh'):
            for label in (0,1):
                group=[r for r in sources if r['language']==lang and r['label']==label and r['output_samples']>=400]
                rng.shuffle(group);counts[lang+'/'+str(label)]=min(256,len(group))
                for r in group[:256]:
                    x=teacher(_read_verified_wave(r))
                    if len(x):pool.append(normalize(x[rng.choice(len(x),min(32,len(x)),replace=False)]))
                print('XEUS codebook samples '+lang+'/'+str(label),flush=True)
        frames=np.concatenate(pool)
        if len(frames)<128:raise ValueError('Insufficient balanced Train frames for 128 acoustic units')
        km=MiniBatchKMeans(128,random_state=seed,n_init=3,batch_size=4096,max_iter=100).fit(frames)
        centers=km.cluster_centers_.astype(np.float32);save_array(centers_path,centers=centers)
        atomic_json(output/'centers.json',dict(sha256=digest(centers_path),owner=fingerprint(owner),sources=counts,frames=len(frames)))
        del frames,pool
    shards=[];arrays=[]
    for start in range(0,len(sources),64):
        part=sources[start:start+64];path=output/f'hist_{start:07d}.npz';meta=path.with_suffix('.json')
        identity=fingerprint([dict(id=r['id'],sha=r['audio_sha256']) for r in part])
        if path.exists() and meta.exists():
            expected=read_json(meta)
            if expected!=dict(owner=fingerprint(owner),identity=identity,sha256=digest(path)):raise ValueError('XEUS histogram shard changed')
            # Re-read source hashes on a resumed preparation to catch stale audio.
            for r in part:
                if digest(r['audio'])!=r['audio_sha256']:raise ValueError('Original Train audio changed')
            h=np.load(path,allow_pickle=False)['histograms']
        else:
            values=[]
            for r in part:
                key=(r['canonical_source'],r['id'],r['audio_sha256'])
                if reused is not None and key in reused['histograms']:
                    if digest(r['audio'])!=r['audio_sha256']:raise ValueError('Original Train bytes changed')
                    values.append(reused['histograms'][key])
                else:values.append(histogram(teacher(_read_verified_wave(r)),centers))
            h=np.stack(values)
            save_array(path,histograms=h)
            atomic_json(meta,dict(owner=fingerprint(owner),identity=identity,sha256=digest(path)))
        arrays.append(h);shards.append(dict(file=path.name,sha256=digest(path)))
        print(f'XEUS Train sources {min(start+64,len(sources))}/{len(sources)}',flush=True)
    h=np.concatenate(arrays);pairs,summary=neighbors(sources,h)
    atomic_json(output/'neighbors.json',pairs)
    audit=dict(neighbors=summary,label_dependency_probe=quality_probe(sources,h,seed),zero_histograms=int((h.sum(1)==0).sum()),
        occupied_units=int((h.sum(0)>0).sum()),acoustic_units_are_not_phonemes=True,
        guarantees=['Train only','same language','opposite labels','different registered PCM source'],
        limitation='Acoustic histogram similarity does not establish equal transcript, speaker or generator')
    atomic_json(output/'quality.json',audit)
    files=shards+[dict(file=n,sha256=digest(output/n)) for n in ('centers.npz','centers.json','neighbors.json','quality.json')]
    atomic_json(output/'manifest.json',dict(owner=owner,complete=True,files=files,quality=audit))
    return bind(output/'manifest.json',rows)


def bind(path,rows):
    path=Path(path).resolve();data=read_json(path)
    if data.get('complete') is not True or data['owner']['rows']!=fingerprint(rows):raise ValueError('XEUS index does not match registered Train rows')
    for entry in data['files']:
        file=path.parent/entry['file']
        if file.parent!=path.parent or digest(file)!=entry['sha256']:raise ValueError('XEUS index file changed')
    sources={r['canonical_source']:(r['language'],r['label']) for r in originals(rows)}
    pairs=read_json(path.parent/'neighbors.json')
    for real,items in pairs.items():
        if real not in sources or sources[real][1]!=1:raise ValueError('Non-Train Real neighbor anchor')
        for item in items:
            fake=item['source']
            if fake not in sources or fake==real or sources[fake]!=(sources[real][0],0):raise ValueError('Invalid opposite-class Train neighbor')
    return dict(path=str(path),sha256=digest(path),neighbors=pairs,quality=data['quality'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run');p.add_argument('--output',required=True)
    p.add_argument('--reuse-run');p.add_argument('--reuse-manifest')
    p.add_argument('--model-cache',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--hf-endpoint');p.add_argument('--local-files-only',action='store_true');a=p.parse_args()
    from w2v_v31813.teacher import assets,Teacher
    from w2v_aasist.launch import run_lock
    parent=resolve_parent(a.parent_run)
    with run_lock(Path(a.output)/'prepare.lock'):
        paths,proof=assets(a.model_cache,a.hf_endpoint,a.local_files_only)
        result=prepare(parent,a.output,Teacher(paths,a.device),proof,reuse_run=a.reuse_run,reuse_manifest=a.reuse_manifest)
    print('XEUS_MANIFEST='+result['path'],flush=True)

if __name__=='__main__':main()
