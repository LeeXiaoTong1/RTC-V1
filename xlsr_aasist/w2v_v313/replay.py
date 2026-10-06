"""Fresh LAST references: tiny scalar cache, no teacher during training."""
import hashlib
import json
from pathlib import Path

import numpy as np

from w2v_v39.common import atomic_json, digest, read_json, verify_files, announce
from w2v_v39.metrics import measure
from w2v_v312.replay import fp32_inference, delta
from .probes import run_probes
from .state import identity


def rows_signature(rows):
    names = ('id', 'source_id', 'group_id', 'condition', 'language', 'label', 'audio_sha256')
    data = [{k:r.get(k) for k in names} for r in rows]
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def training_references(model, rows, cfg, run, infer):
    """One resumable frozen forward sweep; write only 2 logits + RMS per row."""
    run = Path(run)
    folder = run/'train_scalar_references'; folder.mkdir(exist_ok=True)
    signature = dict(schema='v313_train_scalars_v1', identity=identity(cfg), rows=rows_signature(rows))
    names = []
    logits, rms = [], []
    size = cfg['reference_batch']
    total = (len(rows)+size-1)//size
    for number, start in enumerate(range(0,len(rows),size)):
        path = folder/f'block_{number:05d}.npz'
        marker = path.with_suffix('.json')
        block = rows[start:start+size]
        expected = dict(signature, start=start, count=len(block))
        if marker.is_file():
            saved = read_json(marker)
            if saved['signature'] != expected:
                raise ValueError('Train scalar cache identity changed')
            verify_files({str(path):saved['sha256']})
            with np.load(path,allow_pickle=False) as data:
                z, r = data['logits'].copy(), data['rms'].copy()
        else:
            z, features = infer(model,block,cfg,f'V3.13 source-LAST reference {number+1}/{total}',capture=True)
            r = np.sqrt(np.mean(features.astype(np.float64)**2,axis=1)).astype(np.float32)
            del features
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as output:
                np.savez_compressed(output,logits=z.astype(np.float32),rms=r)
            tmp.replace(path)
            atomic_json(marker,dict(signature=expected,sha256=digest(path)))
        if z.shape != (len(block),2) or r.shape != (len(block),) or not np.isfinite(z).all() or not np.isfinite(r).all():
            raise ValueError('Invalid aligned Train scalar reference')
        logits.append(z); rms.append(r); names.extend((path,marker))
        print(f'V313_REFERENCE blocks={number+1}/{total}; scalar_only=True',flush=True)
    z, norms = np.concatenate(logits), np.concatenate(rms)
    labels = np.array([r['label'] for r in rows])
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Only official Train can supply training targets')
    margin = z[np.arange(len(rows)),labels]-z[np.arange(len(rows)),1-labels]
    targets = np.where(margin >= cfg['retention_min_margin'],
                       np.minimum(cfg['retention_cap'],margin-cfg['retention_slack']),0.)
    ceiling = np.maximum(cfg['margin_soft_floor'],cfg['margin_reference_factor']*np.abs(margin)+cfg['margin_reference_slack'])
    output = [dict(row,retention_target=float(t),reference_rms=float(max(n,cfg['feature_rms_floor'])),
                   margin_ceiling=float(c)) for row,t,n,c in zip(rows,targets,norms,ceiling)]
    report = dict(rows=len(rows),protected=int((targets>0).sum()),starting_checkpoint=cfg['starting_checkpoint'],
        starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
        reference='fresh frozen V3.12 LAST in FP32; not historical V3.7 cached vectors',
        new_audio_bytes=0,new_frame_cache_bytes=0,scalar_values=3*len(rows),
        actual_cache_bytes=sum(p.stat().st_size for p in names),
        cost='one frozen Train forward sweep, resumable; zero extra teacher encoder passes per update')
    return output, report


def baseline(model, dev, probe_rows, split, cfg, run, infer):
    run=Path(run); marker=run/'baseline_complete.json'
    signature=dict(identity=identity(cfg),dev=rows_signature(dev['rows']),probe=rows_signature(probe_rows),split=split)
    if marker.is_file():
        saved=read_json(marker)
        if saved['signature'] != signature:
            raise ValueError('Starting-LAST baseline identity changed')
        verify_files({str(run/name):sha for name,sha in saved['files'].items()})
        with np.load(run/'dev_scores_baseline.npz',allow_pickle=False) as data:
            z=data['logits'].copy()
        return z,read_json(run/'baseline_metrics.json'),read_json(run/'baseline_probes.json')
    if (run/'last.pt').is_file():
        raise ValueError('Cannot resume without committed starting-LAST baseline')
    announce('V3.13 measuring submitted V3.12 LAST, never its selected baseline best')
    diagnostics={}
    z,_=infer(model,dev['rows'],cfg,'V3.13 V3.12-LAST baseline Dev',diagnostics=diagnostics)
    source=Path(cfg['source_run'])
    previous=source/('dev_scores_'+cfg['starting_tag']+'.npz')
    comparison={'status':'unavailable','reason':'source validation score file absent; weight hashes remain verified'}
    if previous.is_file() and (source/'dev_rows.json').is_file():
        old_rows=read_json(source/'dev_rows.json')
        keys=('id','source_id','group_id','condition','language','label')
        if [{k:r[k] for k in keys} for r in dev['rows']] != [{k:r[k] for k in keys} for r in old_rows]:
            raise ValueError('Source LAST Dev inventory differs')
        with np.load(previous,allow_pickle=False) as data:
            comparison=delta(data['logits'],z)
        if not comparison['allclose'] or comparison['decision_changes']:
            raise ValueError('Starting LAST replay differs from its committed validation')
        comparison['status']='passed'
    atomic_json(run/'startup_replay.json',dict(starting_kind='trained_v312_last',starting_tag=cfg['starting_tag'],comparison=comparison))
    _,features=infer(model,probe_rows,cfg,'V3.13 initial language diagnostic',capture=True)
    probes=run_probes(features,probe_rows,split,cfg)
    metrics=measure(dev['rows'],z,target=cfg['matched_fake_recall'])
    atomic_json(run/'baseline_metrics.json',metrics)
    atomic_json(run/'baseline_probes.json',probes)
    atomic_json(run/'baseline_diagnostics.json',diagnostics)
    np.savez_compressed(run/'dev_scores_baseline.npz',logits=z)
    names=('baseline_metrics.json','baseline_probes.json','baseline_diagnostics.json','dev_scores_baseline.npz','startup_replay.json')
    atomic_json(marker,dict(signature=signature,files={name:digest(run/name) for name in names}))
    return z,metrics,probes


def paired_errors(rows, logits):
    """Measure processing-added mistakes on Train sources excluded from adaptation."""
    table={}
    predicted=np.asarray(logits).argmax(1)
    for row,pred in zip(rows,predicted):
        table.setdefault(row['source_id'],{})[row['condition']]=(row,int(pred)==row['label'])
    out={}
    for views in table.values():
        original,correct=views['offline']
        for condition,(row,processed_correct) in views.items():
            if condition=='offline':continue
            key=f'{condition}/{row["language"]}/{row["label"]}'
            item=out.setdefault(key,dict(pairs=0,original_errors=0,processed_errors=0,new_processing_errors=0,rescued_errors=0))
            item['pairs']+=1; item['original_errors']+=int(not correct);item['processed_errors']+=int(not processed_correct)
            item['new_processing_errors']+=int(correct and not processed_correct)
            item['rescued_errors']+=int(not correct and processed_correct)
    return dict(groups=out,scope='Train sources held out from V3.13 adaptation; earlier models saw Train; diagnostic only')
