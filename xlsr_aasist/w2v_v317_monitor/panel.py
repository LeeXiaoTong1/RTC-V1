"""Full Train replay: official views and one fixed generated view per source."""
from collections import Counter
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from live_progress import Phase
from w2v_v39.common import atomic_json, digest, read_json
from w2v_v313.data import _inventory
from w2v_v313.replay import rows_signature, fp32_inference
from w2v_v317.state import capture_rng, restore_rng
from w2v_v317.output import infer
from w2v_v315.augment import recipe
from w2v_v316_tfcl.data import Triplets, Ticket, _loader, tensors
from w2v_v3.model import microbatches
from .metrics import summarize


@contextmanager
def observation_mode(model):
    rng=capture_rng(); modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with fp32_inference(next(model.parameters()).device): yield
    finally:
        for module,training in modes:module.training=training
        restore_rng(rng)


def definition(rows,cfg):
    inventory=_inventory(rows)
    originals=[item['indices']['offline'] for _,item in sorted(inventory.items())]
    official=[r for r in rows if r['condition'] in ('offline','online')]
    if any(r['split']!='train' for r in official):raise ValueError('Full Train observation cannot contain Dev/Test')
    source_ids=[rows[i]['source_id'] for i in originals]
    if len(set(source_ids))!=len(originals):raise ValueError('Duplicate original sources')
    if len(originals)!=sum(r['condition']=='offline' for r in official):raise ValueError('Uncovered Offline sources')
    code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}
    value=dict(schema='v317_full_train_monitor_v1', seed=cfg['seed'], sources=len(originals),
        official_rows_signature=rows_signature(official), original_sources=hashlib.sha256(json.dumps(source_ids).encode()).hexdigest(),
        augmentation_runtime=cfg['augmentation_runtime'], augmentation_files=cfg['augmentation_files'],
        code=code, generated_views_per_source=1, generated_split='train', generated_warm=1.,
        scope='full Train; each official view once; one fixed generated noisy view per original source',
        conditions=dict(Counter(r['condition'] for r in official)), new_audio_cache_bytes=0)
    signature=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return originals,[r for r in official if r['condition']=='online'],value,signature


def tickets(rows,indices,cfg):
    result=[]; digest_recipes=hashlib.sha256()
    for phase,index in enumerate(indices):
        r=rows[index];occurrence='full-train-monitor:'+r['source_id']
        config=recipe(cfg['seed'],occurrence,phase,r['group_id'],split='train',warm=1.)
        raw=json.dumps(config,sort_keys=True);digest_recipes.update(raw.encode())
        # Official Online is evaluated independently, without generating it again.
        result.append(Ticket(index,None,occurrence,phase,1.,raw))
    return result,digest_recipes.hexdigest()


def evaluate_panel(model,rows,cfg,run,tag,checkpoint_sha256=None):
    run=Path(run);out=run/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    if (run/'execution_plan.json').exists():
        cfg=dict(cfg,**read_json(run/'execution_plan.json')['selected'])
    indices,online,description,signature=definition(rows,cfg)
    manifest=out/'full_train_definition.json'
    if manifest.exists() and read_json(manifest)['signature']!=signature:
        raise ValueError('Fixed full Train definition/code changed; do not join incomparable curves')
    atomic_json(manifest,dict(signature=signature,definition=description))
    weights=hashlib.sha256()
    for name,value in model.named_parameters():
        if name.startswith('head.') or name.endswith(('lora_A','lora_B')):
            weights.update(name.encode());weights.update(value.detach().cpu().contiguous().numpy().tobytes())
    weights_sha256=weights.hexdigest()
    marker=out/('panel_'+tag+'.json')
    if marker.exists():
        prior=read_json(marker)
        if prior['panel_signature']!=signature or prior['weights_sha256']!=weights_sha256:
            raise ValueError('Saved full Train metrics belong to different weights/definition')
        return prior
    selected,recipe_hash=tickets(rows,indices,cfg)
    eval_cfg=dict(cfg,rolling_cache_bytes=0)
    dataset=Triplets(rows,eval_cfg,out,split='train')
    batches=[selected[i:i+cfg['feature_batch']] for i in range(0,len(selected),cfg['feature_batch'])]
    loader=_loader(dataset,eval_cfg,batch_sampler=batches)
    all_rows=[];all_logits=[];started=time.monotonic()
    with observation_mode(model):
        if online:
            print(f'[Train evaluation] full official Online: {len(online)} waveforms',flush=True)
            z=infer(model,online,eval_cfg,'V3.17 full Train Online',out)
            all_rows.extend(online);all_logits.append(z)
        print(f'[Train evaluation] full Offline + fixed Noisy: {len(indices)} sources, {2*len(indices)} waveforms',flush=True)
        phase=Phase('V3.17 full Train Offline/Noisy evaluation',len(batches))
        iterator=iter(loader)
        try:
            for number,examples in enumerate(iterator,1):
                examples=tensors(examples);z=np.empty((len(examples),2),dtype=np.float32);visited=[]
                for ids,x,mask in microbatches(examples,cfg['microbatch'],cfg['frame_budget']):
                    logits,_=model(x.to(cfg['device']),mask.to(cfg['device']))
                    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite full Train inference')
                    z[ids]=logits.float().cpu().numpy();visited.extend(ids)
                if sorted(visited)!=list(range(len(examples))):raise ValueError('Incomplete diagnostic coverage')
                for row in examples:
                    condition='offline' if row['role']=='offline' else 'noisy_train'
                    all_rows.append({k:row[k] for k in ('id','source_id','group_id','language','label','split')}|{'condition':condition})
                all_logits.append(z);phase.update(number)
        finally:
            if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    expected=len(online)+2*len(indices)
    if len(all_rows)!=expected:raise ValueError('Full Train row count differs')
    for condition,expected_count in (('offline',len(indices)),('noisy_train',len(indices)),('online',len(online))):
        subset=[r for r in all_rows if r['condition']==condition]
        if len(subset)!=expected_count or len({r['source_id'] for r in subset})!=expected_count:
            raise ValueError('Missing or repeated full Train view: '+condition)
    scores=np.concatenate(all_logits)
    path=out/('panel_scores_'+tag+'.npz');tmp=path.with_suffix('.tmp')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,logits=scores,ids=np.asarray([r['id'] for r in all_rows]),
                conditions=np.asarray([r['condition'] for r in all_rows]),languages=np.asarray([r['language'] for r in all_rows]),
                labels=np.asarray([r['label'] for r in all_rows],dtype=np.int8))
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
    result=dict(tag=tag,panel_signature=signature,checkpoint_sha256=checkpoint_sha256,weights_sha256=weights_sha256,metrics=summarize(all_rows,scores),
        scope=description['scope'],sources=len(indices),full_train=True,generated_recipe_sha256=recipe_hash,
        score_sha256=digest(path),seconds=time.monotonic()-started,new_audio_cache_bytes=0,
        note='noisy_train is a fixed Train-distribution augmentation, never Dev Seen or Heldout; observation only, excluded from model selection')
    atomic_json(marker,result)
    return result
