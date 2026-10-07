"""Unchanged historical Dev plus exact-source treatment error accounting."""
from collections import defaultdict
import math

import numpy as np
import torch

from w2v_aasist.progress import progress
from w2v_v3.model import microbatches as exact_microbatches
from w2v_v313.replay import fp32_inference
from w2v_v314.diagnostics import offline_rows, offline_metrics, source_replay
from w2v_v36.metrics import evaluate
from .augment import recipe, seed_for
from .data import panel_loader, tensors
from .step import clear_features


def panel_plan(rows, cfg):
    # Every original Dev source gets one immutable recipe. Each language/class
    # stratum cycles all mechanisms/severities; no selection from model errors.
    strata=defaultdict(list)
    for i,r in enumerate(rows):
        strata[(r['language'],r['label'])].append(i)
    output=[None]*len(rows)
    for group,indices in sorted(strata.items()):
        order=sorted(indices,key=lambda i:seed_for(cfg['panel_seed'],rows[i]['group_id']))
        for rank,i in enumerate(order):
            output[i]=recipe(cfg['panel_seed'],'fixed:'+rows[i]['source_id'],rank,rows[i]['group_id'],split='dev')
    return output


def infer_panel(model, rows, recipes, cfg, run, label):
    model.eval(); result=[]; offset=0
    try:
        with fp32_inference(cfg['device']):
            for batch in progress(panel_loader(rows,recipes,cfg,run),
                                  total=math.ceil(len(rows)/cfg['feature_batch']),label=label,every=100):
                examples=tensors(batch)
                z=np.empty((len(examples),2),np.float32)
                for indices,x,mask in exact_microbatches(examples,cfg['microbatch'],cfg['frame_budget']):
                    logits,_=model(x.to(cfg['device']),mask.to(cfg['device']))
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError('Nonfinite robustness panel logits')
                    z[indices]=logits.float().cpu().numpy(); clear_features(model)
                if [r['source_id'] for r in examples]!=[r['source_id'] for r in rows[offset:offset+len(examples)]]:
                    raise ValueError('Fixed robustness panel order changed')
                result.append(z); offset+=len(examples)
    finally:
        clear_features(model)
    if offset!=len(rows):
        raise ValueError('Robustness panel incomplete')
    return np.concatenate(result)


def panel_metrics(rows, recipes, logits):
    def score(indices):
        selected=[dict(rows[i],condition='heldout') for i in indices]
        value=evaluate(selected,logits[indices])
        return dict(f1=value['heldout_f1'],groups=value['groups'])
    indices=list(range(len(rows)))
    result=dict(score(indices),views=len(rows),included_in_weighted=False,
        scope='official Dev; disjoint noise recordings and held-out settings; engines may have been seen by the base')
    result['families']={name:score([i for i in indices if recipes[i]['family']==name])
                        for name in sorted({r['family'] for r in recipes})}
    result['noise_types']={name:score([i for i in indices if recipes[i]['noise_type']==name])
                        for name in sorted({r['noise_type'] for r in recipes})}
    return result


def treatment_errors(original_rows, original_logits, processed_rows, processed_logits):
    """Exact waveform hash join, no inferred pairing from filename or language."""
    originals={}
    for row,z in zip(original_rows,original_logits):
        key=row['audio_sha256']
        value=(row['language'],row['label'],int(np.argmax(z)))
        if key in originals and originals[key]!=value:
            raise ValueError('Duplicated original hash has conflicting labels/predictions')
        originals[key]=value
    output={}; sources=defaultdict(set); errors=defaultdict(set)
    for row,z in zip(processed_rows,processed_logits):
        sha=row['source_sha256']
        if sha not in originals:
            raise ValueError('No verified original counterpart for treatment diagnostics')
        language,label,prediction=originals[sha]
        if (language,label)!=(row['language'],row['label']):
            raise ValueError('Source pair labels/language differ')
        key=f'{row["condition"]}/{language}/'+('fake' if label==0 else 'real')
        cell=output.setdefault(key,dict(views=0,both_correct=0,original_wrong_processed_correct=0,
            original_correct_processed_wrong=0,both_wrong=0))
        a,b=prediction==label,int(np.argmax(z))==label
        name=('both_correct' if b else 'original_correct_processed_wrong') if a else ('original_wrong_processed_correct' if b else 'both_wrong')
        cell[name]+=1; cell['views']+=1; sources[key].add(sha)
        if a and not b:errors[key].add(sha)
    for key,cell in output.items():
        cell['unique_original_sources']=len(sources[key])
        cell['unique_sources_with_new_error']=len(errors[key])
    return dict(groups=output,pairing='exact original waveform SHA256',unit='processed views; unique source counts also reported')
