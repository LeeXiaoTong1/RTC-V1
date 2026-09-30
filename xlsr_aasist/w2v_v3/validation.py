"""Fixed Dev only; keep historical per-band Macro-F1 and probability threshold."""
import json
import os
from pathlib import Path
from collections import defaultdict
import torch
from w2v_aasist.runtime import Metrics
from w2v_aasist.progress import progress
from .data import loader
from .step import predict


@torch.inference_mode()
def validate(model, validation, cfg, device, score_path):
    model.eval()
    meters = defaultdict(Metrics)
    score_path = Path(score_path)
    score_path.parent.mkdir(parents=True,exist_ok=True)
    temporary = score_path.with_suffix(score_path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as output:
        for condition, records in validation.items():
            batches = loader(records,cfg)
            label = score_path.stem.removesuffix('_scores') + ' Dev ' + condition
            for examples in progress(batches,total=len(batches),label=label,every=200):
                logits = predict(model,examples,device,cfg.get('eval_amp','none'),cfg['microbatch'],cfg['frame_budget'])
                for i, ex in enumerate(examples):
                    z = logits[i:i+1]
                    group = ex['domain'] if condition == 'clean' else condition
                    meters[group].update(z,[ex['label']])
                    meters[group+'/'+ex['language']].update(z,[ex['label']])
                    if condition != 'clean':
                        meters[f'{condition}/band{ex["band"]}'].update(z,[ex['label']])
                    output.write(json.dumps(dict(condition=group,source=ex['id'],language=ex['language'],
                        label=ex['label'],band=ex['band'],p_fake=float(z.softmax(1)[0,0]),logits=z[0].tolist()))+'\n')
            output.flush()
    os.replace(temporary,score_path)
    report = {'groups':{k:v.result() for k,v in meters.items()}}
    if 'online' not in report['groups']:
        raise ValueError('Fixed Dev requires Online for the historical Clean metric')
    for condition in ('seen','heldout'):
        keys = [f'{condition}/band{b}' for b in range(4)]
        if any(k not in report['groups'] or min(report['groups'][k]['class_counts']) == 0 for k in keys):
            raise ValueError('Fixed noisy Dev requires all four bands and both classes')
        report[condition+'_f1'] = sum(report['groups'][k]['macro_f1'] for k in keys)/4
    report['clean_f1'] = report['groups']['online']['macro_f1']
    report['noisy_f1'] = (report['seen_f1']+report['heldout_f1'])/2
    report['weighted_f1'] = .3*report['clean_f1']+.7*report['noisy_f1']
    report['metric_note'] = 'Full ordinary Dev; original fixed short noisy Dev. Threshold0.5; not official platform score.'
    return report
