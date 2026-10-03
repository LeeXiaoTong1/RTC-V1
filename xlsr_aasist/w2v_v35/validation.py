"""Online-only Clean and fixed full-wave Seen/Heldout at threshold 0.5."""
from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from w2v_aasist.runtime import Metrics
from w2v_aasist.progress import progress
from w2v_v3.step import predict
from .data import loader


@torch.inference_mode()
def validate(model, validation, cfg, device, score_path):
    model.eval()
    meters = defaultdict(Metrics)
    score_path = Path(score_path); score_path.parent.mkdir(parents=True,exist_ok=True)
    temporary = score_path.with_suffix(score_path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        for condition,records in validation.items():
            if condition not in ('clean','seen','heldout'): raise ValueError('Unexpected V3.5 validation condition')
            if condition=='clean' and any(r['domain']!='online' for r in records):
                raise ValueError('V3.5 Clean selection uses official Online only')
            batches = loader(records,cfg)
            for examples in progress(batches,total=len(batches),label=score_path.stem+' Dev '+condition,every=200):
                logits = predict(model,examples,device,cfg.get('eval_amp','none'),cfg['microbatch'],cfg['frame_budget'])
                for i,example in enumerate(examples):
                    z = logits[i:i+1]
                    group = 'online' if condition=='clean' else condition
                    meters[group].update(z,[example['label']])
                    meters[group+'/'+example['language']].update(z,[example['label']])
                    if condition!='clean': meters[f'{group}/band{example["band"]}'].update(z,[example['label']])
                    stream.write(json.dumps(dict(condition=group,source=example['id'],source_id=example.get('source_id',example['id']),
                        language=example['language'],label=example['label'],band=example.get('band',-1),
                        processing_family=example.get('processing_family'),p_fake=float(z.softmax(1)[0,0]),logits=z[0].tolist()))+'\n')
            stream.flush()
    os.replace(temporary,score_path)
    groups = {name:meter.result() for name,meter in meters.items()}
    for condition in ('online','seen','heldout'):
        if condition not in groups or min(groups[condition]['class_counts']) == 0:
            raise ValueError('Each fixed validation condition requires both classes')
    for condition in ('seen','heldout'):
        if any(f'{condition}/band{band}' not in groups for band in range(4)):
            raise ValueError('Full Dev needs all four SNR bands')
    clean,seen,heldout = (groups[name]['macro_f1'] for name in ('online','seen','heldout'))
    noisy = .5*(seen+heldout)
    return dict(groups=groups,clean_f1=clean,seen_f1=seen,heldout_f1=heldout,noisy_f1=noisy,
        weighted_f1=.3*clean+.7*noisy,metric_schema='v35_full_online_pooled_conditions_v1',
        metric_note='Full official Clean Online; one full simulated Seen and one anlmdn Heldout per Dev Offline source. '
                    'Pooled Macro-F1 per condition, equal Seen/Heldout average; threshold 0.5. '
                    'Same fixed source mixture in both noisy conditions. Not official platform scores or historical short Dev.')
