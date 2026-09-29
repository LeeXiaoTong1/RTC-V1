import os
import json
from collections import defaultdict
from pathlib import Path
import torch
from tqdm import tqdm
from .data import loader
from .runtime import Metrics, predict

@torch.inference_mode()
def validate(model, validation, cfg, device, score_path):
    model.eval()
    meters = defaultdict(Metrics)
    report = {}
    temporary = Path(str(score_path) + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        for condition, records in validation.items():
            for examples in tqdm(loader(records, cfg), desc='Dev ' + condition):
                predictions = predict(model, examples, device, cfg.get('eval_amp', 'none'), cfg['microbatch'], cfg['frame_budget'])
                for i, ex in enumerate(examples):
                    logits = predictions[i:i+1]
                    group = ex['domain'] if condition == 'clean' else condition
                    meters[group].update(logits, [ex['label']])
                    meters[group + '/' + ex['language']].update(logits, [ex['label']])
                    if condition != 'clean':
                        meters[f'{condition}/band{ex["band"]}'].update(logits, [ex['label']])
                    row = {'condition': group, 'source': ex['id'], 'language': ex['language'],
                           'label': ex['label'], 'band': ex['band'],
                           'p_fake': float(logits.softmax(1)[0, 0]), 'logits': logits[0].tolist()}
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
    os.replace(temporary, score_path)
    report['groups'] = {k: v.result() for k, v in meters.items()}
    if 'online' not in report['groups']:
        raise ValueError('Dev protocol needs Online audio for the original clean selection metric')
    for condition in ('seen', 'heldout'):
        keys = [f'{condition}/band{band}' for band in range(4)]
        if any(k not in report['groups'] or min(report['groups'][k]['class_counts']) == 0 for k in keys):
            raise ValueError('Both classes and all four bands are required for noisy Dev')
        # Same mean-of-band F1 convention as the old training engine.
        report[condition + '_f1'] = sum(report['groups'][k]['macro_f1'] for k in keys) / 4
    report['clean_f1'] = report['groups']['online']['macro_f1']
    report['noisy_f1'] = (report['seen_f1'] + report['heldout_f1']) / 2
    report['weighted_f1'] = .3 * report['clean_f1'] + .7 * report['noisy_f1']
    report['metric_note'] = 'Whole-utterance Online/Offline; unchanged 4-s noisy Dev; fixed threshold=0.5. Not platform Eval.'
    return report

