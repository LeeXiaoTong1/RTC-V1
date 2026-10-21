"""Optional replay of recorded hard pairs through receiver intermediate stages.

This diagnostic never contributes training examples or chooses a submission.
Run after pausing training: it loads one model and uses the recorded recipe.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from w2v_v3189.augment import generate, endpoint, NoiseBank, Engines
from w2v_v3189.views import augment
from w2v_v3183r2.data import read_wave
from w2v_v5.processing import preceding_context, prepare_wave
from .diagnostics import evaluation, encode_bounded
from .common import read_json, atomic_json, apply_partial, fingerprint


def stage_waves(base, recipe, bank, engines, context_seconds):
    context=preceding_context(base,context_seconds)
    stream=np.concatenate((context,base)) if len(context) else base
    observed=[]
    noisy,meta=generate(stream,recipe,bank,engines,input_observer=lambda x:observed.append(x.copy()))
    if len(observed)!=1:
        raise ValueError('One pre-RTC noisy input is required')
    clean_input=stream*meta['common_gain']
    clean,_=endpoint(clean_input,recipe,engines)
    offset=len(context)
    result=dict(common_gain_source=clean_input[offset:].copy(),scene_before_rtc=observed[0][offset:].copy(),
                clean_rtc=clean[offset:].copy(),noisy_rtc=noisy[offset:].copy())
    if recipe.get('category')=='echo':
        # Clean is exactly this same-gain/no-echo endpoint; do not introduce a
        # new gain normalization and misattribute its change to echo removal.
        result['noisy_rtc_echo_off']=result['clean_rtc']
    if any(x.shape!=base.shape or not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Diagnostic stage changed source geometry')
    return result,meta


def hard_pairs(path, allowed, per_group):
    selected={}
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            for item in event.get('examples',[]):
                if item.get('view')!='noisy' or item.get('source_id') not in allowed:
                    continue
                if 'recipe' not in item.get('base_augmentation',{}) or 'recipe' not in item.get('augmentation',{}):
                    continue
                key=item['source_id'];previous=selected.get(key)
                if previous is None or item['weighted_excess']>previous['weighted_excess']:
                    selected[key]=copy.deepcopy(item)
    result=[]
    for language,label in (('en',0),('en',1),('zh',0),('zh',1)):
        rows=[x for x in selected.values() if x['language']==language and x['label']==label]
        result.extend(sorted(rows,key=lambda x:(-x['weighted_excess'],x['source_id']))[:per_group])
    return result


def inspect(model,cfg,rows,examples):
    allowed={row['source_id']:row for row in rows if row.get('kind')=='offline' or str(row.get('id','')).startswith('offline/')}
    # Source IDs may be shared by Online/Offline rows. Choose by the recorded
    # Offline id as well, never silently replay the already-received waveform.
    by_id={row['id']:row for row in rows}
    bank=NoiseBank(cfg['noise_records']['train'],cfg['noise_cache_mib']*1024**2,cfg['scene_noise_records']['train'])
    if getattr(bank,'scenes',None) is not None:bank.scenes.cap=cfg.get('scene_cache_mib',32)*1024**2
    engines=Engines(cfg['ffmpeg']);records=[]
    with evaluation(model):
        for example in examples:
            row=by_id.get(example['id'],allowed.get(example['source_id']))
            if row is None or row['source_id']!=example['source_id'] or row['label']!=example['label']:
                raise ValueError('Recorded source is not an allowed matching training source')
            if row.get('split')!='train' or not str(row['id']).startswith('offline/'):
                raise ValueError('Receiver diagnostic must start from allowed Offline')
            base,_=augment(read_wave(row),example['base_augmentation']['recipe'])
            stages,meta=stage_waves(base,example['augmentation']['recipe'],bank,engines,cfg['v5_context_seconds'])
            original=example['augmentation']
            checked=[]
            for key in ('noise_hashes','noise_original_sources','noise_scale','common_gain'):
                if key in original and meta.get(key)!=original[key]:
                    raise ValueError('Recorded noise/gain replay differs: '+key)
                if key in original:checked.append(key)
            official=next((r for r in rows if str(r['id']).startswith('online/') and r['group_id']==row['group_id']
                           and r['label']==row['label'] and r['language']==row['language']),None)
            if official is not None:stages['official_online_separate']=read_wave(official)
            names=list(stages)
            waves=[prepare_wave(stages[name],cfg['v5_input'])[0] for name in names]
            features=encode_bounded(model,waves,cfg)
            logits=model.classify_sequences(features).float().cpu()
            outputs={name:dict(p_true=float(logits[i].softmax(-1)[row['label']]),
                margin=float(logits[i,row['label']]-logits[i,1-row['label']]),
                correct=int(logits[i].argmax())==row['label']) for i,name in enumerate(names)}
            records.append(dict(source=row['source_id'],id=row['id'],group=f'{row["language"]}/{row["label"]}',
                recipe=example['augmentation']['recipe'],recorded_training_p_true=example.get('p_true'),stages=outputs,
                replay_checked_fields=checked,recorded_noise_identity_confirmed='noise_hashes' in checked,
                replay_limit=None if 'noise_hashes' in checked else 'Historical logger omitted noise identity lists; recipe/gain recomputation is verified, original noise hash cannot be retrospectively confirmed'))
    return dict(schema='v54_receiver_stage_replay_v1',records=records,
        scope='Recorded hard candidates, biased selection; same recipe/gain, independent receiver states; not population error rates or unseen evaluation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--hard-log',help='Defaults to the V5.3 parent final/hard_examples.jsonl')
    parser.add_argument('--per-group',type=int,default=4)
    parser.add_argument('--weights',choices=('parent','last','exported'),default='parent')
    args=parser.parse_args()
    if not 1<=args.per_group<=16:raise ValueError('Use 1..16 hard sources per group')
    from w2v_v3191.processes import ensure_idle
    from w2v_aasist.launch import run_lock
    from .common import ROOT
    from .workflow import read_stage
    from .model import load_model
    from .train import initialize
    root=Path(args.run).expanduser().resolve()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        cfg,rows=read_stage(root/'final')
        hard=Path(args.hard_log) if args.hard_log else Path(cfg['parent_init']['run'])/'final'/'hard_examples.jsonl'
        examples=hard_pairs(hard,{r['source_id'] for r in rows},args.per_group)
        if not examples:raise ValueError('No reproducible recorded hard Noisy pairs; pass --hard-log explicitly')
        model=load_model(cfg)
        if args.weights=='parent':initialize(model,cfg)
        else:
            path=root/'final.pt' if args.weights=='exported' else root/'final'/'last.pt'
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['identity']!=fingerprint(cfg):raise ValueError('Checkpoint recipe mismatch')
            apply_partial(model,saved['model'])
        report=inspect(model,cfg,rows,examples)
        report.update(weights=args.weights,config=fingerprint(cfg),source_log=str(hard))
        target=root/('receiver_stage_replay_'+args.weights+'.json')
        atomic_json(target,report)
        print('RECEIVER_STAGE_REPORT='+str(target),flush=True)


if __name__=='__main__':main()
