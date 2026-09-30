"""Fine-tune the complete original AASIST checkpoint with full-wave input."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import time
import torch
from .progress import training_bar, phase
from . import SCHEMA
from .data import build_data, loader
from .model import Detector
from .runtime import (Metrics, atomic_json, atomic_save, load_checkpoint, seed_all,
                      sha256, storage_size, supervised_step)
from .validation import validate


def source_fingerprints():
    root = Path(__file__).resolve().parent.parent
    paths = list((root / 'w2v_aasist').glob('*.py'))
    paths += [root / p for p in ('w2v_rebuild/model.py', 'utils/data_utils.py',
              'utils/RawBoost.py', 'utils/env_noise.py', 'rtc_noisy_v2/cache.py',
              'rtc_noisy_v2/plan.py', 'rtc_noisy/diverse.py')]
    import transformers
    return {**{str(p.relative_to(root)): sha256(p) for p in paths},
            'torch': str(torch.__version__), 'transformers': transformers.__version__}


def initialize(cfg):
    old = load_checkpoint(cfg['baseline'], schema='rtc_w2v_rebuild_v1')
    mc = old['model_config']
    if cfg.get('production_layout', True) and (mc['hidden_size'], mc['num_hidden_layers'],
                                             mc['feature_projection_input_dim']) != (1024, 24, 160):
        raise ValueError('Expected original w2v-BERT 2.0 architecture')
    prep = [v for k, v in old.get('data_fingerprints', {}).items()
            if Path(k).name == 'preprocessor_config.json']
    if len(prep) != 1 or prep[0] != sha256(Path(cfg['ssl_path']) / 'preprocessor_config.json'):
        raise ValueError('Feature extractor differs from the original best')
    print('Loading ALL original w2v-BERT AND AASIST weights (strict=True).', flush=True)
    model = Detector.load(cfg['ssl_path'], config_dict=mc, checkpointing=cfg['checkpointing'])
    model.load_state_dict(old['model'], strict=True)
    return model, old.get('dev', {})


def optimizer_for(model, cfg):
    groups = []
    for name, module, lr in [('encoder', model.backbone, cfg['encoder_lr']),
                             ('head', model.head, cfg['head_lr'])]:
        for decay in (True, False):
            ps = [p for n, p in module.named_parameters() if p.requires_grad and
                  (p.ndim > 1 and not n.endswith('bias')) == decay]
            if ps:
                groups.append({'params': ps, 'name': name, 'initial_lr': lr, 'lr': lr,
                               'weight_decay': cfg['weight_decay'] if decay else 0.})
    return torch.optim.AdamW(groups, eps=1e-8)


def write_report(run, history, best_epoch, noisy_epoch, status, cfg=None):
    lines = ['# w2v-BERT 2.0 + AASIST / full audio', '', f'Status: {status}',
             f'Best weighted epoch: {best_epoch}; best noisy epoch: {noisy_epoch}', '',
             'Epoch 0 evaluates the original 91.68 checkpoint with the NEW full-input policy, before updating weights.',
             'These are fixed Dev proxies, not official platform results. Noisy Dev WAV files are unchanged.', '',
             '| Epoch | Clean | Seen | Heldout | Weighted | Offline EN real | Online EN real | Seen EN real | Heldout EN real |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in history:
        d = r['dev']
        values = [f'{100*d[k]:.3f}' for k in ('clean_f1','seen_f1','heldout_f1','weighted_f1')]
        for group in ('offline/en','online/en','seen/en','heldout/en'):
            value = d['groups'].get(group, {}).get('recall', [None,None])[1]
            values.append('NA' if value is None else f'{100*value:.3f}')
        lines.append('| ' + str(r['epoch']) + ' | ' + ' | '.join(values) + ' |')
    if cfg and cfg.get('full_noisy'):
        lines += ['', 'Noisy training: two complete, continuously processed versions per Offline Train source.',
                  'Version 0 uses 5-15 dB, version 1 uses 15-25 dB; conditions are stratified by language/class.',
                  'Only one complete cached view is sampled per noisy slot; no prefix-tail or temporal stitching.']
    else:
        lines += ['', 'Training compositions: 50% unchanged cached view, 25% same-source condition switch,',
              '25% cached prefix + the SAME source original tail. Short originals fall back to a single cache view.',
              'Prefix-tail contains a processed prefix and an unprocessed tail; it is not fully noisy long audio.']
    lines += ['Training does not overwrite audio/cache files or original/MultiConv checkpoints.']
    (Path(run) / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def train(cfg, run, resume=None, smoke_steps=0):
    phase('Checking weights, inputs and cache metadata')
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if device.type == 'cuda' and cfg['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; explicitly select --amp none')
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original checkpoint SHA256 changed')
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, validation, weights, counts, fingerprints = build_data(cfg)
    codes = source_fingerprints()
    state = load_checkpoint(resume) if resume else None
    if state and (state['config'] != cfg or state['data_fingerprints'] != fingerprints or state['source_hashes'] != codes):
        raise ValueError('Resume requires identical configuration, metadata, code and versions')
    if state:
        model = Detector.load(cfg['ssl_path'], config_dict=state['model_config'], checkpointing=cfg['checkpointing'])
        model.load_state_dict(state['model'], strict=True)
        historical = state['baseline_dev']
    else:
        phase('Loading original AASIST weights')
        model, historical = initialize(cfg)
    model.configure_trainable_layers(cfg['trainable_layers'])
    model.to(device)
    weights = weights.to(device)
    optimizer = optimizer_for(model, cfg)
    if state:
        optimizer.load_state_dict(state['optimizer'])
    model_bytes = storage_size(model.state_dict())
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    needed = 4 * model_bytes + 4 * trainable_bytes + 2 * 1024**3
    if shutil.disk_usage(run).free < needed:
        raise OSError(f'Checkpoint headroom needs {needed / 1024**3:.1f} GiB')
    print(f'AASIST full audio; encoder last {cfg["trainable_layers"]} layers trainable; '
          f'logical={cfg["ordinary_batch"]}+{cfg["noisy_batch"]}; one-pass microbatch<={cfg["microbatch"]}', flush=True)
    print(f'Train fake/real={counts.tolist()}, inverse-frequency weights={weights.tolist()}', flush=True)
    atomic_json(run / 'inputs.json', {'data_fingerprints': fingerprints, 'source_hashes': codes,
                                     'baseline_dev': historical, 'baseline_sha256': cfg['baseline_sha256']})
    history = state['history'] if state else []
    best_epoch = state['best_epoch'] if state else 0
    noisy_epoch = state['noisy_epoch'] if state else 0
    best_key = tuple(state['best_key']) if state else (-1.,-1.)
    noisy_key = tuple(state['noisy_key']) if state else (-1.,-1.)
    stale = state['stale'] if state else 0
    start_epoch = state['epoch'] + 1 if state else 1

    def snapshot(epoch, dev):
        return {'schema': SCHEMA, 'kind': 'weights', 'epoch': epoch, 'model': model.state_dict(),
                **model.architecture(), 'config': cfg, 'dev': dev, 'baseline_dev': historical,
                'baseline_sha256': cfg['baseline_sha256'], 'data_fingerprints': fingerprints,
                'source_hashes': codes}

    if not state and not smoke_steps:
        print('Epoch 0: original weights / FULL-input Dev evaluation; no update.', flush=True)
        dev = validate(model, validation, cfg, device, run / 'epoch_0_scores.jsonl')
        phase('Epoch 0: saving baseline checkpoints')
        record = {'epoch': 0, 'dev': dev, 'train': {}, 'train_groups': {}, 'composition_counts': {}}
        history.append(record)
        atomic_json(run / 'epoch_0.json', record)
        best_key = (dev['weighted_f1'], dev['noisy_f1'])
        noisy_key = (dev['noisy_f1'], dev['weighted_f1'])
        atomic_save(run / 'best_model.pt', snapshot(0, dev))
        atomic_save(run / 'best_noisy.pt', snapshot(0, dev))
        write_report(run, history, best_epoch, noisy_epoch, 'training', cfg)
        print(f'Starting FULL-input Dev Weighted={100*best_key[0]:.3f} Noisy={100*noisy_key[0]:.3f}', flush=True)
        for group in ('offline/en','online/en','seen/en','heldout/en'):
            print('Epoch 0 ' + group + ' recall [fake,real]=' + str(dev['groups'].get(group,{}).get('recall')), flush=True)
    reason = 'epoch_budget'
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        seed_all(cfg['seed'] + epoch * 100003)
        started, aggregate = time.perf_counter(), defaultdict(float)
        train_meters, modes = defaultdict(Metrics), Counter()
        batches = plan.batches(epoch)
        if smoke_steps:
            batches = batches[:smoke_steps]
        progress = training_bar(loader(plan.records, cfg, training=True, epoch=epoch, batches=batches),
                                len(batches), f'AASIST full epoch {epoch}/{cfg["epochs"]}')
        for step, examples in enumerate(progress):
            global_step = (epoch - 1) * plan.steps + step
            warm = max(1, cfg['lr_warmup_steps'])
            if global_step < warm:
                scale = .1 + .9 * (global_step + 1) / warm
            else:
                fraction = (global_step - warm) / max(1, cfg['epochs'] * plan.steps - warm - 1)
                scale = .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., fraction)))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * scale
            stats, logits = supervised_step(model, examples, optimizer, weights, device, cfg['amp'],
                                           cfg['noisy_weight'], cfg['grad_clip'], cfg['microbatch'], cfg['frame_budget'])
            for k, v in stats.items():
                aggregate[k] += v
            for i, ex in enumerate(examples):
                group = ('noisy' if ex['noisy'] else 'ordinary') + '/' + ex['language']
                train_meters[group].update(logits[i:i+1], [ex['label']])
                modes[f'{ex["composition"]}/{ex["language"]}/{ex["label"]}'] += 1
                if ex['noisy']:
                    modes[f'condition/band{ex["band"]}/{ex["language"]}/{ex["label"]}'] += 1
                aggregate['audio_seconds'] += ex['audio_seconds']
            progress.set_postfix(loss=f'{stats["loss"]:.5f}', refresh=False)
            if step == 0 or (step + 1) % 100 == 0 or step+1 == len(batches):
                seconds_per_step = (time.perf_counter()-started)/(step+1)
                print(f'STEP {step+1}/{len(batches)} CE={stats["loss"]:.6f} '
                      f'grad={stats["grad_norm"]:.3f} seconds/step={seconds_per_step:.3f} '
                      f'ETA_min={seconds_per_step*(len(batches)-step-1)/60:.1f}', flush=True)
        if smoke_steps:
            atomic_json(run / 'smoke.json', {'passed': True, 'losses': dict(aggregate), 'steps': len(batches)})
            print('GPU_SMOKE_PASSED=True; no checkpoint saved.', flush=True)
            return
        training_seconds = time.perf_counter() - started
        dev = validate(model, validation, cfg, device, run / f'epoch_{epoch}_scores.jsonl')
        phase(f'Epoch {epoch}: saving checkpoints and report')
        record = {'epoch': epoch, 'seconds': time.perf_counter()-started, 'training_seconds': training_seconds,
                  'train': {k: v / len(batches) for k, v in aggregate.items()},
                  'train_groups': {k: v.result() for k, v in train_meters.items()},
                  'composition_counts': dict(modes), 'dev': dev}
        history.append(record)
        atomic_json(run / f'epoch_{epoch}.json', record)
        key, nkey = (dev['weighted_f1'], dev['noisy_f1']), (dev['noisy_f1'], dev['weighted_f1'])
        improved = key > best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
            atomic_save(run / 'best_model.pt', snapshot(epoch, dev))
        else:
            stale += 1
        if nkey > noisy_key:
            noisy_key, noisy_epoch = nkey, epoch
            atomic_save(run / 'best_noisy.pt', snapshot(epoch, dev))
        atomic_save(run / 'last.pt', {**snapshot(epoch, dev), 'kind': 'training', 'optimizer': optimizer.state_dict(),
                                     'history': history, 'best_epoch': best_epoch, 'noisy_epoch': noisy_epoch,
                                     'best_key': best_key, 'noisy_key': noisy_key, 'stale': stale})
        print(f'Dev Clean={100*dev["clean_f1"]:.3f} Seen={100*dev["seen_f1"]:.3f} '
              f'Heldout={100*dev["heldout_f1"]:.3f} Weighted={100*dev["weighted_f1"]:.3f} best={improved}', flush=True)
        for group in ('offline/en','online/en','seen/en','heldout/en'):
            print(group + ' recall [fake,real]=' + str(dev['groups'].get(group,{}).get('recall')), flush=True)
        print('COMPOSITIONS=' + json.dumps(dict(modes)), flush=True)
        write_report(run, history, best_epoch, noisy_epoch, 'training', cfg)
        if stale >= cfg['patience']:
            reason = 'no_weighted_dev_improvement'
            break
    phase('Final checkpoint and input verification')
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original best changed externally')
    # No cache writing is performed, and all cache manifests/configs must still match.
    if any(sha256(p) != digest for p, digest in fingerprints.items()):
        raise RuntimeError('Input metadata changed during training')
    write_report(run, history, best_epoch, noisy_epoch, 'complete', cfg)
    atomic_json(run / 'completed.json', {'reason': reason, 'best_epoch': best_epoch,
                'best_noisy_epoch': noisy_epoch, 'best_weighted_dev_f1': best_key[0],
                'original_best_preserved': True, 'cache_metadata_unchanged': True})
    print('TRAINING_COMPLETE=True ORIGINAL_BEST_PRESERVED=True', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--resume')
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    from .launch import run_lock
    with run_lock(Path(args.out)/'.lock'):
        train(cfg, args.out, args.resume, args.smoke_steps)


if __name__ == '__main__':
    main()
