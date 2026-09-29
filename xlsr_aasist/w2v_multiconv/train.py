"""Train the complete w2v-BERT/MultiConv adaptation and persist every Dev result."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
from tqdm import tqdm
from . import SCHEMA
from .data import build_data, loader
from .model import Detector
from .runtime import (Metrics, amp_context, atomic_json, atomic_save, load_checkpoint,
                      replay_step, seed_all, sha256, storage_size)


def optimizer_for(model, cfg):
    groups = []
    # Include frozen parameters: Adam allocates moments only when a gradient exists.
    # This preserves the head's moments when the encoder is unfrozen later.
    for name, module, lr in [('encoder', model.backbone, cfg['encoder_lr']), ('head', model.head, cfg['head_lr'])]:
        for decay in (True, False):
            ps = [p for n, p in module.named_parameters() if (p.ndim > 1 and not n.endswith('bias')) == decay]
            if ps:
                groups.append({'params': ps, 'name': name, 'lr': lr,
                               'weight_decay': cfg['weight_decay'] if decay else 0.})
    return torch.optim.AdamW(groups, eps=1e-8)


@torch.inference_mode()
def validate(model, validation, cfg, device, score_path):
    model.eval()
    meters = defaultdict(Metrics)
    report = {}
    temporary = Path(str(score_path) + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        for condition, records in validation.items():
            for examples in tqdm(loader(records, cfg), desc='Dev ' + condition):
                for ex in examples:
                    with amp_context(device, cfg['amp']):
                        logits, _ = model(ex['features'].to(device), ex['mask'].to(device))
                    logits = logits.float().cpu()
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
    report['metric_note'] = 'Fixed Dev proxy, not official Eval score; threshold=0.5, fake=0.'
    return report


def source_fingerprints():
    root = Path(__file__).parent
    result = {p.name: sha256(p) for p in sorted(root.glob('*.py'))}
    import transformers
    result.update(torch_version=str(torch.__version__), transformers_version=transformers.__version__)
    for relative in ('utils/data_utils.py', 'utils/RawBoost.py', 'rtc_noisy_v2/cache.py', 'rtc_noisy_v2/plan.py'):
        path = root.parent / relative
        result[relative] = sha256(path)
    return result


def initialize(cfg):
    print('Loading original best; importing ONLY w2v-BERT weights.', flush=True)
    old = load_checkpoint(cfg['baseline'], schema='rtc_w2v_rebuild_v1')
    mc = old['model_config']
    if (mc['hidden_size'], mc['num_hidden_layers'], mc['feature_projection_input_dim']) != (1024, 24, 160):
        raise ValueError('Original best is not the expected w2v-BERT 2.0 encoder')
    fingerprints = old.get('data_fingerprints', {})
    prep = [v for k, v in fingerprints.items() if Path(k).name == 'preprocessor_config.json']
    if len(prep) != 1 or prep[0] != sha256(Path(cfg['ssl_path']) / 'preprocessor_config.json'):
        raise ValueError('Feature extractor differs from the original best')
    model = Detector.from_config(mc, checkpointing=True)
    encoder = {k[len('backbone.'):]: v for k, v in old['model'].items() if k.startswith('backbone.')}
    model.backbone.load_state_dict(encoder, strict=True)
    historical = old.get('dev', {})
    return model, historical


def write_report(run, records, best_epoch, baseline, status):
    lines = ['# w2v-BERT + MultiConv', '', f'Status: {status}', f'Best epoch: {best_epoch}', '',
             'Metrics below are fixed-Dev proxies at threshold 0.5, not official Eval results.', '',
             '| Epoch | Phase | Clean F1 | Seen F1 | Heldout F1 | Weighted F1 | Online EN-real recall |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        d = r['dev']
        recall = d['groups'].get('online/en', {}).get('recall', [None, None])[1]
        recall_text = f'{recall * 100:.3f}' if recall is not None else 'NA'
        lines.append(f'| {r["epoch"]} | {r["phase"]} | {100*d["clean_f1"]:.3f} | {100*d["seen_f1"]:.3f} | '
                     f'{100*d["heldout_f1"]:.3f} | {100*d["weighted_f1"]:.3f} | {recall_text} |')
    old_score = baseline.get('robust_f1')
    if old_score is not None:
        lines += ['', f'Original AASIST recorded Dev proxy: {old_score * 100:.3f}%.',
                  'Historical comparison: original clean inputs used a 64600-sample prefix; this run uses its saved input policy.']
    lines += ['', 'The new best is the best MultiConv candidate, not an automatic replacement of the original AASIST best.',
              'Only best_model.pt and last.pt are retained; reports include per-source scores and class/language recalls.']
    (run / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def train(cfg, run, resume=None, smoke_steps=0):
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no silent CPU training')
    if device.type == 'cuda' and cfg['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU needs BF16 support, or explicitly select --amp none')
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, validation, weights, counts, fingerprints = build_data(cfg)
    codes = source_fingerprints()
    print(f'Train counts [fake,real]={counts.tolist()}, CE weights={weights.tolist()}', flush=True)
    print(f'Logical batch={cfg["ordinary_batch"]}+{cfg["noisy_batch"]}; physical batch=1; steps={plan.steps}', flush=True)
    print('Computing original-best hash (read-only).', flush=True)
    digest = sha256(cfg['baseline'])
    if digest != cfg['baseline_sha256']:
        raise ValueError('Original best changed after deployment planning')
    if resume:
        state = load_checkpoint(resume)
        if state.get('kind') != 'training' or state['config'] != cfg:
            raise ValueError('Exact resume requires last.pt and an unchanged training configuration')
        if state['data_fingerprints'] != fingerprints or state['source_hashes'] != codes:
            raise ValueError('Data/code changed since last.pt; refusing an inexact resume')
        model = Detector.from_config(state['model_config'], state['head_config'])
        model.load_state_dict(state['model'], strict=True)
        historical = state['baseline_dev']
    else:
        model, historical = initialize(cfg)
        state = None
    model.to(device)
    weights = weights.to(device)
    optimizer = optimizer_for(model, cfg)
    if state:
        optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch'] + 1 if state else 1
    history = list(state['history']) if state else []
    best_key = tuple(state['best_key']) if state else (-1., -1.)
    best_epoch = state['best_epoch'] if state else 0
    stale = state['stale'] if state else 0
    lr_scale = state['lr_scale'] if state else 1.
    global_step = state['global_step'] if state else 0
    model_bytes = storage_size(model.state_dict())
    needed = 4 * model_bytes + 2 * 1024**3  # best + last + atomic replacement and partial optimizer allowance
    if shutil.disk_usage(run).free < needed:
        raise OSError(f'Need at least {needed / 1024**3:.1f} GiB free before training')
    atomic_json(run / 'inputs.json', {'data_fingerprints': fingerprints, 'source_hashes': codes,
                                    'class_counts': counts.tolist(), 'class_weights': weights.cpu().tolist(),
                                    'baseline_dev': historical, 'baseline_sha256': digest})
    total_epochs = cfg['warmup_epochs'] + cfg['joint_epochs']
    stop_reason = 'epoch_budget'
    for epoch in range(start_epoch, total_epochs + 1):
        warm = epoch <= cfg['warmup_epochs']
        phase = 'head_warmup' if warm else 'joint_last_layers'
        if epoch == cfg['warmup_epochs'] + 1:
            stale, lr_scale = 0, 1.
        model.configure_trainable_layers(0 if warm else cfg['trainable_layers'])
        model.train()
        # Epoch-boundary replay is deterministic; workers and sampler also use epoch seeds.
        seed_all(cfg['seed'] + epoch * 100003)
        print(f'Epoch {epoch}/{total_epochs}: {phase}; trainable encoder layers={model.trainable_layers}', flush=True)
        started, aggregate = time.perf_counter(), defaultdict(float)
        train_meters = defaultdict(Metrics)
        batches = plan.batches(epoch)
        if smoke_steps:
            if sha256(cfg['baseline']) != digest:
                raise RuntimeError('Original best changed externally during smoke test')
            batches = batches[:smoke_steps]
        progress = tqdm(loader(plan.records, cfg, training=True, epoch=epoch, batches=batches),
                        total=len(batches), desc=f'MultiConv {phase} {epoch}')
        for step, examples in enumerate(progress):
            phase_step = (epoch - 1 if warm else epoch - cfg['warmup_epochs'] - 1) * plan.steps + step
            ramp = min(1., (phase_step + 1) / max(1, cfg['lr_warmup_steps']))
            for group in optimizer.param_groups:
                lr = cfg['warmup_head_lr'] if warm else cfg['head_lr']
                if group['name'] == 'encoder':
                    lr = 0. if warm else cfg['encoder_lr']
                group['lr'] = lr * lr_scale * (.1 + .9 * ramp)
            cka_weight = cfg['cka_weight'] * min(1., (global_step + 1) / max(1, cfg['lr_warmup_steps']))
            stats, logits = replay_step(model, examples, optimizer, weights, device, cfg['amp'],
                                       cfg['noisy_weight'], cka_weight, cfg['grad_clip'], check_replay=step == 0)
            global_step += 1
            for key, value in stats.items():
                aggregate[key] += value
            for i, ex in enumerate(examples):
                key = ('noisy' if ex['noisy'] else 'ordinary') + '/' + ex['language']
                train_meters[key].update(logits[i:i+1], [ex['label']])
            progress.set_postfix(loss=f'{stats["loss"]:.4f}', ce=f'{stats["ce"]:.4f}', cka=f'{stats["cka"]:.4f}')
            if step == 0 or (step + 1) % 100 == 0:
                print(f'STEP {step+1}/{len(batches)} CE={stats["ce"]:.6f} CKA={stats["cka"]:.6f} '
                      f'weight={cka_weight:.4f} grad={stats["grad_norm"]:.4f}', flush=True)
        if smoke_steps:
            atomic_json(run / 'smoke.json', {'status': 'passed', 'steps': len(batches), 'losses': dict(aggregate)})
            print('GPU_SMOKE_PASSED=True; no trainable checkpoint saved.', flush=True)
            return
        dev = validate(model, validation, cfg, device, run / f'epoch_{epoch}_scores.jsonl')
        record = {'epoch': epoch, 'phase': phase, 'seconds': time.perf_counter() - started,
                  'train': {k: v / len(batches) for k, v in aggregate.items()},
                  'train_groups': {k: v.result() for k, v in train_meters.items()}, 'dev': dev}
        history.append(record)
        atomic_json(run / f'epoch_{epoch}.json', record)
        key = (dev['weighted_f1'], dev['noisy_f1'])
        improved = key > best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
        elif not warm:
            stale += 1
            if stale == 2:
                lr_scale *= .5
        print(f'Dev Clean={100*dev["clean_f1"]:.3f} Seen={100*dev["seen_f1"]:.3f} '
              f'Heldout={100*dev["heldout_f1"]:.3f} Weighted={100*dev["weighted_f1"]:.3f} best={improved}', flush=True)
        for condition in ('online', 'seen', 'heldout'):
            print(condition + ' EN recall [fake,real]=' + str(dev['groups'].get(condition + '/en', {}).get('recall')), flush=True)
        common = {'schema': SCHEMA, 'kind': 'weights', 'epoch': epoch, 'model': model.state_dict(),
                  **model.architecture(), 'config': cfg, 'dev': dev, 'baseline_dev': historical,
                  'baseline_sha256': digest, 'data_fingerprints': fingerprints, 'source_hashes': codes}
        if improved:
            atomic_save(run / 'best_model.pt', common)
        atomic_save(run / 'last.pt', {**common, 'kind': 'training', 'optimizer': optimizer.state_dict(),
                                     'history': history, 'best_key': best_key, 'best_epoch': best_epoch,
                                     'stale': stale, 'lr_scale': lr_scale, 'global_step': global_step})
        write_report(run, history, best_epoch, historical, 'training')
        if not warm and stale >= cfg['patience']:
            stop_reason = 'joint_plateau'
            break
    preserved = sha256(cfg['baseline']) == digest
    if not preserved:
        raise RuntimeError('Original best changed externally during training')
    write_report(run, history, best_epoch, historical, 'complete')
    old = historical.get('robust_f1')
    atomic_json(run / 'completed.json', {'status': 'complete', 'stop_reason': stop_reason,
                'best_epoch': best_epoch, 'best_model': str(run / 'best_model.pt'),
                'best_weighted_dev_f1': best_key[0], 'original_best_preserved': preserved,
                'exceeds_recorded_baseline_proxy': best_key[0] > old if old is not None else None})
    print(f'TRAINING_COMPLETE=True\nBEST_MODEL={run / "best_model.pt"}\nORIGINAL_BEST_PRESERVED=True', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if (run / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Existing last.pt: use --resume, or start a new run')
    from .launch import run_lock
    with run_lock(run / '.training.lock'):
        try:
            train(cfg, run, run / 'last.pt' if args.resume else None, args.smoke_steps)
        except Exception as exc:
            atomic_json(run / 'failed.json', {'error': type(exc).__name__ + ': ' + str(exc),
                                             'time': datetime.now().isoformat()})
            raise


if __name__ == '__main__':
    main()
