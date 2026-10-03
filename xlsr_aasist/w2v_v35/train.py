"""Fresh encoder/head training for Online deployment; epoch-boundary exact resume."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
import torch
from w2v_aasist.progress import phase as publish_phase, training_bar
from w2v_aasist.runtime import Metrics, atomic_json, atomic_save, seed_all, sha256
from w2v_v3.train import read_state, rng_state, restore_rng
from w2v_v33.model import Detector, install_runtime
from w2v_v32.performance import TimedLoader
from . import SCHEMA
from .pretrained import initialize, reference_model
from .data import build_data, loader
from .cache import retire_generations
from .validation import validate
from .optim import optimizer_for, apply_learning_rates, schedule_scale, EMA
from .control import Controller
from .step import supervised_step
from .storage import compatible_code, checkpoint_peak_bytes, check_space


def phase(label):
    publish_phase(label)
    print('V32_EVENT=' + json.dumps(dict(kind='phase', label=label)), flush=True)


def source_fingerprints():
    import transformers
    import numpy as np
    import scipy
    root = Path(__file__).resolve().parent.parent
    files = {str(path.relative_to(root)): sha256(path)
             for folder in ('w2v_aasist', 'w2v_v3', 'w2v_v31', 'w2v_v32', 'w2v_v33', 'w2v_v34', 'w2v_v35')
             for path in sorted((root / folder).glob('*.py')) if not path.name.startswith('test_')}
    return dict(torch=str(torch.__version__), transformers=str(transformers.__version__),
                numpy=str(np.__version__), scipy=str(scipy.__version__), **files)


def verify_protected(cfg):
    files = dict(cfg.get('source_provenance', {}).get('file_fingerprints', {}))
    files[cfg['reference_checkpoint']] = cfg['reference_checkpoint_sha256']
    if cfg.get('baseline') and cfg.get('baseline_sha256'):
        files[cfg['baseline']] = cfg['baseline_sha256']
    for name, digest in files.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError('Protected source checkpoint or metadata changed: ' + name)


def _save_weights(model, cfg, tag, phase_name, epoch, dev, fingerprints, codes):
    return dict(schema=SCHEMA, kind='weights', tag=tag, phase=phase_name, epoch=epoch,
                model=model.state_dict(), **model.architecture(), config=cfg, dev=dev,
                data_fingerprints=fingerprints, source_hashes=codes,
                reference_checkpoint_sha256=cfg['reference_checkpoint_sha256'],
                weight_source='protected_reference' if tag == 'reference' else 'ema')


def _check_resume(state, cfg, fingerprints, codes):
    if (state.get('schema') != SCHEMA or state.get('kind') != 'training'
            or state.get('config') != cfg or state.get('data_fingerprints') != fingerprints
            or not compatible_code(state.get('source_hashes', {}), codes) or 'ema' not in state):
        raise ValueError('Exact V3.5 resume requires matching config, data metadata, code and EMA')


def _checkpoint_tag(path):
    return read_state(path).get('tag') if path.is_file() else None


def recover_winners(run, controller):
    for field, filename in (('best_selected', 'best_model.pt'), ('best_train', 'best_candidate.pt')):
        expected = controller.state[field]
        path = run / filename
        previous = run / (filename + '.previous')
        if expected is None:
            # An uncommitted first candidate is safe to discard on exact resume.
            if path.is_file():
                path.unlink()
            if previous.is_file():
                previous.unlink()
            continue
        if _checkpoint_tag(path) != expected['tag']:
            if _checkpoint_tag(previous) != expected['tag']:
                raise ValueError('Winner checkpoint disagrees with committed training state: ' + filename)
            os.replace(previous, path)
        elif previous.is_file():
            previous.unlink()


def write_report(run, history, controller, status):
    lines = ['# V3.5: Online-focused source-balanced training', '', 'Status: ' + status,
        'Promoted checkpoint: ' + controller.state['best_selected']['tag'],
        'Fixed full Online Dev at threshold 0.5; these local proxies are not platform scores.',
        'One full Offline, available Online and two rotating full noisy views per source.',
        'Four language/class source budgets; 0.1/0.3/0.6 condition loss; noisy 0.5 mean + 0.5 max.',
        'Only EMA is evaluated each epoch. No Offline metric is used as a promotion veto.',
        'Reference is the preserved submitted model, measured on the same new full Dev.', '',
        '| Checkpoint | Phase | Clean Online | Seen | Heldout | Noisy | Weighted | Promoted |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for record in history:
        scores = [f'{100 * record["dev"][key]:.3f}' for key in
                  ('clean_f1', 'seen_f1', 'heldout_f1', 'noisy_f1', 'weighted_f1')]
        lines.append('| ' + ' | '.join([record['tag'], record['phase'], *scores,
                                       str(record['decision'].get('promoted', False))]) + ' |')
    lines += ['', 'best_model.pt: Online-weighted winner, including protected reference fallback.',
              'best_candidate.pt: best learned EMA, even if it has not beaten the reference.',
              'last.pt: raw model, optimizer, EMA and RNG at the last complete epoch; unsaved steps replay.',
              'No guarantee of official 97+ follows from these local validation measurements.']
    temporary = run / 'report.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, run / 'report.md')


def print_validation(record):
    print('V32_EVENT=' + json.dumps(dict(kind='validation', tag=record['tag'])), flush=True)
    print('Dev ' + record['tag'] + ' ' + ' '.join(
        f'{label}={100 * record["dev"][key]:.3f}' for label, key in
        (('Clean', 'clean_f1'), ('Seen', 'seen_f1'), ('Heldout', 'heldout_f1'),
         ('Noisy', 'noisy_f1'), ('Weighted', 'weighted_f1')))
        + f' promoted={record["decision"].get("promoted", False)} action={record["decision"]["action"]}', flush=True)
    for key, metrics in record['dev'].get('groups', {}).items():
        if key in ('offline/en', 'online/en', 'seen/en', 'heldout/en'):
            print(key + ' recall [fake,real]=' + str(metrics['recall']), flush=True)


def train(cfg, run, resume=None, smoke_steps=0):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if not resume and any((run / name).exists() for name in ('last.pt', 'best_model.pt', 'best_candidate.pt')):
        raise ValueError('Run contains checkpoints; resume last.pt or choose a new directory')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if device.type == 'cuda' and cfg.get('amp') == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable')
    if cfg.get('eval_amp', 'none') != 'none' or cfg.get('evals_per_epoch', 1) != 1:
        raise ValueError('V3.5 uses one full FP32 EMA validation per epoch')
    if cfg.get('head_epochs', 1) != 1 or not 1 <= cfg.get('joint_epochs', 5) <= 5:
        raise ValueError('One head epoch and 1..5 joint epochs required')
    if smoke_steps < 0:
        raise ValueError('smoke_steps must be nonnegative')
    phase('V3.5 validating protected reference and full-length data')
    verify_protected(cfg)
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, validation, weights, counts, fingerprints = build_data(cfg)
    codes = source_fingerprints()
    state = read_state(resume) if resume else None
    if state:
        _check_resume(state, cfg, fingerprints, codes)
        if state['source_hashes'] != codes:
            atomic_json(run/'storage_resume_compat.json',dict(patch='v35_disk_peak_v1',
                old_source_hashes=state['source_hashes'],new_source_hashes=codes,
                note='Storage/report ordering only; config, data, model, optimizer, EMA and RNG unchanged.'))
    controller_cfg = {**cfg, 'patience': cfg.get('early_stop_patience', 2)}
    controller = Controller(controller_cfg, state['controller'] if state else None)
    history = state['history'] if state else []
    if state:
        recover_winners(run, controller)
        model = Detector.from_checkpoint(state, checkpointing=cfg.get('checkpointing', True))
    else:
        phase('V3.5 measuring original submitted model on the new full Dev')
        reference = reference_model(cfg).to(device)
        reference_dev = validate(reference, validation, cfg, device, run / 'reference_scores.jsonl')
        controller.initialize(reference_dev)
        reference_record = dict(tag='reference', phase='reference', arm='V3.5', epoch=0, cursor=0,
            global_steps=0, dev=reference_dev, train={}, train_groups={}, learning_rates={},
            decision=dict(save=['best_safe'], promoted=True, warnings=[], action='reference_preserved'))
        history.append(reference_record)
        atomic_save(run / 'best_model.pt', _save_weights(reference, cfg, 'reference', 'reference',
                                                        0, reference_dev, fingerprints, codes))
        atomic_json(run / 'reference.json', reference_record)
        del reference
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Constructing/evaluating the independent reference cannot change the fresh run's RNG.
        seed_all(cfg['seed'])
        model = initialize(cfg)
        print_validation(reference_record)
    model.to(device)
    model = install_runtime(model, chunk_layers=cfg.get('fusion_chunk_layers', 5))
    phase_name = state['phase'] if state else 'head'
    optimizer = optimizer_for(model, cfg, phase_name)
    ema_device = cfg.get('ema_device', 'model')
    ema = EMA(model, cfg.get('ema_decay', .999), device=None if ema_device == 'model' else ema_device)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'], model)
        restore_rng(state['rng'])
    epoch = state['epoch'] if state else 0
    global_steps = state['global_steps'] if state else 0
    phase_steps = state['phase_steps'] if state else 0
    complete = state.get('complete', False) if state else False
    atomic_json(run / 'inputs.json', dict(data_fingerprints=fingerprints, source_hashes=codes,
        source_group_counts=counts.tolist() if hasattr(counts, 'tolist') else counts,
        source_group_weights=weights, reference_checkpoint=cfg['reference_checkpoint'],
        reference_checkpoint_sha256=cfg['reference_checkpoint_sha256']))
    if hasattr(plan, 'coverage'):
        atomic_json(run / 'planned_coverage.json', plan.coverage())

    def training_snapshot(tag, dev):
        return {**_save_weights(model, cfg, tag, phase_name, epoch, dev, fingerprints, codes),
            'kind': 'training', 'weight_source': 'raw_training', 'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict(), 'controller': controller.dump(), 'history': history,
            'global_steps': global_steps, 'phase_steps': phase_steps, 'rng': rng_state(), 'complete': complete}

    if not state:
        atomic_save(run / 'last.pt', training_snapshot('bootstrap', history[0]['dev']))
    else:
        for committed_record in history:
            atomic_json(run / (committed_record['tag'] + '.json'), committed_record)
        # A crash can occur after last.pt commits but before retirement. Preserve
        # its epoch and any already-generated next epoch; remove older owned data.
        retire_generations(cfg, keep_epochs=[epoch, epoch + 1])
    write_report(run, history, controller, 'complete' if complete else 'training')
    while not complete:
        epoch += 1
        wanted_phase = 'head' if epoch <= cfg.get('head_epochs', 1) else 'joint'
        if wanted_phase != phase_name:
            old_optimizer = optimizer
            optimizer = optimizer_for(model, cfg, wanted_phase)
            # Preserve the learned head's Adam moments; newly unfrozen blocks start fresh.
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter in old_optimizer.state:
                        optimizer.state[parameter] = old_optimizer.state[parameter]
            del old_optimizer
            phase_name, phase_steps = wanted_phase, 0
            ema.add_trainable(model)
        # last.pt at epoch N resumes N+1; it never needs N's derived noisy WAVs.
        # No training loader is alive here. Retain a partial upcoming generation
        # for deterministic replay, retiring only older regenerable generations.
        retire_generations(cfg, keep_epochs=[epoch])
        check_space(cfg, run, epoch, checkpoint_peak_bytes(model, ema))
        atomic_json(run / ('optimizer_groups_' + phase_name + '.json'), [dict(name=group['name'],
            base_lr=group['base_lr'], weight_decay=group['weight_decay'],
            parameters=sum(p.numel() for p in group['params'])) for group in optimizer.param_groups])
        batches = plan.batches(epoch)
        if not batches:
            raise ValueError('No complete training-source epoch exists')
        total_phase_steps = len(batches) * cfg.get(phase_name + '_epochs', 1 if phase_name == 'head' else 5)
        seed_all(cfg['seed'] + epoch * 100003)
        model.train()
        aggregate, composition, meters = defaultdict(float), Counter(), defaultdict(Metrics)
        phase(f'V3.5 {phase_name} epoch {epoch} steps 1-{len(batches)}')
        selected_batches = batches[:smoke_steps] if smoke_steps else batches
        timed = TimedLoader(loader(plan, cfg, training=True, epoch=epoch, batches=selected_batches))
        progress = training_bar(timed, len(selected_batches), f'V3.5 {phase_name} epoch {epoch}')
        for cursor, examples in enumerate(progress, 1):
            started = time.perf_counter()
            if len({row['source_id'] for row in examples}) != len(selected_batches[cursor - 1]):
                raise RuntimeError('Prepared views changed the unique-source exposure budget')
            scale = schedule_scale(phase_steps, total_phase_steps, cfg.get('warmup_fraction', .05),
                                   cfg.get('min_lr_scale', .1))
            apply_learning_rates(optimizer, scale)
            stats, logits = supervised_step(model, examples, optimizer, weights, device, cfg.get('amp', 'bf16'),
                source_denominator=cfg.get('source_batch_size', cfg.get('source_batch', 16)),
                source_microbatch=cfg.get('source_chunk', 4),
                offline_weight=cfg.get('offline_weight', .1), online_weight=cfg.get('online_weight', .3),
                noisy_weight=cfg.get('noisy_weight', .6), grad_clip=cfg.get('grad_clip', 1.),
                microbatch=cfg.get('microbatch', 4), frame_budget=cfg.get('frame_budget', 1600),
                offload_activations=cfg.get('offload_activations', True),
                activation_budget_gib=cfg.get('activation_budget_gib', 0.),
                gpu_activation_gib=cfg.get('gpu_activation_gib', 18.), gpu_reserve_gib=cfg.get('gpu_reserve_gib', 8.))
            ema.update(model)
            if stats['source_count'] != len(selected_batches[cursor - 1]):
                raise RuntimeError('Prepared views changed the unique-source exposure budget')
            global_steps += 1; phase_steps += 1
            elapsed = time.perf_counter() - started
            for key, value in stats.items():
                aggregate[key] += value
            for index, row in enumerate(examples):
                group = row['condition'] + '/' + row['language'] + '/full'
                meters[group].update(logits[index:index + 1], [row['label']])
                composition[row['condition'] + '/' + row['language'] + '/' + str(row['label'])] += 1
            performance = dict(global_step=global_steps, epoch=epoch, phase=phase_name,
                sources=stats['source_count'], encoded_views=len(examples),
                data_wait_seconds=timed.last_wait, compute_seconds=elapsed,
                wall_seconds=elapsed + timed.last_wait,
                sources_per_second=stats['source_count'] / max(1e-9, elapsed + timed.last_wait), **stats)
            with (run / 'performance.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(performance) + '\n')
            atomic_json(run / 'performance_latest.json', performance)
            progress.set_postfix(loss=f'{stats["loss"]:.5f}')
        if smoke_steps:
            atomic_json(run / 'smoke.json', dict(steps=len(selected_batches), optimizer_updates=global_steps,
                metrics=dict(aggregate), saved_training_boundary='bootstrap_or_last_completed_epoch'))
            return
        del progress, timed
        tag = 'epoch_' + str(epoch)
        phase('V3.5 validating ' + tag + ' EMA (one inference pass)')
        replacing = []
        with ema.average_parameters(model):
            dev = validate(model, validation, cfg, device, run / (tag + '_scores.jsonl'))
            decision = controller.observe(dev, tag, phase_name)
            # Scalars survive a later large-checkpoint failure. This is explicitly
            # pending and does not claim that last.pt or selection has committed.
            atomic_json(run/(tag+'_pending.json'),dict(tag=tag,phase=phase_name,
                epoch=epoch,dev=dev,decision=decision,checkpoint_committed=False))
            print('Dev measured '+tag+' '+ ' '.join(f'{key}={100*dev[key]:.3f}'
                for key in ('clean_f1','noisy_f1','weighted_f1'))+'; checkpoint pending',flush=True)
            for name in decision['save']:
                filename = 'best_model.pt' if name == 'best_model' else 'best_candidate.pt'
                path, previous = run / filename, run / (filename + '.previous')
                if previous.exists():
                    raise RuntimeError('Unreconciled checkpoint transaction; resume last.pt')
                if path.exists():
                    os.replace(path, previous)
                replacing.append(filename)
                atomic_save(path, _save_weights(model, cfg, tag, phase_name, epoch, dev, fingerprints, codes))
        rates = {group['name']: group['lr'] for group in optimizer.param_groups}
        if decision['promoted']:
            decision['save'].append('best_safe')  # Shared terminal viewer's promotion label.
        record = dict(tag=tag, phase=phase_name, arm='V3.5', epoch=epoch, cursor=len(batches), global_steps=global_steps,
            dev=dev, decision=decision, weight_source='ema', learning_rates=rates,
            train={key: value / len(batches) for key, value in aggregate.items()},
            train_groups={key: meter.result() for key, meter in meters.items()}, composition_counts=dict(composition))
        history.append(record)
        complete = controller.state['completed']
        phase('V3.5 saving selected EMA and exact raw training state')
        atomic_save(run / 'last.pt', training_snapshot(tag, dev))
        for filename in replacing:
            previous = run / (filename + '.previous')
            if previous.exists():
                previous.unlink()
        atomic_json(run / (tag + '.json'), record)
        write_report(run, history, controller, 'complete' if complete else 'training')
        print_validation(record)
        pending=run/(tag+'_pending.json')
        if pending.exists():
            pending.unlink()
        # Workers have been exhausted and the new resume boundary is durable.
        retire_generations(cfg, keep_epochs=[epoch])
    verify_protected(cfg)
    completed = dict(selection=controller.dump(), selected_tag=controller.state['best_selected']['tag'],
        eligible_improvement=controller.state['best_selected']['tag'] != 'reference', reference_preserved=True,
        reason='joint_plateau' if controller.state['joint_stale_epochs'] >= cfg.get('early_stop_patience', 2)
               else 'joint_budget', epochs=epoch, global_steps=global_steps)
    atomic_json(run / 'completed.json', completed)
    print('V35_TRAINING_COMPLETE=True REFERENCE_PRESERVED=True', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    from w2v_aasist.launch import run_lock
    with run_lock(Path(args.out) / '.lock'):
        train(json.loads(Path(args.config).read_text(encoding='utf-8')), args.out, args.resume, args.smoke_steps)


if __name__ == '__main__':
    main()
