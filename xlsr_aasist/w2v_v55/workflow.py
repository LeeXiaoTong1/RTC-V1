"""V5.1 submitted-final continuation with no screening on trained Dev."""
import copy
from datetime import datetime
from pathlib import Path
import secrets
import signal
import sys
import traceback
import torch
from .common import (ROOT, RECIPE, atomic_json, read_json, digest, fingerprint,
    verify_files, apply_partial, atomic_save, schema_for)
from .config import parser, validate, training_configuration, code_inventory, verify_inputs, compact_configuration
from .parent import metadata, load as load_parent, locked_weights
from .train import run_stage, make_plan
from w2v_v5.manifests import apply_coverage_budgets, bind_parent_phonemes
from .resources import audit_training
from w2v_v5.weights import average_states, invariant_keys
from w2v_v31810.identity import verify_registry
from w2v_aasist.launch import run_lock
from w2v_v3191.processes import ensure_idle


def persist_stage(run, cfg, rows, registry):
    run.mkdir(parents=True, exist_ok=True)
    files = {'train_rows.json': rows, 'source_registry.json': registry}
    for name, value in files.items():
        atomic_json(run / name, value)
    cfg['saved_manifests'] = {name: digest(run/name) for name in files}
    atomic_json(run/'config.json', cfg)
    atomic_json(run/'run_summary.json', compact_configuration(cfg))


def read_stage(run):
    cfg = read_json(run/'config.json')
    schema_for(cfg)
    if any(Path(name).name != name for name in cfg['saved_manifests']):
        raise ValueError('Manifest path escapes the run')
    verify_files({str(run/name): h for name, h in cfg['saved_manifests'].items()})
    rows = read_json(run/'train_rows.json')
    registry = read_json(run/'source_registry.json')
    verify_registry(rows, registry, cfg, {'source_registry_hash': registry['registry_hash']})
    verify_inputs(cfg)
    return cfg, rows


def continuation_lock(root, cfg):
    origin = cfg['parent_init']
    inherited = origin['locked_recipe']
    lock = dict(version='5.5', recipe=RECIPE, base_config=fingerprint(cfg),
        parent_checkpoint=origin['checkpoint_sha256'], parent_recipe=origin['recipe_sha256'],
        input=copy.deepcopy(inherited['input']), epochs=cfg['epochs'],
        context_seconds=cfg['v5_context_seconds'], weights=locked_weights(inherited),
        selection=None, candidates=[],
        final_policy='Continue exact V5.1 final; inherited weight rule anchored to V5.1; no Dev selection',
        hard_positive={k: v for k, v in cfg.items() if k.startswith('hp_')})
    if cfg['v5_input'] != lock['input']:
        raise ValueError('Preserve the V5.1 input policy')
    path = root/'recipe_lock.json'
    if path.exists():
        if read_json(path) != lock:
            raise ValueError('Locked continuation recipe changed')
    else:
        atomic_json(path, lock)
    return lock


def final_fit(root, cfg):
    lock = continuation_lock(root, cfg)
    stage = root/'final'
    if not (stage/'config.json').exists():
        _, origin, rows, dev = metadata(cfg['parent_init']['run'])
        if origin != cfg['parent_init'] or dev:
            raise ValueError('V5.1 lineage changed or trained Dev was reused for selection')
        final_cfg = copy.deepcopy(cfg)
        final_cfg['v5_recipe_lock_sha256'] = digest(root/'recipe_lock.json')
        bind_parent_phonemes(final_cfg, rows)
        coverage = apply_coverage_budgets(final_cfg, rows)
        if final_cfg['source_draws_per_stream'] != cfg['source_draws_per_stream']:
            raise ValueError('V5.1 final source exposure budget changed')
        # Exact same verified final rows and budgets were audited at preparation.
        final_cfg['window_resource_audit'] = copy.deepcopy(cfg['window_resource_audit'])
        atomic_json(root/'final_data_audit.json', dict(mode='exact_V5.1_Train_Dev',
            parent=origin['run'], training_rows=len(rows), evaluation_rows=0, budget=coverage))
        persist_stage(stage, final_cfg, rows, read_json(origin['source_registry']))
    final_cfg, rows = read_stage(stage)
    if final_cfg['v5_recipe_lock_sha256'] != digest(root/'recipe_lock.json'):
        raise ValueError('Locked recipe changed after fitting started')
    plan = make_plan(rows, final_cfg)
    print(f'[V5.5 final budget] {plan.steps} updates/epoch; V5.1 source coverage; '
          'Online25/CleanRTC25/Noisy50; up to 4 extra Noisy views/update', flush=True)
    model, state = run_stage(final_cfg, stage, rows)
    if state['status'] != 'complete':
        return False
    _, parent, _, _ = load_parent(cfg['parent_init']['run'], cfg['parent_init'])
    trained = state['tail_model'] if lock['weights']['source'] == 'tail' else state['model']
    alpha = lock['weights']['parent_weight']
    merged = average_states([parent['best_model'], trained], [alpha, 1-alpha],
                            constant_keys=invariant_keys(model, trained))
    tag = 'final_' + lock['weights']['name']
    atomic_save(root/'final.pt', dict(schema=schema_for(final_cfg), identity=fingerprint(final_cfg),
        model=merged, tag=tag, recipe=lock), cfg.get('free_reserve_bytes', 0))
    atomic_json(root/'final_config.json', final_cfg)
    atomic_json(root/'completed.json', dict(status='complete', version='5.5', selected=tag,
        checkpoint_sha256=digest(root/'final.pt'), final_config_sha256=digest(root/'final_config.json'),
        recipe_sha256=digest(root/'recipe_lock.json'), evaluation=None,
        note='Single model, fixed V5.1 weight rule; trained Dev cannot validate generalization.'))
    return True


def main():
    args = parser().parse_args()
    validate(args, sys.argv[1:])
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
        if not cuda_available or not bf16_supported:
            reason = 'CUDA initialization failed or no GPU is available' if not cuda_available else 'the active GPU/runtime does not support BF16'
            raise RuntimeError(
                f'V5.5 requires the pinned Linux BF16 CUDA environment: {reason}.\n'
                f'Python: {sys.executable}\n'
                f'PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}\n'
                'For the original project server, activate the full environment prefix:\n'
                '  conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318\n'
                'The same-named ~/.conda/envs/sdd-v318 is a different environment. '
                'On another server, use the full prefix of its verified V5.1 environment.'
            )
        if args.resume:
            root = Path(args.resume).expanduser().resolve()
            cfg, _ = read_stage(root)
        else:
            old, origin, rows, dev = metadata(args.parent_run)
            if dev:
                raise ValueError('Only completed V5.1 final continuation is supported')
            cfg = training_configuration(old, origin, args)
            bind_parent_phonemes(cfg, rows)
            cfg['window_resource_audit'] = audit_training(rows, [], cfg)
            cfg['code_fingerprints'] = code_inventory()
            verify_inputs(cfg)
            root = ROOT/'exp'/('w2v_v55_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(2))
            root.mkdir(parents=True, exist_ok=False)
            persist_stage(root, cfg, rows, read_json(origin['source_registry']))
        (ROOT/'exp'/'.latest_v55_run').write_text(str(root)+'\n', encoding='utf-8')
        print('V5.5_RUN=' + str(root), flush=True)
        print('[V5.5 rollback] exact V5.1 final: ' + cfg['parent_init']['checkpoint'], flush=True)
        print('[V5.5 architecture] V5.1 static layer fusion + AASIST 384d readout; '
              'no V5.2 contrast bank, V5.3 conditioner, V5.4 repair/KD', flush=True)
        print('[V5.5 metric] squared Euclidean, 2 positives x 3 negatives; '
              f'm={cfg["hp_guidance_margin"]}, alpha={cfg["hp_triplet_margin"]}, '
              f'lambdaG={cfg["hp_guidance_weight"]}, lambdaT={cfg["hp_triplet_weight"]}, '
              f'ramp={cfg["hp_ramp_updates"]}, mode={cfg["hp_mode"]}', flush=True)
        (root/'STOP_REQUESTED').unlink(missing_ok=True)
        (root/'paused.json').unlink(missing_ok=True)
        def request_stop(signum, frame):
            (root/'STOP_REQUESTED').touch()
            print('\n[V5.5] stopping at next committed update; checkpoint will be retained', flush=True)
        previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if not (root/'completed.json').is_file() and not final_fit(root, cfg):
                atomic_json(root/'paused.json', dict(status='paused', resume=str(root)))
                print('V5.5_PAUSED=' + str(root), flush=True)
                return
            from .delivery import finish
            finish(root, args)
            (root/'failure.log').unlink(missing_ok=True)
        except BaseException:
            (root/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


if __name__ == '__main__':
    main()
