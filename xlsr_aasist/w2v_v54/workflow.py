"""Completed V5.3 continuation with a fixed recipe and no trained-Dev selection."""
import copy
from datetime import datetime
import gc
from pathlib import Path
import secrets
import signal
import sys
import traceback
import torch
from .common import (ROOT, RECIPE, atomic_json, read_json, digest, fingerprint,
                     verify_files, apply_partial, atomic_save, schema_for)
from .config import parser, validate, training_configuration, code_inventory, verify_inputs, compact_configuration
from w2v_v5.manifests import apply_coverage_budgets, bind_parent_phonemes
from w2v_v5.weights import average_states, invariant_keys
from .train import run_stage, make_plan
from w2v_v5.resources import audit_training
from .model import parent_anchor
from .parent import metadata, load as load_parent, is_final_parent, locked_weights
from w2v_aasist.launch import run_lock
from w2v_v3191.processes import ensure_idle


def persist_stage(run, cfg, rows, registry, dev=None):
    run.mkdir(parents=True, exist_ok=True)
    files = {"train_rows.json": rows, "source_registry.json": registry}
    if dev is not None:
        files["dev_rows.json"] = dev
    for name, value in files.items():
        atomic_json(run / name, value)
    cfg["saved_manifests"] = {name: digest(run/name) for name in files}
    atomic_json(run / "config.json", cfg)
    atomic_json(run / "run_summary.json", compact_configuration(cfg))


def read_stage(run):
    cfg = read_json(run / "config.json")
    verify_files({str(run/name): h for name, h in cfg["saved_manifests"].items()})
    rows = read_json(run / "train_rows.json")
    from w2v_v31810.identity import verify_registry
    registry = read_json(run / "source_registry.json")
    verify_registry(rows, registry, cfg, {"source_registry_hash": registry["registry_hash"]})
    verify_inputs(cfg)
    return cfg, rows


def interrupted(root):
    if (root / "STOP_REQUESTED").exists():
        atomic_json(root / "paused.json", dict(status="paused", resume=str(root)))
        print("V5.4_PAUSED=" + str(root), flush=True)
        return True
    return False


def continuation_lock(root, cfg):
    """Fix export rules before training; the anchor is this submitted parent."""
    origin = cfg["parent_init"]
    inherited = origin["locked_recipe"]
    lock = dict(version="5.4", recipe=RECIPE, base_config=fingerprint(cfg),
        parent_checkpoint=origin["checkpoint_sha256"], parent_recipe=origin["recipe_sha256"],
        input=copy.deepcopy(inherited["input"]), epochs=cfg["epochs"],
        context_seconds=cfg["v5_context_seconds"], weights=locked_weights(inherited),
        selection=None, candidates=[],
        final_policy="Continue exact submitted V5 final; inherited weight rule anchored to that final; no Dev selection",
        evidence="Parent trained on all Train+Dev; continuation quality remains unmeasured")
    if cfg["v5_input"] != lock["input"]:
        raise ValueError("Continuation must retain the submitted parent's input policy")
    path = root / "recipe_lock.json"
    if path.exists():
        if read_json(path) != lock:
            raise ValueError("Inherited continuation recipe changed")
    else:
        atomic_json(path, lock)
    return lock


def final_fit(root, cfg):
    if not is_final_parent(cfg):
        raise ValueError('V5.4 requires completed V5.3 parent')
    continuation_lock(root, cfg)
    lock = read_json(root / "recipe_lock.json")
    if lock["base_config"] != fingerprint(cfg):
        raise ValueError("Final recipe belongs to another run")
    stage = root / "final"
    if not (stage / "config.json").exists():
        old, origin, train, dev = metadata(cfg["parent_init"]["run"])
        if origin != cfg["parent_init"]:
            raise ValueError("Parent changed before final fitting")
        final_cfg = copy.deepcopy(cfg)
        final_cfg.update(v5_stage="final", v5_input=lock["input"],
                         v5_recipe_lock_sha256=digest(root/"recipe_lock.json"))
        print("[V5.4 continuation] reusing submitted parent's ALL Train+Dev; no Dev selection", flush=True)
        registry = read_json(origin["source_registry"])
        audit = dict(mode="reuse_verified_parent_final", parent=origin["run"],
                     training_rows=len(train), evaluation_rows=0,
                     source_registry_hash=registry["registry_hash"])
        if dev:
            raise ValueError("Final fitting cannot retain trained Dev as selection data")
        bind_parent_phonemes(final_cfg, train)
        coverage = apply_coverage_budgets(final_cfg, train)
        final_cfg["window_resource_audit"] = audit_training(train, [], final_cfg)
        plan = make_plan(train, final_cfg)
        print(f'[V5.4 final budget] {plan.steps} updates/epoch; all canonical sources covered; '
              'loss mass stays Online25/CleanRTC25/Noisy50', flush=True)
        atomic_json(root / "final_data_audit.json", dict(manifests=audit, budget=coverage))
        persist_stage(stage, final_cfg, train, registry)
    final_cfg, train = read_stage(stage)
    if final_cfg["v5_recipe_lock_sha256"] != digest(root/"recipe_lock.json"):
        raise ValueError("Locked recipe changed after final fitting started")
    model, state = run_stage(final_cfg, stage, train)
    if state["status"] != "complete":
        return False
    _, parent, _, _ = load_parent(cfg["parent_init"]["run"], cfg["parent_init"])
    trained = state["tail_model"] if lock["weights"]["source"] == "tail" else state["model"]
    alpha = lock["weights"]["parent_weight"]
    merged = average_states([parent_anchor(model, parent["best_model"]), trained], [alpha, 1-alpha],
                            constant_keys=invariant_keys(model, trained))
    def repair_norms(values):
        return {part: float(torch.stack([v.float().square().sum() for name,v in values.items()
            if name.startswith('repair.'+part)]).sum().sqrt()) for part in ('out.', 'gate.')}
    atomic_json(root/'export_repair_audit.json', dict(parent_weight=alpha,
        trained=repair_norms(trained), exported=repair_norms(merged),
        includes_repair=True, frame_scale_equal=torch.equal(trained['repair.frame_scale'], merged['repair.frame_scale']),
        note='Locked parameter averaging includes repair. Parameter norms are not quality metrics; nonlinear output may be attenuated.'))
    tag = "final_" + lock["weights"]["name"]
    payload = dict(schema=schema_for(final_cfg), identity=fingerprint(final_cfg),
                   model=merged, tag=tag, recipe=lock)
    atomic_save(root/"final.pt", payload, cfg.get("free_reserve_bytes", 0))
    atomic_json(root/"final_config.json", final_cfg)
    atomic_json(root/"completed.json", dict(status="complete", version="5.4", selected=tag,
        checkpoint_sha256=digest(root/"final.pt"), final_config_sha256=digest(root/"final_config.json"),
        recipe_sha256=digest(root/"recipe_lock.json"), evaluation=None,
        note="Single merged model. Dev entered final training; no post-fit validation score."))
    del model, state, parent, merged
    gc.collect()
    torch.cuda.empty_cache()
    return True


def run_training(root, cfg, train, dev, registry, phase):
    if not is_final_parent(cfg) or dev or phase not in ('all', 'final'):
        raise ValueError('V5.4 continues completed V5.3 Train+Dev without selection')
    return final_fit(root, cfg)


def main():
    args = parser().parse_args()
    validate(args, sys.argv[1:])
    ensure_idle()
    with run_lock(ROOT/"exp"/".aasist-launch.lock"):
        ensure_idle()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("V5.4 production training requires the pinned Linux BF16 CUDA environment")
        if args.resume:
            root = Path(args.resume).expanduser().resolve()
            cfg, train = read_stage(root)
            dev = read_json(root/"dev_rows.json")
            registry = read_json(root/"source_registry.json")
        else:
            old, origin, train, dev = metadata(args.parent_run)
            cfg = training_configuration(old, origin, args)
            registry = read_json(origin["source_registry"])
            audit = dict(mode="reuse_verified_parent_final", parent=origin["run"],
                training_rows=len(train), evaluation_rows=0,
                source_registry_hash=registry["registry_hash"])
            bind_parent_phonemes(cfg, train)
            cfg["window_resource_audit"] = audit_training(train, dev, cfg)
            cfg["code_fingerprints"] = code_inventory()
            verify_inputs(cfg)
            root = ROOT/"exp"/("w2v_v54_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+secrets.token_hex(2))
            root.mkdir(parents=True, exist_ok=False)
            persist_stage(root, cfg, train, registry, dev)
            atomic_json(root/"continuation_data_audit.json", audit)
        (ROOT/"exp"/".latest_v54_run").write_text(str(root)+"\n", encoding="utf-8")
        print("V5.4_RUN="+str(root), flush=True)
        print('[V5.4 objective] CE Online25/CleanRTC25/Noisy50; '
              f'PCL noise={cfg["pcl_noise_weight"]:.8g}, platform={cfg["pcl_platform_weight"]:.8g}; '
              f'arm={cfg["repair_arm"]}; LF={cfg["repair_feature_weight"]}, KD={cfg["repair_kd_weight"]}', flush=True)
        if is_final_parent(cfg):
            origin = cfg["parent_init"]
            print(f'[V5.4 parent] {origin["checkpoint"]} | {origin["tag"]} | '
                  f'sha256={origin["checkpoint_sha256"]}', flush=True)
            print(f'[V5.4 continuation] {cfg["epochs"]} epoch(s) TOTAL; no input/weight screen; '
                  f'export rule={locked_weights(origin["locked_recipe"])["name"]}', flush=True)
        (root/"STOP_REQUESTED").unlink(missing_ok=True)
        (root/"paused.json").unlink(missing_ok=True)
        def request_stop(signum, _frame):
            (root/"STOP_REQUESTED").touch()
            print("\n[V5.4] stopping at next committed update/evaluation boundary; checkpoint will be retained", flush=True)
        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if not (root/"completed.json").is_file():
                if not run_training(root, cfg, train, dev, registry, args.phase):
                    interrupted(root)
                    return
            from .delivery import finish, report
            if (root/"completed.json").is_file():
                finish(root, args)
            else:
                report(root, args.download_dir, upload=args.upload_temp)
                print("V5.4_SCREEN_COMPLETE="+str(root), flush=True)
            (root/"failure.log").unlink(missing_ok=True)
        except BaseException:
            (root/"failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
