"""One screen continuation, then one fixed Train+Dev refit from V3.18.9 best."""
import copy
from datetime import datetime
import gc
from pathlib import Path
import secrets
import signal
import sys
import traceback
import numpy as np
import torch
from .common import (ROOT, RECIPE, atomic_json, read_json, digest, fingerprint,
                     verify_files, apply_partial, atomic_save, schema_for)
from .config import parser, validate, training_configuration, code_inventory, verify_inputs
from .manifests import prepare_manifests, apply_coverage_budgets, bind_parent_phonemes
from .processing import candidate_recipes
from .selection import score, balanced_source_subset, choose_candidate
from .inference import infer
from .weights import average_states, invariant_keys
from .train import run_stage, initialize, make_plan
from .resources import audit_training
from w2v_v3191.model import load_model
from w2v_v3191.parent import metadata, load as load_parent
from w2v_v3184.records import dev_partition
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


def read_stage(run):
    cfg = read_json(run / "config.json")
    verify_files({str(run/name): h for name, h in cfg["saved_manifests"].items()})
    rows = read_json(run / "train_rows.json")
    from w2v_v31810.identity import verify_registry
    registry = read_json(run / "source_registry.json")
    verify_registry(rows, registry, cfg, {"source_registry_hash": registry["registry_hash"]})
    verify_inputs(cfg)
    return cfg, rows


def checked_eval(model, rows, cfg, root, name, checkpoint_identity):
    """Restartable evaluation bound to actual checkpoint and input policy."""
    folder = root / "evaluations"
    folder.mkdir(exist_ok=True)
    path = folder / (name + ".json")
    identity = fingerprint(dict(checkpoint=checkpoint_identity, cfg=fingerprint(cfg),
                                rows=[r["id"] for r in rows], name=name))
    if path.is_file():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError("Cached evaluation inputs changed: " + name)
        verify_files({str(folder/saved["scores"]): saved["scores_sha256"]})
        return saved["metrics"]
    print(f'[V5 evaluate] {name}: {len(rows)} recordings', flush=True)
    z, _ = infer(model, rows, cfg)
    metrics = score(rows, z)
    scores = folder / (name + ".npz")
    np.savez_compressed(scores, logits=z)
    atomic_json(path, dict(identity=identity, checkpoint=checkpoint_identity, metrics=metrics,
        scores=scores.name, scores_sha256=digest(scores), input_policy=cfg["v5_input"],
        scope="Historical Dev partition; previously viewed in earlier experiments, not a fresh test"))
    print(f'[V5 evaluate] {name} Clean={metrics["clean_f1"]*100:.3f} '
          f'Noisy={metrics["noisy_f1"]*100:.3f} Weighted={metrics["weighted_f1"]*100:.3f}', flush=True)
    return metrics


def interrupted(root):
    if (root / "STOP_REQUESTED").exists():
        atomic_json(root / "paused.json", dict(status="paused", resume=str(root)))
        print("V5_PAUSED=" + str(root), flush=True)
        return True
    return False


def screen_input(root, cfg, dev, partition):
    path = root / "input_lock.json"
    if path.is_file():
        lock = read_json(path)
        if lock["base_config"] != fingerprint(cfg):
            raise ValueError("Input screen binding changed")
        return lock["policy"]
    model = load_model(cfg)
    initialize(model, cfg)
    parent_id = cfg["parent_init"]["checkpoint_sha256"] + ":" + cfg["parent_init"]["tag"]
    subset = balanced_source_subset(dev, partition["select"], cfg["v5_screen_sources_per_group"],
                                    cfg["sampling_seed"], source_keys=partition["source_keys"])
    rows = [dev[i] for i in subset]
    results = []
    for candidate in candidate_recipes():
        if interrupted(root):
            return None
        selected_cfg = dict(cfg, v5_input=candidate["policy"])
        metrics = checked_eval(model, rows, selected_cfg, root, "input_small_"+candidate["name"], parent_id)
        results.append(dict(candidate, metrics=metrics))
    provisional = choose_candidate(results)
    full_rows = [dev[i] for i in partition["select"]]
    choices = [candidate_recipes()[0]]
    if provisional["name"] != "identity":
        choices.append({k: provisional[k] for k in ("name", "policy")})
    confirmed = []
    for candidate in choices:
        if interrupted(root):
            return None
        metrics = checked_eval(model, full_rows, dict(cfg, v5_input=candidate["policy"]),
                               root, "input_full_"+candidate["name"], parent_id)
        confirmed.append(dict(candidate, metrics=metrics))
    chosen = choose_candidate(confirmed)
    lock = dict(base_config=fingerprint(cfg), policy=chosen["policy"], name=chosen["name"],
        subset_indices=subset, small_screen=results, full_confirmation=confirmed,
        selection_scope="Dev Select only; identity fallback included; no test audio inspected")
    atomic_json(path, lock)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[V5 input locked] " + chosen["name"], flush=True)
    return chosen["policy"]


def weight_candidates(parent, state, constants):
    return [
        dict(name="anchor_tail_50", source="tail", parent_weight=.5, conservative_rank=0,
             model=average_states([parent, state["tail_model"]], [.5, .5], constant_keys=constants)),
        dict(name="tail_mean", source="tail", parent_weight=0., conservative_rank=1,
             model=state["tail_model"]),
        dict(name="last", source="last", parent_weight=0., conservative_rank=2,
             model=state["model"])]


def locked_audit(root, cfg, stage_cfg, model, state, dev, partition):
    """Resume this report without reopening selection or inspecting final Train."""
    path = root/"screen_completed.json"
    lock = read_json(root/"recipe_lock.json")
    if lock["base_config"] != fingerprint(cfg):
        raise ValueError("Locked recipe belongs to different screening inputs")
    checkpoint_id = digest(root/"screen"/"last.pt")
    if checkpoint_id != lock["screen_checkpoint"]:
        raise ValueError("Screen checkpoint changed after selection")
    if path.exists():
        if read_json(path)["recipe_sha256"] != digest(root/"recipe_lock.json"):
            raise ValueError("Audit belongs to a different locked recipe")
        return True
    if interrupted(root):
        return False
    _, parent, _, _ = load_parent(cfg["parent_init"]["run"], cfg["parent_init"])
    trained = state["tail_model"] if lock["weights"]["source"] == "tail" else state["model"]
    alpha = lock["weights"]["parent_weight"]
    selected = average_states([parent["best_model"], trained], [alpha, 1-alpha],
                              constant_keys=invariant_keys(model, trained))
    apply_partial(model, selected)
    audit = checked_eval(model, [dev[i] for i in partition["calibration"]],
                         stage_cfg, root, "locked_audit", checkpoint_id+":"+lock["weights"]["name"])
    atomic_json(path, dict(recipe_sha256=digest(root/"recipe_lock.json"), audit=audit,
                          audit_used_for_selection=False, status="complete"))
    return not interrupted(root)


def screen(root, cfg, rows, dev, registry):
    partition = dev_partition(dev, cfg["dev_pairs"], cfg["seed"], cfg["calibration_fraction"])
    if partition != read_json(Path(cfg["parent_init"]["run"]) / "dev_partition.json"):
        raise ValueError("The source-isolated parent Dev partition changed")
    atomic_json(root / "dev_partition.json", partition)
    policy = screen_input(root, cfg, dev, partition)
    if policy is None:
        return False
    stage = root / "screen"
    if not (stage / "config.json").exists():
        selected_cfg = dict(cfg, v5_input=policy)
        persist_stage(stage, selected_cfg, rows, registry)
    stage_cfg, rows = read_stage(stage)
    model, state = run_stage(stage_cfg, stage, rows)
    if state["status"] != "complete":
        return False
    if (root / "recipe_lock.json").exists():
        complete = locked_audit(root, cfg, stage_cfg, model, state, dev, partition)
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
        return complete
    _, parent_state, _, _ = load_parent(cfg["parent_init"]["run"], cfg["parent_init"])
    parent = parent_state["best_model"]
    constants = invariant_keys(model, parent)
    select_rows = [dev[i] for i in partition["select"]]
    checkpoint_id = digest(stage / "last.pt")
    candidates = weight_candidates(parent, state, constants)
    results = []
    for candidate in candidates:
        if interrupted(root):
            return False
        apply_partial(model, candidate["model"])
        metrics = checked_eval(model, select_rows, stage_cfg, root, "weights_"+candidate["name"],
                               checkpoint_id+":"+candidate["name"])
        results.append(dict((k, v) for k, v in candidate.items() if k != "model") | {"metrics": metrics})
    chosen = choose_candidate(results)
    input_lock = read_json(root / "input_lock.json")
    baseline = next(c for c in input_lock["full_confirmation"] if c["policy"] == policy)["metrics"]
    lock = dict(version="5", recipe=RECIPE, base_config=fingerprint(cfg),
        screen_checkpoint=checkpoint_id, input=policy, epochs=cfg["epochs"],
        context_seconds=cfg["v5_context_seconds"],
        weights={k: chosen[k] for k in ("name", "source", "parent_weight")},
        selection=chosen["metrics"], parent_same_input=baseline, candidates=results,
        screen_gain=chosen["metrics"]["weighted_f1"]-baseline["weighted_f1"],
        final_policy="Restart parent best; fixed budget on all Train+Dev; no Dev selection afterward",
        evidence="Local Dev proxies only; no guarantee of Progress improvement")
    atomic_json(root / "recipe_lock.json", lock)
    if lock["screen_gain"] <= 0:
        print("[V5 evidence] No screened continuation exceeded the parent with the same input policy. "
              "The fixed full-Dev refit is still scheduled; improvement has NOT been established.", flush=True)
    # Informational only; recipe is already persisted and cannot be changed by Audit.
    complete = locked_audit(root, cfg, stage_cfg, model, state, dev, partition)
    print(f'[V5 recipe locked] {chosen["name"]}; Dev Select gain={lock["screen_gain"]*100:+.3f} points. '
          'Final Train+Dev result remains unmeasured.', flush=True)
    del model, state, candidates, parent_state, parent
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def final_fit(root, cfg):
    lock = read_json(root / "recipe_lock.json")
    if lock["base_config"] != fingerprint(cfg):
        raise ValueError("Final recipe belongs to another run")
    if digest(root / "screen" / "last.pt") != lock["screen_checkpoint"]:
        raise ValueError("Screen checkpoint changed after recipe lock")
    stage = root / "final"
    if not (stage / "config.json").exists():
        old, origin, train, dev = metadata(cfg["parent_init"]["run"])
        if origin != cfg["parent_init"]:
            raise ValueError("Parent changed before final fitting")
        final_cfg = copy.deepcopy(cfg)
        final_cfg.update(v5_stage="final", v5_input=lock["input"],
                         v5_recipe_lock_sha256=digest(root/"recipe_lock.json"))
        print("[V5 final] preparing ALL official Train+Dev; Dev will no longer be used to select", flush=True)
        train, evaluation, registry, audit = prepare_manifests(final_cfg, train, dev, phase="final")
        if evaluation:
            raise ValueError("Final fitting cannot retain trained Dev as selection data")
        bind_parent_phonemes(final_cfg, train)
        coverage = apply_coverage_budgets(final_cfg, train)
        final_cfg["window_resource_audit"] = audit_training(train, [], final_cfg)
        plan = make_plan(train, final_cfg)
        print(f'[V5 final budget] {plan.steps} updates/epoch; all canonical sources covered; '
              'loss mass stays Noisy50/Online25/CleanRTC25', flush=True)
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
    merged = average_states([parent["best_model"], trained], [alpha, 1-alpha],
                            constant_keys=invariant_keys(model, trained))
    tag = "final_" + lock["weights"]["name"]
    payload = dict(schema=schema_for(final_cfg), identity=fingerprint(final_cfg),
                   model=merged, tag=tag, recipe=lock)
    atomic_save(root/"final.pt", payload, cfg.get("free_reserve_bytes", 0))
    atomic_json(root/"final_config.json", final_cfg)
    atomic_json(root/"completed.json", dict(status="complete", version="5", selected=tag,
        checkpoint_sha256=digest(root/"final.pt"), final_config_sha256=digest(root/"final_config.json"),
        recipe_sha256=digest(root/"recipe_lock.json"), evaluation=None,
        note="Single merged model. Dev entered final training; no post-fit validation score."))
    del model, state, parent, merged
    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    args = parser().parse_args()
    validate(args, sys.argv[1:])
    ensure_idle()
    with run_lock(ROOT/"exp"/".aasist-launch.lock"):
        ensure_idle()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("V5 production training requires the pinned Linux BF16 CUDA environment")
        if args.resume:
            root = Path(args.resume).expanduser().resolve()
            cfg, train = read_stage(root)
            dev = read_json(root/"dev_rows.json")
            registry = read_json(root/"source_registry.json")
            if args.phase == "final" and not (root/"recipe_lock.json").is_file():
                raise ValueError("Complete screening first; final cannot guess a recipe")
        else:
            old, origin, train, dev = metadata(args.parent_run)
            cfg = training_configuration(old, origin, args)
            print("[V5 setup] verify real PCM source identities and inherited assets", flush=True)
            train, dev, registry, audit = prepare_manifests(cfg, train, dev, phase="screen")
            bind_parent_phonemes(cfg, train)
            cfg["window_resource_audit"] = audit_training(train, dev, cfg)
            cfg["code_fingerprints"] = code_inventory()
            verify_inputs(cfg)
            root = ROOT/"exp"/("w2v_v5_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+secrets.token_hex(2))
            root.mkdir(parents=True, exist_ok=False)
            persist_stage(root, cfg, train, registry, dev)
            atomic_json(root/"screen_data_audit.json", audit)
        (ROOT/"exp"/".latest_v5_run").write_text(str(root)+"\n", encoding="utf-8")
        print("V5_RUN="+str(root), flush=True)
        (root/"STOP_REQUESTED").unlink(missing_ok=True)
        (root/"paused.json").unlink(missing_ok=True)
        def request_stop(signum, _frame):
            (root/"STOP_REQUESTED").touch()
            print("\n[V5] stopping at next committed update/evaluation boundary; checkpoint will be retained", flush=True)
        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if not (root/"completed.json").is_file():
                if args.phase != "final" or not (root/"screen_completed.json").exists():
                    if not screen(root, cfg, train, dev, registry):
                        interrupted(root)
                        return
                if args.phase != "screen":
                    if not final_fit(root, cfg):
                        interrupted(root)
                        return
            from .delivery import finish, report
            if (root/"completed.json").is_file():
                finish(root, args)
            else:
                report(root, args.download_dir, upload=args.upload_temp)
                print("V5_SCREEN_COMPLETE="+str(root), flush=True)
            (root/"failure.log").unlink(missing_ok=True)
        except BaseException:
            (root/"failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
