"""Matched condition controls; persist only committed optimizer boundaries."""
from collections import defaultdict, deque
from pathlib import Path
import time
import numpy as np
import torch
from .common import (schema_for, atomic_json, read_json, digest, fingerprint, seed_all,
                     partial_state, apply_partial, atomic_save, to_cpu, capture_rng, restore_rng, append_json)
from .model import load_model, optimizer_for, adapter_inventory
from .sampling import Plan
from .data import Waves, TicketSampler, loader, close
from .inference import infer
from .step import train_step
from .schedule import schedule
from .precision import scaler_for
from .monitor import summarize, measure, print_epoch, render
from .calibration import write_identity
from w2v_v3184.records import dev_partition
from w2v_v3184.recovery import trim_step_log
from .robust_eval import prepare_panel, evaluate_panel, read_scores
from .ce import classification_weights
from .execution import diagnostic_step, record_execution


def promotion_check(candidate, parent, best_score, old_metrics, parent_metrics, cfg):
    """Only Select can promote; group guards are relative to the fixed parent."""
    if candidate.get("split") != "select" or parent.get("split") != "select":
        raise ValueError("Audit cannot participate in checkpoint selection")
    checks = dict(
        improves_selected_score=candidate["score"] > best_score+cfg.get("robust_min_gain", .001),
        official_clean_preserved=old_metrics["clean_f1"] >=
            parent_metrics["clean_f1"]-cfg.get("robust_clean_guard", .002),
        worst_mechanism_preserved=candidate["worst_mechanism_f1"] >=
            parent["worst_mechanism_f1"]-cfg.get("robust_clean_guard", .002))
    for key, values in parent["groups"].items():
        checks["recall_"+key] = candidate["groups"][key]["recall"] >= (
            values["recall"]-cfg.get("robust_recall_guard", .005))
    return dict(passed=all(checks.values()), checks=checks, split="select",
                minimum_gain=cfg.get("robust_min_gain", .001))


def indexed_scores(count, indices, logits):
    """Preserve manifest indices without fabricating unobserved Audit scores."""
    values = np.asarray(logits)
    if values.shape != (len(indices), 2) or not np.isfinite(values).all():
        raise ValueError("Select inference must return finite two-class logits")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("Invalid scored source indices")
    result = np.full((count, 2), np.nan, dtype=values.dtype)
    result[indices] = values
    return result


def save(run, cfg, state, model, optimizer, current=None):
    if getattr(model, "_v31810_virtual_active", False):
        raise RuntimeError("Refusing to persist temporary meta parameters")
    state.update(model=partial_state(model) if current is None else current,
                 optimizer=to_cpu(optimizer.state_dict()),
                 grad_scaler=scaler_for(optimizer, cfg).state_dict(), rng=capture_rng())
    if state["next_step"] == 0 and state["best_tag"] == state["last_tag"]:
        state["best_model"] = state["model"]
    atomic_save(Path(run) / "last.pt", state, cfg.get("free_reserve_bytes", 0) + cfg.get("disk_margin_bytes", 0))
    atomic_json(Path(run) / "training_history.json", state["history"])


def pause(run, cfg, state, model, optimizer, dev, partition):
    save(run, cfg, state, model, optimizer)
    checksum = write_identity(run, state, paused=True)
    value = dict(status="paused", version="3.19.1", recipe=cfg["recipe"],
                 completed_epochs=state["epoch"], next_step=state["next_step"],
                 best_tag=state["best_tag"], last_tag=state["last_tag"], checkpoint_sha256=checksum)
    atomic_json(run / "paused.json", value)
    atomic_json(run / "progress.json", dict(value, phase="Paused at saved outer-update boundary"))
    print("PAUSED_RUN=" + str(run), flush=True)
    return value


def run_experiment(cfg, run, train, dev, model_factory=load_model, baseline_only=False, infer_fn=infer):
    classification_weights(cfg)
    print('[Objective] CE: Noisy=50%, official Online=25%, Clean-RTC=25%; '
          'equal language/class groups within each view', flush=True)
    run = Path(run)
    seed_all(cfg.get("sampling_seed", cfg["seed"]))
    request, paused = run / "stop_requested.json", run / "paused.json"
    if paused.is_file() and request.is_file() and request.stat().st_mtime_ns <= paused.stat().st_mtime_ns:
        request.unlink()
    model = model_factory(cfg)
    optimizer = optimizer_for(model, cfg)
    plan = Plan(train, cfg["stream_sources"], cfg.get("sampling_seed", cfg["seed"]), cfg["group_quotas"],
                cfg["view_factors"], cfg["refresh_base_variants"], cfg=cfg)
    partition = dev_partition(dev, cfg["dev_pairs"], cfg["seed"], cfg["calibration_fraction"])
    if partition != read_json(Path(cfg["parent_init"]["run"]) / "dev_partition.json"):
        raise ValueError("Parent Dev partition changed")
    atomic_json(run / "dev_partition.json", partition)
    select = partition["select"]
    select_rows = [dev[i] for i in select]
    state = dict(schema=schema_for(cfg), identity=fingerprint(cfg), epoch=0, next_step=0, cursor=0,
                 alpha=0., epoch_accumulator=None,
                 best_tag=None, last_tag=None, best_metrics=None, best_model=None,
                 best_logits=None, last_logits=None, history=[], steps=[])
    if (run / "last.pt").is_file():
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if state.get("schema") != schema_for(cfg) or state.get("identity") != fingerprint(cfg):
            raise ValueError("Resume config or schema differs")
        apply_partial(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("grad_scaler"):
            scaler_for(optimizer, cfg).load_state_dict(state["grad_scaler"])
        restore_rng(state["rng"])
        print(f'[Resume] epoch={state["epoch"]}; next_step={state["next_step"]}; alpha={state["alpha"]}', flush=True)
    else:
        from .parent import initialize
        baseline, weights, logits, audit = initialize(model, cfg, dev, partition)
        state.update(best_tag=baseline["tag"], last_tag=baseline["tag"], best_model=weights,
                     best_logits=logits, last_logits=logits, best_metrics=baseline["selection_metrics"],
                     history=[baseline], initialization_audit=audit)
        atomic_json(run / "initialization_audit.json", audit)
        atomic_json(run / "parent_baseline.json", baseline)
        np.savez_compressed(run / "parent_baseline_scores.npz", logits=logits.numpy())
        save(run, cfg, state, model, optimizer, weights)
        print_epoch(baseline, state)
    panel = prepare_panel(cfg, run, dev, partition)
    if "robust_parent_select" not in state:
        # Must run at epoch zero; never invent a parent score from trained weights.
        if state["epoch"] or state["next_step"]:
            raise ValueError("Missing fixed parent processing-panel baseline on a resumed training run")
        initial = evaluate_panel(model, cfg, run, panel, split="select",
                                 checkpoint_tag=state["best_tag"], reference_logits=state["best_logits"].numpy())
        state.update(robust_parent_select=initial, robust_best_select=initial)
        atomic_json(run/"robust_parent_select.json", initial)
        save(run, cfg, state, model, optimizer)
    atomic_json(run / "parameter_inventory.json", dict(
        encoder_adaptation=adapter_inventory(model),
        optimizer_groups=[dict(name=g["name"], initial_lr=g["initial_lr"],
                               parameters=sum(p.numel() for p in g["params"])) for g in optimizer.param_groups]))
    trim_step_log(run / "steps.jsonl", state["cursor"])
    record_execution(run, cfg, state)
    if baseline_only:
        print("INITIALIZATION_VERIFIED_RUN=" + str(run), flush=True)
        return dict(status="initialization_verified", version="3.19.1")
    paused.unlink(missing_ok=True)
    if request.is_file():
        return pause(run, cfg, state, model, optimizer, dev, partition)
    sampler = TicketSampler()
    batches = None
    try:
        for epoch in range(state["epoch"], cfg["epochs"]):
            model.set_phase(True)
            model.train()
            if cfg["consistency"] == "dual" and not model.pcl_scale.is_frozen:
                raise ValueError("Parent PCL scale must remain frozen")
            if batches is None:
                batches = loader(Waves(train, cfg), cfg, batch_sampler=sampler)
            coverage = plan.coverage(epoch)
            atomic_json(run / f"coverage_epoch_{epoch+1}.json", coverage)
            sampler.batches = list(plan.batches(epoch))[state["next_step"]:]
            totals = defaultdict(float, state.get("epoch_accumulator") or {})
            recent_seconds = deque(maxlen=50)
            start = time.monotonic()
            if cfg["device"].startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(cfg["device"])
            print(f'[Train] V3.19.1 epoch {epoch+1}/{cfg["epochs"]} phase=paired_condition updates={plan.steps}', flush=True)
            for step, units in enumerate(batches, start=state["next_step"]):
                wait = time.monotonic() - start
                schedule(optimizer, cfg, epoch, step, plan.steps)
                beta = cfg["meta_beta"]  # fixed A/B contribution; no temporary update
                began = time.monotonic()
                audit = step % cfg.get("gradient_audit_interval", 200) == 0
                retries = []
                for attempt in range(4):
                    try:
                        runtime = dict(cfg, _candidate_diagnostics=diagnostic_step(step))
                        stats = train_step(model, optimizer, units, runtime, state["alpha"], beta, joint=True, audit=audit)
                    except torch.OutOfMemoryError:
                        atomic_json(run / "oom_context.json", dict(epoch=epoch+1, step=step+1,
                                    last_saved_checkpoint=str(run / "last.pt"),
                                    last_encode=getattr(model, "last_encode_geometry", None),
                                    temporary_parameters_restored=not getattr(model, "_v31810_virtual_active", False)))
                        raise
                    if not stats["amp"]["skipped_update"]:
                        break
                    retries.append(stats["amp"])
                else:
                    raise FloatingPointError("Repeated AMP overflow: the failed task did not consume the update budget")
                seconds = time.monotonic() - began
                recent_seconds.append(seconds + wait)
                for key in ("loss", "classification_loss", "views", "encoded_views", "auxiliary_reference_views"):
                    totals[key] += stats.get(key, 0.)
                totals["updates"] += 1
                totals["compute_seconds"] += seconds
                totals["wait_seconds"] += wait
                totals["amp_retries"] += len(retries)
                totals["profiled_updates"] += 1
                totals["scoring_seconds"] += stats["selection"]["scoring_seconds"]
                totals["scoring_encoded_views"] += stats["selection"]["scoring_encoded_views"]
                totals["training_encoded_views"] += stats["training_encoded_views"]
                state.update(cursor=state["cursor"]+1, next_step=step+1, epoch_accumulator=dict(totals))
                append_json(run / "steps.jsonl", dict(update=state["cursor"], epoch=epoch+1, step=step+1,
                            **stats, compute_seconds=seconds, wait_seconds=wait, amp_retry_attempts=retries))
                if audit or step+1 == plan.steps:
                    state["steps"].append(dict(update=state["cursor"], loss=stats["loss"],
                                               raw_ce=stats["raw_ce_sum"]/stats["views"]))
                eta = (plan.steps-step-1) * sum(recent_seconds)/len(recent_seconds)
                atomic_json(run / "progress.json", dict(status="training", phase="paired_condition", epoch=epoch+1,
                            epochs=cfg["epochs"], step=step+1, steps=plan.steps, loss=stats["loss"],
                            alpha=state["alpha"], beta=beta, eta=eta,
                            eta_basis="last up to 50 updates in this session; epoch evaluation excluded"))
                if step % 10 == 0 or step+1 == plan.steps:
                    print(f'Epoch {epoch+1}/{cfg["epochs"]} {step+1}/{plan.steps} | loss={stats["loss"]:.5f} '
                          f'CE={stats["classification_loss"]:.5f} arm={cfg["robust_arm"]} '
                          f'score_views={stats["selection"]["scoring_encoded_views"]} '
                          f'score={stats["selection"]["scoring_seconds"]:.2f}s '
                          f'compute={seconds:.2f}s wait={wait:.2f}s ETA={eta/60:.1f}m', flush=True)
                if request.is_file():
                    return pause(run, cfg, state, model, optimizer, dev, partition)
                if state["cursor"] % cfg.get("checkpoint_interval", 200) == 0:
                    save(run, cfg, state, model, optimizer)
                start = time.monotonic()
            if totals["updates"] != plan.steps:
                raise RuntimeError("Incomplete source exposure epoch")
            tag = f"epoch_{epoch+1}_step_{plan.steps}"
            atomic_json(run / "progress.json", dict(status="evaluating", phase="Same parent Dev audio; no adaptation",
                        epoch=epoch+1, epochs=cfg["epochs"], step=plan.steps, steps=plan.steps))
            # Audit sources must not enter any epoch's inference, visible
            # metrics, or diagnostic plots. Missing scores remain explicit NaN.
            select_logits, _ = infer_fn(model, select_rows, cfg)
            selected = measure(select_rows, select_logits)
            metrics = selected
            logits = indexed_scores(len(dev), select, select_logits)
            current = partial_state(model)
            scores = torch.from_numpy(logits.copy())
            robust = evaluate_panel(model, cfg, run, panel, split="select", checkpoint_tag=tag,
                                    baseline_logits=read_scores(run, state["robust_parent_select"]),
                                    reference_logits=logits)
            decision = promotion_check(robust, state["robust_parent_select"],
                                       state["robust_best_select"]["score"], selected,
                                       state["history"][0]["selection_metrics"], cfg)
            promoted = decision["passed"]
            if promoted:
                state.update(best_tag=tag, best_model=current, best_logits=scores,
                             best_metrics=selected, robust_best_select=robust)
            atomic_json(run/f"robust_select_{tag}.json", dict(robust, promotion=decision))
            training = {key: value/plan.steps for key, value in totals.items()}
            observed = totals["profiled_updates"]
            training["execution_profile"] = dict(
                observed_updates=int(observed),
                mean_scoring_seconds=totals["scoring_seconds"]/max(1, observed),
                mean_scoring_views=totals["scoring_encoded_views"]/max(1, observed),
                mean_training_views=totals["training_encoded_views"]/max(1, observed),
                scope="Only updates recorded by the optimized executor; other epoch means may include pre-upgrade steps")
            training.update(alpha=state["alpha"], beta=beta, total_updates=plan.steps,
                            A_objective="paired balanced CE + inherited dual PCL",
                            B_objective="paired balanced CE only", robust_arm=cfg["robust_arm"])
            if cfg["device"].startswith("cuda"):
                training["cuda_peak_memory"] = dict(allocated_gib=torch.cuda.max_memory_allocated(cfg["device"])/1024**3,
                                                    reserved_gib=torch.cuda.max_memory_reserved(cfg["device"])/1024**3)
            entry = dict(epoch=epoch+1, tag=tag, phase="paired_condition", metrics=metrics, selection_metrics=selected,
                         dev=summarize(select_rows, select_logits), training=training, promoted=promoted,
                         evaluation_scope="Historical Select sources only; Audit unscored until final checkpoint lock",
                         scored_indices=select,
                         robust_select=robust, promotion_checks=decision, fusion=model.fusion.describe())
            state["history"].append(entry)
            state.update(epoch=epoch+1, next_step=0, epoch_accumulator=None, last_tag=tag, last_logits=scores)
            np.savez_compressed(run / f"dev_scores_{tag}.npz", logits=logits,
                                scored_indices=np.asarray(select, dtype=np.int64),
                                scope=np.asarray("select_only; NaN outside scored_indices"))
            save(run, cfg, state, model, optimizer, current)
            print_epoch(entry, state)
            render(run, state["history"], state["steps"])
            if request.is_file():
                return pause(run, cfg, state, model, optimizer, dev, partition)
    finally:
        if batches is not None:
            close(batches)
    # Freeze the choice before exposing the audit subset. Never promote on Audit.
    atomic_json(run/"robust_selection_locked.json", dict(chosen_checkpoint_tag=state["best_tag"],
                selection_split="select", audit_used_for_selection=False, panel_hash=panel["hash"]))
    final_path = run/"robust_final.json"
    if not final_path.is_file():
        last_weights = partial_state(model)
        from .parent import initialize
        _, _, parent_logits, _ = initialize(model, cfg, dev, partition)
        audit_cfg = dict(cfg, robust_final_audit=True)
        parent_audit = evaluate_panel(model, audit_cfg, run, panel, split="audit",
                                     checkpoint_tag="parent", reference_logits=parent_logits.numpy())
        apply_partial(model, state["best_model"])
        # Complete historical raw scores only for the already selected model.
        # Parent fallback already has known inherited scores; no new pass is
        # needed in that case. Never evaluate an unselected last on Audit.
        best_logits = state["best_logits"].numpy().copy()
        missing = np.flatnonzero(~np.isfinite(best_logits).all(axis=1)).tolist()
        audit_indices = partition.get("audit", partition.get("calibration", []))
        if missing:
            if set(missing) != set(audit_indices):
                raise ValueError("Unobserved final scores must be exactly the locked Audit sources")
            final_logits, _ = infer_fn(model, [dev[i] for i in audit_indices], cfg)
            if np.asarray(final_logits).shape != (len(audit_indices), 2) or not np.isfinite(final_logits).all():
                raise ValueError("Final historical Audit logits are invalid")
            best_logits[audit_indices] = final_logits
            state["best_logits"] = torch.from_numpy(best_logits)
            if state["best_tag"] == state["last_tag"]:
                state["last_logits"] = state["best_logits"]
        atomic_json(run/"historical_best_dev_metrics.json", dict(
            selected=state["best_tag"], scope="Final locked best only; historical Dev, no score fitting",
            metrics=measure(dev, best_logits),
            audit_metrics=measure([dev[i] for i in audit_indices], best_logits[audit_indices])))
        final_audit = evaluate_panel(model, audit_cfg, run, panel, split="audit", bootstrap=True,
                                    checkpoint_tag=state["best_tag"],
                                    baseline_logits=read_scores(run, parent_audit),
                                    reference_logits=state["best_logits"].numpy())
        apply_partial(model, last_weights)
        # Save the completed score record with the LAST model and its unchanged
        # optimizer, so a post-Audit interruption cannot corrupt resumability.
        save(run, cfg, state, model, optimizer, last_weights)
        atomic_json(final_path, dict(
            chosen_checkpoint_tag=state["best_tag"], selection_split="select",
            audit_used_for_selection=False, select=state["robust_best_select"], audit=final_audit,
            parent_select=state["robust_parent_select"], parent_audit=parent_audit,
            panel_hash=panel["hash"], robust_arm=cfg["robust_arm"],
            compare_budget=dict(epochs=cfg["epochs"], optimizer_updates=state["cursor"], steps_per_epoch=plan.steps,
                                source_draws_per_stream=cfg["source_draws_per_stream"],
                                candidate_count=cfg["robust_candidate_count"], paired_classification=True),
            limitation="Historical Dev was used during prior development; processor holdout is synthetic, not unknown-platform proof."))
    else:
        final = read_json(final_path)
        if (final.get("chosen_checkpoint_tag") != state["best_tag"]
                or final.get("panel_hash") != panel["hash"]):
            raise ValueError("Existing final Audit differs from the frozen checkpoint/panel")
        read_scores(run, final["audit"])
    checksum = write_identity(run, state)
    fallback = state["best_tag"].startswith("parent_v3189_")
    done = dict(status="complete", version="3.19.1", recipe=cfg["recipe"], completed_epochs=state["epoch"],
                planned_epochs=cfg["epochs"], best_tag=state["best_tag"], last_tag=state["last_tag"],
                checkpoint_sha256=checksum, baseline_fallback=fallback,
                best_origin="v3189_parent" if fallback else "v3191_trained",
                inner_alpha=state["alpha"], selection=cfg["selection"], initialization=cfg["initialization"])
    atomic_json(run / "completed.json", done)
    atomic_json(run / "progress.json", dict(done, phase="complete"))
    print(f'[Complete] V3.19.1 best={state["best_tag"]}; origin={done["best_origin"]}; last={state["last_tag"]}', flush=True)
    return done
