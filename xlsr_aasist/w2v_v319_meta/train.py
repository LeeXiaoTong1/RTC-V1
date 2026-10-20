"""One-epoch parent continuation; only committed outer states are persisted."""
from collections import defaultdict
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
from .meta import train_step
from .schedule import schedule
from .precision import scaler_for
from .monitor import summarize, measure, print_epoch, render
from w2v_v3189.train import promote, calibrations
from w2v_v3184.records import dev_partition
from w2v_v3184.recovery import trim_step_log


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
    checksum = calibrations(run, state, dev, partition, paused=True)
    value = dict(status="paused", version="3.19", recipe=cfg["recipe"],
                 completed_epochs=state["epoch"], next_step=state["next_step"],
                 best_tag=state["best_tag"], last_tag=state["last_tag"], checkpoint_sha256=checksum)
    atomic_json(run / "paused.json", value)
    atomic_json(run / "progress.json", dict(value, phase="Paused at saved outer-update boundary"))
    print("PAUSED_RUN=" + str(run), flush=True)
    return value


def run_experiment(cfg, run, train, dev, model_factory=load_model, baseline_only=False, infer_fn=infer):
    run = Path(run)
    seed_all(cfg["seed"])
    request, paused = run / "stop_requested.json", run / "paused.json"
    if paused.is_file() and request.is_file() and request.stat().st_mtime_ns <= paused.stat().st_mtime_ns:
        request.unlink()
    model = model_factory(cfg)
    optimizer = optimizer_for(model, cfg)
    plan = Plan(train, cfg["stream_sources"], cfg["seed"], cfg["group_quotas"],
                cfg["view_factors"], cfg["refresh_base_variants"], cfg=cfg)
    partition = dev_partition(dev, cfg["dev_pairs"], cfg["seed"], cfg["calibration_fraction"])
    if partition != read_json(Path(cfg["parent_init"]["run"]) / "dev_partition.json"):
        raise ValueError("Parent Dev partition changed")
    atomic_json(run / "dev_partition.json", partition)
    select = partition["select"]
    select_rows = [dev[i] for i in select]
    state = dict(schema=schema_for(cfg), identity=fingerprint(cfg), epoch=0, next_step=0, cursor=0,
                 alpha=0. if cfg["experiment"] == "E0" else None, epoch_accumulator=None,
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
    atomic_json(run / "parameter_inventory.json", dict(
        encoder_adaptation=adapter_inventory(model),
        optimizer_groups=[dict(name=g["name"], initial_lr=g["initial_lr"],
                               parameters=sum(p.numel() for p in g["params"])) for g in optimizer.param_groups]))
    trim_step_log(run / "steps.jsonl", state["cursor"])
    if baseline_only:
        print("INITIALIZATION_VERIFIED_RUN=" + str(run), flush=True)
        return dict(status="initialization_verified", version="3.19")
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
            if state["alpha"] is None:
                from .calibration import calibrate

                def task_factory():
                    dataset = Waves(train, cfg)
                    for tickets in plan.calibration_batches():
                        yield [dataset[t] for t in tickets]

                def progress(done, total, report):
                    atomic_json(run / "progress.json", dict(status="calibrating", phase="Train-only inner-step calibration",
                                epoch=epoch+1, epochs=cfg["epochs"], step=done, steps=total))
                    print(f"Meta Train calibration {done}/{total}", flush=True)

                try:
                    report = calibrate(model, optimizer, task_factory, cfg, progress=progress)
                except Exception as exc:
                    atomic_json(run / "inner_calibration_failure.json",
                                dict(error=str(exc), details=getattr(exc, "report", None),
                                     parent_preserved=True, silent_zero_step_fallback=False))
                    raise
                state["alpha"] = report["selected_alpha"]
                state["calibration"] = report
                atomic_json(run / "inner_calibration.json", report)
                save(run, cfg, state, model, optimizer)
                if request.is_file():
                    return pause(run, cfg, state, model, optimizer, dev, partition)
            if batches is None:
                batches = loader(Waves(train, cfg), cfg, batch_sampler=sampler)
            coverage = plan.coverage(epoch)
            atomic_json(run / f"coverage_epoch_{epoch+1}.json", coverage)
            sampler.batches = list(plan.batches(epoch))[state["next_step"]:]
            totals = defaultdict(float, state.get("epoch_accumulator") or {})
            start = time.monotonic()
            if cfg["device"].startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(cfg["device"])
            print(f'[Train] V3.19 epoch {epoch+1}/{cfg["epochs"]} phase=metadg updates={plan.steps}', flush=True)
            for step, units in enumerate(batches, start=state["next_step"]):
                wait = time.monotonic() - start
                _, beta = schedule(optimizer, cfg, epoch, step, plan.steps)
                began = time.monotonic()
                audit = step % cfg.get("gradient_audit_interval", 200) == 0
                retries = []
                for attempt in range(4):
                    try:
                        stats = train_step(model, optimizer, units, cfg, state["alpha"], beta, joint=True, audit=audit)
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
                for key in ("loss", "classification_loss", "views", "encoded_views", "auxiliary_reference_views"):
                    totals[key] += stats.get(key, 0.)
                totals["updates"] += 1
                totals["compute_seconds"] += seconds
                totals["wait_seconds"] += wait
                totals["amp_retries"] += len(retries)
                state.update(cursor=state["cursor"]+1, next_step=step+1, epoch_accumulator=dict(totals))
                append_json(run / "steps.jsonl", dict(update=state["cursor"], epoch=epoch+1, step=step+1,
                            **stats, compute_seconds=seconds, wait_seconds=wait, amp_retry_attempts=retries))
                if audit or step+1 == plan.steps:
                    state["steps"].append(dict(update=state["cursor"], loss=stats["loss"],
                                               raw_ce=stats["raw_ce_sum"]/stats["views"]))
                eta = (plan.steps-step-1) * (totals["compute_seconds"]+totals["wait_seconds"])/totals["updates"]
                atomic_json(run / "progress.json", dict(status="training", phase="metadg", epoch=epoch+1,
                            epochs=cfg["epochs"], step=step+1, steps=plan.steps, loss=stats["loss"],
                            alpha=state["alpha"], beta=beta, eta=eta))
                if step % 10 == 0 or step+1 == plan.steps:
                    print(f'Epoch {epoch+1}/{cfg["epochs"]} {step+1}/{plan.steps} | loss={stats["loss"]:.5f} '
                          f'CE={stats["classification_loss"]:.5f} alpha={state["alpha"]:.3g} beta={beta:.3f} ETA={eta/60:.1f}m', flush=True)
                if request.is_file():
                    return pause(run, cfg, state, model, optimizer, dev, partition)
                start = time.monotonic()
            if totals["updates"] != plan.steps:
                raise RuntimeError("Incomplete source exposure epoch")
            tag = f"epoch_{epoch+1}_step_{plan.steps}"
            atomic_json(run / "progress.json", dict(status="evaluating", phase="Same parent Dev audio; no adaptation",
                        epoch=epoch+1, epochs=cfg["epochs"], step=plan.steps, steps=plan.steps))
            logits, _ = infer_fn(model, dev, cfg)
            metrics, selected = measure(dev, logits), measure(select_rows, logits[select])
            current = partial_state(model)
            scores = torch.from_numpy(logits.copy())
            promoted = promote(state, tag, current, selected, scores)
            training = {key: value/plan.steps for key, value in totals.items()}
            training.update(alpha=state["alpha"], beta=beta, total_updates=plan.steps,
                            A_objective="balanced CE + inherited dual PCL", B_objective="balanced CE only")
            if cfg["device"].startswith("cuda"):
                training["cuda_peak_memory"] = dict(allocated_gib=torch.cuda.max_memory_allocated(cfg["device"])/1024**3,
                                                    reserved_gib=torch.cuda.max_memory_reserved(cfg["device"])/1024**3)
            entry = dict(epoch=epoch+1, tag=tag, phase="metadg", metrics=metrics, selection_metrics=selected,
                         dev=summarize(dev, logits), training=training, promoted=promoted, fusion=model.fusion.describe())
            state["history"].append(entry)
            state.update(epoch=epoch+1, next_step=0, epoch_accumulator=None, last_tag=tag, last_logits=scores)
            np.savez_compressed(run / f"dev_scores_{tag}.npz", logits=logits)
            save(run, cfg, state, model, optimizer, current)
            print_epoch(entry, state)
            render(run, state["history"], state["steps"])
            if request.is_file():
                return pause(run, cfg, state, model, optimizer, dev, partition)
    finally:
        if batches is not None:
            close(batches)
    checksum = calibrations(run, state, dev, partition)
    fallback = state["best_tag"].startswith("parent_v3189_")
    done = dict(status="complete", version="3.19", recipe=cfg["recipe"], completed_epochs=state["epoch"],
                planned_epochs=cfg["epochs"], best_tag=state["best_tag"], last_tag=state["last_tag"],
                checkpoint_sha256=checksum, baseline_fallback=fallback,
                best_origin="v3189_parent" if fallback else "v319_meta_trained",
                inner_alpha=state["alpha"], selection=cfg["selection"], initialization=cfg["initialization"])
    atomic_json(run / "completed.json", done)
    atomic_json(run / "progress.json", dict(done, phase="complete"))
    print(f'[Complete] V3.19 best={state["best_tag"]}; origin={done["best_origin"]}; last={state["last_tag"]}', flush=True)
    return done
