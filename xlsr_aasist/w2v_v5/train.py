"""Bounded continuation, committed-update resume, and four tail snapshots."""
from collections import defaultdict, deque
from pathlib import Path
import time
import torch
from .common import (schema_for, fingerprint, seed_all, partial_state, apply_partial,
                     atomic_save, atomic_json, capture_rng, restore_rng, to_cpu, append_json)
from .data import Waves, TicketSampler, loader, close
from .sampling import Plan
from .weights import average_states, invariant_keys
from .selection import plan_tail_updates
from w2v_v3191.model import load_model, optimizer_for, adapter_inventory
from w2v_v3191.parent import load as load_parent
from w2v_v3191.schedule import schedule
from w2v_v3191.step import train_step
from w2v_v3191.precision import scaler_for
from w2v_v3184.recovery import trim_step_log


def make_plan(rows, cfg):
    return Plan(rows, cfg["stream_sources"], cfg["sampling_seed"], cfg["group_quotas"],
                cfg["view_factors"], cfg["refresh_base_variants"], cfg=cfg)


def initialize(model, cfg):
    _, parent, _, _ = load_parent(cfg["parent_init"]["run"], cfg["parent_init"])
    apply_partial(model, parent["best_model"])
    if cfg.get("consistency") == "dual" and not model.pcl_scale.is_frozen:
        raise ValueError("Parent best must include its frozen PCL scales")
    return parent["best_model"]


def save(path, cfg, state, model, optimizer):
    state.update(model=partial_state(model), optimizer=to_cpu(optimizer.state_dict()),
                 grad_scaler=scaler_for(optimizer, cfg).state_dict(), rng=capture_rng())
    atomic_save(path, state, cfg.get("free_reserve_bytes", 0) + cfg.get("disk_margin_bytes", 0))


def run_stage(cfg, run, rows, *, model_factory=load_model, step_fn=train_step,
              initializer=initialize):
    """No evaluation occurs here, especially after Dev has entered final Train."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    root = run.parent
    seed_all(cfg["sampling_seed"])
    model = model_factory(cfg)
    optimizer = optimizer_for(model, cfg)
    plan = make_plan(rows, cfg)
    count = plan.steps * cfg["epochs"]
    tail_updates = plan_tail_updates(count, cfg["v5_tail_snapshots"], cfg["v5_tail_start"])
    # Frozen 3B base tensors are not stored in the detector-only checkpoint.
    # Only invariants actually present in these snapshots enter soup checks.
    constants = invariant_keys(model, partial_state(model))
    checkpoint = run / "last.pt"
    state = dict(schema=schema_for(cfg), identity=fingerprint(cfg), epoch=0,
                 next_step=0, cursor=0, tail_count=0, tail_model=None, status="training",
                 totals={}, tail_updates=tail_updates)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["schema"] != schema_for(cfg) or state["identity"] != fingerprint(cfg):
            raise ValueError("V5 stage settings changed; refusing incompatible resume")
        apply_partial(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler_for(optimizer, cfg).load_state_dict(state["grad_scaler"])
        restore_rng(state["rng"])
        print(f'[Resume {cfg["v5_stage"]}] epoch={state["epoch"]} next_step={state["next_step"]}', flush=True)
    else:
        initializer(model, cfg)
        save(checkpoint, cfg, state, model, optimizer)
    trim_step_log(run / "steps.jsonl", state["cursor"])
    atomic_json(run / "parameter_inventory.json", adapter_inventory(model))
    atomic_json(run / "budget.json", dict(updates=count, steps_per_epoch=plan.steps,
                epochs=cfg["epochs"], tail_updates=tail_updates, stage=cfg["v5_stage"],
                no_candidate_scoring=True, no_virtual_meta_update=True))
    if state["status"] == "complete":
        return model, state
    sampler = TicketSampler()
    batches = None
    try:
        for epoch in range(state["epoch"], cfg["epochs"]):
            if (root / "STOP_REQUESTED").exists():
                state["status"] = "paused"
                save(checkpoint, cfg, state, model, optimizer)
                return model, state
            if batches is None:
                batches = loader(Waves(rows, cfg), cfg, batch_sampler=sampler)
            sampler.batches = list(plan.batches(epoch))[state["next_step"]:]
            atomic_json(run / f"coverage_epoch_{epoch+1}.json", plan.coverage(epoch))
            totals = defaultdict(float, state.get("totals") or {})
            times = deque(maxlen=30)
            began = time.monotonic()
            print(f'[V5 {cfg["v5_stage"]}] epoch {epoch+1}/{cfg["epochs"]}; {plan.steps} updates', flush=True)
            for step, units in enumerate(batches, start=state["next_step"]):
                schedule(optimizer, cfg, epoch, step, plan.steps)
                for attempt in range(4):
                    stats = step_fn(model, optimizer, units,
                                    dict(cfg, _candidate_diagnostics=False), audit=step % 200 == 0)
                    if not stats["amp"]["skipped_update"]:
                        break
                else:
                    raise FloatingPointError("Four AMP failures; no update consumed")
                now = time.monotonic()
                times.append(now-began)
                began = now
                totals["loss"] += stats["loss"]
                totals["ce"] += stats["classification_loss"]
                totals["updates"] += 1
                state.update(cursor=state["cursor"]+1, next_step=step+1, totals=dict(totals),
                             status="training")
                if state["cursor"] in tail_updates:
                    snapshot = partial_state(model)
                    state["tail_model"] = (snapshot if state["tail_count"] == 0 else
                        average_states([state["tail_model"], snapshot],
                                       [state["tail_count"], 1], constant_keys=constants))
                    state["tail_count"] += 1
                append_json(run / "steps.jsonl", dict(update=state["cursor"], epoch=epoch+1,
                                                      step=step+1, **stats))
                eta = (plan.steps-step-1)*sum(times)/len(times)/60
                atomic_json(root / "progress.json", dict(stage=cfg["v5_stage"], epoch=epoch+1,
                    epochs=cfg["epochs"], step=step+1, steps=plan.steps,
                    loss=stats["loss"], eta_minutes=eta, status="training"))
                if step % 10 == 0 or step+1 == plan.steps:
                    print(f'V5 {cfg["v5_stage"]} epoch {epoch+1}/{cfg["epochs"]} '
                          f'{step+1}/{plan.steps} loss={stats["loss"]:.5f} '
                          f'CE={stats["classification_loss"]:.5f} ETA={eta:.1f}m', flush=True)
                paused = (root / "STOP_REQUESTED").exists()
                if paused or state["cursor"] % cfg["checkpoint_interval"] == 0:
                    state["status"] = "paused" if paused else "training"
                    save(checkpoint, cfg, state, model, optimizer)
                if paused:
                    return model, state
            atomic_json(run / f"epoch_{epoch+1}.json", dict(epoch=epoch+1,
                loss=totals["loss"]/totals["updates"], ce=totals["ce"]/totals["updates"],
                updates=int(totals["updates"]),
                evaluation="None; final stage has no untrained Dev selection set"))
            state.update(epoch=epoch+1, next_step=0, totals={})
            save(checkpoint, cfg, state, model, optimizer)
        if state["tail_count"] != len(tail_updates):
            raise RuntimeError("Tail checkpoint coverage incomplete")
        state["status"] = "complete"
        save(checkpoint, cfg, state, model, optimizer)
        return model, state
    finally:
        if batches is not None:
            close(batches)
