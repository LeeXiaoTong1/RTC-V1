"""Bounded continuation, committed-update resume, and four tail snapshots."""
from collections import defaultdict, deque
from pathlib import Path
import time
import torch
from .common import (schema_for, fingerprint, seed_all, partial_state, apply_partial,
                     atomic_save, atomic_json, capture_rng, restore_rng, to_cpu, append_json,read_json)
from .data import Waves, TicketSampler, loader, close
from w2v_v5.sampling import Plan
from w2v_v5.weights import average_states, invariant_keys
from w2v_v5.selection import plan_tail_updates
from .model import load_model, optimizer_for, adapter_inventory, parent_anchor
from .parent import load as load_parent
from .schedule import schedule, warmup_steps
from .calibration import calibrate
from .step import train_step
from w2v_v3191.precision import scaler_for
from w2v_v3184.recovery import trim_step_log


def make_plan(rows, cfg):
    return Plan(rows, cfg["stream_sources"], cfg["sampling_seed"], cfg["group_quotas"],
                cfg["view_factors"], cfg["refresh_base_variants"], cfg=cfg)


def initialize(model, cfg):
    _, parent, _, _ = load_parent(cfg["parent_init"]["run"], cfg["parent_init"])
    anchor = parent_anchor(model, parent["best_model"])
    apply_partial(model, anchor)
    if cfg.get("consistency") == "dual" and not model.pcl_scale.is_frozen:
        raise ValueError("Parent best must include its frozen PCL scales")
    return anchor


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
            raise ValueError("V5.4 stage settings changed; refusing incompatible resume")
        apply_partial(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler_for(optimizer, cfg).load_state_dict(state["grad_scaler"])
        restore_rng(state["rng"])
        print(f'[Resume {cfg["v5_stage"]}] epoch={state["epoch"]} next_step={state["next_step"]}', flush=True)
    else:
        initializer(model, cfg)
        save(checkpoint, cfg, state, model, optimizer)
    if not bool(model.repair.scale_ready):
        completed = calibrate(model, cfg, rows, plan, run)
        if completed is False:
            state['status'] = 'paused'
            save(checkpoint, cfg, state, model, optimizer)
            return model, state
        save(checkpoint, cfg, state, model, optimizer)
    trim_step_log(run / "steps.jsonl", state["cursor"])
    trim_step_log(run / "fusion_diagnostics.jsonl", state["cursor"])
    trim_step_log(run / 'hard_examples.jsonl', state['cursor'])
    trim_step_log(run / 'repair_diagnostics.jsonl', state['cursor'])
    atomic_json(run / "parameter_inventory.json", dict(adapter_inventory(model),
                adaptive_fusion=model.fusion.describe(),
                repair_parameters=sum(p.numel() for p in model.repair.parameters()),
                repair_arm=cfg['repair_arm'],
                conditioner_parameters=sum(p.numel() for p in model.fusion.conditioner.parameters()),
                conditioner_trainable=sum(p.numel() for p in model.fusion.conditioner.parameters() if p.requires_grad)))
    atomic_json(run / "budget.json", dict(updates=count, steps_per_epoch=plan.steps,
                epochs=cfg["epochs"], tail_updates=tail_updates, stage=cfg["v5_stage"],
                no_candidate_scoring=True, no_virtual_meta_update=True,
                repair_warmup_updates=warmup_steps(cfg,count), repair_arm=cfg['repair_arm']))
    if state["status"] == "complete":
        return model, state
    sampler = TicketSampler()
    batches = None
    smoke = read_json(run/'runtime_smoke.json') if (run/'runtime_smoke.json').is_file() else {}
    smoke_complete = smoke.get('status')=='passed' and smoke.get('config')==fingerprint(cfg)
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
            print(f'[V5.4 {cfg["v5_stage"]}] epoch {epoch+1}/{cfg["epochs"]}; {plan.steps} updates', flush=True)
            if str(cfg['device']).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(torch.device(cfg['device']))
            for step, units in enumerate(batches, start=state["next_step"]):
                if cfg.get('repair_audit_revision',0)>=2 and not smoke_complete:
                    from .smoke import check
                    print('[V5.4 check] testing actual warm/joint paths; temporary updates will be restored',flush=True)
                    check(model,optimizer,units,cfg,plan.steps,run/'runtime_smoke.json')
                    smoke_complete=True
                    began=time.monotonic()
                    if (root/'STOP_REQUESTED').exists():
                        state['status']='paused';save(checkpoint,cfg,state,model,optimizer)
                        return model,state
                update_started = time.monotonic()
                schedule(optimizer, cfg, epoch, step, plan.steps)
                for attempt in range(4):
                    stats = step_fn(model, optimizer, units,
                                    dict(cfg, _candidate_diagnostics=False), audit=state['cursor'] % 100 == 0)
                    if not stats["amp"]["skipped_update"]:
                        break
                else:
                    raise FloatingPointError("Four AMP failures; no update consumed")
                now = time.monotonic()
                stats['timing'] = dict(update_seconds=now-update_started, cycle_seconds=now-began,
                    peak_allocated_bytes=(torch.cuda.max_memory_allocated(torch.device(cfg['device']))
                                          if str(cfg['device']).startswith('cuda') else None))
                times.append(now-began)
                began = now
                totals["loss"] += stats["loss"]
                totals["ce"] += stats["classification_loss"]
                repair = stats['repair_losses']
                totals['repair_feature'] += repair['feature']
                totals['repair_kd'] += repair['kd']
                pcl = stats.get('pcl_losses', {})
                for key, value in dict(pcl_noise=pcl.get('noise', {}).get('weighted_loss', 0.),
                    pcl_platform=pcl.get('platform', {}).get('weighted_loss', 0.)).items():
                    totals[key] += value
                totals["updates"] += 1
                state.update(cursor=state["cursor"]+1, next_step=step+1, totals=dict(totals),
                             status="training")
                if state["cursor"] in tail_updates:
                    snapshot = partial_state(model)
                    state["tail_model"] = (snapshot if state["tail_count"] == 0 else
                        average_states([state["tail_model"], snapshot],
                                       [state["tail_count"], 1], constant_keys=constants))
                    state["tail_count"] += 1
                hard = stats.pop('hard_examples')
                append_json(run / 'hard_examples.jsonl', dict(update=state['cursor'], **hard))
                append_json(run / 'repair_diagnostics.jsonl', dict(update=state['cursor'],
                    stage=stats['repair_stage'], **repair, parameter_update=stats.get('repair_update')))
                # Keep full per-source provenance once, not twice in two logs.
                stats['repair_losses']={k:v for k,v in repair.items() if k!='pairs'}
                if hard.get('validated') and hard.get('examples'):
                    worst = hard['examples'][0]
                    if not worst['correct']:
                        print(f'[V5.4 hard] update={state["cursor"]} errors={hard["errors"]}/{hard["views"]} '
                              f'worst={worst["view"]}/{worst["language"]}/{worst["true_class"]} '
                              f'p_true={worst["p_true"]:.5f} pair={worst.get("pair_outcome","unpaired")} '
                              f'id={worst.get("id",worst.get("source_id"))}', flush=True)
                append_json(run / "steps.jsonl", dict(update=state["cursor"], epoch=epoch+1,
                                                      step=step+1, **stats))
                if 'adaptive_fusion' in stats:
                    diagnostics = stats['adaptive_fusion']
                    append_json(run / 'fusion_diagnostics.jsonl', dict(update=state['cursor'],
                        epoch=epoch+1, step=step+1, **diagnostics))
                    print(f'[V5.4 fusion] mode={cfg["adaptive_fusion_mode"]} '
                          f'input_std={diagnostics.get("input_weight_std", 0.):.6g} '
                          f'delta={diagnostics.get("mean_absolute_change", 0.):.6g} '
                          f'gate_update={diagnostics["conditioner_update_norm"]:.6g}', flush=True)
                eta = (plan.steps-step-1)*sum(times)/len(times)/60
                # The detailed append-only log still records every committed
                # step. Refresh the small UI file with the console cadence.
                if step % 10 == 0 or step+1 == plan.steps:
                    atomic_json(root / "progress.json", dict(stage=cfg["v5_stage"], epoch=epoch+1,
                        epochs=cfg["epochs"], step=step+1, steps=plan.steps,
                        loss=stats["loss"], eta_minutes=eta, status="training"))
                    print(f'V5.4 {cfg["v5_stage"]} epoch {epoch+1}/{cfg["epochs"]} '
                          f'{step+1}/{plan.steps} loss={stats["loss"]:.5f} '
                          f'CE={stats["classification_loss"]:.5f} '
                          f'LF={repair["feature"]:.6f} KD={repair["kd"]:.6f} '
                          f'PCL-p={pcl.get("platform", {}).get("weighted_loss", 0.):.6f} '
                          f'stage={stats["repair_stage"]} '
                          f'encoded={stats.get("encoded_views", 0)} ETA={eta:.1f}m', flush=True)
                paused = (root / "STOP_REQUESTED").exists()
                if paused or state["cursor"] % cfg["checkpoint_interval"] == 0:
                    state["status"] = "paused" if paused else "training"
                    save(checkpoint, cfg, state, model, optimizer)
                    from .telemetry import summarize
                    summarize(run, state['cursor'])
                if paused:
                    return model, state
                if cfg.get('repair_audit_revision',0) >= 2 and state['cursor'] in {warmup_steps(cfg,count),count//2,count}:
                    # A named, bounded diagnostic boundary, not a second
                    # encoder pass hidden inside ordinary gradient updates.
                    save(checkpoint,cfg,state,model,optimizer)
                    from .diagnostics import probe
                    probe(model,cfg,run,'pair_probe_update_'+str(state['cursor']))
                    began = time.monotonic()  # Exclude diagnostics from train ETA.
            atomic_json(run / f"epoch_{epoch+1}.json", dict(epoch=epoch+1,
                loss=totals["loss"]/totals["updates"], ce=totals["ce"]/totals["updates"],
                updates=int(totals["updates"]),
                training_recipe_base='5.1', contrastive_enabled=False,
                pcl_noise=totals['pcl_noise']/totals['updates'], pcl_platform=totals['pcl_platform']/totals['updates'],
                repair_feature=totals['repair_feature']/totals['updates'], repair_kd=totals['repair_kd']/totals['updates'],
                evaluation="None; final stage has no untrained Dev selection set"))
            state.update(epoch=epoch+1, next_step=0, totals={})
            save(checkpoint, cfg, state, model, optimizer)
        if state["tail_count"] != len(tail_updates):
            raise RuntimeError("Tail checkpoint coverage incomplete")
        state["status"] = "complete"
        save(checkpoint, cfg, state, model, optimizer)
        from .telemetry import summarize
        summarize(run, state['cursor'])
        return model, state
    finally:
        if batches is not None:
            close(batches)
