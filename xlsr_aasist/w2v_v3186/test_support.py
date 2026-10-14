"""Small deterministic CPU workflow fixtures; no GPU or recognizer claim."""
import copy
import hashlib
from pathlib import Path
import soundfile as sf
import torch
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout, make_fixture
from w2v_v3185.test_support import tiny_cfg as prior_cfg, tiny_model as prior_model, scene_fixture
from w2v_v3185.common import schema_for as prior_schema, RECIPE as PRIOR_RECIPE
from w2v_v3185.sampling import Plan as ParentPlan
from w2v_v3183r2.monitor import panel_rows
from w2v_v3184.records import dev_partition
from w2v_v3184.inference import infer
from .common import RECIPE, atomic_json, digest, partial_state, fingerprint, seed_all
from .model import Detector
from .sampling import Plan, probe_tickets


def tiny_cfg():
    cfg = prior_cfg()
    cfg.update(version='3.18.6', recipe=RECIPE, pooling='whole', local_evidence=False,
        warm_epochs=0, epochs=2, consistency='dual', pcl_platform_weight=.15, pcl_noise_weight=.15,
        pcl_ramp_updates=2, platform_sources_per_group=1, gradient_audit_interval=2,
        whole_max_frames=3000, whole_graph_budget=1_000_000, whole_batch=8,
        microbatch=8, frame_budget=1000, inference_requires_phonemes=False)
    return cfg


def tiny_model(cfg):
    return Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)


def save_inputs(run, cfg, train, dev):
    run.mkdir()
    atomic_json(run / 'train_rows.json', train)
    atomic_json(run / 'dev_rows.json', dev)
    cfg['saved_manifests'] = {name: digest(run / name) for name in ('train_rows.json', 'dev_rows.json')}
    atomic_json(run / 'config.json', cfg)


def completed_v3185(root):
    return completed_parent(root)


def completed_aug5(root):
    return completed_parent(root,aug5=True)


def completed_parent(root,aug5=False):
    parent_config,parent_model,parent_schema,parent_plan=prior_cfg,prior_model,prior_schema,ParentPlan
    if aug5:
        from w2v_v3184_aug5.test_support import tiny_cfg as parent_config,tiny_model as parent_model
        from w2v_v3184_aug5.common import schema_for as parent_schema
        from w2v_v3184_aug5.sampling import Plan as parent_plan
    cfg, train, dev = make_fixture(root / 'audio')
    cfg.update(parent_config())
    # Existing legacy fixtures deliberately give each stream an independent
    # identity. This fixture explicitly records the official same-source map.
    offline = {r['group_id']: r for r in train if r['condition'] == 'offline'}
    for row in train:
        off = offline[row['group_id'].removeprefix('online:')]
        group = hashlib.sha256(off['id'].encode()).hexdigest()
        row.update(source_id=off['id'], group_id=group, source_sha256=group,
            audio_sha256=group if row['condition'] == 'offline' else hashlib.sha256(row['id'].encode()).hexdigest(),
            output_samples=int(sf.info(row['audio']).frames))
    parent = root / 'parent'
    save_inputs(parent, cfg, train, dev)
    atomic_json(parent / 'dev_partition.json', dev_partition(dev, cfg['dev_pairs'], cfg['seed'], cfg['calibration_fraction']))
    plan = parent_plan(train, cfg['stream_sources'], cfg['seed'], cfg['group_quotas'],
                       cfg['view_factor'] if aug5 else cfg['view_factors'])
    atomic_json(parent / 'train_probe_tickets.json', probe_tickets(plan, cfg['train_probe_per_group']))
    rows, tickets = panel_rows(cfg)
    atomic_json(parent / 'independent_panel.json', dict(rows=rows, tickets=tickets))
    seed_all(cfg['seed'])
    model = parent_model(cfg)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith('.lora_b'):
                value.normal_(0, .002)
        for name, value in model.named_buffers():
            if name.endswith('running_var'):
                value.fill_(1.1)
            if name.endswith('num_batches_tracked'):
                value.fill_(5)
    best = partial_state(model)
    last = copy.deepcopy(best)
    last['fusion.layer_logits'].fill_(.5)
    logits, _ = infer(model, dev, cfg)
    torch.save(dict(schema=parent_schema(cfg), identity=fingerprint(cfg), best_tag='epoch_1_step_3',
        last_tag='epoch_2_step_3', best_model=best, model=last, best_logits=torch.from_numpy(logits),
        last_logits=torch.from_numpy(-logits), history=[{'tag': 'epoch_1_step_3'}, {'tag': 'epoch_2_step_3'}]),
        parent / 'last.pt')
    atomic_json(parent / 'completed.json', dict(version=cfg['version'], recipe=cfg['recipe'], status='complete',
        best_tag='epoch_1_step_3', last_tag='epoch_2_step_3', checkpoint_sha256=digest(parent / 'last.pt')))
    return parent, cfg, train, dev, best


def fixture_phones(dataset, row):
    # A manually supplied segmentation tests orchestration only. Production
    # CTC extraction, cache identity and masks have separate phoneme tests.
    return [dict(instance=i, token=token, start_sample=start, end_sample=end,
                 posterior=.9, valid=True)
            for i, (token, start, end) in enumerate((('a', 500, 2200), ('b', 2800, 4500), ('c', 5000, 7600)))]
