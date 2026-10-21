"""Attach observation to the unmodified, checkpoint-pinned V5.3 workflow."""
import hashlib
import json
from pathlib import Path
import torch
from w2v_v53.common import ROOT, atomic_json, append_json, read_json
from w2v_v3184.recovery import trim_step_log
from . import VERSION
from .observe import observed_step, MAX_EXAMPLES_PER_UPDATE, SAMPLE_EXCESS_THRESHOLD
from .summary import summarize


def text_digest(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()


def pipeline_identity():
    """Fail before training if the source-to-logit order contract has changed."""
    folder = Path(__file__).parent
    expected = read_json(folder/'pipeline.json')
    for name, checksum in expected.items():
        if text_digest(ROOT/name) != checksum:
            raise ValueError('Hard-example observer needs review for changed pipeline: ' + name)
    return dict(schema=VERSION, pinned_pipeline=expected,
        observer_code={p.name: text_digest(p) for p in sorted(folder.glob('*.py'))
                       if not p.name.startswith('test_')},
        pipeline_manifest_sha256=text_digest(folder/'pipeline.json'))


def checkpoint_cursor(stage):
    checkpoint = Path(stage)/'last.pt'
    if not checkpoint.exists():
        return 0
    # Memory-map tensor storages: no second materialized copy of model/Adam state.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    return int(state['cursor'])


class Recorder:
    def __init__(self, stage, cursor, checkpoint_interval=100):
        self.stage, self.cursor = Path(stage), cursor
        self.interval = checkpoint_interval
        self.stage.mkdir(parents=True, exist_ok=True)
        trim_step_log(self.stage/'hard_examples.jsonl', cursor)

    def commit(self, stats):
        if stats['amp']['skipped_update']:
            return stats
        self.cursor += 1
        observation = stats['hard_examples']
        event = dict(schema=VERSION, update=self.cursor, total_loss=stats['loss'], **observation)
        append_json(self.stage/'hard_examples.jsonl', event)
        # Avoid duplicating sample recipes in the existing large steps.jsonl.
        stats['hard_examples'] = {k:v for k,v in observation.items() if k not in ('examples', 'cells')}
        if not observation['validated']:
            print(f'[V5.3 hard] update={self.cursor} attribution FAILED; IDs suppressed', flush=True)
        elif observation['examples'] and (observation['errors'] or observation['weighted_excess'] >= .02):
            row = observation['examples'][0]
            print(f'[V5.3 hard] update={self.cursor} CE={observation["classification_loss"]:.5f} '
                  f'errors={observation["errors"]}/{observation["views"]} '
                  f'worst={row["view"]}/{row["language"]}/{row["true_class"]} '
                  f'p_true={row["p_true"]:.5f} reason={row["reason"]} '
                  f'pair={row.get("pair_outcome", "unpaired")} '
                  f'id={row.get("id", row.get("source_id"))}', flush=True)
        if self.cursor % self.interval == 0:
            summarize(self.stage)
        return stats


def make_run_stage(original, step_fn, identity):
    def run_stage(cfg, run, rows, **kwargs):
        if 'step_fn' in kwargs:
            raise ValueError('Observed production stage cannot replace the pinned training step')
        stage = Path(run)
        cursor = checkpoint_cursor(stage)
        recorder = Recorder(stage, cursor, cfg['checkpoint_interval'])
        policy_path = stage/'hard_examples_policy.json'
        old = read_json(policy_path) if policy_path.exists() else {}
        if old and old.get('schema') != VERSION:
            raise ValueError('Hard-example telemetry schema changed')
        # Keep historical configurations, recipe locks and checkpoints byte-for-byte.
        atomic_json(policy_path, dict(identity,
            first_enabled_after_update=old.get('first_enabled_after_update', cursor),
            latest_resume_checkpoint=cursor, max_examples_per_update=MAX_EXAMPLES_PER_UPDATE,
            sample_ce_excess_threshold=SAMPLE_EXCESS_THRESHOLD,
            extra_encoder_passes=0, changes_training_objective=False,
            sample_selection_affects_training=False, historical_config_unchanged=True))
        print(f'[V5.3 hard] observing existing forwards from update {cursor+1}; '
              'no extra encoder pass; historical updates are not replayed', flush=True)

        def step(model, optimizer, units, runtime, **step_kwargs):
            stats = observed_step(step_fn, model, optimizer, units, runtime, **step_kwargs)
            return recorder.commit(stats)

        outcome = None
        try:
            outcome = original(cfg, run, rows, step_fn=step, **kwargs)
            return outcome
        finally:
            # An abrupt failure can leave observations beyond the disk checkpoint.
            # The summary must not describe rolled-back updates as durable history.
            durable = int(outcome[1]['cursor']) if outcome is not None else checkpoint_cursor(stage)
            summarize(stage, maximum=durable)
    return run_stage


def main():
    from w2v_v53 import workflow
    from w2v_v53.step import train_step
    identity = pipeline_identity()
    original = workflow.run_stage
    workflow.run_stage = make_run_stage(original, train_step, identity)
    try:
        workflow.main()
    finally:
        workflow.run_stage = original
