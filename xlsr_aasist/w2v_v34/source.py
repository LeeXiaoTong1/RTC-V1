"""Read-only binding of a completed V3.3 winner to its exported submission."""
import copy
import json
from pathlib import Path
import re

import torch

from w2v_aasist.runtime import sha256
from w2v_v33 import SCHEMA as SOURCE_SCHEMA
from w2v_v33.control import quality


DEFAULT_SUBMISSION_ROOT = Path('/home/ubuntu/LXT/temp')


def _json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required V3.3 source evidence is missing: ' + str(path))
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def _same_path(recorded, expected):
    return isinstance(recorded, str) and Path(recorded).expanduser().resolve() == expected.resolve()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def resolve_source(source_run, submission_meta=None):
    """Resolve only the checkpoint the default V3.3 export command selected.

    The metadata proves which weights produced the exported file; it does not
    authenticate an official leaderboard score or assume the paired arm won.
    No configuration builder, cache generator, cleanup, or model inference runs.
    """
    fingerprints = {}

    def read_evidence(path):
        path = Path(path).resolve()
        if not path.is_file():
            return _json(path)  # Preserve the actionable missing-file error.
        before = sha256(path)
        value = _json(path)
        if sha256(path) != before:
            raise RuntimeError('V3.3 source evidence changed while reading: '+str(path))
        fingerprints[str(path)] = before
        return value

    run = Path(source_run).expanduser().resolve()
    comparison_path, root_config_path = run/'comparison.json', run/'config.json'
    comparison, root_cfg = read_evidence(comparison_path), read_evidence(root_config_path)
    if root_cfg.get('version') != '3.3':
        raise ValueError('Source must be the V3.3 experiment root, not a V3 run or arm directory')
    if comparison.get('status') != 'complete':
        raise ValueError('V3.3 comparison is not complete; finish/resume it before starting V3.4')
    arm = comparison.get('selected_arm')
    if arm not in ('control', 'candidate') or arm not in root_cfg.get('arms', ()):
        raise ValueError('V3.3 comparison does not identify a requested selected arm')
    checkpoint = (run/arm/'best_model.pt').resolve()
    if not _same_path(comparison.get('checkpoint'), checkpoint) or not checkpoint.is_file():
        raise ValueError('V3.3 selected checkpoint path changed or is missing')
    # A symlink must not redirect a claimed arm winner out of this experiment.
    checkpoint.relative_to(run)
    arm_cfg_path, completed_path = run/arm/'config.json', run/arm/'completed.json'
    cfg, completed = read_evidence(arm_cfg_path), read_evidence(completed_path)
    expected_cfg = dict(root_cfg, arm=arm)
    if arm == 'control':
        expected_cfg['pair_weight'] = 0.
    if cfg != expected_cfg:
        raise ValueError('Selected arm configuration differs from the completed V3.3 root recipe')
    selection = completed.get('selection', {})
    best, anchor = selection.get('best_safe'), selection.get('anchor')
    reported = comparison.get('arms', {}).get(arm, {})
    if (not isinstance(best, dict) or not isinstance(anchor, dict)
            or reported.get('selected') != best or reported.get('anchor') != anchor
            or not _same_path(reported.get('checkpoint'), checkpoint)):
        raise ValueError('V3.3 comparison differs from the completed arm selection')
    tag = best.get('tag')
    if not isinstance(tag, str) or not tag:
        raise ValueError('Completed V3.3 selection has no checkpoint tag')

    meta_path = (Path(submission_meta).expanduser().resolve() if submission_meta else
                 (DEFAULT_SUBMISSION_ROOT/(run.name+'_submission')/'submission_meta.json').resolve())
    if not meta_path.is_file():
        raise FileNotFoundError('Cannot verify the submitted checkpoint: missing '+str(meta_path)+
            '. Pass --submission-meta /path/to/the/original/submission_meta.json from the V3.3 export.')
    meta = read_evidence(meta_path)
    checkpoint_hash = sha256(checkpoint)
    if meta.get('checkpoint_sha256') != checkpoint_hash or meta.get('checkpoint_tag') != tag:
        raise ValueError('Submission metadata does not match the selected V3.3 checkpoint SHA256 and tag')
    if (meta.get('score') != 'P(fake)' or meta.get('threshold') != .5
            or meta.get('input_policy') != 'full utterance' or meta.get('eval_amp') != 'none'
            or type(meta.get('count')) is not int or meta['count'] <= 0
            or not _digest(meta.get('zip_sha256')) or not _digest(meta.get('protocol_sha256'))):
        raise ValueError('Submission metadata has an incompatible or incomplete V3.3 inference policy')
    archive = meta_path.parent/'submission.zip'
    archive_verified = archive.is_file()
    if archive_verified and sha256(archive) != meta['zip_sha256']:
        raise ValueError('submission.zip differs from its recorded export SHA256')

    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(state, dict) or state.get('schema') != SOURCE_SCHEMA
            or state.get('kind') != 'weights' or state.get('tag') != tag):
        raise ValueError('V3.4 requires selected V3.3 weight schema/kind/tag; last.pt is not a warm source')
    if state.get('config') != cfg or state.get('baseline_sha256') != cfg.get('baseline_sha256'):
        raise ValueError('Selected checkpoint configuration differs from its saved arm configuration')
    if not isinstance(state.get('model'), dict) or not state['model']:
        raise ValueError('Selected V3.3 checkpoint has no model weights')
    actual_quality = quality(state.get('dev', {}))
    if any(key not in best or abs(actual_quality[key]-best[key]) > 1e-10 for key in actual_quality):
        raise ValueError('Selected checkpoint Dev metrics differ from completed best_safe selection')
    data_fingerprints = state.get('data_fingerprints')
    if not isinstance(data_fingerprints, dict) or not data_fingerprints:
        raise ValueError('Selected V3.3 checkpoint has no input fingerprints')
    if cfg.get('preparation_fingerprints') != data_fingerprints:
        raise ValueError('V3.3 prepared inputs differ from the selected checkpoint fingerprints')
    del state
    if any(sha256(path) != digest for path, digest in fingerprints.items()):
        raise RuntimeError('V3.3 source evidence changed while resolving the submitted winner')
    # Checkpoint immutability is checked again by the training workflow before use.
    provenance = dict(format='rtc_v34_submitted_source_v1', source_v33_run=str(run),
        selected_arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        checkpoint_tag=tag, baseline_fallback=tag == 'baseline',
        inherited_source_run=cfg.get('source_run'), inherited_pair_weight=cfg.get('pair_weight'),
        submission_meta=str(meta_path), submission_meta_sha256=fingerprints[str(meta_path)],
        submission_metadata=meta, submission_zip=str(archive), submission_zip_verified=archive_verified,
        file_fingerprints=fingerprints, source_data_fingerprints=copy.deepcopy(data_fingerprints),
        note='SHA256/tag bind the warm start to the exported submission. An official score is not read or optimized here; baseline fallback is allowed and explicitly recorded.')
    return dict(config=copy.deepcopy(cfg), checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash, checkpoint_tag=tag, selected_arm=arm,
                provenance=provenance)
