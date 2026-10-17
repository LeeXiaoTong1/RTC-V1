"""Best-only fixed-probability exports and CPU-only independent upload retries."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import soundfile as sf
import torch

from .common import RECIPE, schema_for, fingerprint, digest, atomic_json, read_json, partial_state
from .evaluate import export, selected
from .delivery import check_archive, deliver_export, finish_request
from .reports import retry_outputs


class Fusion(torch.nn.Linear):
    def __init__(self): super().__init__(1, 1)
    def describe(self): return dict(kind='test fusion')


class Detector(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head = torch.nn.Module(); self.head.aasist = torch.nn.Module()
        self.head.aasist.out_layer = torch.nn.Linear(384, 2)
        self.head.aasist.normalization = 'GN/LN'
        self.fusion = Fusion()
    def set_phase(self, joint): pass
    def forward(self, waves):
        return self.head.aasist.out_layer.bias.expand(len(waves), -1)


def fixture(root):
    root = Path(root); run = root/'run'; run.mkdir()
    cfg = dict(version='3.18.10', recipe=RECIPE, seed=3181001, pooling='asp', local_evidence=True,
               window=128, hop=64, window_batch=16, consistency='none', omni_arch='3b',
               noisy_views_per_source=1, rtc_activity_policy='mixed', epochs=15, experiment='E2',
               initialization=dict(mode='pretrained', old_detector_loaded=False),
               device='cpu', workers=0, eval_workers=0, microbatch=2, frame_budget=9000,
               eval_batch=2, whole_max_frames=3000, raw_audio_cache_mib=1,
               backend_precision='fp32', omni_provenance=dict(fixture=True), lora_layers=60,
               classification_objective='source_isolated_metadg_ce', validation_protocol_hash='locked-protocol',
               saved_manifests={})
    atomic_json(run/'config.json', cfg)
    model = Detector(cfg)
    with torch.no_grad(): model.head.aasist.out_layer.bias.copy_(torch.tensor([1., -1.]))
    best = partial_state(model)
    with torch.no_grad(): model.head.aasist.out_layer.bias.copy_(torch.tensor([-1., 1.]))
    last = partial_state(model)
    history = [dict(epoch=9, tag='epoch_9_step_1403'), dict(epoch=15, tag='epoch_15_step_1403')]
    state = dict(schema=schema_for(cfg), identity=fingerprint(cfg), history=history,
                 best_tag=history[0]['tag'], last_tag=history[1]['tag'], best_model=best, model=last)
    torch.save(state, run/'last.pt'); atomic_json(run/'training_history.json', history)
    done = dict(status='complete', version='3.18.10', recipe=RECIPE, best_tag=state['best_tag'],
                last_tag=state['last_tag'], completed_epochs=15, planned_epochs=15,
                checkpoint_sha256=digest(run/'last.pt'))
    atomic_json(run/'completed.json', done)
    audit = dict(identity=dict(checkpoint_sha256=done['checkpoint_sha256'], selected=done['best_tag'],
                               protocol_sha256=cfg['validation_protocol_hash']),
                 metrics=dict(split='audit', score=None, marker='already evaluated once'), used_for_selection=False)
    atomic_json(run/'locked_audit.json', audit)
    audio = root/'progress'; audio.mkdir(); ids = ['three.wav', 'one.wav', 'two.wav']
    for n, name in enumerate(ids): sf.write(audio/name, np.linspace(-.02, .03, 4800+n*320), 16000)
    protocol = root/'progress.txt'; protocol.write_text('\n'.join(ids)+'\n')
    destination = root/'downloads'; destination.mkdir()
    atomic_json(run/'report_request.json', dict(destination=str(destination), upload=True))
    (run/'forbidden.wav').write_bytes(b'not report audio')
    return run, cfg, protocol, audio, destination


def export_fixture(root):
    run, cfg, protocol, audio, destination = fixture(root)
    out = Path(root)/'submission'
    archive = export(run, out, device='cpu', workers=0, protocol=protocol, audio_root=audio,
                     model_factory=Detector, verify=False)
    meta = read_json(out/'submission_meta.json')
    request = dict(version='3.18.10', recipe=RECIPE, kind='best', protocol=str(protocol),
                   protocol_sha256=digest(protocol), audio_root=str(audio), out=str(out),
                   count=3, id_order_sha256=meta['id_order_sha256'], upload=True)
    atomic_json(run/'submission_request.json', request)
    return run, cfg, out, destination, request, archive


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_real_export_selects_best_and_raw_fake_probability_without_fit(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run, cfg, out, _, _, archive = export_fixture(directory)
            before = digest(run/'last.pt'); _, meta = check_archive(out)
            with zipfile.ZipFile(archive) as stream:
                self.assertEqual(stream.namelist(), ['scores.txt'])
                rows = [line.split() for line in stream.read('scores.txt').decode().splitlines()]
            self.assertEqual([row[0] for row in rows], ['three.wav', 'one.wav', 'two.wav'])
            expected = torch.tensor([1., -1.]).softmax(0)[0].item()
            for _, value in rows: self.assertAlmostEqual(float(value), expected, places=6)
            self.assertEqual(meta['selected'], 'epoch_9_step_1403')
            self.assertEqual(meta['calibration']['selected_candidate'], 'fixed_identity')
            self.assertFalse(meta['calibration']['fit']); self.assertFalse(meta['target_adaptation'])
            self.assertFalse(meta['inference_requires_phonemes']); self.assertFalse(meta['inference_requires_language'])
            self.assertEqual(meta['classifier_input_dim'], 384)
            self.assertEqual(digest(run/'last.pt'), before)
            _, last, selected_last = selected(run, 'last')
            self.assertEqual(selected_last['selected'], 'epoch_15_step_1403')
            self.assertEqual(last['head.aasist.out_layer.bias'].tolist(), [-1., 1.])

    def test_locked_audit_export_reuses_saved_metrics_without_model_or_audio_inference(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run, _, _, _, _ = fixture(directory)
            with patch('w2v_v31810.evaluate.infer', side_effect=AssertionError('No inference for saved audit')):
                target = export(run, Path(directory)/'audit', audit=True,
                                model_factory=lambda cfg: self.fail('Do not instantiate a model for audit reuse'))
            self.assertEqual(read_json(target), read_json(run/'locked_audit.json'))
            changed = read_json(run/'locked_audit.json'); changed['identity']['selected'] = 'wrong-best'
            atomic_json(run/'locked_audit.json', changed)
            with self.assertRaisesRegex(ValueError, 'verified locked audit'):
                export(run, Path(directory)/'bad-audit', audit=True)

    def test_tampered_or_uncompleted_best_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _, _, _ = fixture(directory)
            done = read_json(run/'completed.json'); done['best_tag'] = 'epoch_16_step_1'
            atomic_json(run/'completed.json', done)
            with self.assertRaisesRegex(ValueError, 'completed, recorded'):
                selected(run, 'best')
            with (run/'last.pt').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint hash changed'):
                selected(run, 'best')

    def test_calibrated_metadata_cannot_be_reused_as_fixed_identity(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            _, _, out, _, _, _ = export_fixture(directory)
            meta = read_json(out/'submission_meta.json')
            meta['calibration']['bias'] = -.7; atomic_json(out/'submission_meta.json', meta)
            with self.assertRaisesRegex(ValueError, 'fixed|identity|calibration|unfitted'):
                check_archive(out)

    def test_independent_link_retry_never_reloads_checkpoint_and_preserves_archives(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run, _, out, destination, _, archive = export_fixture(directory)
            before = digest(run/'last.pt'); scores_hash = digest(archive)
            with patch('w2v_v31810.delivery.upload_archive', side_effect=RuntimeError('submission network fixture')), \
                 patch('w2v_v31810.reports.upload_archive', return_value='https://temp.sh/r/report.zip') as report_upload, \
                 patch('torch.load', side_effect=AssertionError('Retries must not load checkpoint')):
                with self.assertRaisesRegex(RuntimeError, 'Artifact delivery incomplete'):
                    deliver_export(run, out, destination)
                self.assertEqual(report_upload.call_count, 1)
                self.assertEqual(read_json(run/'artifact_links.json')['report_url'], 'https://temp.sh/r/report.zip')
            with patch('w2v_v31810.delivery.upload_archive', return_value='https://temp.sh/s/submission.zip') as submit_upload, \
                 patch('w2v_v31810.reports.upload_archive', return_value='https://temp.sh/r2/report.zip') as report_upload, \
                 patch('torch.load', side_effect=AssertionError('Retries must not load checkpoint')):
                links = deliver_export(run, out, destination)
                again = deliver_export(run, out, destination)
                self.assertEqual(links, again); self.assertEqual(submit_upload.call_count, 1)
                self.assertEqual(report_upload.call_count, 1)
            self.assertEqual(links['submission_url'], 'https://temp.sh/s/submission.zip')
            self.assertEqual(links['report_url'], 'https://temp.sh/r2/report.zip')
            self.assertEqual(digest(archive), scores_hash); self.assertEqual(digest(run/'last.pt'), before)
            with zipfile.ZipFile(links['report_archive']) as stream:
                names = set(stream.namelist())
                self.assertIn('locked_audit.json', names); self.assertIn('training_history.json', names)
                self.assertNotIn('last.pt', names); self.assertNotIn('forbidden.wav', names)

    def test_completed_retry_checks_existing_best_and_keeps_other_link_on_report_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run, cfg, out, _, request, _ = export_fixture(directory)
            with patch('w2v_v31810.delivery.ROOT', Path(directory)), \
                 patch('w2v_v31810.delivery.upload_archive', return_value='https://temp.sh/s/submission.zip'), \
                 patch('torch.load', side_effect=AssertionError('No GPU/checkpoint load for existing scores')):
                result = finish_request(run, cfg, request, exporter=lambda *a, **k: self.fail('Must reuse valid export'))
                self.assertTrue(result['uploaded'])
            with patch('w2v_v31810.reports.upload_archive', side_effect=RuntimeError('report network fixture')), \
                 patch('torch.load', side_effect=AssertionError('No checkpoint load for retry')):
                with self.assertRaisesRegex(RuntimeError, 'report:'): retry_outputs(run)
            links = read_json(run/'artifact_links.json')
            self.assertEqual(links['submission_url'], 'https://temp.sh/s/submission.zip')
            self.assertIsNone(links['report_url'])
            with (run/'last.pt').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
                finish_request(run, cfg, request)


if __name__=='__main__': unittest.main()
