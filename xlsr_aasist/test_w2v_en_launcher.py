"""CPU-only launcher checks: no speech, model download, GPU or network."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import start_w2v_en as en
from start_w2v_adapt import adapt_config
from w2v_rebuild.core import sha256


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'old run;$(nope)'
        self.stage = self.source/'stage3'
        self.stage.mkdir(parents=True)
        self.baseline = self.root/'original'/'stage3'/'best_model.pt'
        self.baseline.parent.mkdir(parents=True)
        self.baseline.write_bytes(b'original-best-fixture')
        self.cache = self.root/'diverse train'
        self.cache.mkdir()
        (self.cache/'config.json').write_text(json.dumps({'role':'train', 'processing':{'profile':'diverse'}, 'generation':1}), encoding='utf-8')
        (self.cache/'manifest.jsonl').write_text('', encoding='utf-8')
        self.config = {'stage':3, 'adaptation':True, 'baseline_path':str(self.baseline),
                       'init_sha256':sha256(self.baseline), 'extra_train_noisy_cache':[str(self.cache)],
                       'train_noisy_cache':str(self.root/'old cache'),
                       'dev_noisy_cache':str(self.root/'dev seen'),
                       'dev_heldout_cache':str(self.root/'dev heldout'),
                       'dev_protocol':str(self.root/'dev protocol'), 'dev_data_path':str(self.root/'dev audio'),
                       'train_protocol':'train protocol', 'train_data_path':'train audio', 'rtc_pairs':'pairs.json',
                       'train_noise_manifest':'noise.json', 'ssl_path':'ssl model', 'feature_cache':'features',
                       'seed':2345, 'algo':5, 'SNRmin':10, 'SNRmax':40, 'pair_warmup_epochs':2.,
                       'noise_environment':{'RTC_B_NOISE_PROB':'0.5'}, 'microbatch':4, 'eval_microbatch':4,
                       'eval_batch':16, 'data_fingerprints':{'dev protocol':'dev-sha', 'seen manifest':'seen-sha'}}
        (self.stage/'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        (self.stage/'candidate_best.pt').write_bytes(b'candidate-must-be-kept')
        (self.stage/'completed.json').write_text(json.dumps({'candidate_epoch':1}), encoding='utf-8')
        self.rows = [{'epoch':1, 'candidate_saved':True, 'dev':{'robust_f1':.95550}},
                     {'epoch':2, 'candidate_saved':False, 'dev':{'robust_f1':.95454}}]
        self.write_rows()

    def write_rows(self):
        (self.stage/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows), encoding='utf-8')

    def test_only_recipe_change_is_language_and_single_epoch(self):
        original = copy.deepcopy(self.config)
        expected = adapt_config(self.config, self.cache)
        actual = en.english_config(self.config, self.cache)
        expected.update(epochs=1, language_weighting=True, en_real_budget=.35, en_fake_budget=.4)
        self.assertEqual(actual, expected)
        self.assertEqual(self.config, original)
        self.assertEqual(actual['noise_environment'], self.config['noise_environment'])
        self.assertEqual((actual['seed'],actual['pair_warmup_epochs'],actual['algo']), (2345,2.,5))
        self.assertEqual((actual['noisy_extra_fraction'],actual['noisy_mix_warmup_epochs']), (.2,1.))
        self.assertEqual((actual['trainable_encoder_layers'],actual['encoder_lr'],actual['head_lr'],actual['real_ce_weight']), (4,1e-7,2e-6,1.25))
        self.assertEqual(actual['ordinary_sampling'],'legacy')

    def test_baseline_hash_and_source_checks(self):
        before = self.baseline.read_bytes()
        self.assertEqual(en.verified_baseline(self.config,self.source)[0],self.baseline.resolve())
        bad = dict(self.config, init_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            en.verified_baseline(bad,self.source)
        with self.assertRaisesRegex(ValueError,'reference run'):
            en.verified_baseline(self.config,self.baseline.parent.parent)
        self.assertEqual(self.baseline.read_bytes(),before)

    def test_output_never_inside_source_or_cache(self):
        for protected in (self.source,self.cache,self.baseline.parent):
            with self.assertRaises(ValueError):
                en.guard_output(protected/'new run',[protected])
        en.guard_output(self.root/'exp'/'new',[self.source,self.cache,self.baseline.parent])

    def test_reference_uses_candidate_epoch_not_last(self):
        result = en.reference_metrics(self.source,self.config)
        self.assertTrue(result['available'])
        self.assertEqual(result['epoch'],1)
        self.assertEqual(result['metrics']['dev']['robust_f1'],.95550)
        self.assertEqual(result['data_fingerprints'],self.config['data_fingerprints'])
        self.assertEqual((result['eval_microbatch'],result['eval_batch']),(4,16))
        self.assertEqual(result['dev_inputs']['ssl_path'],self.config['ssl_path'])
        self.assertEqual(result['dev_inputs']['dev_heldout_cache'],self.config['dev_heldout_cache'])

    def test_reference_requires_recorded_content_and_inference_settings(self):
        for update in ({'data_fingerprints':None}, {'data_fingerprints':{}},
                       {'data_fingerprints':{'dev protocol':'changed'}}, {'eval_microbatch':2}, {'eval_batch':None}):
            result=en.reference_metrics(self.source,dict(self.config,**update))
            self.assertFalse(result['available'])
            self.assertTrue(result['reason'])
        reference=dict(self.config)
        reference.pop('data_fingerprints')
        (self.stage/'config.json').write_text(json.dumps(reference),encoding='utf-8')
        result=en.reference_metrics(self.source,self.config)
        self.assertFalse(result['available'])
        self.assertIn('Reference data_fingerprints missing',result['reason'])

    def test_reference_rejects_missing_duplicate_or_unsaved_row(self):
        for rows in ([self.rows[1]], [self.rows[0],self.rows[0]], [dict(self.rows[0],candidate_saved=False)]):
            self.rows=rows
            self.write_rows()
            self.assertFalse(en.reference_metrics(self.source,self.config)['available'])

    def test_reference_rejects_changed_baseline_or_dev(self):
        for key in ('init_sha256','dev_protocol','dev_noisy_cache'):
            bad = dict(self.config, **{key:'changed'})
            self.assertFalse(en.reference_metrics(self.source,bad)['available'])

    def test_command_preserves_literal_paths_and_language_flags(self):
        values = en.english_config(self.config,self.cache)
        command = en.english_command(values,self.root/'out with spaces;$(nope)',self.baseline)
        self.assertIn('--language_weighting',command)
        self.assertEqual(command[command.index('--en_real_budget')+1],'0.35')
        self.assertEqual(command[command.index('--en_fake_budget')+1],'0.4')
        self.assertEqual(command[command.index('--epochs')+1],'1')
        self.assertEqual(command[command.index('--finetune_from')+1],str(self.baseline))
        self.assertEqual(command[command.index('--out')+1],str(self.root/'out with spaces;$(nope)'))
        self.assertNotIn('--preflight',command)
        self.assertNotIn('--resume',command)

    def test_incomplete_engine_update_fails_loudly(self):
        with patch.object(en,'training_command',return_value=['python','-m','train']):
            with self.assertRaisesRegex(RuntimeError,'complete English-weighting update'):
                en.english_command({},'out','baseline')

    def test_preview_has_no_training_or_outputs(self):
        before = self.baseline.read_bytes()
        with patch.object(en,'ROOT',self.root), patch.object(en,'run_training') as child, redirect_stdout(io.StringIO()):
            self.assertEqual(en.main(['--from-run',str(self.source)]),0)
        child.assert_not_called()
        self.assertFalse((self.root/'exp').exists())
        self.assertEqual(self.baseline.read_bytes(),before)

    def test_failed_training_still_exports_with_failed_status(self):
        before = {p:p.read_bytes() for p in [self.baseline,self.stage/'candidate_best.pt']}
        with patch.object(en,'ROOT',self.root), patch.object(en,'recovery_lock',return_value=nullcontext()), \
                patch.object(en,'run_training',return_value=7), patch.object(en,'finish_report') as report, redirect_stdout(io.StringIO()):
            self.assertEqual(en.main(['--from-run',str(self.source),'--run']),7)
        self.assertFalse(report.call_args.args[3]['training_complete'])
        self.assertEqual(report.call_args.args[3]['status'],'failed')
        self.assertTrue(report.call_args.args[3]['original_best_preserved'])
        for path,contents in before.items():
            self.assertEqual(path.read_bytes(),contents)

    def test_report_export_and_upload_failure_does_not_hide_gpu_success(self):
        run = self.root/'new'
        run.mkdir()
        metadata = {'report':'report.md','archive':'local.zip','download_archive':'download.zip'}
        with patch('w2v_rebuild.group_metrics.export_language_report',return_value=metadata) as export, \
                patch('audit_w2v_train.upload_report',side_effect=OSError('network unavailable')) as upload, redirect_stdout(io.StringIO()):
            result = en.finish_report(run,{'epoch':1},self.root/'downloads',{'status':'complete'},True)
        self.assertEqual(result,metadata)
        upload.assert_called_once_with('download.zip')
        self.assertEqual(export.call_args.kwargs['reference_metrics'],{'epoch':1})
        self.assertTrue((run/'stage3'/'launcher_status.json').is_file())

    def test_training_success_is_retained_if_report_export_fails(self):
        def complete(command, env, log_path):
            stage = Path(log_path).parent/'stage3'
            stage.mkdir()
            (stage/'completed.json').write_text(json.dumps({'status':'candidate_only',
                'best_model':str(stage/'best_model.pt'),'candidate_model':str(stage/'candidate_best.pt')}),encoding='utf-8')
            return 0
        with patch.object(en,'ROOT',self.root), patch.object(en,'recovery_lock',return_value=nullcontext()), \
                patch.object(en,'run_training',side_effect=complete), \
                patch.object(en,'finish_report',side_effect=OSError('report path unavailable')), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(en.main(['--from-run',str(self.source),'--run']),0)
        self.assertIn('REPORT_EXPORT_FAILED=OSError',output.getvalue())
        self.assertIn('EN_TRAINING_RESULT=candidate_only',output.getvalue())

    def test_shell_forwards_arguments_without_eval(self):
        shell = Path(en.__file__).with_name('run_w2v_en.sh').read_text(encoding='utf-8')
        self.assertIn('--from-run "$SOURCE" --run "$@"',shell)
        self.assertNotIn('eval ',shell)
        self.assertIn('exp/.latest_en_pid',shell)


if __name__ == '__main__':
    unittest.main()
