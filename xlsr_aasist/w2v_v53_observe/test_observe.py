import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from w2v_v53.test_step import fixture
from w2v_v53.step import train_step
from w2v_v53.model import optimizer_for
from w2v_v3191.meta import rng_state, restore_rng
from w2v_v3191.test_step import assert_rng_equal
from .observe import (classification_plan, capture_classifier, analyze, observed_step,
                      entropy_floor)
from .runtime import Recorder, make_run_stage, checkpoint_cursor, pipeline_identity
from .summary import summarize


def synthetic(cfg, units, modifier=None):
    batches = classification_plan(units, cfg, 1)
    captured, cells = [], {}
    total = 0.
    for batch in batches:
        logits = []
        for item in batch:
            label = item['source']['label']
            probabilities = [.01, .01]
            probabilities[label] = .99
            lp = [math.log(p) for p in probabilities]
            if modifier:
                lp = modifier(item, lp)
            logits.append(lp)
            normalized = torch.log_softmax(torch.tensor(lp, dtype=torch.float64), -1).tolist()
            ce = -(1-cfg['label_smoothing'])*normalized[label] - cfg['label_smoothing']*sum(normalized)/2
            total += ce*item['mass']
            key = f'{item["view"]}/{item["source"]["language"]}/{label}'
            cell = cells.setdefault(key, dict(views=0, correct=0, raw_ce_sum=0., objective=0.))
            cell['views'] += 1
            cell['correct'] += int((0 if normalized[0] >= normalized[1] else 1) == label)
            cell['raw_ce_sum'] -= normalized[label]
            cell['objective'] += ce*item['mass']
        captured.append(torch.tensor(logits, dtype=torch.float32))
    stats = dict(classification_loss=total, loss=total, cells=cells, amp=dict(skipped_update=False))
    return captured, batches, stats


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5311)

    def test_observation_preserves_actual_losses_adam_rng_and_encoder_calls(self):
        cfg, units, model = fixture()
        # Exercise existing dropout as well as the real AASIST and both PCLs.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .1
        other = copy.deepcopy(model)
        opt, other_opt = optimizer_for(model,cfg), optimizer_for(other,cfg)
        initial = rng_state()
        expected = train_step(other, other_opt, units, cfg, audit=True)
        expected_rng = rng_state()
        restore_rng(initial)
        calls = []
        hook = model.encoder.register_forward_hook(lambda m,args,out: calls.append(out.shape[0]))
        actual = observed_step(train_step, model, opt, units, cfg, audit=True)
        hook.remove()
        self.assertEqual(sum(calls), expected['encoded_views'])
        self.assertEqual(actual['loss'], expected['loss'])
        self.assertEqual(actual['classification_loss'], expected['classification_loss'])
        self.assertTrue(actual['hard_examples']['validated'], actual['hard_examples'])
        self.assertEqual(actual['hard_examples']['views'], 24)  # 28 encodes: 4 PCL-only refs.
        assert_rng_equal(self, expected_rng, rng_state())
        self.assertNotIn('classify_sequences', model.__dict__)
        for name,value in other.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        for key,state in other_opt.state_dict()['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=0, atol=0)

    def test_logit_order_weights_and_clean_noisy_pair_attribution(self):
        cfg, units, _ = fixture()
        changed = []
        def difficult(item, lp):
            if item['view'] == 'noisy' and item['source']['label'] == 1 and not changed:
                changed.append(item['source']['source_id'])
                return [math.log(.995), math.log(.005)]
            return lp
        captured, batches, stats = synthetic(cfg, units, difficult)
        self.assertAlmostEqual(sum(i['mass'] for b in batches for i in b), 1.)
        result = analyze(captured, batches, cfg, stats)
        self.assertTrue(result['validated'])
        self.assertEqual(result['errors'], 1)
        row = result['examples'][0]
        self.assertEqual(row['source_id'], changed[0])
        self.assertEqual(row['view'], 'noisy')
        self.assertEqual(row['true_class'], 'real')
        self.assertEqual(row['reason'], 'misclassified')
        self.assertEqual(row['pair_outcome'], 'noisy_only_wrong')
        self.assertAlmostEqual(row['p_true'], .005, places=7)
        self.assertTrue(row['paired']['correct'])
        self.assertEqual(row['paired']['view'], 'clean_rtc')
        self.assertAlmostEqual(result['ce_floor'], .0560015343548473)
        self.assertNotIn('wave', json.dumps(result))
        for batch in batches:
            views = [item['view'] for item in batch]
            if 'clean_rtc' in views:
                self.assertTrue(all(view == 'clean_rtc' for view in views[views.index('clean_rtc'):]))

    def test_extreme_correct_predictions_are_not_reported_as_wrong(self):
        cfg, units, _ = fixture()
        changed = []
        def overconfident(item, lp):
            if not changed:
                changed.append(True)
                return [1000. if i == item['source']['label'] else 0. for i in (0,1)]
            return lp
        captured,batches,stats = synthetic(cfg, units, overconfident)
        result = analyze(captured,batches,cfg,stats)
        self.assertTrue(result['validated'])
        self.assertEqual(result['errors'], 0)
        self.assertEqual(result['examples'][0]['reason'], 'overconfident_correct')
        self.assertEqual(result['examples'][0]['raw_ce'], 0.)
        self.assertGreater(result['examples'][0]['smoothed_ce'], 1.)

    def test_disagreeing_group_attribution_suppresses_sample_ids(self):
        cfg,units,_ = fixture()
        captured,batches,stats = synthetic(cfg,units)
        next(iter(stats['cells'].values()))['correct'] = -1
        result = analyze(captured,batches,cfg,stats)
        self.assertFalse(result['validated'])
        self.assertEqual(result['examples'], [])

    def test_capture_restores_instance_override_on_exception(self):
        _,_,model = fixture()
        override = lambda seq: torch.zeros(len(seq),2)
        model.classify_sequences = override
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with capture_classifier(model, [[{}]]) as rows:
                self.assertIsNot(model.classify_sequences, override)
                model.classify_sequences([torch.ones(2,128)])
                self.assertEqual(len(rows), 1)
                raise RuntimeError('injected')
        self.assertIs(model.classify_sequences, override)

    def test_amp_failed_attempt_has_no_observation_and_no_commit(self):
        cfg,units,model = fixture()
        optimizer = optimizer_for(model,cfg)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        hook = model.fusion.conditioner[-1].weight.register_hook(lambda g:g*float('nan'))
        with patch('w2v_v53.step.scaler_for', return_value=scaler):
            stats = observed_step(train_step,model,optimizer,units,cfg)
        hook.remove()
        self.assertTrue(stats['amp']['skipped_update'])
        self.assertNotIn('hard_examples', stats)
        self.assertNotIn('classify_sequences', model.__dict__)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            recorder = Recorder(folder, 461)
            recorder.commit(stats)
            self.assertEqual(recorder.cursor, 461)
            self.assertFalse((Path(folder)/'hard_examples.jsonl').exists())

    def test_pinned_pipeline_and_changed_pipeline_rejection(self):
        identity = pipeline_identity()
        self.assertEqual(len(identity['pinned_pipeline']), 4)
        with patch('w2v_v53_observe.runtime.text_digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed pipeline'):
                pipeline_identity()

    def test_roll_back_log_tail_and_report_only_observed_history(self):
        cfg,units,_ = fixture()
        captured,batches,stats = synthetic(cfg,units)
        observation = analyze(captured,batches,cfg,stats)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            stage = Path(folder)
            recorder = Recorder(stage, 461)
            for _ in range(3):
                recorder.commit(dict(stats, hard_examples=copy.deepcopy(observation)))
            with (stage/'hard_examples.jsonl').open('a') as output:
                output.write('{incomplete')
            resumed = Recorder(stage, 462)
            resumed.commit(dict(stats, hard_examples=copy.deepcopy(observation)))
            rows = [json.loads(s) for s in (stage/'hard_examples.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in rows], [462,463])
            result = summarize(stage)
            self.assertEqual(result['first_observed_update'], 462)
            self.assertEqual(result['observed_updates'], 2)
            self.assertEqual(result['classification_views'], 48)
            self.assertEqual(result['classification_errors'], 0)
            self.assertTrue((stage/'hard_examples_summary.json').exists())

    def test_enable_observer_when_resuming_existing_checkpoint_preserves_training(self):
        from w2v_v53 import train
        from w2v_v53.test_lifecycle import (StageLifecycleTests, configuration, TinyDetector,
            optimization_step, initialize_tiny)
        helper = StageLifecycleTests()
        cfg = configuration()
        empty = dict(validated=True, views=1, ce_floor=0., classification_loss=0.,
            reconstructed_ce=0., errors=0, raw_ce_mean=0., max_raw_ce=0., weighted_excess=0.,
            sample_excess_threshold=.05, cells={}, candidate_count=0, recorded_count=0,
            omitted_candidates=0, recorded_excess_fraction=0., examples=[])
        def instrument(fn, model, optimizer, ticket, runtime, **kwargs):
            result = fn(model, optimizer, ticket, runtime, **kwargs)
            result['hard_examples'] = dict(empty, classification_loss=result['classification_loss'])
            return result
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, helper.patches():
            root = Path(folder)
            _, reference = helper.run_tiny(root/'reference', cfg)
            paused_root = root/'paused'
            calls = []
            def pause(model,opt,ticket,runtime,**kwargs):
                result = optimization_step(model,opt,ticket,runtime,**kwargs)
                calls.append(ticket)
                if len(calls) == 13:
                    (paused_root/'STOP_REQUESTED').touch()
                return result
            _, old = helper.run_tiny(paused_root,cfg,step=pause)
            self.assertEqual(old['cursor'], 13)
            self.assertEqual(checkpoint_cursor(paused_root/'stage'), 13)
            (paused_root/'STOP_REQUESTED').unlink()
            original_cfg = copy.deepcopy(cfg)
            run = make_run_stage(train.run_stage, optimization_step, dict(schema='v53_hard_examples_v1'))
            with patch('w2v_v53_observe.runtime.observed_step', side_effect=instrument):
                _, resumed = run(cfg,paused_root/'stage',[dict(id='train')],
                    model_factory=TinyDetector, initializer=lambda *a:self.fail('parent restarted'))
            self.assertEqual(cfg, original_cfg)
            for name, value in reference['model'].items():
                torch.testing.assert_close(resumed['model'][name],value,rtol=0,atol=0)
            for name,value in reference['tail_model'].items():
                torch.testing.assert_close(resumed['tail_model'][name],value,rtol=0,atol=0)
            self.assertEqual(resumed['cursor'], 16)
            result = summarize(paused_root/'stage')
            self.assertEqual(result['first_observed_update'], 14)
            self.assertEqual(result['observed_updates'], 3)

    def test_existing_report_includes_bounded_summary(self):
        from w2v_v53.delivery import _report_files
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder)
            stage = run/'final'
            stage.mkdir()
            summarize(stage)
            included, skipped = _report_files(run)
            self.assertIn('final/hard_examples_summary.json', {r['file'] for r in included})


if __name__ == '__main__':
    unittest.main()
