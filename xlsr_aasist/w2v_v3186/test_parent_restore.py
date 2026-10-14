"""Exact parent tensors and an actual fixed-Dev identity check before training."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from torch import nn
from .common import atomic_json, digest, partial_state, read_json
from .parent import baseline_check, transfer


class TinyFusion(nn.Linear):
    def __init__(self):
        super().__init__(3, 4)

    def describe(self):
        return {'fixture': True}


class TinyRestore(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.aasist = nn.Module()
        self.head.aasist.out_layer = nn.Linear(320, 2)
        self.head.aasist.norm = nn.BatchNorm1d(4)
        self.head.contribution = nn.Linear(160, 1)
        self.head.attention_delta = nn.Linear(160, 1)
        self.fusion = TinyFusion()
        self.adapter = nn.Module()
        self.adapter.lora_a = nn.Parameter(torch.randn(2, 3))
        self.adapter.lora_b = nn.Parameter(torch.randn(3, 2))
        self.joint = False

    def set_phase(self, joint):
        self.joint = joint


def metric(rows, logits):
    # Deliberately sensitive to small score changes so the test can distinguish
    # measured new scores from an accidentally recycled parent metric.
    return {'weighted_f1': float(np.asarray(logits)[:, 0].mean()), 'count': len(rows)}


class ParentRestoreTests(unittest.TestCase):
    def test_all_tensors_including_std_pooling_and_bn_are_copied_exactly(self):
        source, target = TinyRestore(), TinyRestore()
        with torch.no_grad():
            source.head.aasist.out_layer.weight[:, 160:].fill_(.731)
            source.head.aasist.out_layer.bias.fill_(-.21)
            source.head.aasist.norm.running_mean.fill_(1.2)
            source.head.aasist.norm.running_var.fill_(2.3)
            source.head.aasist.norm.num_batches_tracked.fill_(51)
            source.head.contribution.weight.fill_(.17)
            source.head.attention_delta.weight.fill_(.29)
        values = partial_state(source)
        result = transfer(target, values)
        self.assertTrue(target.joint)
        self.assertTrue(result['initial_function_preserved'])
        self.assertTrue(result['all_partial_tensors_identical'])
        self.assertTrue(result['external_window_pooling'])
        self.assertFalse(result['optimizer_inherited'])
        self.assertEqual(result['removed_tensors'], [])
        self.assertEqual(result['bn_statistics_reset'], [])
        self.assertEqual(result['copied_tensor_count'], len(values))
        for key, value in partial_state(target).items():
            torch.testing.assert_close(value, values[key], rtol=0, atol=0)

    def test_invalid_inventory_shape_dtype_and_nonfinite_never_partially_load(self):
        model = TinyRestore()
        before = partial_state(model)
        source = partial_state(TinyRestore())
        cases = []
        missing = dict(source); missing.pop('head.aasist.norm.running_mean'); cases.append(missing)
        extra = dict(source, unexpected=torch.ones(1)); cases.append(extra)
        short = dict(source); short['head.aasist.out_layer.weight'] = short['head.aasist.out_layer.weight'][:, :160]; cases.append(short)
        wrong_type = dict(source); wrong_type['fusion.weight'] = wrong_type['fusion.weight'].double(); cases.append(wrong_type)
        invalid = copy.deepcopy(source); invalid['fusion.weight'][0, 0] = float('nan'); cases.append(invalid)
        for values in cases:
            with self.subTest(keys=list(values)):
                with self.assertRaises(ValueError):
                    transfer(model, values)
                for key, value in partial_state(model).items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def fixture(self, root):
        parent, run = root / 'parent', root / 'new'
        parent.mkdir(); run.mkdir()
        dev = [dict(id='r'+str(i), condition='online', language='en', label=i % 2) for i in range(4)]
        partition = dict(select=[0, 1], calibration=[2, 3])
        expected = np.array([[.4, -.4], [.3, -.3], [.2, -.2], [.1, -.1]], dtype=np.float32)
        torch.save(dict(best_tag='epoch_4_step_2417', best_logits=torch.from_numpy(expected)), parent / 'last.pt')
        atomic_json(parent / 'config.json', dict(microbatch=2, frame_budget=40, eval_batch=4))
        atomic_json(parent / 'dev_rows.json', dev)
        atomic_json(parent / 'dev_partition.json', partition)
        cfg = dict(microbatch=8, frame_budget=120, eval_batch=16,
                   initialization=dict(run=str(parent), tag='epoch_4_step_2417', checkpoint_sha256=digest(parent / 'last.pt')))
        actual_rows = [dict(row, row_index=i) for i, row in enumerate(dev)]
        return parent, run, cfg, dev, partition, expected, actual_rows

    def test_actual_zero_update_pass_and_partition_metrics_are_saved(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            parent, run, cfg, dev, partition, expected, actual_rows = self.fixture(Path(directory))
            actual = expected + np.float32(.0001)
            infer = Mock(return_value=(actual, actual_rows))
            result = baseline_check(TinyRestore(), cfg, run, dev, partition, infer, metric)
            infer.assert_called_once()
            eval_cfg = infer.call_args.args[2]
            self.assertEqual([eval_cfg[k] for k in ('microbatch', 'frame_budget', 'eval_batch')], [2, 40, 4])
            self.assertEqual([cfg[k] for k in ('microbatch', 'frame_budget', 'eval_batch')], [8, 120, 16])
            self.assertTrue(result['verified'])
            self.assertTrue(result['new_model_reproduces_parent'])
            self.assertFalse(result['reference_only'])
            self.assertEqual(result['rows'], 4)
            self.assertEqual(result['metrics'], metric(dev, actual))
            self.assertNotEqual(result['metrics'], result['reference_metrics'])
            self.assertEqual(result['selection_metrics']['count'], 2)
            self.assertEqual(result['calibration_metrics']['count'], 2)
            with np.load(run / 'parent_baseline_scores.npz') as scores:
                np.testing.assert_array_equal(scores['logits'], actual)
            with np.load(run / 'parent_reference_scores.npz') as scores:
                np.testing.assert_array_equal(scores['logits'], expected)
            self.assertTrue(read_json(run / 'parent_baseline.json')['verified'])

    def test_prediction_mismatch_is_recorded_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            parent, run, cfg, dev, partition, expected, actual_rows = self.fixture(Path(directory))
            actual = expected + .2
            with self.assertRaisesRegex(ValueError, 'No training update made'):
                baseline_check(TinyRestore(), cfg, run, dev, partition, Mock(return_value=(actual, actual_rows)), metric)
            result = read_json(run / 'parent_baseline.json')
            self.assertFalse(result['verified'])
            self.assertFalse(result['new_model_reproduces_parent'])
            self.assertGreater(result['max_abs_logit_error'], .19)
            with np.load(run / 'parent_baseline_scores.npz') as scores:
                np.testing.assert_array_equal(scores['logits'], actual)

    def test_wrong_manifest_and_partition_are_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, run, cfg, dev, partition, expected, actual_rows = self.fixture(Path(directory))
            infer = Mock()
            with self.assertRaisesRegex(ValueError, 'manifest/order'):
                baseline_check(TinyRestore(), cfg, run, dev[::-1], partition, infer, metric)
            with self.assertRaisesRegex(ValueError, 'partition differs'):
                baseline_check(TinyRestore(), cfg, run, dev, dict(select=[0, 2], calibration=[1, 3]), infer, metric)
            infer.assert_not_called()

    def test_actual_row_order_and_nonfinite_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            parent, run, cfg, dev, partition, expected, actual_rows = self.fixture(Path(directory))
            altered = copy.deepcopy(actual_rows); altered[0]['label'] = 1
            for rows in (actual_rows[::-1], altered):
                with self.assertRaisesRegex(ValueError, 'row identity/order'):
                    baseline_check(TinyRestore(), cfg, run, dev, partition, Mock(return_value=(expected, rows)), metric)
            invalid = expected.copy(); invalid[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                baseline_check(TinyRestore(), cfg, run, dev, partition, Mock(return_value=(invalid, actual_rows)), metric)


if __name__ == '__main__':
    unittest.main()
