"""Budget conservation, Train-only head fitting, microbatch gradients and selection."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from w2v_v39.metrics import measure
from .data import SourcePlan, head_weights, loss_weights
from .head import fit
from .train import acceptance, train_step, update_selections
from .state import atomic_save


def rows_fixture(count=3):
    rows = []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for number in range(count):
                source = f'offline/{language}/{label}_{number}.wav'
                sha = hashlib.sha256(source.encode()).hexdigest()
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    if condition == 'online' and number % 2:
                        continue
                    rows.append(dict(id=source if condition != 'online' else source.replace('offline/', 'online/'),
                        source_id=source, group_id=sha, source_sha256=sha, audio_sha256=sha,
                        language=language, label=label, condition=condition, split='train', view='full', full_length=True))
    return rows


class CoreTests(unittest.TestCase):
    def test_offline_diagnostic_cannot_change_weighted_score(self):
        rows, logits = [], []
        for condition in ('online','seen','heldout'):
            for lang in ('en','zh'):
                for label in (0,1):
                    rows.append(dict(condition=condition,language=lang,label=label))
                    logits.append([2.,-2.] if label==0 else [-2.,2.])
        before = measure(rows,np.array(logits))
        after = measure(rows+[dict(condition='offline',language='en',label=1)],np.array(logits+[[100.,-100.]]))
        self.assertEqual(before['weighted_f1'], after['weighted_f1'])
        self.assertEqual(after['groups']['offline/en']['recall'][1],0.)

    def test_missing_online_never_changes_noisy_or_class_language_budget_and_resume_order(self):
        rows = rows_fixture(9)
        plan = SourcePlan(rows, 16, 31401)
        for epoch in range(3):
            full = plan.batches(epoch)
            self.assertEqual(full[1:], plan.batches(epoch, 1))
            for batch in full:
                examples = [dict(rows[t.index], ce_weight=t.ce_weight) for t in batch]
                weights = loss_weights(examples)
                self.assertAlmostEqual(sum(w for r,w in zip(examples, weights) if r['condition'].startswith('noisy_')), .5)
                self.assertEqual(len(batch), 32)
                self.assertEqual(len({t.pair_occurrence for t in batch}), 16)
        weights = head_weights(rows)
        self.assertAlmostEqual(weights.sum(), 1.)
        for lang in ('en', 'zh'):
            for label in (0, 1):
                self.assertAlmostEqual(sum(w for r,w in zip(rows, weights) if r['language']==lang and r['label']==label), .25)
        self.assertAlmostEqual(sum(w for r,w in zip(rows, weights) if r['condition'].startswith('noisy_')), .5)
        changed = copy.deepcopy(examples)
        changed[0]['ce_weight'] *= 2
        with self.assertRaises(ValueError):
            loss_weights(changed)

    def test_convex_head_updates_only_its_parameters_and_rejects_dev_rows(self):
        rows = rows_fixture(2)
        x = np.array([[1. if r['label']==0 else -1., .3 if r['language']=='en' else -.3] for r in rows], dtype=np.float32)
        original = dict(weight=torch.tensor([[-.5, .1], [.5, -.1]]), bias=torch.zeros(2))
        frozen = copy.deepcopy(original)
        cfg = dict(device='cpu', head_steps=20, head_anchor=.1, head_chunk=7, head_grad_tolerance=1e-7)
        output, report = fit(dict(rows=rows, x=x), original, cfg)
        self.assertLess(report['final']['objective'], report['initial']['objective'])
        self.assertGreater(report['parameter_displacement_l2'], .1)
        for key in original:
            torch.testing.assert_close(original[key], frozen[key], rtol=0, atol=0)
        logits = torch.from_numpy(x) @ output['weight'].T + output['bias']
        self.assertTrue(np.all(logits.argmax(-1).numpy() == [r['label'] for r in rows]))
        with self.assertRaisesRegex(ValueError, 'Train only'):
            fit(dict(rows=[dict(r, split='dev') for r in rows], x=x), original, cfg)

    def test_microbatch_splitting_preserves_global_gradient_and_one_optimizer_update(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Module()
                self.head.classifier = nn.Linear(2, 2)
            def forward(self, x, mask):
                return self.head.classifier((x*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)), None
        torch.manual_seed(37)
        a, b = Tiny(), Tiny()
        b.load_state_dict(a.state_dict())
        rows = rows_fixture()
        plan = SourcePlan(rows, 16, 1)
        examples = [dict(rows[t.index], ce_weight=t.ce_weight, features=torch.randn(1, 4, 2), mask=torch.ones(1,4, dtype=torch.long))
                    for t in plan.batches(0)[0]]
        cfg = dict(device='cpu', amp='none', microbatch=1, frame_budget=500, max_grad_norm=1.)
        optim_a = torch.optim.SGD(a.parameters(), lr=.1)
        optim_b = torch.optim.SGD(b.parameters(), lr=.1)
        train_step(a, optim_a, examples, cfg)
        train_step(b, optim_b, examples, dict(cfg, microbatch=32))
        for x,y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x,y,atol=2e-8,rtol=1e-6)

    def test_best_weighted_is_retained_even_when_rejected_by_classification_guards(self):
        rows, z0, z1 = [], [], []
        for condition in ('online','seen','heldout'):
            for lang in ('en','zh'):
                for label in (0,1):
                    for j in range(20):
                        rows.append(dict(condition=condition, language=lang, label=label))
                        base_pred = 0 if label==0 or j<6 else 1
                        candidate_pred = 1 if label==1 or j==0 else 0
                        z0.append([2.,-2.] if base_pred==0 else [-2.,2.])
                        z1.append([2.,-2.] if candidate_pred==0 else [-2.,2.])
        baseline, current = measure(rows,np.array(z0)), measure(rows,np.array(z1))
        cfg = dict(min_gain=.0002,max_clean_drop=.003,max_fake_drop=.005,max_real_drop=.015,
                   max_auc_drop=.002,max_matched_real_drop=.02)
        eligible,reasons = acceptance(baseline,current,cfg)
        self.assertFalse(eligible)
        self.assertTrue(any('fake_recall_drop' in r for r in reasons))
        selectors = dict(best_weighted='starting_last',best_guarded='starting_last')
        scores = {k:baseline['weighted_f1'] for k in selectors}
        bank,promoted = update_selections(selectors,scores,{},'new',current,eligible,dict(kind='head',state={}))
        self.assertEqual(promoted,['best_weighted'])
        self.assertEqual(selectors['best_guarded'],'starting_last')
        self.assertEqual(set(bank),{'new'})

    def test_atomic_failure_keeps_previous_checkpoint_and_shared_tensor_storage_is_not_duplicated(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            value = torch.randn(10000)
            atomic_save(path,dict(model=value,candidates={'same':value}),0)
            self.assertLess(path.stat().st_size,60000)
            before = path.read_bytes()
            with patch('w2v_v313.state.shutil.disk_usage',return_value=type('Disk',(),dict(free=0))()):
                with self.assertRaises(OSError):
                    atomic_save(path,dict(model=torch.zeros(10000)),0)
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
