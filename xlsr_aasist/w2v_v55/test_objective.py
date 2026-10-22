import copy
import json
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v3189.test_model import tiny_cfg
from w2v_v3191.model import Detector, optimizer_for
from w2v_v3191.test_step import fixture as source_fixture
from w2v_v3191.side import side_pass as parent_side
from w2v_v3191.meta import rng_state, restore_rng
from w2v_v3191.precision import scaler_for
from .sampling import annotate
from .objective import paper_losses, classify_with_readout
from .batching import microbatches, layout
from .side import side_pass
from .step import train_step


def fixture(marked=True, pcl=True):
    cfg, units = source_fixture(copies=3)
    cfg.update(tiny_cfg(3))
    cfg.update(retained_lora_layers=3, lora_layers=2, encoder_layers=3, lora_dropout=0.,
        consistency='dual' if pcl else 'none', microbatch=16, frame_budget=9000,
        robust_arm='baseline', _candidate_diagnostics=False, hp_ramp_updates=1,
        hp_guidance_weight=.01, hp_triplet_weight=.01, hp_mode='hard_positive',
        pcl_ramp_updates=1, pcl_noise_weight=.15, pcl_platform_weight=.045,
        pcl_min_frames=1, pcl_variance_weight=.01, pcl_variance_min_sources=1,
        pcl_variance_min_phones=1, pcl_variance_target=.5)
    rows, tickets = [], []
    for i, unit in enumerate(units):
        r = unit[0]
        r.pop('robust_candidates', None)
        rows.append(r)
        tickets.append(dict(index=i, kind='offline' if r['role'] == 'noisy' else 'online',
            epoch=0, step=0, meta_side=r['meta_side'], canonical_source=r['canonical_source'], occurrence=r['id']))
        if r['role'] == 'noisy':
            r['pcl_refs']['clean']['wave'] = r['pcl_refs']['clean']['wave'] + .1*np.sin(np.arange(4000)*.03).astype(np.float32)
            r['pcl_refs']['official'] = dict(r['pcl_refs']['clean'], role='official_ref')
            r['pcl_refs']['official']['wave'] = r['wave'] + .1*np.cos(np.arange(4000)*.09).astype(np.float32)
            r.update(pcl_noisy_index=0, pcl_platform_budget=3,
                pcl_pairs={'noise': [[400, 3000, 400, 3000]], 'platform': [[400, 3000, 400, 3000]]})
    for t in annotate(tickets, rows, 123) if marked else []:
        r = rows[t['index']]
        for key in ('hp_group', 'hp_member_role', 'hp_skip'):
            if key in t:
                r[key] = t[key]
                for ref in r.get('pcl_refs', {}).values():
                    ref[key] = t[key]
        if t.get('hp_member_role') == 'anchor':
            r['hp_positive2'] = dict(r, wave=r['wave'] + .3 * np.sin(np.arange(len(r['wave']))*.07).astype(np.float32))
            r['hp_positive2'].pop('pcl_refs', None)
            r['hp_positive2'].pop('pcl_pairs', None)
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList([TinyLayer() for _ in range(3)])
    model = Detector(TinyFrontend(), encoder, cfg, TinyLayout)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.
    model.pcl_scale.observe(torch.randn(6, 128), 's1')
    model.pcl_scale.observe(torch.randn(6, 128), 's2')
    model.pcl_scale.freeze()
    return cfg, units, model


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(55)

    def test_exact_equations_hard_selection_and_all_six_combinations(self):
        a = torch.tensor([0., .2], requires_grad=True)
        p = torch.tensor([[1., 1.], [2., 1.]], requires_grad=True)
        n = torch.tensor([[.1, .3], [.2, .4], [.3, .5]], requires_grad=True)
        g, t, s = paper_losses(a, p, n)
        ap = torch.stack([sum((a[k]-r[k])**2 for k in range(2)) for r in p])
        an = torch.stack([sum((a[k]-r[k])**2 for k in range(2)) for r in n])
        hn = torch.stack([sum((p[1,k]-r[k])**2 for k in range(2)) for r in n])
        torch.testing.assert_close(g, torch.log1p(torch.exp(ap[1]-hn+.2)).mean())
        torch.testing.assert_close(t, torch.stack([torch.clamp(ap[i]-an[j]+.3, min=0)
            for i in range(2) for j in range(3)]).mean())
        self.assertEqual(int(s['hardest']), 1)
        self.assertEqual(int(s['triplet_violations']), 6)
        (g+t).backward()
        for value in (a.grad, p.grad, n.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.norm().item(), 0.)
        self.assertTrue((p.grad.norm(dim=1)>0).all())
        self.assertTrue((n.grad.norm(dim=1)>0).all())

    def test_softplus_still_positive_when_margin_satisfied_and_hinge_zero(self):
        g, t, s = paper_losses(torch.zeros(2), torch.tensor([[.1, 0], [.2, 0]]),
                              torch.tensor([[2., 0], [3., 0], [4., 0]]))
        self.assertGreater(g.item(), 0.)
        self.assertEqual(t.item(), 0.)
        self.assertEqual(int(s['guidance_violations']), 0)
        with self.assertRaises(ValueError):
            paper_losses(torch.zeros(2), torch.zeros(1, 2), torch.zeros(3, 2))

    def test_readout_is_pre_dropout_and_does_not_change_parent_model(self):
        cfg, _, model = fixture(marked=False)
        model.eval()
        # A long utterance traverses multiple AASIST windows, yet must still
        # yield exactly one utterance readout.
        x = [torch.randn(4000), torch.randn(45000)]
        before = copy.deepcopy(model.state_dict())
        expected = model(x)
        z, e = classify_with_readout(model, model.encode(x))
        torch.testing.assert_close(z, expected, rtol=0, atol=0)
        self.assertEqual(e.shape, (2, 384))
        self.assertEqual(set(before), set(model.state_dict()))
        self.assertFalse(any('conditioner' in k or 'repair' in k for k in before))
        self.assertEqual(len(model.head.aasist.drop._forward_pre_hooks), 0)

    def test_no_hard_groups_exactly_preserves_parent_losses_gradients_and_rng(self):
        cfg, units, model = fixture(marked=False)
        other = copy.deepcopy(model)
        units = [u for u in units if u[0]['meta_side'] == 'A']
        opt, opt2 = optimizer_for(model, cfg), optimizer_for(other, cfg)
        initial = rng_state()
        expected = parent_side(other, units, cfg, scaler_for(opt2,cfg), backward=True)
        expected_rng = rng_state()
        restore_rng(initial)
        actual = side_pass(model, units, cfg, scaler_for(opt,cfg), backward=True)
        self.assertAlmostEqual(actual['loss'], expected['loss'], places=6)
        self.assertEqual(actual['encoded_views'], expected['encoded_views'])
        self.assertTrue(torch.equal(expected_rng['torch'], rng_state()['torch']))
        for (name, p), (_, p2) in zip(model.named_parameters(), other.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, p2.grad, rtol=0, atol=0, msg=name)

    def test_one_encoder_pass_per_view_both_losses_reach_fusion_and_lora(self):
        cfg, units, model = fixture()
        # This fixture must exercise an ACTIVE hinge to verify its backward
        # path. A deliberately large test-only margin creates violations;
        # the separate zero-hinge test covers already-satisfied margins.
        cfg['hp_triplet_margin'] = 10000.
        calls = []
        hook = model.encoder.register_forward_hook(lambda m,a,y: calls.append(y.shape[0]))
        before = copy.deepcopy(model.state_dict())
        result = train_step(model, optimizer_for(model,cfg), units,cfg,audit=True)
        hook.remove()
        self.assertEqual(result['hard_positive']['groups'], 4)
        self.assertEqual(result['hard_positive']['extra_encoded_views'], 4)
        self.assertEqual(result['encoded_views'], 88)  # 72 CE + 12 official PCL + 4 positives.
        self.assertEqual(sum(calls), result['encoded_views'])
        self.assertEqual(result['hard_positive']['extra_scoring_passes'], 0)
        for name in ('noise', 'platform'):
            self.assertGreater(result['pcl_losses'][name]['weighted_loss'], 0.)
        reconstructed = (result['classification_loss'] + sum(v['weighted_loss'] for v in result['pcl_losses'].values())
            + result['pcl_variance']['weighted_loss'] + result['en_real_confidence']['weighted_loss']
            + result['hard_positive']['guidance_weighted'] + result['hard_positive']['triplet_weighted'])
        self.assertAlmostEqual(result['loss'], reconstructed, places=5)
        self.assertGreater(result['hard_positive']['guidance_weighted'], 0.)
        changed = [name for name, value in model.state_dict().items() if not torch.equal(value, before[name])]
        self.assertTrue(any(name.endswith('lora_b') for name in changed))
        self.assertTrue(any(name.startswith('fusion.') for name in changed))
        norms = result['hard_positive_gradient_audit']['A']['norms']
        self.assertGreater(norms['guidance'], 0.)
        self.assertGreater(norms['triplet'], 0.)
        for side in result['hard_positive']['sides'].values():
            self.assertEqual(len(side['records']), 2)
            self.assertTrue(all(len(set(r['negatives'])) == 3 for r in side['records']))
        json.dumps(result, allow_nan=False)

    def test_control_has_same_views_and_zero_metric_gradient(self):
        cfg, units, model = fixture()
        result = train_step(model, optimizer_for(model,cfg), units,dict(cfg,hp_mode='control'),audit=True)
        self.assertEqual(result['hard_positive']['groups'], 4)
        self.assertEqual(result['hard_positive']['extra_encoded_views'], 4)
        self.assertEqual(result['hard_positive']['guidance_weighted'], 0.)
        self.assertEqual(result['hard_positive']['triplet_weighted'], 0.)
        self.assertEqual(result['hard_positive_gradient_audit']['A']['norms']['guidance'], 0.)

    def test_extra_positive_preserves_ce_source_mass_25_25_50(self):
        from unittest.mock import patch
        import w2v_v55.side as side_module
        cfg, units, model = fixture(pcl=False)
        units = [u for u in units if u[0]['meta_side'] == 'A']
        cfg['classification_ce_weights'] = dict(online=.25, clean_rtc=.25, noisy=.5)
        # Known, different logits for all views; verify source/group averaging
        # independently of micro_losses and the selected physical packing.
        observed = {}
        def known_logits(model, sequences):
            values = torch.arange(len(sequences), dtype=torch.float32) / 7
            logits = torch.stack((values, -values), dim=1)
            observed['logits'] = logits
            return logits, torch.zeros(len(sequences),384)
        original = side_module.micro_objective
        expected = []
        def checking_readouts(readouts, rows, *args, **kwargs):
            logits = observed['logits']
            raw = F.cross_entropy(logits,torch.tensor([r['label'] for r in rows]),
                                  label_smoothing=cfg['label_smoothing'],reduction='none')
            for row,value in zip(rows,raw):
                kind = 'clean_rtc' if row['role'] == 'clean_ref' else row['role']
                sources = sum(u[0]['language']==row['language'] and u[0]['label']==row['label']
                              and u[0]['role']==('noisy' if kind=='clean_rtc' else kind) for u in units)
                weight = cfg['classification_ce_weights'][kind] / (4*sources)
                if kind=='noisy' and row.get('hp_member_role')=='anchor':
                    weight *= .5
                expected.append(float(value)*weight)
            return original(readouts,rows,*args,**kwargs)
        opt = optimizer_for(model,cfg)
        with patch.object(side_module,'classify_with_readout',side_effect=known_logits), \
             patch.object(side_module,'micro_objective',side_effect=checking_readouts):
            actual = side_pass(model,units,dict(cfg,hp_mode='control'),scaler_for(opt,cfg),backward=False)
        self.assertAlmostEqual(actual['classification_loss'],sum(expected),places=6)

    def test_batch_keeps_every_source_and_never_splits_metric_group(self):
        cfg, units, _ = fixture()
        for side in ('A', 'B'):
            source = [u for u in units if u[0]['meta_side'] == side]
            weights = dict(noise=.15, platform=.045) if side == 'A' else dict(noise=0.,platform=0.)
            batches = list(microbatches(source,16,9000,weights=weights))
            self.assertEqual(sorted(id(u) for b in batches for u in b), sorted(map(id,source)))
            seen = set()
            for batch in batches:
                groups = {u[0]['hp_group'] for u in batch if 'hp_group' in u[0]}
                self.assertFalse(seen & groups)
                seen |= groups
                rows = [r for u in batch for r in layout(u,weights)[0]]
                self.assertLessEqual(len(rows),16)
            with self.assertRaisesRegex(ValueError,'complete hard-positive'):
                list(microbatches(source,4,9000,weights=weights))

    def test_nonfinite_gradients_rollback_model_optimizer_and_rng(self):
        cfg, units, model = fixture()
        opt = optimizer_for(model,cfg)
        train_step(model,opt,units,cfg)
        before = copy.deepcopy(model.state_dict())
        adam = copy.deepcopy(opt.state_dict())
        rng = rng_state()
        hook = model.fusion.projection.weight.register_hook(lambda g: g*float('nan'))
        with self.assertRaises(FloatingPointError):
            train_step(model,opt,units,cfg)
        hook.remove()
        for name,value in before.items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        for key,state in adam['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(opt.state_dict()['state'][key][name],value,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng['torch'],rng_state()['torch']))


if __name__ == '__main__':
    unittest.main()
