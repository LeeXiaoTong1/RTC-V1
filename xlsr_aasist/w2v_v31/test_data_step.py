"""Waveform crop/CMVN checks and exact full-short source-budget gradients."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from w2v_aasist.data import FeatureCollator, read_wave
from w2v_v3.data import loader as old_loader
from w2v_v3.model import diversity_cka, microbatches
from w2v_v3.test_model_step import ToyDetector, tiny_detector
from .data import AudioDataset, ViewCollator, build_data, crop_spec, loader
from .step import loss_function, supervised_step, view_components

torch.set_num_threads(1)


def expanded_examples(dim=5):
    result = []
    for i, length in enumerate((13, 9, 15, 16)):
        common = dict(id=f'{i}.wav', label=i%2, noisy=i>=2, source_group=str(i))
        result.append({**common, 'view':'full', 'view_weight':1. if i==1 else .7,
                       'features':torch.randn(1,length,dim), 'mask':torch.ones(1,length).long()})
        if i != 1:
            shorter = 7+i%2
            result.append({**common, 'view':'short', 'view_weight':.3,
                           'features':torch.randn(1,shorter,dim), 'mask':torch.ones(1,shorter).long()})
    return result


def write_audio(root, name, samples=160000):
    import soundfile as sf
    wave = np.random.default_rng(91).normal(0,.03,samples).astype(np.float32)
    path = root/name; sf.write(path, wave, 16000, subtype='FLOAT')
    return dict(id=name, audio=str(path), label=1, noisy=False, domain='offline',
                language='en', band=-1), wave


class ViewDataTests(unittest.TestCase):
    def test_one_read_one_augmentation_then_consistent_two_waveforms(self):
        with tempfile.TemporaryDirectory() as td:
            row, wave = write_audio(Path(td), 'long.wav')
            ds = AudioDataset([row], training=True, epoch=1, seed=11, rawboost=5,
                              rawboost_probability=1., short_prefix_probability=1.)
            with patch('w2v_aasist.data.read_wave', wraps=read_wave) as reader, \
                 patch('utils.data_utils.process_rawboost_feature', side_effect=lambda w,*_:w*2) as augment:
                full, short = ds[0]
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(augment.call_count, 1)
            self.assertEqual(full['source_group'], short['source_group'])
            self.assertAlmostEqual(full['view_weight']+short['view_weight'], 1.)
            self.assertEqual(short['crop_start_sample'], 0)
            self.assertEqual(full['augmentation'], short['augmentation'])
            np.testing.assert_array_equal(full['wave'], wave*2)
            np.testing.assert_array_equal(short['wave'], full['wave'][:short['crop_samples']])
            self.assertGreaterEqual(short['crop_samples'], 3*16000)
            self.assertLessEqual(short['crop_samples'], 6*16000)

    def test_crop_is_source_deterministic_and_symmetric_across_metadata_versions(self):
        with tempfile.TemporaryDirectory() as td:
            row, wave = write_audio(Path(td), 'shared.wav')
            rows = [dict(row, label=label, language=language, noisy=noisy, version=version,
                         full_length=noisy, output_samples=len(wave), source_sha256='same')
                    for label,language,noisy,version in [(0,'en',False,0),(1,'zh',False,0),
                                                         (0,'en',True,0),(1,'zh',True,1)]]
            ds = AudioDataset(rows, training=True, epoch=2, seed=17)
            views = [ds[(i,i,'full',i)] if r['noisy'] else ds[i] for i,r in enumerate(rows)]
            choices = {(v[1]['crop_start_sample'],v[1]['crop_samples']) for v in views}
            self.assertEqual(len(choices), 1)
            self.assertEqual(len({v[0]['source_group'] for v in views}), len(rows))
            for a,b in zip(ds[0],ds[0]): np.testing.assert_array_equal(a['wave'],b['wave'])
            next_epoch = AudioDataset(rows, training=True, epoch=3, seed=17)[0][1]
            self.assertNotIn((next_epoch['crop_start_sample'],next_epoch['crop_samples']), choices)

    def test_crop_distribution_bounds_prefix_and_random_offsets(self):
        choices = [crop_spec(str(i),160000,seed=9,epoch=2) for i in range(256)]
        self.assertTrue(all(48000 <= n <= 96000 and 0 <= start <= 160000-n for start,n,_ in choices))
        self.assertTrue(90 < sum(mode=='prefix' for _,_,mode in choices) < 166)
        self.assertAlmostEqual(sum(n for _,n,_ in choices)/256/16000,4.5,delta=.25)
        self.assertEqual(crop_spec('a',160000,seed=1,epoch=1,short_prefix_probability=1.)[0],0)

    def test_short_recording_consolidates_without_four_second_tiling(self):
        with tempfile.TemporaryDirectory() as td:
            row, wave = write_audio(Path(td), 'short.wav', samples=12800)
            views = AudioDataset([row], training=True, epoch=1)[0]
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0]['view_weight'], 1.)
            self.assertEqual(views[0]['audio_seconds'], .8)
            np.testing.assert_array_equal(views[0]['wave'], wave)

    def test_waveform_cropping_precedes_independent_fbank_normalization(self):
        from transformers import SeamlessM4TFeatureExtractor
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ssl=root/'ssl'; SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            row,_ = write_audio(root,'cmvn.wav',samples=128000)
            rng=np.random.default_rng(7)
            wave=np.concatenate([rng.normal(0,.001,64000),rng.normal(0,.4,64000)]).astype(np.float32)
            sf.write(row['audio'],wave,16000,subtype='FLOAT')
            views=AudioDataset([row],training=True,epoch=1,short_min_seconds=3.,
                               short_max_seconds=3.,short_prefix_probability=1.)[0]
            full,short=ViewCollator(ssl)([views])
            expected=FeatureCollator(ssl)([views[1]])[0]
            torch.testing.assert_close(short['features'],expected['features'],rtol=0,atol=0)
            self.assertFalse(torch.allclose(full['features'][:,:short['features'].shape[1]],
                                            short['features'],atol=1e-4,rtol=1e-4))

    def test_validation_is_bitwise_unchanged_and_no_second_view_is_emitted(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ssl=root/'ssl'; SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            row,_=write_audio(root,'eval.wav',samples=12800)
            cfg=dict(ssl_path=str(ssl),seed=2,workers=0,rawboost=0,raw_config={},eval_batch=2)
            new=next(iter(loader([row],cfg,training=False)))
            old=next(iter(old_loader([row],cfg,training=False)))
            self.assertEqual(len(new),1)
            self.assertEqual(set(new[0]),set(old[0]))
            self.assertNotIn('source_group',new[0])
            torch.testing.assert_close(new[0]['features'],old[0]['features'],rtol=0,atol=0)
            torch.testing.assert_close(new[0]['mask'],old[0]['mask'],rtol=0,atol=0)

    def test_cache_validation_and_epoch_plan_are_reused(self):
        sentinel=object(); cfg={'short_min_seconds':3.}
        with patch('w2v_v31.data.full_build_data',return_value=sentinel) as checked:
            self.assertIs(build_data(cfg),sentinel)
            checked.assert_called_once_with(cfg)


class ViewStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)

    def test_source_coefficients_preserve_class_and_component_mass(self):
        ex=expanded_examples()
        ow,nw=torch.tensor([.7,1.4]),torch.tensor([.55,2.2])
        labels,noisy,_,_,coeff,n,s,full,vw=view_components(ex,ow,nw,'cpu',.5)
        self.assertEqual((n,s),(2,2))
        self.assertEqual(int(full.sum()),4)
        self.assertEqual(len(ex),7)
        for group in ('0','1','2','3'):
            indices=[i for i,e in enumerate(ex) if e['source_group']==group]
            e=ex[indices[0]]
            expected=.5*(nw if e['noisy'] else ow)[e['label']]/2
            torch.testing.assert_close(coeff[indices].sum(),expected)

    def test_global_gradient_matches_manual_full_short_objective_and_cka_full_only(self):
        for cka_weight in (0.,.17):
            with self.subTest(cka_weight=cka_weight):
                reference=ToyDetector(); actual=copy.deepcopy(reference); ex=expanded_examples()
                ow,nw=torch.tensor([.7,1.4]),torch.tensor([.55,2.2])
                z,h=zip(*(reference(e['features'],e['mask']) for e in ex))
                z,h=torch.cat(z),torch.cat(h)
                ce=F.cross_entropy(z,torch.tensor([e['label'] for e in ex]),reduction='none')
                manual=z.sum()*0
                for i,e in enumerate(ex):
                    component=(.45*nw[e['label']] if e['noisy'] else .55*ow[e['label']])/2
                    manual=manual+component*ce[i]*e['view_weight']
                only_full=torch.tensor([e['view']=='full' for e in ex])
                reference_cka=diversity_cka(h[only_full])
                manual=manual+cka_weight*reference_cka
                manual.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
                torch.optim.SGD(reference.parameters(),lr=.03).step()
                seen=[];hook=actual.register_forward_pre_hook(lambda _m,args:seen.append(len(args[0])))
                stats,scores=supervised_step(actual,ex,torch.optim.SGD(actual.parameters(),lr=.03),
                    ow,'cpu',amp='none',noisy_weight=.45,cka_weight=cka_weight,noisy_class_weights=nw,
                    microbatch=2,frame_budget=26)
                hook.remove()
                self.assertEqual(sum(seen),len(ex))
                self.assertEqual(actual.calls,len(list(microbatches(ex,2,26))))
                self.assertEqual(stats['full_views'],4);self.assertEqual(stats['short_views'],3)
                self.assertEqual(stats['cka_full_views'],4 if cka_weight else 0)
                self.assertAlmostEqual(stats['loss'],float(manual),places=6)
                if cka_weight:self.assertAlmostEqual(stats['cka'],float(reference_cka),places=6)
                torch.testing.assert_close(scores,z.detach(),atol=2e-6,rtol=2e-5)
                for a,b in zip(reference.parameters(),actual.parameters()):
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_cka_has_no_gradient_through_short_embeddings(self):
        ex=expanded_examples(); z=torch.randn(len(ex),2,requires_grad=True)
        h=torch.randn(len(ex),4,6,requires_grad=True)
        loss,_=loss_function(z,h,ex,torch.ones(2),cka_weight=.2)
        loss.backward()
        full=torch.tensor([e['view']=='full' for e in ex])
        self.assertEqual(float(h.grad[~full].abs().sum()),0.)
        self.assertGreater(float(h.grad[full].abs().sum()),0.)

    def test_invalid_view_groups_rejected_before_optimizer_update(self):
        model=ToyDetector(); before=copy.deepcopy(model.state_dict()); ex=expanded_examples()
        ex[1]['view_weight']=.6
        with self.assertRaisesRegex(ValueError,'sum to one'):
            supervised_step(model,ex,torch.optim.SGD(model.parameters(),lr=.1),torch.ones(2),'cpu')
        self.assertEqual(model.calls,0)
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        ex=expanded_examples();ex[1]['label']=1
        with self.assertRaisesRegex(ValueError,'disagree'):
            view_components(ex,torch.ones(2),None,'cpu',.5)

    def test_real_hf_joint_step_uses_native_short_lengths_and_one_forward_per_view(self):
        model=tiny_detector(checkpointing=True).configure_trainable_layers(1).train()
        ex=expanded_examples(160)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        seen=[];hook=model.backbone.register_forward_pre_hook(lambda _m,args,kwargs:seen.append(len(kwargs['input_features'])),with_kwargs=True)
        stats,_=supervised_step(model,ex,torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001),
                                torch.ones(2),'cpu',amp='none',microbatch=2,frame_budget=26)
        hook.remove()
        self.assertEqual(sum(seen),len(ex))
        self.assertTrue(np.isfinite(stats['loss']))
        for n,p in model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
