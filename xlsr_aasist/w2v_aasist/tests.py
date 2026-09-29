"""CPU checks for checkpoint parity, one-pass gradients, cache composition and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from torch import nn
from .composition import compose, CUT
from .data import AudioDataset, EpochPlan, FeatureCollator, safe_audio_path
from .model import Detector, microbatches
from .runtime import Metrics, atomic_save, load_checkpoint, predict, seed_all, sha256, supervised_step
from .evaluate import package_scores

torch.set_num_threads(1)


def tiny_detector():
    from transformers import Wav2Vec2BertConfig
    cfg = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, feature_projection_input_dim=160,
                           conv_depthwise_kernel_size=7, hidden_dropout=.1,
                           attention_dropout=.1, activation_dropout=.1,
                           feat_proj_dropout=0., layerdrop=0., apply_spec_augment=False,
                           num_conv_pos_embedding_groups=2)
    return Detector.load(None, config_dict=cfg.to_dict(), checkpointing=False)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_all(51)

    def test_original_checkpoint_and_logits_are_identical(self):
        from w2v_rebuild.model import Detector as OldDetector
        model = tiny_detector().eval()
        old = OldDetector.load(None, config_dict=model.backbone.config.to_dict(), checkpointing=False).eval()
        old.load_state_dict(model.state_dict(), strict=True)
        for frames in (25, 201, 350):
            f, m = torch.randn(1, frames, 160), torch.ones(1, frames, dtype=torch.long)
            with torch.no_grad():
                actual, reference = model(f, m), old(f, m)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_microbatches_never_pad_or_lose_examples(self):
        examples = [dict(features=torch.randn(1, n, 160), mask=torch.ones(1, n).long())
                    for n in (25, 51, 25, 201, 51)]
        batches = list(microbatches(examples, 4, 80))
        self.assertEqual(sorted(i for ids, _, _ in batches for i in ids), list(range(5)))
        for ids, f, m in batches:
            self.assertEqual(len({examples[i]['features'].shape[1] for i in ids}), 1)
            self.assertTrue(bool(m.all()))
        model = tiny_detector().eval()
        a = predict(model, examples, 'cpu', 'none', 1)
        b = predict(model, examples, 'cpu', 'none', 4)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_frontend_freeze_and_padding_rejection(self):
        model = tiny_detector().configure_trainable_layers(1).train()
        for p in model.backbone.encoder.layers[0].parameters():
            self.assertFalse(p.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        self.assertFalse(model.backbone.encoder.layers[0].training)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 25, 160), torch.zeros(1, 25).long())

    def test_one_pass_gradients_match_logical_weighted_ce(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__(); self.linear = nn.Linear(5, 2); self.seen = 0
            def forward(self, x, mask):
                self.seen += len(x)
                z = self.linear(x.mean(1)); return z, z
        a = Toy(); b = copy.deepcopy(a)
        examples = [dict(features=torch.randn(1, 14 + i % 2, 5), mask=torch.ones(1, 14+i%2).long(),
                         label=i % 2, noisy=i >= 4) for i in range(6)]
        weights = torch.tensor([.7, 1.8])
        opt_a, opt_b = torch.optim.SGD(a.parameters(), .02), torch.optim.SGD(b.parameters(), .02)
        labels = torch.tensor([e['label'] for e in examples])
        z = torch.cat([a(e['features'], e['mask'])[0] for e in examples])
        ce = torch.nn.functional.cross_entropy(z, labels, reduction='none')
        loss = .7 * (ce[:4] * weights[labels[:4]]).mean() + .3 * ce[4:].mean()
        loss.backward(); opt_a.step()
        stats, _ = supervised_step(b, examples, opt_b, weights, torch.device('cpu'), 'none', .3, 100., 4)
        self.assertEqual(b.seen, len(examples))
        self.assertAlmostEqual(stats['loss'], loss.item(), places=6)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)


class CompositionTests(unittest.TestCase):
    def test_single_and_switch_preserve_time_and_inputs(self):
        a = np.linspace(-.4, .4, CUT, dtype=np.float32)
        b = a + .2
        x, y = a.copy(), b.copy()
        out, mode = compose(a, None, None, 'single', np.random.default_rng(1))
        np.testing.assert_array_equal(a, out)
        out, mode = compose(a, b, None, 'switch', np.random.default_rng(1))
        self.assertEqual(out.shape, a.shape)
        np.testing.assert_array_equal(out[:CUT//4-320], a[:CUT//4-320])
        np.testing.assert_array_equal(out[3*CUT//4+320:], b[3*CUT//4+320:])
        self.assertTrue(np.all((out >= a - 1e-6) & (out <= b + 1e-6)))
        np.testing.assert_array_equal(a, x); np.testing.assert_array_equal(b, y)

    def test_prefix_tail_preserves_original_content_coordinates(self):
        a = np.full(CUT, .3, dtype=np.float32)
        original = np.linspace(-.1, .1, CUT+19000, dtype=np.float32)
        out, mode = compose(a, None, original, 'prefix_tail', np.random.default_rng(1))
        self.assertEqual(len(out), len(original))
        np.testing.assert_array_equal(out[:CUT-320], a[:CUT-320])
        np.testing.assert_array_equal(out[CUT:], original[CUT:])
        out, mode = compose(a, None, original[:5000], 'prefix_tail', np.random.default_rng(1))
        self.assertEqual(mode, 'single_short_source')
        np.testing.assert_array_equal(out, a)

    def test_dataset_rejects_cross_source_composition_and_dev_tickets(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'cache.wav'; sf.write(p, np.zeros(CUT), 16000)
            rows = [dict(audio=str(p), id=str(i), source_sha256='same', label=1, noisy=True) for i in range(2)]
            with self.assertRaisesRegex(ValueError, 'SAME'):
                AudioDataset(rows, training=True)[(0,1,'switch',0)]
            with self.assertRaisesRegex(ValueError, 'Train-only'):
                AudioDataset(rows)[(0,1,'switch',0)]

    def test_full_audio_and_cache_bytes_are_unchanged(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            source, cached = Path(td)/'source.wav', Path(td)/'cache.wav'
            wave = np.random.default_rng(4).normal(0,.03,CUT+19000).astype(np.float32)
            sf.write(source,wave,16000,subtype='FLOAT'); sf.write(cached,wave[:CUT],16000,subtype='FLOAT')
            digests = [sha256(p) for p in (source,cached)]
            row = dict(audio=str(source),id='offline/en/source.wav',label=1,noisy=False)
            cache = dict(row,audio=str(cached),source_audio=str(source),source_sha256=digests[0],noisy=True)
            actual = AudioDataset([row,cache],training=True)[(1,1,'prefix_tail',0)]
            self.assertEqual(len(actual['wave']),len(wave))
            np.testing.assert_array_equal(actual['wave'][CUT:],wave[CUT:])
            self.assertEqual([sha256(p) for p in (source,cached)],digests)
            self.assertEqual(len(AudioDataset([row])[0]['wave']),len(wave))
            self.assertEqual(len(AudioDataset([row],legacy_prefix=True)[0]['wave']),CUT)

    def test_plan_full_coverage_balanced_labels_and_same_mode_budgets(self):
        ordinary = [dict(id=str(i),label=i%2) for i in range(37)]
        noisy = [dict(id=str(i),label=i%2,band=b) for i in range(12) for b in range(4)]
        plan = EpochPlan(ordinary,[noisy],ordinary_batch=6,noisy_batch=4)
        batches = plan.batches(1)
        self.assertEqual(sorted(i for batch in batches for i in batch if isinstance(i,int)),list(range(37)))
        for batch in batches:
            by_label={0:[],1:[]}
            for ticket in [x for x in batch if isinstance(x,tuple)]:
                i,j,mode,_=ticket
                self.assertEqual(plan.records[i]['id'],plan.records[j]['id'])
                self.assertNotEqual(plan.records[i]['band'],plan.records[j]['band'])
                by_label[plan.records[i]['label']].append(mode)
            self.assertEqual(by_label[0],by_label[1])
        self.assertEqual(batches,plan.batches(1))
        self.assertNotEqual(batches,plan.batches(2))


class StorageTests(unittest.TestCase):
    def test_cleanup_archives_changed_scripts_and_preserves_data(self):
        from .maintenance import clean_code
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); obsolete=root/'old'/'audit.py';obsolete.parent.mkdir()
            obsolete.write_bytes(b'# user changes preserved in the archive\n')
            data=root/'exp'/'best.pt';data.parent.mkdir();data.write_bytes(b'protected')
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'files':[{'path':'old/audit.py'}]}),encoding='utf-8')
            before=obsolete.read_bytes();result=clean_code(root,manifest,True)
            self.assertFalse(obsolete.exists());self.assertEqual(data.read_bytes(),b'protected')
            with zipfile.ZipFile(result['archive']) as z:self.assertEqual(z.read('old/audit.py'),before)
            manifest.write_text(json.dumps({'files':[{'path':'exp/best.pt'}]}),encoding='utf-8')
            with self.assertRaises(ValueError):clean_code(root,manifest,True)
            self.assertTrue(data.exists())

    def test_submission_format_and_score_direction(self):
        with tempfile.TemporaryDirectory() as td:
            archive=package_scores(['en/a.wav','zh/b.wav'],[.75,.125],td)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(),['scores.txt'])
                self.assertEqual(z.read('scores.txt').decode(),'en/a.wav 0.7500000000\nzh/b.wav 0.1250000000\n')
        m=Metrics(); m.update(torch.tensor([[4.,-4.],[-4.,4.],[0.,0.]]),[0,1,1])
        self.assertEqual(m.result()['confusion'],[[1,0],[1,1]])

    def test_failed_atomic_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'best.pt'; atomic_save(path,{'x':torch.ones(3)}); digest=sha256(path)
            def fail(_,temp):
                Path(temp).write_bytes(b'partial'); raise OSError('full disk')
            with patch('w2v_aasist.runtime.torch.save',side_effect=fail):
                with self.assertRaises(OSError): atomic_save(path,{'x':torch.ones(9)})
            self.assertEqual(sha256(path),digest)

    def test_protected_checkpoint_retention_never_deletes_sources(self):
        from .checkpoints import retain
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); paths=[]
            for i in range(5):
                p=root/f'run{i}'/'best_model.pt'; p.parent.mkdir()
                atomic_save(p,{'schema':'rtc_w2v_multiconv_v1' if i==1 else 'rtc_w2v_rebuild_v1',
                               'epoch':1,'dev':{'weighted_f1':.91+i*.01},'model':{'x':torch.tensor(i)}})
                paths.append(p)
            before={str(p):sha256(p) for p in paths}
            selected=retain(paths[0],[root],root/'retained',True,expected_sha=before[str(paths[0])])
            self.assertEqual(len(selected),4)
            self.assertEqual({str(p):sha256(p) for p in paths},before)
            for row in selected:self.assertEqual(sha256(row['retained_copy']),row['sha256'])

    def test_wrong_checkpoint_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'mc.pt'; atomic_save(p,{'schema':'rtc_w2v_multiconv_v1'})
            with self.assertRaises(ValueError):load_checkpoint(p)
            with self.assertRaises(ValueError):safe_audio_path(td,'../../outside.wav')


class IntegrationTests(unittest.TestCase):
    def test_complete_tiny_hf_training_resume_baseline_preservation_and_export(self):
        import soundfile as sf
        from transformers import SeamlessM4TFeatureExtractor
        from . import train as training
        from . import evaluate
        from .launch import RAW
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ssl=root/'ssl'; SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            model=tiny_detector(); baseline=root/'original.pt'
            fingerprint={str(ssl/'preprocessor_config.json'):sha256(ssl/'preprocessor_config.json')}
            atomic_save(baseline,{'schema':'rtc_w2v_rebuild_v1','model_config':model.backbone.config.to_dict(),
                                  'model':model.state_dict(),'data_fingerprints':fingerprint,'dev':{}})
            ordinary,noisy=[],[]
            for i in range(8):
                wave=np.random.default_rng(i).normal(0,.03,13000+i*80).astype(np.float32)
                path=root/f'{i}.wav';sf.write(path,wave,16000,subtype='FLOAT')
                row=dict(id=f'{i}.wav',audio=str(path),label=i%2,language='en',domain='online' if i<4 else 'offline',noisy=False,band=-1)
                ordinary.append(row)
                if i>=4:
                    cache=root/f'cache{i}.wav';sf.write(cache,np.tile(wave,6)[:CUT],16000,subtype='FLOAT')
                    for b in range(4):noisy.append(dict(row,audio=str(cache),noisy=True,band=b,source_audio=str(path),source_sha256=sha256(path)))
            plan=EpochPlan(ordinary,[noisy],ordinary_batch=4,noisy_batch=2,seed=7)
            val={'clean':ordinary,'seen':noisy,'heldout':noisy}
            cfg=dict(device='cpu',amp='none',seed=71,baseline=str(baseline),baseline_sha256=sha256(baseline),
                     ordinary_batch=4,noisy_batch=2,epochs=2,trainable_layers=1,head_lr=.0001,encoder_lr=.00001,
                     weight_decay=.0001,workers=0,ssl_path=str(ssl),max_seconds=0.,input_policy='full utterance',
                     rawboost=0,raw_config=RAW,lr_warmup_steps=1,noisy_weight=.3,grad_clip=1.,patience=3,
                     checkpointing=False,microbatch=2,frame_budget=500,production_layout=False,eval_batch=8)
            legacy = torch.load(baseline, weights_only=True)
            legacy['config'] = {'ssl_path': str(ssl), 'seed': 71}
            atomic_save(baseline, legacy)
            cfg['baseline_sha256'] = sha256(baseline)
            first,resumed=root/'first',root/'resumed'
            discovery=lambda _:(plan,val,torch.ones(2),torch.tensor([4,4]),fingerprint)
            with patch.object(training,'build_data',discovery):
                training.train(cfg,first)
                real_validate=training.validate
                def interrupt(model,val,cfg,device,path):
                    if path.name.startswith('epoch_2'):raise RuntimeError('interrupted epoch 2')
                    return real_validate(model,val,cfg,device,path)
                with patch.object(training,'validate',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):training.train(cfg,resumed)
                training.train(cfg,resumed,resumed/'last.pt')
            a,b=load_checkpoint(first/'last.pt'),load_checkpoint(resumed/'last.pt')
            self.assertEqual(a['epoch'],2)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            for key,state in a['optimizer']['state'].items():
                for k,v in state.items():torch.testing.assert_close(v,b['optimizer']['state'][key][k],rtol=0,atol=0)
            self.assertEqual(sha256(baseline),cfg['baseline_sha256'])
            for k,v in model.state_dict().items():
                if k.startswith('backbone.encoder.layers.0.'):
                    torch.testing.assert_close(a['model'][k],v,rtol=0,atol=0)
            self.assertTrue((first/'epoch_0.json').exists())
            self.assertTrue((first/'best_noisy.pt').exists())
            protocol=root/'eval.txt';protocol.write_text('1.wav\n0.wav\n',encoding='utf-8')
            argv=['evaluate','--checkpoint',str(first/'best_model.pt'),'--protocol',str(protocol),
                  '--audio-root',str(root),'--out',str(root/'submission'),'--device','cpu','--workers','0']
            with patch('sys.argv',argv):evaluate.main()
            with zipfile.ZipFile(root/'submission'/'submission.zip') as z:
                self.assertEqual([x.split()[0] for x in z.read('scores.txt').decode().splitlines()],['1.wav','0.wav'])
            legacy_argv = list(argv)
            legacy_argv[legacy_argv.index('--checkpoint') + 1] = str(baseline)
            legacy_argv[legacy_argv.index('--out') + 1] = str(root/'legacy_submission')
            with patch('sys.argv', legacy_argv): evaluate.main()
            meta = json.loads((root/'legacy_submission'/'submission_meta.json').read_text())
            self.assertEqual(meta['input_policy'], 'legacy 64600-sample prefix/repeat')
            self.assertEqual(meta['checkpoint_sha256'], cfg['baseline_sha256'])


if __name__ == '__main__':
    unittest.main()
