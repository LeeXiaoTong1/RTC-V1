"""Real WAV/codec checks for full caches, retirement and non-TTY progress."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from rtc_noisy.simulator import LocalRTC
from rtc_noisy_v2.plan import BANDS, PLAN_ID, SCHEMA, plan_definition, settings_for
from w2v_aasist.full_cache import prepare, read_index, assignments
from w2v_aasist.data import read_protocol, build_data, loader, AudioDataset
from w2v_aasist.runtime import atomic_json, sha256, supervised_step
from w2v_aasist.cache_retirement import retire
from w2v_aasist.progress import progress, training_bar


FFMPEG = os.environ.get('W2V_TEST_FFMPEG') or shutil.which('ffmpeg')


class FullNoisyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FFMPEG: raise unittest.SkipTest('Set W2V_TEST_FFMPEG for real codec validation')
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.project = cls.root/'project'; cls.project.mkdir()
        cls.dataset = cls.root/'official'
        cls.version = LocalRTC(FFMPEG).version
        cls.source = {}
        for split in ('train','dev'):
            names = []
            kinds = [(lang,label) for lang in ('en','zh') for label in (0,1)]
            for i,(lang,label) in enumerate(kinds):
                for domain in ('offline','online'):
                    name = f'{domain}/{lang}/{split}_{i}.wav'
                    p = cls.dataset/'wav'/split/name;p.parent.mkdir(parents=True,exist_ok=True)
                    frames = 12000 + i*26000
                    wave = np.random.default_rng(i+4).normal(0,.06,frames).astype(np.float32)
                    sf.write(p,wave,16000,subtype='PCM_16')
                    names.append(f'{name} {"real" if label else "fake"}')
            protocol = cls.dataset/f'{split}_label.txt'
            protocol.write_text('\n'.join(names)+'\n',encoding='utf-8')
            cls.source[split+'_protocol']=str(protocol)
            cls.source[split+'_data_path']=str(cls.dataset/'wav'/split)
        noise = cls.root/'background.wav'
        sf.write(noise,np.random.default_rng(42).normal(0,.03,16000*8),16000,subtype='PCM_16')
        short_noise=cls.root/'short_background.wav'
        sf.write(short_noise,np.random.default_rng(43).normal(0,.03,16000*2),16000,subtype='PCM_16')
        cls.manifest=cls.root/'noise.jsonl'
        cls.manifest.write_text(json.dumps(dict(path=str(noise),split='train',original_recording='train-noise'))+'\n'+
                                json.dumps(dict(path=str(short_noise),split='train',original_recording='short-noise'))+'\n')
        from rtc_noisy.common import noise_catalog
        cls.catalog=noise_catalog(cls.manifest,'train')[1]
        cls.old=cls.dataset/'rtc_noisy_cache_v2'/'train_g0'
        cls.seen=cls.root/'dev_seen';cls.held=cls.root/'dev_heldout'
        for role,path,key in [('train',cls.old,'train_noisy_cache'),('dev_seen',cls.seen,'dev_noisy_cache'),
                              ('dev_heldout',cls.held,'dev_heldout_cache')]:
            cls.make_legacy(role,path)
            cls.source[key]=str(path)
        from transformers import SeamlessM4TFeatureExtractor
        cls.ssl=cls.root/'ssl';SeamlessM4TFeatureExtractor().save_pretrained(cls.ssl)
        cls.source['ssl_path']=str(cls.ssl)
        cls.full=cls.project/'data'/'rtc_noisy_full2_v1'/'train'
        prepare(cls.source,cls.full,cls.manifest,FFMPEG,workers=2)
        cls.ordinary=read_protocol(cls.source['train_protocol'],cls.source['train_data_path'])

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    @classmethod
    def make_legacy(cls,role,path):
        split='train' if role=='train' else 'dev'
        records=read_protocol(cls.source[split+'_protocol'],cls.source[split+'_data_path'])
        records=[r for r in records if r['domain']=='offline']
        path.mkdir(parents=True,exist_ok=True)
        config=dict(format='rtc_noisy_pair_cache_v1',optimization_schema=SCHEMA,role=role,split=split,
                    seed=1234,generation=0,plan_id=PLAN_ID,plan=plan_definition(),
                    allowed_settings=[list(x) for x in settings_for(role)],
                    protocol_sha256=sha256(cls.source[split+'_protocol']),cut=64600,sr=16000,
                    snr_bands=[list(x) for x in BANDS],limit=0,offline_count=len(records),ffmpeg_version=cls.version,
                    noise=cls.catalog if split=='train' else {'recording_ids':['dev-noise'],'file_sha256':['d'*64]})
        atomic_json(path/'config.json',config)
        rows=[]
        for record in records:
            key=hashlib.sha256(record['id'].encode()).hexdigest()[:24]
            for band in range(4):
                audio=path/'audio'/key[:2]/f'{key}_b{band}.wav';audio.parent.mkdir(parents=True,exist_ok=True)
                sf.write(audio,np.random.default_rng(band).normal(0,.04,64600),16000,subtype='FLOAT')
                setting=dict(zip(('noise_reduction','max_gain','bitrate'),settings_for(role)[0]))
                row=dict(source=record['id'],label=record['label'],band=band,snr_db=sum(BANDS[band])/2,
                         source_sha256=sha256(record['audio']),role=role,generation=0,mix_id='a'*64,
                         audio=audio.relative_to(path).as_posix(),rtc=setting)
                rows.append(row);atomic_json(audio.with_suffix('.json'),row)
        (path/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

    def config(self):
        return dict(self.source,train_caches=[str(self.full)],full_noisy=True,baseline=str(self.root/'original.pt'),
                    seed=1234,ordinary_batch=4,noisy_batch=2,max_seconds=0.,workers=0,rawboost=0,raw_config={},
                    eval_batch=8)

    def test_generated_views_keep_all_samples_and_resume_without_rewriting(self):
        rows,(raw,cfg)=read_index(self.full,'train',self.source['train_protocol'],self.ordinary,verify_audio_hash=True)
        self.assertEqual(cfg['continuous_noise_recordings'],1)
        self.assertTrue(all(Path(r['noise']['noise_path']).name=='background.wav' for r in raw))
        self.assertEqual(len(rows),8)
        self.assertTrue(any(r['output_samples']>64600 for r in rows))
        self.assertTrue(any(r['output_samples']<64600 for r in rows))
        before={r['audio']:(sha256(r['audio']),Path(r['audio']).stat().st_mtime_ns) for r in rows}
        prepare(self.source,self.full,self.manifest,FFMPEG,workers=2)
        self.assertEqual(before,{p:(sha256(p),Path(p).stat().st_mtime_ns) for p in before})
        for r in rows:
            np.testing.assert_equal(len(AudioDataset([r])[0]['wave']),r['output_samples'])

    def test_full_training_reader_single_view_and_actual_gradient_step(self):
        cfg=self.config();plan,_,weights,_,_=build_data(cfg)
        self.assertTrue(plan.full)
        self.assertEqual(sum(isinstance(x,int) for b in plan.batches(1) for x in b),8)
        for batch in plan.batches(1):
            for ticket in (x for x in batch if isinstance(x,tuple)):
                self.assertEqual(ticket[0],ticket[1]);self.assertEqual(ticket[2],'full')
        examples=next(iter(loader(plan.records,cfg,training=True,epoch=1,batches=plan.batches(1))))
        self.assertEqual(sum(e['noisy'] for e in examples),2)
        from w2v_aasist.tests import tiny_detector
        model=tiny_detector().configure_trainable_layers(1).train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        stats,_=supervised_step(model,examples,optimizer,weights,'cpu','none',microbatch=2)
        self.assertTrue(np.isfinite(stats['loss']))

    def test_assignments_stratify_severity_and_exclude_heldout_processing(self):
        rows=[dict(id=f'{lang}/{label}/{i}',domain='offline',language=lang,label=label)
              for lang in ('en','zh') for label in (0,1) for i in range(101)]
        values=assignments(rows,1234)
        self.assertEqual(values,assignments(list(reversed(rows)),1234))
        held=set(settings_for('dev_heldout'))
        for lang in ('en','zh'):
            for label in (0,1):
                bands=[0]*4
                for r in rows:
                    if r['language']!=lang or r['label']!=label: continue
                    pair=values[r['id']]
                    self.assertLess(pair[0]['band'],2);self.assertGreaterEqual(pair[1]['band'],2)
                    self.assertNotEqual(pair[0]['rtc'],pair[1]['rtc'])
                    for view in pair:
                        bands[view['band']]+=1
                        self.assertNotIn(tuple(view['rtc'].values()),held)
                self.assertLessEqual(max(bands)-min(bands),1)

    def test_incomplete_replacement_blocks_cleanup_and_low_space_preserves_old(self):
        cfg=self.config();cfg['train_caches']=[str(self.root/'missing')]
        before=sha256(self.old/'manifest.jsonl')
        with self.assertRaises(FileNotFoundError):retire(self.source,cfg,self.project,lambda:None)
        with patch('w2v_aasist.full_cache.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(OSError):prepare(self.source,self.root/'no-space',self.manifest,FFMPEG,workers=1)
        self.assertEqual(sha256(self.old/'manifest.jsonl'),before)
        self.assertTrue(any(self.old.rglob('*.wav')))

    def test_wrong_role_and_manifest_mutation_rejected(self):
        with self.assertRaises(ValueError):read_index(self.full,'dev_seen',self.source['train_protocol'],self.ordinary)
        manifest=self.full/'manifest.jsonl';original=manifest.read_bytes()
        try:
            manifest.write_bytes(original+b'\n')
            with self.assertRaises(ValueError):read_index(self.full,'train',self.source['train_protocol'],self.ordinary)
        finally:manifest.write_bytes(original)

    def test_z_retirement_preserves_new_dev_sources_and_unknown_files(self):
        cfg=self.config()
        protected=[Path(r['audio']) for r in self.ordinary]+list(self.full.rglob('*.wav'))+list(self.seen.rglob('*.wav'))+list(self.held.rglob('*.wav'))
        before={str(p):sha256(p) for p in protected}
        unknown=self.old/'best_model.pt';unknown.write_bytes(b'keep unknown file')
        feature=self.project/'data'/'w2v_feature_cache'/('a'*64)/'bb'/('b'*64+'.npy')
        feature.parent.mkdir(parents=True);np.save(feature,np.zeros((201,160),np.float32))
        retire(self.source,cfg,self.project,lambda:None)
        self.assertFalse(any(self.old.rglob('*.wav')));self.assertFalse(feature.exists())
        self.assertTrue(unknown.exists());self.assertTrue((self.old/'manifest.jsonl').exists())
        self.assertEqual(before,{str(p):sha256(p) for p in protected})


class ProgressTests(unittest.TestCase):
    def test_background_progress_is_bounded_and_has_no_carriage_returns(self):
        output=io.StringIO()
        with patch('w2v_aasist.progress.sys.stderr.isatty',return_value=False),redirect_stdout(output):
            self.assertEqual(len(list(progress(range(1000),total=1000,label='Test',every=100))),1000)
            bar=training_bar(range(5),5,'Train')
            self.assertTrue(bar.disable)
            for _ in bar:bar.set_postfix(loss=1,refresh=False)
        self.assertNotIn('\r',output.getvalue())
        self.assertLessEqual(len(output.getvalue().splitlines()),12)


if __name__=='__main__':unittest.main()
