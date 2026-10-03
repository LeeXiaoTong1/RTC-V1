"""Deterministic full-wave rotation, role isolation and bounded cache lifetime."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from w2v_aasist.data import read_protocol
from w2v_aasist.runtime import sha256
from . import cache


class FakeRTC:
    version='fixture-ffmpeg'
    webrtc_version='0.1.3'
    calls=[]
    def __init__(self,*args): pass
    def process(self,wave,assignment):
        self.calls.append(dict(assignment))
        return np.asarray(wave*(.8 if assignment['family']=='anlmdn' else .9),np.float32)


def fixture(root, per_group=4):
    root=Path(root)
    cfg=dict(seed=91,dev_seed=871,train_cache_root=str(root/'train_cache'),full_dev_cache_root=str(root/'dev_cache'),
        workers=0,cache_workers=2,source_batch=3,microbatch=2,frame_budget=2400,rawboost=0,raw_config={},
        cache_free_floor_bytes=0,device='cpu',ssl_path=str(root/'ssl'))
    Path(cfg['ssl_path']).mkdir()
    (Path(cfg['ssl_path'])/'preprocessor_config.json').write_text('{}')
    all_records={}
    for split_index,split in enumerate(('train','dev')):
        audio_root=root/(split+'_audio'); audio_root.mkdir()
        protocol=root/(split+'.txt'); lines=[]; pair_rows=[]
        for language_index,language in enumerate(('en','zh')):
            for label in (0,1):
                for i in range(per_group):
                    base=f'{language}/{label}_{i}.wav'
                    for domain_index,domain in enumerate(('offline','online')):
                        name=domain+'/'+base
                        path=audio_root/name; path.parent.mkdir(parents=True,exist_ok=True)
                        seed=10000*split_index+1000*language_index+100*label+10*i+domain_index
                        wave=np.random.default_rng(seed).normal(0,.08,8000+100*i).astype(np.float32)
                        sf.write(path,wave,16000,subtype='FLOAT')
                        lines.append(name+' '+('real' if label else 'fake'))
                    pair_rows.append('offline/'+base+',online/'+base)
        protocol.write_text('\n'.join(lines)+'\n')
        cfg[split+'_protocol']=str(protocol);cfg[split+'_data_path']=str(audio_root)
        all_records[split]=read_protocol(protocol,audio_root)
        if split=='train':
            pairs=root/'train_offline_online_pairs.csv'
            pairs.write_text('offline_id,online_id\n'+'\n'.join(pair_rows)+'\n')
            cfg['train_pair_manifest']=str(pairs)
        noise_path=root/(split+'_noise.wav')
        sf.write(noise_path,np.random.default_rng(551+split_index).normal(0,.1,20000).astype(np.float32),16000,subtype='FLOAT')
        noise_manifest=root/(split+'_noise.jsonl')
        noise_manifest.write_text(json.dumps(dict(path=str(noise_path),split=split,
            original_recording=split+'-noise',sha256=sha256(noise_path)))+'\n')
        cfg[split+'_noise_manifest']=str(noise_manifest)
    return cfg,all_records


class CacheTests(unittest.TestCase):
    def setUp(self): FakeRTC.calls=[]

    def test_assignments_balance_classes_and_rotate_without_heldout_family_leak(self):
        records=[dict(id=f'offline/{language}/{label}_{i}',language=language,label=label)
                 for language in ('en','zh') for label in (0,1) for i in range(96)]
        one=cache.assignments(records,1,1)
        self.assertEqual(one,cache.assignments(list(reversed(records)),1,1))
        self.assertNotEqual(one,cache.assignments(records,1,2))
        groups=defaultdict(Counter)
        for row in records:
            pair=one[row['id']]
            self.assertEqual({a['family'] for a in pair},{'ffmpeg','webrtc'})
            self.assertEqual((pair[1]['band']-pair[0]['band'])%4,2)
            for a in pair: groups[(row['language'],row['label'])][(a['family'],a['mode'],a['band'])]+=1
        self.assertTrue(all(c==next(iter(groups.values())) for c in groups.values()))
        for pair in cache.assignments(records,1,0,'dev').values():
            self.assertEqual(pair[1]['family'],'anlmdn')
            self.assertEqual(pair[0]['band'],pair[1]['band'])
            self.assertNotIn(tuple(pair[1]['rtc'][k] for k in ('noise_reduction','max_gain','bitrate')),cache.settings_for('train'))

    def test_noise_pool_selected_per_source_and_short_noise_crossfades(self):
        class Noise:
            records=[dict(path='short',original_recording='s'),dict(path='long',original_recording='l')]
            def _read_noise(self,path,sr):
                return np.random.default_rng(1 if path=='short' else 2).normal(0,.1,8000 if path=='short' else 16000).astype(np.float32)
        for length,eligible,repeated in ((4000,2,False),(12000,1,False),(40000,2,True)):
            wave=np.random.default_rng(3).normal(0,.1,length).astype(np.float32)
            mixed,info=cache.full_mix(Noise(),[.5,1.],wave,np.random.RandomState(8),10.)
            self.assertEqual(len(mixed),length)
            self.assertEqual(info['eligible_noise_count'],eligible)
            self.assertEqual(info['crossfade_repeated'],repeated)
            self.assertTrue(np.isfinite(mixed).all())
        noise=np.ones(3000,np.float32)
        expanded,info=cache.extend_noise(noise,12000,np.random.RandomState(8))
        np.testing.assert_array_equal(expanded,np.ones(12000,np.float32))

    def test_snr_scaling_uses_common_gain_without_hard_clipping(self):
        class Noise:
            records=[dict(path='constant',original_recording='n')]
            def _read_noise(self,path,sr):return np.ones(20000,np.float32)*.8
        wave=np.ones(16000,np.float32)*.8
        result,info=cache.full_mix(Noise(),[1.25],wave,np.random.RandomState(1),10.)
        self.assertLess(info['gain'],1.)
        self.assertLessEqual(float(np.abs(result).max()),.990001)
        recovered_noise=result.astype(np.float64)/info['gain']-wave
        measured=10*np.log10(np.mean(wave.astype(np.float64)**2)/np.mean(recovered_noise**2))
        self.assertAlmostEqual(float(measured),10.,places=4)

    def test_lazy_generation_resume_and_bounded_retirement(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            cache.prepare_epoch(cfg,1)
            store=cache.EpochCache(cfg,1)
            row=records['train'][0]
            wave=cache.read_wave(row['audio'])
            first=store.get(row,wave)
            self.assertEqual(len(FakeRTC.calls),2)
            second=store.get(row,wave)
            self.assertEqual(len(FakeRTC.calls),2)
            for a,b in zip(first,second): np.testing.assert_array_equal(a[0],b[0])
            # Losing a cache file after restart is deterministic reconstruction,
            # not a different training view or an implicit random draw.
            Path(first[0][2]).unlink()
            again=store.get(row,wave)
            np.testing.assert_array_equal(again[0][0],first[0][0])
            cache.prepare_epoch(cfg,2)
            with self.assertRaisesRegex(RuntimeError,'Two Train'): cache.prepare_epoch(cfg,3)
            store.close()
            removed=cache.retire_generations(cfg,[2])
            self.assertEqual(len(removed),1)
            cache.prepare_epoch(cfg,3)
            self.assertEqual(len(list(Path(cfg['train_cache_root']).glob('epoch_*'))),2)

    def test_full_dev_pair_shares_noise_and_is_fixed(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            dev=[r for r in records['dev'] if r['domain']=='offline']
            first=cache.prepare_dev(cfg,dev)
            count=len(FakeRTC.calls)
            second=cache.prepare_dev(cfg,dev)
            self.assertEqual(first,second)
            self.assertEqual(len(FakeRTC.calls),count)
            for a,b in zip(first['seen'],first['heldout']):
                x=json.loads(Path(a['audio']).with_suffix('.json').read_text())
                y=json.loads(Path(b['audio']).with_suffix('.json').read_text())
                self.assertEqual(x['mixture_sha256'],y['mixture_sha256'])
                self.assertEqual(x['snr_db'],y['snr_db'])
                self.assertEqual(sf.info(a['audio']).frames,sf.info(a['source_audio']).frames)
            self.assertEqual(len(first['seen']),len(dev))

    def test_reject_split_overlap_and_changed_official_audio(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            cache.prepare_epoch(cfg,1)
            store=cache.EpochCache(cfg,1)
            path=Path(records['train'][0]['audio'])
            with path.open('ab') as stream: stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'source changed'): store.get(records['train'][0])
            store.close()
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            entry=json.loads(Path(cfg['train_noise_manifest']).read_text());entry['split']='dev'
            Path(cfg['dev_noise_manifest']).write_text(json.dumps(entry)+'\n')
            with self.assertRaisesRegex(ValueError,'overlap'): cache.prepare_base(cfg,records['train'],records['dev'])

    def test_generation_cannot_adopt_unmarked_folder_or_changed_recipe(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            root=Path(cfg['train_cache_root']);root.mkdir();(root/'precious').write_text('keep')
            with self.assertRaisesRegex(ValueError,'unrecognized'): cache.prepare_base(cfg,records['train'],records['dev'])
            self.assertTrue((root/'precious').exists())
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            folder=Path(cfg['train_cache_root'])/'epoch_001';folder.mkdir()
            (folder/'keep').write_text('keep')
            with self.assertRaisesRegex(ValueError,'unmarked'):cache.prepare_epoch(cfg,1)
            self.assertTrue((folder/'keep').is_file())

    @unittest.skipUnless(os.name=='posix','Linux shared lifetime locks')
    def test_retirement_refuses_open_worker_generation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            cache.prepare_epoch(cfg,1);cache.prepare_epoch(cfg,2)
            reader=cache.EpochCache(cfg,1)
            with self.assertRaises(BlockingIOError):cache.retire_generations(cfg,[2])
            self.assertTrue((Path(cfg['train_cache_root'])/'epoch_001').exists())
            reader.close()
            self.assertEqual(len(cache.retire_generations(cfg,[2])),1)

    @unittest.skipUnless(shutil.which('ffmpeg'),'Real FFmpeg integration runs on the server')
    def test_real_ffmpeg_modes_preserve_full_length(self):
        from rtc_noisy.simulator import LocalRTC
        engine=object.__new__(cache.DiverseRTC)
        LocalRTC.__init__(engine,'ffmpeg')
        wave=np.random.default_rng(99).normal(0,.1,16123).astype(np.float32)
        for family in ('ffmpeg','anlmdn'):
            for mode in ('identity','codec','denoise','gain','full'):
                assignment=dict(family=family,mode=mode,rtc=dict(noise_reduction=12,max_gain=4,bitrate=24000))
                result=engine.process(wave,assignment)
                self.assertEqual(result.shape,wave.shape)
                self.assertTrue(np.isfinite(result).all())

    def test_actual_dispatch_enables_only_requested_webrtc_modules(self):
        calls=[]
        class APM:
            def __init__(self,**kwargs):calls.append(kwargs)
            def set_stream_format(self,*args):pass
            def set_ns_level(self,*args):pass
            def set_agc_target(self,*args):pass
            def process_stream(self,value):return value
        engine=object.__new__(cache.DiverseRTC);engine.apm_class=APM
        wave=np.ones(8013,np.float32)*.2
        for mode in ('denoise','gain'):
            result=engine.process(wave,dict(family='webrtc',mode=mode,ns_level=2,agc_target_dbfs=12,rtc={}))
            self.assertEqual(result.shape,wave.shape)
        self.assertEqual(calls,[dict(enable_ns=True,agc_type=0,enable_vad=False),
                                dict(enable_ns=False,agc_type=1,enable_vad=False)])


if __name__=='__main__': unittest.main()
