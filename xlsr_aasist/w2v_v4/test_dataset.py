import copy
from pathlib import Path
import tempfile
import unittest
import soundfile as sf
from w2v_v31810.test_validation import validation_fixture, _write
from w2v_v31810.validation import build_protocol, _AudioIdentity
from w2v_v31810.identity import registry_config, verify_registry
from w2v_v3187.phonemes import inventory
from .dataset import official_rows, transfer
from .validation import language_balanced


def prepared_fixture(root,duplicate=False,extra_train=0):
    cfg,train,dev=validation_fixture(root)
    template=copy.deepcopy(train)
    for n in range(extra_train):
        for row in template:
            r=dict(row);r['id']=row['id'].replace('.wav',f'_extra{n}.wav')
            r['source_id']=row['source_id'].replace('.wav',f'_extra{n}.wav')
            r.update(_write(Path(root)/'train'/r['id'],r['id']))
            train.append(r)
    off={r['id']:r for r in train if r['condition']=='offline'}
    for r in train:
        r.update(group_id=off[r['source_id']]['audio_sha256'],output_samples=4800)
    if duplicate:
        path=Path(cfg['official_dev_root'])/'offline/en/dev_en_1_0.wav'
        dup=path.with_name('duplicate.wav');wave,sr=sf.read(path,dtype='float32')
        sf.write(dup,wave,sr,format='WAVEX',subtype='FLOAT')
        with Path(cfg['official_dev_protocol']).open('a') as stream:stream.write('offline/en/duplicate.wav real\n')
        with Path(cfg['dev_pairs']).open('a') as stream:stream.write('dev/offline/en/duplicate.wav,\n')
    protocol,train,registry=build_protocol(cfg,train,dev,root,return_registry=True)
    cfg['source_registry']=registry_config(registry)
    return cfg,train,official_rows(cfg,registry),protocol,registry


class DatasetTests(unittest.TestCase):
    def test_whole_sources_fixed_evaluation_and_original_pcl_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,train,dev,protocol,registry=prepared_fixture(tmp)
            original=copy.deepcopy((train,dev,protocol,registry))
            rows,remaining,new,reg,report=transfer(train,dev,protocol,registry,cfg)
            self.assertEqual((train,dev,protocol,registry),original)
            self.assertTrue(report['moved_sources'])
            moved=set(report['moved_sources']);retained={r['canonical_source'] for r in remaining}
            self.assertFalse(moved&retained)
            for r in dev:self.assertEqual(r['canonical_source'] in moved,any(x['id']==r['id'] for x in rows))
            inventory(rows)  # real Train-only PCL lineage validator, not a stub
            verify_registry(rows,reg,dict(cfg,source_registry=registry_config(reg)),new)
            for split in ('select','audit'):
                old={(r['id'],r['validation_panel']):t for r,t in zip(protocol[split]['rows'],protocol[split]['tickets'])}
                for i,(r,t) in enumerate(zip(new[split]['rows'],new[split]['tickets'])):
                    self.assertNotIn(r['canonical_source'],moved)
                    expected=dict(old[r['id'],r['validation_panel']],index=i)
                    self.assertEqual(t,expected)
                self.assertEqual(new[split]['scene_noise_records'],protocol[split]['scene_noise_records'])
            self.assertEqual(report,transfer(train,dev,protocol,registry,cfg)[-1])

    def test_different_containers_same_pcm_are_never_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,train,dev,protocol,registry=prepared_fixture(tmp,True)
            original=next(r for r in dev if r['id']=='offline/en/dev_en_1_0.wav')
            duplicate=next(r for r in dev if r['id']=='offline/en/duplicate.wav')
            self.assertNotEqual(original['audio_sha256'],duplicate['audio_sha256'])
            self.assertEqual(original['decoded_pcm_sha256'],duplicate['decoded_pcm_sha256'])
            self.assertEqual(original['canonical_source'],duplicate['canonical_source'])
            rows,remaining,_,_,report=transfer(train,dev,protocol,registry,cfg)
            ids={r['id'] for r in rows}
            self.assertEqual(original['id'] in ids,duplicate['id'] in ids)

    def test_holdout_cannot_be_removed_or_identity_silently_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,train,dev,protocol,registry=prepared_fixture(tmp)
            with self.assertRaises(ValueError):transfer(train,dev,protocol,registry,cfg,1.)
            modified=copy.deepcopy(dev);modified[0]['decoded_pcm_sha256']='0'*64
            with self.assertRaises(ValueError):transfer(train,modified,protocol,registry,cfg)
            rows,remaining,_,_,report=transfer(train,dev,protocol,registry,cfg,0.)
            self.assertFalse(report['moved_sources']);self.assertEqual(len(rows),len(train));self.assertEqual(len(remaining),len(dev))

    def test_language_selection_is_not_weighted_by_remaining_counts(self):
        panels=('official_online','regular_noisy','heldout_noisy')
        value=dict(split='select',score=.98,panels={p:dict(macro_f1=.98) for p in panels},
            groups={p+'/'+lang:dict(macro_f1=.7 if lang=='en' else .9) for p in panels for lang in ('en','zh')})
        result=language_balanced(value)
        self.assertAlmostEqual(result['score'],.8);self.assertEqual(result['pooled_legacy_score'],.98)

    def test_real_sampler_consumes_migrated_registry_without_changing_exposures(self):
        from collections import Counter
        from w2v_v31813.pairing import Plan
        with tempfile.TemporaryDirectory() as tmp:
            cfg,train,dev,protocol,registry=prepared_fixture(tmp,extra_train=6)
            rows,_,protocol,registry,report=transfer(train,dev,protocol,registry,cfg)
            cfg.update(source_registry=registry_config(registry),source_draws_per_stream=[32]*4,
                stream_sources=8,group_quotas=[2]*4,view_factors=[1,5,1,5],refresh_base_variants=True,
                source_epoch_offset=0,mode='baseline',experiment='E0',warm_epochs=1,consistency='none')
            plan=Plan(rows,cfg,{},{});counts=Counter();seen=set()
            for batch in plan.epoch_batches(0):
                self.assertEqual(len(batch),16);self.assertEqual(len({t['canonical_source'] for t in batch}),16)
                for ticket in batch:
                    r=rows[ticket['index']];counts[ticket['kind'],r['language'],r['label']]+=1;seen.add(r['canonical_source'])
            self.assertEqual(set(counts.values()),{32})
            self.assertTrue(set(report['moved_sources'])<=seen)


if __name__=='__main__':unittest.main()
