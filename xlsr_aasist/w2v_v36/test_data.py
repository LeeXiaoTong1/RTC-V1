"""Old full Train cache plus fixed Dev; no generation on the production path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from w2v_aasist.runtime import atomic_json,sha256
from w2v_v33 import cache as old_cache
from w2v_v35 import cache as dev_cache
from w2v_v35.test_cache import fixture as base_fixture,FakeRTC
from .data import build_records


def fixture(root,missing_online=False,duplicate_source=False):
    cfg,records=base_fixture(root,per_group=2 if duplicate_source else 1)
    if duplicate_source:
        rows=[r for r in records['train'] if r['domain']=='offline' and r['language']=='en' and r['label']==0]
        Path(rows[1]['audio']).write_bytes(Path(rows[0]['audio']).read_bytes())
    if missing_online:
        removed=next(r['id'] for r in records['train'] if r['domain']=='online')
        protocol=Path(cfg['train_protocol'])
        protocol.write_text('\n'.join(line for line in protocol.read_text().splitlines() if not line.startswith(removed+' '))+'\n')
        pair=Path(cfg['train_pair_manifest']);pair.write_text(pair.read_text().replace(','+removed,','))
        records['train']=[r for r in records['train'] if r['id']!=removed]
    for key in ('dev_noisy_cache','dev_heldout_cache'):
        folder=Path(root)/key;folder.mkdir()
        atomic_json(folder/'config.json',dict(ffmpeg_version=FakeRTC.version,
            noise=dict(recording_ids=['dev-noise'],file_sha256=['dev-sha'],manifest_sha256='dev-manifest')))
        cfg[key]=str(folder)
    cfg['paired_cache_v33']=str(Path(root)/'paired_train')
    with patch.object(old_cache,'PairedRTC',FakeRTC),patch.object(dev_cache,'DiverseRTC',FakeRTC):
        old_cache.prepare_cache(cfg,cfg['paired_cache_v33'],workers=1)
        dev_cache.prepare_base(cfg,records['train'],records['dev'])
        dev_cache.prepare_dev(cfg,[r for r in records['dev'] if r['domain']=='offline'])
    return cfg


class DataTests(unittest.TestCase):
    def test_complete_reuse_missing_online_and_source_grouping_without_generation(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d,missing_online=True)
            before={str(p):sha256(p) for p in Path(d).rglob('*') if p.is_file()}
            with patch.object(old_cache,'prepare_cache',side_effect=AssertionError('no new Train audio')), \
                 patch.object(dev_cache,'prepare_dev',side_effect=AssertionError('no new Dev audio')):
                result=build_records(cfg,cfg)
            self.assertEqual(len(result['train']),15);self.assertEqual(len(result['dev']),12)
            self.assertEqual(result['coverage']['missing_online'],1)
            self.assertEqual(result['coverage']['train_conditions'],dict(offline=4,online=3,noisy_a=4,noisy_b=4))
            for source in {r['source_id'] for r in result['train']}:
                views=[r for r in result['train'] if r['source_id']==source]
                self.assertEqual(len({r['group_id'] for r in views}),1)
                self.assertTrue(all(r['view']=='full' and r['full_length'] for r in views))
            after={str(p):sha256(p) for p in Path(d).rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_missing_dev_audio_and_incomplete_manifest_fail_without_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d);root=Path(cfg['full_dev_cache_root'])/'epoch_000'
            manifest=json.loads((root/'manifest.json').read_text())
            audio=Path(manifest['seen'][0]['audio']);audio.unlink()
            with patch.object(dev_cache,'prepare_dev',side_effect=AssertionError('no regeneration')):
                with self.assertRaises((FileNotFoundError,RuntimeError)):build_records(cfg,cfg)
            self.assertFalse(audio.exists())
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d);complete=Path(cfg['full_dev_cache_root'])/'epoch_000'/'complete.json'
            complete.unlink()
            with self.assertRaises(FileNotFoundError):build_records(cfg,cfg)

    def test_wrong_dev_owner_and_changed_noisy_audio_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d);owner=Path(cfg['full_dev_cache_root'])/'owner.json'
            value=json.loads(owner.read_text());value['role']='train';atomic_json(owner,value)
            with self.assertRaisesRegex(ValueError,'owner'):build_records(cfg,cfg)
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d);folder=Path(cfg['paired_cache_v33'])
            raw=json.loads((folder/'manifest.jsonl').read_text().splitlines()[0]);audio=folder/raw['audio']
            payload=bytearray(audio.read_bytes());payload[-1]^=1;audio.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,'audio hash'):build_records(cfg,cfg)

    def test_duplicate_recording_ids_share_one_holdout_group(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d,duplicate_source=True);result=build_records(cfg,cfg)
            rows=[r for r in result['train'] if r['condition']=='offline' and r['language']=='en' and r['label']==0]
            self.assertNotEqual(rows[0]['source_id'],rows[1]['source_id'])
            self.assertEqual(rows[0]['group_id'],rows[1]['group_id'])
            self.assertEqual(result['coverage']['train_unique_source_hashes'],7)

    def test_actual_train_dev_waveform_overlap_rejected(self):
        from . import data
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(d);noisy,metadata,files=data._fixed_dev(cfg,
                data.read_protocol(cfg['dev_protocol'],cfg['dev_data_path']))
            train=data.read_protocol(cfg['train_protocol'],cfg['train_data_path'])
            metadata[next(iter(metadata))]['audio_sha256']=sha256(train[0]['audio'])
            with patch.object(data,'_fixed_dev',return_value=(noisy,metadata,files)):
                with self.assertRaisesRegex(ValueError,'Train and Dev'):build_records(cfg,cfg)


if __name__=='__main__':unittest.main()
