"""Raw-only cleanup: complete audio verification, boundaries and no model deletion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from w2v_v3184.test_support import scene_fixture
from w2v_v3184.noise import sha256,bind_manifest
from w2v_v3184.prepare_noise import atomic_json,preparation_lock
from v3184_storage import noise_plan,apply_plan,checkpoint_inventory,checked_path


def fixture(root):
    bank=root/'bank';scene_fixture(bank);audio=bank/'audio';audio.mkdir()
    manifest=json.loads((bank/'manifest.json').read_text())
    for row in manifest['records']:
        path=bank/row['file'];path.replace(audio/path.name);row['file']='audio/'+path.name
    raw=bank/'raw';selected=raw/'rnnoise_selected';selected.mkdir(parents=True)
    def source(path,data):
        path.write_bytes(data);return dict(bytes=len(data),sha256=sha256(path))
    archive=source(raw/'rnnoise_contributions.tar.gz',b'archive fixture')
    vad=source(raw/'silero_vad.jit',b'vad fixture')
    esc=dict(source(raw/'esc50-test.wav',b'esc fixture'),key='esc50-test')
    sample=source(selected/'1500000000000-office.raw',b'office raw fixture')
    sources=root/'sources.json';atomic_json(sources,dict(archive=archive,vad=vad,esc50=[esc]))
    identity=dict(sources_sha256=sha256(sources))
    manifest.update(variant='v3184-published-noise-sources-v1',preparation_identity=identity)
    atomic_json(bank/'manifest.json',manifest);atomic_json(bank/'preparation_report.json',dict(identity=identity))
    atomic_json(selected/'index.json',dict(identity=dict(archive_sha256=archive['sha256']),
        records=[dict(file='1500000000000-office.raw',bytes=sample['bytes'],source_sha256=sample['sha256'])]))
    (raw/'unrecognized.tmp').write_bytes(b'keep unknown files')
    checkpoint=root/'exp'/'parent';checkpoint.mkdir(parents=True)
    (checkpoint/'last.pt').write_bytes(b'composite checkpoint fixture')
    (checkpoint/'best.pt').write_bytes(b'older standalone best fixture')
    return bank,sources


class StorageTests(unittest.TestCase):
    def test_preview_apply_revalidation_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bank,sources=fixture(root)
            before={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
            plan=noise_plan(bank,sources)
            self.assertEqual(len(plan['candidates']),4)
            self.assertEqual(before,{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})
            self.assertEqual({p['action'] for p in checkpoint_inventory(root)},{'KEEP'})
            with preparation_lock(bank/'.prepare.lock'):result=apply_plan(plan,lambda text:None)
            self.assertEqual(result['removed_files'],4)
            for name,digest in before.items():
                if (root/name).exists():self.assertEqual(sha256(root/name),digest)
            self.assertTrue((bank/'raw/rnnoise_selected/index.json').is_file())
            self.assertTrue((bank/'raw/unrecognized.tmp').is_file())
            self.assertEqual(len(bind_manifest(bank/'manifest.json')['scene_noise_records']['train']),5)
            self.assertEqual(noise_plan(bank,sources)['candidates'],[])
            self.assertEqual(apply_plan(noise_plan(bank,sources))['status'],'already_clean')
            self.assertEqual(len(checkpoint_inventory(root)),2)

    def test_corrupt_or_unfinished_retained_audio_prevents_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            bank,sources=fixture(Path(d));manifest=json.loads((bank/'manifest.json').read_text())
            manifest['status']='running';atomic_json(bank/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'incomplete'):noise_plan(bank,sources)
            manifest['status']='complete';atomic_json(bank/'manifest.json',manifest)
            (bank/manifest['records'][0]['file']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Prepared noise changed'):noise_plan(bank,sources)
            self.assertTrue((bank/'raw/rnnoise_contributions.tar.gz').exists())

    def test_changed_raw_or_completion_marker_removes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            bank,sources=fixture(Path(d));plan=noise_plan(bank,sources)
            raw=bank/'raw/silero_vad.jit';original=raw.read_bytes();raw.write_bytes(b'x'*len(original))
            with self.assertRaisesRegex(ValueError,'no files removed'):apply_plan(plan,lambda text:None)
            self.assertTrue((bank/'raw/rnnoise_contributions.tar.gz').exists())
            raw.write_bytes(original);plan=noise_plan(bank,sources)
            (bank/'preparation_report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'metadata changed'):apply_plan(plan,lambda text:None)
            self.assertTrue(raw.exists())

    def test_escape_unknown_raw_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bank,sources=fixture(Path(d))
            for name in ('../outside','/absolute','raw/../outside','raw\\outside','C:/outside'):
                with self.assertRaises(ValueError):checked_path(bank,name,missing=True)
            index_path=bank/'raw/rnnoise_selected/index.json';index=json.loads(index_path.read_text())
            index['records'][0]['file']='../../../exp/parent/last.pt';atomic_json(index_path,index)
            with self.assertRaisesRegex(ValueError,'Unexpected selected'):noise_plan(bank,sources)
            with patch.object(Path,'is_symlink',return_value=True):
                with self.assertRaisesRegex(ValueError,'symlink'):checked_path(bank,'raw/silero_vad.jit')

    def test_metadata_only_catalog_changes_still_require_pinned_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            bank,sources=fixture(Path(d));catalog=json.loads(sources.read_text())
            catalog['description']='Updated prose, same published recordings';atomic_json(sources,catalog)
            plan=noise_plan(bank,sources)
            self.assertNotEqual(plan['recorded_source_catalog_sha256'],plan['current_source_catalog_sha256'])
            self.assertEqual(len(plan['candidates']),4)
            catalog['archive']['sha256']='0'*64;atomic_json(sources,catalog)
            with self.assertRaisesRegex(ValueError,'different archive'):noise_plan(bank,sources)


if __name__=='__main__':unittest.main()
