"""CPU-only tests: diagnostics archive excludes all audio and model weights."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from .delivery import pack,deliver,sha256_file


class DeliveryTests(unittest.TestCase):
    def make_outputs(self, root):
        out=root/'run123'/'v31811_diagnostics'
        (out/'scores').mkdir(parents=True)
        (out/'features').mkdir()
        (out/'scores'/'score_summary.json').write_text('{"n": 2}\n')
        (out/'features'/'feature_probe_summary.json').write_text('{"auc": 0.7}\n')
        (out/'features'/'train_features.npz').write_bytes(b'fake-feature-test-bytes')
        (out/'features'/'select_features.npz').write_bytes(b'fake-select-test-bytes')
        feature_files=['feature_probe_summary.json','train_features.npz','select_features.npz']
        (out/'features'/'feature_manifest.json').write_text(json.dumps(dict(
            schema='v31811_source_bound_features_v2',
            files={name:sha256_file(out/'features'/name) for name in feature_files})))
        (out/'completed.json').write_text('{"status": "done"}\n')
        (out/'last.pt').write_bytes(b'never include checkpoint')
        (out/'record.wav').write_bytes(b'never include waveform')
        (out/'submission.zip').write_bytes(b'never include Progress')
        return out

    def test_zip_only_contains_allowlisted_diagnostics(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);out=self.make_outputs(base)
            archive,identity,manifest=pack(out,base/'temp')
            self.assertEqual(identity,sha256_file(archive))
            with zipfile.ZipFile(archive) as z:
                files=set(z.namelist())
                self.assertEqual(files,{'manifest.json','completed.json',
                    'scores/score_summary.json','features/feature_probe_summary.json',
                    'features/train_features.npz','features/select_features.npz',
                    'features/feature_manifest.json'})
                info=json.loads(z.read('manifest.json'))
                self.assertEqual(info['version'],'3.18.11')
                self.assertFalse(any(s.endswith(('.wav','.pt')) for s in files))
            self.assertEqual(len(manifest['files']),6)

    def test_upload_url_written_and_retry_avoids_new_network(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);out=self.make_outputs(base);sent=[]
            def uploader(path):
                sent.append(path)
                return 'https://temp.sh/abc123.zip'
            r=deliver(out,base/'temp',uploader=uploader)
            self.assertEqual(r['url'],'https://temp.sh/abc123.zip')
            self.assertEqual(len(sent),1)
            self.assertEqual((out/'temp_download_url.txt').read_text().strip(),r['url'])
            cached=deliver(out,base/'temp',uploader=lambda _: self.fail('Unexpected upload'))
            self.assertTrue(cached['cached'])
            self.assertEqual(len(sent),1)
            (out/'scores'/'score_summary.json').write_text('{"n": 3}\n')
            updated=deliver(out,base/'temp',uploader=uploader)
            self.assertFalse(updated['cached'])
            self.assertEqual(len(sent),2)

    def test_cannot_package_symlink_to_private_wave(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);out=self.make_outputs(base)
            (out/'features'/'audit_features.npz').symlink_to(out/'record.wav')
            manifest_path=out/'features'/'feature_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            manifest['files']['audit_features.npz']='0'*64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'symlink|manifest'):
                pack(out,base/'temp')

    def test_partial_failure_can_be_packaged(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);out=self.make_outputs(base)
            (out/'completed.json').unlink()
            (out/'failure.json').write_text('{"error": "diagnostic failed"}\n')
            archive,_,_=pack(out,base/'temp')
            with zipfile.ZipFile(archive) as z:
                self.assertIn('failure.json',z.namelist())
                self.assertNotIn('completed.json',z.namelist())

    def test_unverified_legacy_features_are_not_shipped(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);out=self.make_outputs(base)
            (out/'features'/'feature_manifest.json').unlink()
            archive,_,_=pack(out,base/'temp')
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn('features/train_features.npz',z.namelist())
                self.assertIn('scores/score_summary.json',z.namelist())


if __name__=='__main__':unittest.main()
