"""Order-sensitive paired RTC tests using actual finite waveform fixtures."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf

from .augment import NoiseBank, make_recipe, generate, clean
from .views import FAMILIES, recipe, augment
from w2v_v3184.noise import RECORDED


class Engine:
    def __init__(self):
        self.inputs, self.recipes = [], []
    def __call__(self, wave, r):
        self.inputs.append(wave.copy())
        self.recipes.append(copy.deepcopy(r))
        return np.tanh(wave*1.3).astype(np.float32)


def banks(directory):
    path = Path(directory)/'noise.wav'
    t = np.arange(16000, dtype=np.float32)
    sf.write(path, .07*np.sin(t*.041)+.035*np.sin(t*.139), 16000, subtype='FLOAT')
    stat = path.stat()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    base = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha)
    scenes = [dict(base, category=c, source_group=c, original_sources=[c], license='fixture', split='train') for c in RECORDED]
    return NoiseBank([dict(base, original_recording='fixture', split='train')], 1024**2, scenes)


class AugmentationTests(unittest.TestCase):
    def test_six_views_deterministic_finite_distinct_and_geometry_shared(self):
        x = (.15*np.sin(np.arange(64000)*.06)+.03*np.cos(np.arange(64000)*.121)).astype(np.float32)
        outputs = []
        for variant in range(6):
            r = recipe(19, 'source', 'offline', variant)
            y, meta = augment(x, r)
            again, _ = augment(x, recipe(19, 'source', 'offline', variant))
            np.testing.assert_array_equal(y, again)
            self.assertTrue(np.isfinite(y).all())
            self.assertEqual(meta['independent_new_recording'], False)
            self.assertLessEqual(float(np.abs(y).max()), .98+1e-7)
            if variant:
                self.assertFalse(len(y) == len(x) and np.array_equal(x, y))
            outputs.append((len(y), hashlib.sha256(y.tobytes()).hexdigest()))
        self.assertEqual(len(set(outputs)), 6)
        self.assertEqual(len(augment(x, recipe(19, 'source', 'online', 5))[0]), len(x))

    def test_noise_precedes_same_rtc_and_common_gain_with_six_scenes(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = banks(folder)
            x = (.15*np.sin(np.arange(48000)*.06)).astype(np.float32)
            for category in (*RECORDED, 'echo'):
                r = make_recipe(dict(seed=31901, scene_probability=1.), 'source:0', 's', scene=category)
                engine = Engine()
                ref, noisy, meta = generate(x, r, bank, engine)
                self.assertEqual(engine.recipes[0], engine.recipes[1])
                self.assertEqual(len(ref), len(x))
                self.assertEqual(len(noisy), len(x))
                self.assertTrue(meta['same_settings'] and meta['shared_base'] and meta['fresh_state_per_utterance'])
                np.testing.assert_allclose(engine.inputs[0], x*meta['common_gain'], rtol=1e-6, atol=1e-8)
                self.assertFalse(np.array_equal(engine.inputs[0], engine.inputs[1]))
                # A nonlinear RTC proves we process x+n, rather than adding n
                # to an already processed R(x) output.
                np.testing.assert_array_equal(noisy, np.tanh(engine.inputs[1]*1.3).astype(np.float32))
                if category != 'echo':
                    self.assertAlmostEqual(meta['measured_input_snr_db'], r['snr_db'], places=5)
                    self.assertEqual(meta['noise_original_sources'], [category])
                else:
                    self.assertIsNone(meta['measured_input_snr_db'])
                single = Engine()
                absent, only, same = generate(x, r, bank, single, include_reference=False)
                self.assertIsNone(absent)
                self.assertEqual(len(single.inputs), 1)
                np.testing.assert_array_equal(only, noisy)
                self.assertEqual(same, meta)

    def test_known_loss_mask_and_independent_clean_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            x = (.13*np.sin(np.arange(48000)*.09)).astype(np.float32)
            cfg = dict(seed=31901, scene_probability=0.)
            r = make_recipe(cfg, 'pair', 'source')
            r['dropout'] = dict(position=.3, seconds=.06, gain=0.)
            engine = Engine()
            ref, noisy, meta = generate(x, r, banks(folder), engine)
            start, stop = meta['erased_spans'][0]
            self.assertTrue(np.all(noisy[start:stop] == 0))
            self.assertTrue(np.any(ref[start:stop] != 0))
            fallback = make_recipe(cfg, 'pair:missing-online', 'source')
            self.assertNotEqual(r['seed'], fallback['seed'])
            other = clean(x, fallback, engine)
            self.assertEqual(len(other), len(x))
            self.assertFalse(np.array_equal(other, x))

    def test_recipe_coverage_and_no_warmup(self):
        cfg = dict(seed=31901, scene_probability=.8, rtc_policy='three-profile')
        counts, families = {}, {}
        for i in range(1000):
            r = make_recipe(cfg, str(i), 'source')
            counts[r.get('category', 'legacy')] = counts.get(r.get('category', 'legacy'), 0)+1
            families[r['family']] = families.get(r['family'], 0)+1
            self.assertTrue(5 <= r['snr_db'] <= 25)
            self.assertEqual(r['codec'], 'opus')
            self.assertEqual(r['warm'], 1.)
        self.assertEqual(set(counts), set(RECORDED)|{'echo', 'legacy'})
        self.assertEqual(set(families), {'webrtc', 'ffmpeg', 'light'})
        self.assertTrue(150 < counts['legacy'] < 250)
        explicit = make_recipe(cfg, 'dev', 'source', split='dev', scene='keyboard')
        self.assertEqual(explicit['category'], 'keyboard')
        self.assertEqual(explicit['split'], 'dev')


if __name__ == '__main__':
    unittest.main()
