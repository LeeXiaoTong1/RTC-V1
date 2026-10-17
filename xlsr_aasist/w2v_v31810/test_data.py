import copy
import unittest
from unittest.mock import patch
import numpy as np
from w2v_v3186.test_sampling import rows_for
from .data import Waves, make_recipe, ready, microbatches
from .sampling import Plan


def configuration(**extra):
    return dict(seed=3181001, warm_epochs=1, rtc_policy='three-profile',
        rtc_activity_policy='mixed', scene_probability=1., rtc_stress_probability=.35,
        rtc_stress_ramp_epochs=2., source_draws_per_stream=(113, 89, 89, 89), **extra)


class DataTests(unittest.TestCase):
    def test_forced_domains_survive_three_profile_and_non_domain_controls_match(self):
        rows = rows_for((23, 23, 23, 23)); cfg = configuration()
        for batch in Plan(rows, cfg=cfg).batches(1):
            controls = {}
            for ticket in batch:
                if ticket['kind'] != 'offline': continue
                before = copy.deepcopy(ticket); rec = make_recipe(ticket, rows[ticket['index']], cfg)
                self.assertEqual(rec['family'], ticket['domain_override']['family'])
                self.assertEqual(rec['category'], ticket['domain_override']['scene'])
                keys = ('snr_db', 'input_gain_db', 'codec', 'bitrate', 'cutoff', 'nr', 'gain', 'ns', 'target', 'activity')
                values = {key: rec[key] for key in keys}
                rank = ticket['condition_rank']
                if rank in controls: self.assertEqual(values, controls[rank])
                controls[rank] = values
                self.assertEqual(ticket, before)

    def test_e1_generates_identical_actual_audio_recipes_and_only_changes_sides(self):
        rows = rows_for((23, 23, 23, 23)); base = configuration()
        e1 = Plan(rows, cfg=dict(base, experiment='E1')); e2 = Plan(rows, cfg=dict(base, experiment='E2'))
        dataset = Waves(rows, base); dataset.original = lambda row: np.sin(np.arange(5000, dtype=np.float32) * .013) * .1
        dataset.bank = object(); dataset.engines = object()
        # Deterministic source/noise-dependent result checks the complete data path,
        # including base refresh; native processor is checked separately below.
        def generate(x, r, *_):
            rng = np.random.default_rng(r['seed'])
            return (x + rng.normal(0, .001, len(x))).astype(np.float32), {'recipe': r}
        aa = next(e1.batches(2)); bb = next(e2.batches(2))
        with patch('w2v_v31810.data.generate', side_effect=generate):
            for a, b in zip(aa, bb):
                left = dataset[a][0]; right = dataset[b][0]
                np.testing.assert_array_equal(left['wave'], right['wave'])
                self.assertEqual(left.get('augmentation'), right.get('augmentation'))
                self.assertEqual(left['base_augmentation'], right['base_augmentation'])
                self.assertNotIn('pcl_refs', left); self.assertNotIn('pcl_pairs', left)

    def test_only_one_noise_generation_no_reference_work_and_short_reflection_is_last(self):
        source = np.linspace(-.1, .1, 1000, dtype=np.float32)
        rows = rows_for((23, 23, 23, 23)); cfg = configuration()
        dataset = Waves(rows, cfg); dataset.original = lambda _: source.copy()
        dataset.bank = object(); dataset.engines = object()
        ticket = next(t for t in next(Plan(rows, cfg=cfg).batches(1)) if t['kind'] == 'offline')
        ticket.update(base_variant=0, base_family='original', base_generation=0, base_dynamic=False)
        seen = []
        def generate(x, rec, *_):
            seen.append(len(x)); return x * .5, {'recipe': rec}
        with patch('w2v_v31810.data.generate', side_effect=generate): actual = dataset[ticket]
        self.assertEqual(seen, [1000]); self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['role'], 'noisy'); self.assertEqual(actual[0]['original_samples'], 1000)
        self.assertEqual(len(actual[0]['wave']), 3920)
        np.testing.assert_array_equal(actual[0]['wave'][:1000], source * .5)
        online = dataset[dict(ticket, kind='online')][0]
        np.testing.assert_array_equal(online['wave'][:1000], source)
        self.assertFalse(hasattr(dataset, 'phonemes'))

    def test_fixed_holdout_never_becomes_training_family_or_activity(self):
        row = rows_for((23, 23, 23, 23))[0]; cfg = configuration()
        for family in ('anlmdn', 'g711_mulaw'):
            ticket = dict(index=0, kind='offline', epoch=0, occurrence='fixed', family=family, scene='rain')
            rec = make_recipe(ticket, row, cfg, 'dev')
            self.assertEqual(rec['family'], family); self.assertEqual(rec['category'], 'rain')
            self.assertEqual(rec['codec'], 'none'); self.assertEqual(rec['activity']['profile'], 'off')

    def test_inference_integer_rows_keep_original_order_and_pack_no_references(self):
        rows = rows_for((23, 23, 23, 23))[:6]; dataset = Waves(rows, configuration(), 'dev')
        lengths = {row['id']: 1000 + 3200 * index for index, row in enumerate(rows)}
        dataset.original = lambda row: np.full(lengths[row['id']], .01, np.float32)
        units = [dataset[index] for index in range(len(rows))]
        self.assertEqual([u[0]['row_index'] for u in units], list(range(len(rows))))
        packed = list(microbatches(units, 3, 1000))
        actual = [row for batch in packed for unit in batch for row in unit]
        self.assertEqual(len(actual), len(rows)); self.assertEqual({r['row_index'] for r in actual}, set(range(len(rows))))
        self.assertTrue(all(len(unit) == 1 for batch in packed for unit in batch))
        self.assertTrue(all('pcl_refs' not in row for row in actual))

    def test_native_ffmpeg_and_opus_use_declared_family_once_preserve_entire_waveform(self):
        from w2v_v318.runtime import bundled_ffmpeg
        from w2v_v3189.augment import Engines
        from w2v_v3189.test_activity import speech_fixture
        rows = rows_for((23, 23, 23, 23)); cfg = configuration()
        cfg['ffmpeg'] = bundled_ffmpeg()
        dataset = Waves(rows, cfg); original = speech_fixture()
        dataset.original = lambda _: original.copy(); dataset.bank = object()
        native = Engines(cfg['ffmpeg']); calls = []
        def engine(wave, rec):
            calls.append((rec['family'], rec['codec'])); return native(wave, rec)
        dataset.engines = engine
        ticket = next(t for t in next(Plan(rows, cfg=cfg).batches(1)) if t['kind'] == 'offline')
        ticket.update(domain_override={'scene': 'echo', 'family': 'ffmpeg'},
                      base_variant=0, base_family='original', base_generation=0, base_dynamic=False)
        row = dataset[ticket][0]
        self.assertEqual(len(row['wave']), len(original)); self.assertTrue(np.isfinite(row['wave']).all())
        self.assertEqual(row['augmentation']['recipe']['family'], 'ffmpeg')
        self.assertEqual(sum(f == 'ffmpeg' for f, _ in calls), 1)
        self.assertEqual(sum(codec == 'opus' for _, codec in calls), 1)
        self.assertFalse(np.array_equal(row['wave'], original))


if __name__ == '__main__': unittest.main()
