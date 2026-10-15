import unittest
from types import SimpleNamespace
from unittest.mock import patch
from .resources import audit_training,audit_inference


def row(identity,samples,condition='offline',source='off'):
    return dict(id=identity,output_samples=samples,condition=condition,source_id=source,split='train',audio='unused.wav')


class ResourceTests(unittest.TestCase):
    def test_full_encoder_limit_with_bounded_window_graphs(self):
        n=400+320*(3000-1)
        out=audit_inference([row('limit',n)],{})
        self.assertEqual(out['maxima']['classification_frames'],3000)
        self.assertNotIn('whole_graph_budget',out['minimum_required'])
        self.assertEqual(out['aasist_windows']['max_temporal_graph_cells_per_window'],42**2)
        # The 319 extra samples do not create another encoder frame.
        audit_inference([row('tail',n+319)],{})
        with self.assertRaisesRegex(ValueError,'whole-max-frames 3001'):
            audit_inference([row('too-long',n+320)],{})

    def test_short_recordings_use_ready_padding(self):
        out=audit_inference([row('tiny',10)],{})
        self.assertEqual(out['maxima']['classification_frames'],12)

    def test_all_depth_resource_metadata_is_not_memory_claim(self):
        out=audit_inference([row('short',16000)],{})
        self.assertEqual(out['limits']['microbatch'],16)
        self.assertEqual(out['encoder_adaptation']['lora_layers'],60)
        self.assertEqual(out['encoder_adaptation']['checkpoint_layers'],60)
        self.assertIn('not measured GPU-memory guarantees',out['estimate'])

    def test_header_only_resampling_ceil(self):
        r=row('stereo',1);del r['output_samples']
        with patch('soundfile.info',return_value=SimpleNamespace(frames=48002,samplerate=48000,channels=2)) as info, \
                patch('soundfile.read',side_effect=AssertionError('Do not decode audio')):
            out=audit_inference([r],{})
        info.assert_called_once_with(__import__('pathlib').Path('unused.wav'))
        self.assertEqual(out['maxima']['classification_frames'],49)
        self.assertEqual(out['source_metadata'],{'audio_header':1})

    def test_speed_expansion_checked_before_training(self):
        # 59 seconds fits before augmentation, exceeds 60 seconds after slow.
        train=[row('off',59*16000)]
        with self.assertRaisesRegex(ValueError,'Train.*off.*whole-max-frames'):
            audit_training(train,[],{})

    def test_longer_official_aux_controls_atomic_budget(self):
        train=[row('off',16000),row('on',32000,'online')]
        with self.assertRaisesRegex(ValueError,'frame-budget 396'):
            audit_training(train,[],{'frame_budget':300})
        out=audit_training(train,[],{'frame_budget':500})
        self.assertEqual(out['maxima']['atomic_views'],4)
        self.assertEqual(out['maxima']['atomic_padded_frames'],396)

    def test_ce_only_and_missing_official_have_different_unit_costs(self):
        train=[row('off',16000),row('on',16000,'online')]
        out=audit_training(train,[],{'consistency':'none','microbatch':2})
        self.assertEqual(out['maxima']['atomic_views'],2)
        out=audit_training(train[:1],[],{'microbatch':3})
        self.assertEqual(out['maxima']['atomic_views'],3)
        with self.assertRaisesRegex(ValueError,'microbatch 4'):
            audit_training(train,[],{'microbatch':3})

    def test_fixed_dev_included(self):
        with self.assertRaisesRegex(ValueError,'Inference.*dev-long'):
            audit_training([row('off',16000)],[row('dev-long',80*16000)],{})


if __name__=='__main__':unittest.main()
