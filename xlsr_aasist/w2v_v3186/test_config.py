"""Production defaults and resume must not silently change the experiment."""
import unittest
from .config import parser,validate,validate_locked_cli,training_configuration


class ConfigTests(unittest.TestCase):
    def test_joint_defaults_and_targeted_expansion(self):
        a=parser().parse_args([]);validate(a)
        self.assertEqual((a.epochs,a.warm_epochs,a.consistency),(6,0,'dual'))
        self.assertEqual(a.view_factors,[5,5,1,3]);self.assertEqual(a.group_quotas,[4,3,5,4])
        self.assertEqual((a.pcl_platform_weight,a.pcl_noise_weight),(.15,.15))

    def test_resume_rejects_training_overrides_and_accepts_delivery(self):
        for flag in ('--epochs=8','--consistency=none','--pcl-noise-weight=.2'):
            argv=['--resume','fixture',flag];a=parser().parse_args(argv)
            with self.assertRaises(ValueError):validate_locked_cli(a,argv)
        argv=['--resume','fixture','--submit-best','--upload-submission']
        a=parser().parse_args(argv);validate(a);validate_locked_cli(a,argv)

    def test_invalid_expansion_and_nonfinite_weights_rejected(self):
        for argv in (['--view-factors','5','5','5','5'],['--pcl-noise-weight','nan'],
                     ['--upload-submission'],['--platform-sources-per-group','5']):
            with self.subTest(argv=argv),self.assertRaises(ValueError):validate(parser().parse_args(argv))

    def test_aug5_restores_complete_pooling_but_uses_targeted_expansion(self):
        from w2v_v3184_aug5.test_support import tiny_cfg
        from .parent_discovery import validate_parent_config
        old=tiny_cfg();old['encoder_layers']=60
        args=parser().parse_args([])
        cfg=training_configuration(old,args,validate_parent_config(old))
        self.assertEqual(old['view_factor'],5)
        self.assertNotIn('view_factor',cfg)
        self.assertEqual(cfg['view_factors'],[5,5,1,3])
        self.assertEqual(cfg['group_quotas'],[4,3,5,4])
        self.assertEqual((cfg['version'],cfg['epochs'],cfg['warm_epochs']),('3.18.6',6,0))
        self.assertEqual(cfg['pooling'],'asp');self.assertTrue(cfg['local_evidence'])
        self.assertTrue(cfg['initialization']['initial_function_preserved'])
        for name in ('window','hop','window_batch'):self.assertEqual(cfg[name],old[name])
        self.assertEqual((cfg['evidence_lr'],cfg['pool_lr']),(3e-5,3e-5))
        self.assertEqual(cfg['initialization']['parent_version'],'3.18.4')
        self.assertEqual(cfg['initialization']['parent_view_factors'],[5,5,5,5])
        self.assertEqual(cfg['fusion_layers'],list(range(1,61)))
        self.assertEqual(old['pooling'],'asp')

    def test_old_whole_recipe_cannot_resume_as_windows(self):
        from .common import schema_for
        from .test_support import tiny_cfg
        cfg=tiny_cfg();schema_for(cfg)
        cfg.update(recipe='v3186_whole_dual_pcl_v1',pooling='whole',local_evidence=False)
        with self.assertRaisesRegex(ValueError,'start a new run'):schema_for(cfg)


if __name__=='__main__':unittest.main()
