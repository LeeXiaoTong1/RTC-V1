"""Production defaults and resume must not silently change the experiment."""
import unittest
from .config import parser,validate,validate_locked_cli


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


if __name__=='__main__':unittest.main()
