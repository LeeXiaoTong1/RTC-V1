"""Launch-budget regressions: defaults, explicit overrides and actionable failures."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from .arguments import parser,validate_arguments


class ArgumentTests(unittest.TestCase):
    def test_default_four_and_explicit_eight_without_changing_other_budgets(self):
        defaults=parser().parse_args([])
        self.assertEqual((defaults.epochs,defaults.workers,defaults.microbatch,defaults.frame_budget),(4,4,18,10800))
        args=parser().parse_args(['--epochs','8','--microbatch','40','--workers','0'])
        self.assertEqual((args.epochs,args.microbatch,args.workers,args.frame_budget),(8,40,0,10800))
        self.assertIs(validate_arguments(args),args)

    def test_each_bad_budget_reports_its_name_value_and_bound(self):
        for flag,value,minimum in (('--epochs','0',1),('--workers','-1',0),('--microbatch','2',3),('--frame-budget','0',1)):
            with self.subTest(flag=flag),redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as failure:parser().parse_args([flag,value])
                self.assertEqual(failure.exception.code,2)
                self.assertIn(f'{flag}={value}',output.getvalue())
                self.assertIn(f'integer >= {minimum}',output.getvalue())

    def test_multiple_invalid_budgets_are_reported_together(self):
        args=parser().parse_args([]);args.epochs=-2;args.microbatch=1
        with self.assertRaises(ValueError) as failure:validate_arguments(args)
        self.assertIn('--epochs=-2',str(failure.exception));self.assertIn('--microbatch=1',str(failure.exception))

    def test_configuration_accepts_eight_and_rejects_bad_budget_before_opening_parent(self):
        from . import config
        class ReachedParent(Exception):pass
        with patch.object(config,'source_configuration',side_effect=ReachedParent) as parent:
            args=parser().parse_args(['--parent-run','fixture','--epochs','8'])
            with self.assertRaises(ReachedParent):config.configuration(args)
            parent.assert_called_once_with('fixture','best_guarded')
            parent.reset_mock();args.workers=-2
            with self.assertRaisesRegex(ValueError,'--workers=-2'):config.configuration(args)
            parent.assert_not_called()

    def test_standalone_validation_needs_no_torch_or_dataset(self):
        code=('import runpy,sys; sys.modules["torch"]=None; '
              'sys.argv=["budget_check","--epochs","8","--workers","0"]; '
              'runpy.run_module("w2v_v316_tfcl.arguments",run_name="__main__")')
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('epochs=8 workers=0',result.stdout)


if __name__=='__main__':unittest.main()
