"""V3.7 pins the submitted best and adds Train-only language distillation."""
from pathlib import Path
import os

from w2v_aasist.launch import ROOT
from w2v_aasist.runtime import sha256
from w2v_v36.config import parser as base_parser, configuration as base_configuration, verify_files


def parser():
    p = base_parser()
    p.description = __doc__
    p.add_argument('--feature-run',help='Reuse completed V3.6/V3.7 hidden cache splits read-only')
    p.add_argument('--language-model-dir',default=str(ROOT/'models'/'v37_language_teacher'))
    p.add_argument('--fit-threads',type=int,default=min(8,os.cpu_count() or 1))
    return p


def code_fingerprints():
    paths = []
    for folder in ('w2v_v37','w2v_v36'):
        paths += [p for p in (ROOT/folder).glob('*.py') if not p.name.startswith('test_')]
    for folder,names in (
        ('w2v_v3',('model.py','data.py','step.py')),
        ('w2v_aasist',('model.py','data.py','runtime.py')),
        ('w2v_rebuild',('model.py',))):
        paths += [ROOT/folder/name for name in names]
    return {str(p.resolve()):sha256(p) for p in sorted(set(paths))}


def configuration(args):
    if args.fit_threads < 1:
        raise ValueError('fit-threads must be positive')
    cfg = base_configuration(args)
    cfg.update(version='3.7',seed=3701,
        feature_run=str(Path(args.feature_run).expanduser().resolve()) if args.feature_run else None,
        language_model_dir=str(Path(args.language_model_dir).expanduser().resolve()),
        language_cache_dir=str(Path(args.language_model_dir).expanduser().resolve()),
        fit_threads=args.fit_threads,alpha_grid=[.25,.5,.75],ridge_grid=[.1,1.],
        student_epochs=30,student_hidden=64,student_batch_rows=4096,
        student_learning_rate=.002,student_weight_decay=.0001,
        min_en_real_gain=.005,max_real_recall_drop=.005,
        algorithm='Train-only LID teacher -> nonlinear language branch -> gated centered real-only ridge correction -> anchored binary classifier',
        inference_policy='one frozen w2v-BERT + MultiConv pass; internal language branch; no teacher at inference',
        code_fingerprints=code_fingerprints())
    return cfg
