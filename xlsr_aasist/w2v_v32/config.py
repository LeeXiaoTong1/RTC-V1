"""Communication-condition coverage with GPU-first activation storage."""
import math
from pathlib import Path
from w2v_v31.config import configuration as previous_configuration, parser as previous_parser
from w2v_aasist.launch import ROOT, BASELINE_SHA256
from w2v_aasist.runtime import sha256


def parser():
    p = previous_parser()
    p.description = __doc__
    p._option_string_actions['--resume'].help = 'V3.2 run directory; restores the saved recipe exactly'
    p.set_defaults(source_run=None, workers=6)
    p.add_argument('--gpu-activation-gib', type=float, default=18.)
    p.add_argument('--gpu-reserve-gib', type=float, default=8.)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--condition-probability', type=float, default=.5)
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--fusion-chunk-layers', type=int, default=5)
    return p


def default_source():
    pointer = ROOT/'exp'/'.latest_v31_run'
    if pointer.is_file():
        path = Path(pointer.read_text(encoding='utf-8').strip()).expanduser().resolve()
        if (path/'best_model.pt').is_file() and (path/'config.json').is_file():
            return path
    from w2v_v31.config import DEFAULT_SOURCE
    return DEFAULT_SOURCE


def configuration(args):
    if args.source_run is None:
        args.source_run = str(default_source())
    if not all(math.isfinite(v) and v >= 0 for v in (args.gpu_activation_gib, args.gpu_reserve_gib)):
        raise ValueError('GPU activation budget and reserve must be finite and nonnegative')
    if args.gpu_reserve_gib < 2 or args.prefetch_factor < 1 or args.fusion_chunk_layers < 1:
        raise ValueError('Reserve >=2 GiB; positive prefetch and fusion chunk required')
    if not math.isfinite(args.condition_probability) or not 0 <= args.condition_probability <= 1:
        raise ValueError('Condition probability must be in [0,1]')
    cfg = previous_configuration(args)
    cfg.update(version='3.2', gpu_activation_gib=args.gpu_activation_gib,
        gpu_reserve_gib=args.gpu_reserve_gib, prefetch_factor=args.prefetch_factor,
        condition_probability=args.condition_probability,
        processing_enabled=args.condition_probability>0,
        processing_identity_probability=1-args.condition_probability,
        processing_single_probability=.8*args.condition_probability,
        fusion_chunk_layers=args.fusion_chunk_layers,
        checkpointing=not args.no_gradient_checkpointing,
        coverage_policy='Full official Train traversal; two existing noisy versions; label-independent on-read processing; full/short views share source CE budget',
        algorithm='w2v-BERT 2.0 + MultiConv; V3.2 processing-condition coverage; full/short CE and full-view CKA',
        activation_policy='GPU-first bounded saved activations; host spill; original logical batch and objective')
    print('V32_SOURCE_RUN='+cfg['source_run'], flush=True)
    return cfg
