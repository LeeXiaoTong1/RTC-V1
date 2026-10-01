"""Explicit V3.3 recipe; never silently adopt a newer warm checkpoint."""
import json
import math
from pathlib import Path
from w2v_v32.config import parser as previous_parser, configuration as previous_configuration
from w2v_v31.config import DEFAULT_SOURCE
from w2v_aasist.launch import ROOT
from w2v_aasist.runtime import sha256

EXPECTED_TAG = 'epoch_2_step_2417'


def parser():
    p = previous_parser()
    p.description = __doc__
    p.set_defaults(source_run=str(DEFAULT_SOURCE), epochs=1)
    p.add_argument('--arm', choices=('both', 'control', 'candidate'), default='both')
    p.add_argument('--source-batch', type=int, default=16)
    p.add_argument('--train-pair-manifest')
    p.add_argument('--noise-manifest')
    p.add_argument('--ffmpeg')
    p.add_argument('--cache-workers', type=int, default=4)
    p.add_argument('--full-noisy-cache', default=str(ROOT/'data'/'rtc_noisy_v33'/'train'))
    p.add_argument('--pair-weight', type=float, default=.02)
    p.add_argument('--pair-warmup-fraction', type=float, default=.1)
    p.add_argument('--aux-max-tokens', type=int, default=256)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--keep-old-caches', action='store_true', help='Keep replaced full2 Train cache WAVs')
    return p


def configuration(args):
    if args.source_batch < 2 or args.cache_workers < 1 or args.aux_max_tokens < 4:
        raise ValueError('source-batch >=2, cache-workers >=1 and aux-max-tokens >=4 required')
    if not math.isfinite(args.pair_weight) or not 0 <= args.pair_weight <= .1:
        raise ValueError('pair-weight must be in [0, .1]')
    if not math.isfinite(args.pair_warmup_fraction) or not 0 < args.pair_warmup_fraction <= 1:
        raise ValueError('pair-warmup-fraction must be in (0,1]')
    cfg = previous_configuration(args)
    # Verify actual metadata, not directory dates or a report that might be stale.
    from w2v_v3.train import read_state
    state = read_state(cfg['warm_checkpoint'])
    if (state.get('schema') != 'rtc_w2v_multiconv_v3' or state.get('kind') != 'weights'
            or state.get('tag') != EXPECTED_TAG):
        raise ValueError('V3.3 requires V3 weights tagged '+EXPECTED_TAG+'; no latest-run fallback')
    del state
    legacy = list(cfg['train_caches'])
    cache = str(Path(args.full_noisy_cache).expanduser().resolve())
    if cache in [str(Path(p).resolve()) for p in legacy]:
        raise ValueError('V3.3 cache must be separate from the original full2 cache')
    cfg.update(version='3.3', arm=args.arm, expected_warm_tag=EXPECTED_TAG,
        source_batch=args.source_batch, pair_weight=args.pair_weight,
        pair_warmup_fraction=args.pair_warmup_fraction, aux_max_tokens=args.aux_max_tokens,
        pair_temperature=.1, pair_time_prior=.25,
        train_pair_manifest=args.train_pair_manifest,
        train_noise_manifest=args.noise_manifest or '/home/ubuntu/LXT/RTC/external_noise/noise_split/train.jsonl',
        train_noisy_cache_v33=cache, train_caches=[cache],
        legacy_full_cache=legacy[0], legacy_train_caches=legacy,
        ffmpeg=args.ffmpeg, cache_workers=args.cache_workers,
        diagnose_aux_steps=[0, 1, 9, 99],
        success_noisy_gain=.003, success_en_real_gain=.02,
        en_real_tolerance=.005, noisy_fake_tolerance=.003, clean_tolerance=.002,
        selection_min_delta=.0001, pair_full_only=True,
        cka_source_view='offline/full', short_loss_weight=.3,
        processing_enabled=False, condition_probability=0.,
        processing_identity_probability=1., processing_single_probability=0.,
        processing_silence_probability=.05,
        algorithm='w2v-BERT 2.0 + MultiConv; V3.3 source-paired full/short CE and temporal/structural consistency',
        coverage_policy='Each canonical Offline source once; available official Online and two full noisy conditions; per-source CE budget',
        pair_policy='TFCL-inspired parameter-free soft temporal alignment and token relations; not a reproduction of frequency CKA',
        short_rawboost_probability=.25, keep_old_caches=args.keep_old_caches)
    cfg['train_noise_manifest'] = str(Path(cfg['train_noise_manifest']).expanduser().resolve())
    return cfg
