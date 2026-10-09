"""Continue a completed V3.16 LAST for an additional, explicit epoch budget."""
import argparse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed w2v_v316_tfcl run; initialization is always its LAST')
    p.add_argument('--resume', help='Resume an interrupted V3.16.1 continuation with its recorded configuration')
    p.add_argument('--epochs', type=int, default=4, help='ADDITIONAL epochs; default 4, no metric early stopping')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=24, help='Physical full views; logical source batch stays 16')
    p.add_argument('--frame-budget', type=int, default=14400)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-autotune', action='store_true')
    p.add_argument('--encoder-lr', type=float, default=5e-6)
    p.add_argument('--head-lr', type=float, default=2.5e-5)
    p.add_argument('--tfcl-lr', type=float, default=1e-4)
    p.add_argument('--time-weight', type=float, default=.15)
    p.add_argument('--structure-weight', type=float, default=.15)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def validate(args):
    import math
    if args.epochs < 1 or args.workers < 0 or args.microbatch < 3 or args.frame_budget < 1:
        raise ValueError('epochs>=1, workers>=0, microbatch>=3 and frame-budget>=1 required')
    if any(not math.isfinite(x) or x <= 0 for x in (args.encoder_lr, args.head_lr, args.tfcl_lr,
                                                  args.time_weight, args.structure_weight)):
        raise ValueError('Learning rates and TFCL weights must be finite and positive')


if __name__ == '__main__':
    validate(parser().parse_args())
