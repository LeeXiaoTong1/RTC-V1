"""Fresh LoRA/forensic-head training; V3.16 is used only as a data manifest."""
import argparse
import math


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run', help='Existing V3.16 or V3.16.1 run: data configuration ONLY')
    p.add_argument('--resume', help='Resume a V3.17 run from its last committed epoch')
    p.add_argument('--ssl-path', help='Original public w2v-BERT directory, never a task checkpoint')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=24)
    p.add_argument('--frame-budget', type=int, default=14400)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-layers', type=int, default=8)
    p.add_argument('--lora-lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--tfcl-lr', type=float, default=1e-4)
    p.add_argument('--time-weight', type=float, default=.15)
    p.add_argument('--structure-weight', type=float, default=.15)
    p.add_argument('--seed', type=int, default=31701)
    p.add_argument('--no-autotune', action='store_true')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def validate(args):
    if not 1 <= args.epochs <= 4 or args.workers < 0 or args.microbatch < 3 or args.frame_budget < 1:
        raise ValueError('epochs=1..4, workers>=0, microbatch>=3 and frame-budget>=1 required')
    if not 1 <= args.lora_layers <= 24 or not 1 <= args.lora_rank <= 1024 or args.seed < 0:
        raise ValueError('Invalid LoRA layers/rank or seed')
    if any(not math.isfinite(v) or v <= 0 for v in (args.lora_lr,args.head_lr,args.tfcl_lr)):
        raise ValueError('Learning rates must be finite and positive')
    if any(not math.isfinite(v) or v < 0 for v in (args.time_weight,args.structure_weight)):
        raise ValueError('Auxiliary weights must be finite and nonnegative')


if __name__ == '__main__': validate(parser().parse_args())
