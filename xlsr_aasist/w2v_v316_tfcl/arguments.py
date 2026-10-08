"""Validate launch budgets without importing torch, opening assets or spawning a job."""
import argparse


def validate_arguments(args):
    errors=[]
    for field,minimum,hint in (
        ('epochs',1,'default is 4; an explicit larger epoch budget is allowed'),
        ('workers',0,'0 runs the data loader in the main process'),
        ('microbatch',3,'one complete source can contain Offline, Online and Noisy'),
        ('frame_budget',1,'padded-frame budget must be positive')):
        value=getattr(args,field)
        if type(value) is not int or value<minimum:
            flag='--'+field.replace('_','-')
            errors.append(f'{flag}={value!r}: must be an integer >= {minimum} ({hint})')
    if errors:raise ValueError('Invalid training arguments: '+'; '.join(errors))
    return args


class TrainingParser(argparse.ArgumentParser):
    def parse_args(self,args=None,namespace=None):
        result=super().parse_args(args,namespace)
        try:return validate_arguments(result)
        except ValueError as exc:self.error(str(exc))


def parser():
    p=TrainingParser(description='V3.16 TFCL: reliable Offline references and explicit initialization.')
    p.add_argument('--parent-run','--source-run',dest='parent_run')
    p.add_argument('--parent-checkpoint',choices=('best_guarded','best_weighted','last'))
    p.add_argument('--init',dest='init_mode',choices=('pretrained','parent'))
    p.add_argument('--tfcl',choices=('weighted','uniform','original','ce'))
    p.add_argument('--ssl-path',help='Local original official w2v-BERT weights for pretrained initialization')
    p.add_argument('--resume');p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=4,help='Positive maximum epoch budget (default: 4); explicit larger values are allowed')
    p.add_argument('--workers',type=int,default=4,help='Data workers >= 0; 0 is supported')
    p.add_argument('--microbatch',type=int,default=18,help='Physical view limit >= 3; keep one complete source together')
    p.add_argument('--frame-budget',type=int,default=10800,help='Positive padded-frame budget')
    p.add_argument('--no-autotune',action='store_true');p.add_argument('--seed',type=int,default=31601)
    p.add_argument('--fixed-budget',action='store_true',help='Run all requested epochs for equal-budget controls; nonfinite safety still applies')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp');p.add_argument('--upload-temp',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.resume:
        print('V316_TFCL_ARGUMENTS_VALID=True; resume restores budgets from saved config',flush=True)
    else:
        print(f'V316_TFCL_ARGUMENTS_VALID=True epochs={args.epochs} workers={args.workers} '
              f'microbatch={args.microbatch} frame_budget={args.frame_budget}',flush=True)


if __name__=='__main__':main()
