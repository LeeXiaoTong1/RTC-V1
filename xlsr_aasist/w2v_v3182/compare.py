"""Compare the CE-only control and contrastive run with verified shared warmup."""
import argparse
from pathlib import Path
from .common import read_json, atomic_json, digest
from .warmup import signature


def read_run(path):
    from .evaluate import selected
    path=Path(path).resolve();cfg,weights,done=selected(path,'best')
    del weights
    if cfg.get('version')!='3.18.2' or done.get('status')!='complete':
        raise ValueError('Both runs must be completed V3.18.2 experiments')
    if digest(path/'last.pt')!=done['checkpoint_sha256']:
        raise ValueError('Completed checkpoint hash differs')
    best=next(e for e in read_json(path/'training_history.json') if e['tag']==done['best_tag'])
    parent=Path(cfg.get('warmup_parent',path))
    warm=read_json(parent/'warmup.json')
    if digest(parent/'warmup.pt')!=warm['checkpoint_sha256']:
        raise ValueError('Shared warmup hash differs')
    return cfg,done,best,warm['checkpoint_sha256']


def compare(control,treatment):
    a,ad,ae,aw=read_run(control);b,bd,be,bw=read_run(treatment)
    if a['contrastive_weight']!=0 or b['contrastive_weight']<=0:
        raise ValueError('Control must be CE-only; treatment must enable contrastive loss')
    if signature(a)!=signature(b) or aw!=bw:
        raise ValueError('A/B architecture, data, schedule, code or shared warmup differs')
    if ad['completed_epochs']!=bd['completed_epochs']:
        raise ValueError('A/B update budgets differ; finish both before comparing')
    groups={}
    for name,left in ae['dev']['groups'].items():
        right=be['dev']['groups'][name]
        fields={'fake_recall':lambda v:v['recall'][0], 'real_recall':lambda v:v['recall'][1],
                'fake_precision':lambda v:v['precision'][0], 'real_precision':lambda v:v['precision'][1],
                'auc':lambda v:v['auc'], 'f1':lambda v:v['macro_f1'],
                'real_at_95_fake':lambda v:v['matched_95']['real_recall'],
                'real_at_99_fake':lambda v:v['matched_99']['real_recall']}
        groups[name]={key:dict(control=fn(left),treatment=fn(right),delta_pp=100*(fn(right)-fn(left)))
                      for key,fn in fields.items()}
    scores={}
    for part in ('selection_metrics','metrics'):
        scores[part]={key:dict(control=ae[part][key],treatment=be[part][key],
                             delta_pp=100*(be[part][key]-ae[part][key]))
                      for key in ('clean_f1','noisy_f1','weighted_f1')}
    return dict(version='3.18.2',control=str(Path(control).resolve()),treatment=str(Path(treatment).resolve()),
        checkpoints={'control':ad['best_tag'],'treatment':bd['best_tag']},shared_warmup_sha256=aw,
        completed_epochs=ad['completed_epochs'],raw_scores=scores,groups=groups,
        note='Each checkpoint selected on the same Dev selection sources. Full-Dev group diagnostics are not an independent test or a significance claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',required=True);p.add_argument('--treatment',required=True)
    p.add_argument('--out',required=True,help='Output JSON report path')
    args=p.parse_args();result=compare(args.control,args.treatment)
    out=Path(args.out).resolve()
    for run in (Path(args.control).resolve(),Path(args.treatment).resolve()):
        if out==run or run in out.parents:raise ValueError('Write comparison outside both run directories')
    if out.exists():raise FileExistsError('Comparison destination already exists')
    atomic_json(out,result)
    print('Validated shared warmup and equal training budgets.')
    print('Selection Weighted delta (B-A), pp: '+str(result['raw_scores']['selection_metrics']['weighted_f1']['delta_pp']))
    for name,values in result['groups'].items():
        print(name+' real@95fake delta='+str(values['real_at_95_fake']['delta_pp'])+
              ' pp; real@99fake delta='+str(values['real_at_99_fake']['delta_pp'])+' pp')
    print('COMPARISON_JSON='+str(out))


if __name__=='__main__':main()
