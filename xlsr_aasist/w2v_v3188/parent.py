"""Read-only, hash-verified transfer of the complete V3.18.7 selected best."""
import argparse
from pathlib import Path
import torch
from .common import ROOT,read_json,digest,fingerprint,verify_files,apply_partial,partial_state
from w2v_v3187.common import schema_for as parent_schema


def resolve(value=None,root=ROOT):
    if value:
        path=Path(value).expanduser().resolve()
    else:
        marker=Path(root)/'exp'/'.latest_v3187_run'
        if not marker.is_file() or not marker.read_text().strip():
            raise ValueError('Missing exp/.latest_v3187_run; pass --parent-run PATH to the stopped V3.18.7 v2 run')
        path=Path(marker.read_text().strip()).expanduser()
        if not path.is_absolute():path=Path(root)/path
        path=path.resolve()
    for name in ('config.json','last.pt','train_rows.json','dev_rows.json','dev_partition.json'):
        if not (path/name).is_file():raise FileNotFoundError('Missing parent file: '+str(path/name))
    return path


def load(run,expected=None):
    run=Path(run);cfg=read_json(run/'config.json');parent_schema(cfg)
    checkpoint=run/'last.pt';checksum=digest(checkpoint)
    if expected and (checksum!=expected['checkpoint_sha256'] or digest(run/'config.json')!=expected['config_sha256']):
        raise ValueError('Parent changed after preparation; keep the selected source run stopped')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if state.get('schema')!=parent_schema(cfg) or state.get('identity')!=fingerprint(cfg):
        raise ValueError('Parent checkpoint/config mismatch')
    tag=state.get('best_tag')
    entries=[e for e in state.get('history',[]) if e.get('tag')==tag]
    if not tag or len(entries)!=1 or entries[0]['epoch']<1 or not state.get('best_model'):
        raise ValueError('Parent has no verified completed best epoch')
    if expected and tag!=expected['tag']:raise ValueError('Parent selected best changed')
    if digest(checkpoint)!=checksum:raise ValueError('Parent checkpoint changed while reading')
    return cfg,state,entries[0],checksum


def metadata(value=None):
    run=resolve(value);cfg,state,entry,checksum=load(run)
    verify_files(cfg['code_fingerprints'])
    verify_files({str(run/name):checksum for name,checksum in cfg['saved_manifests'].items()})
    from .sampling import quota_schedule
    steps=len(quota_schedule(tuple(cfg['source_draws_per_stream']),cfg['stream_sources']))
    origin=dict(run=str(run),checkpoint=str(run/'last.pt'),checkpoint_sha256=checksum,
        config_sha256=digest(run/'config.json'),tag=entry['tag'],epoch=int(entry['epoch']),steps=steps,
        manifest_files={str(run/name):digest(run/name) for name in ('config.json','train_rows.json','dev_rows.json','dev_partition.json')})
    return cfg,origin,read_json(run/'train_rows.json'),read_json(run/'dev_rows.json')


def initialize(model,cfg,dev,partition):
    origin=cfg['parent_init'];old,state,entry,_=load(origin['run'],origin)
    values=state['best_model']
    apply_partial(model,values)
    if cfg.get('consistency')=='dual' and not model.pcl_scale.is_frozen:
        raise ValueError('Parent best lacks frozen Train PCL scale; do not silently recalibrate it')
    logits=state['best_logits']
    if logits.shape!=(len(dev),2) or not torch.isfinite(logits).all():
        raise ValueError('Parent best Dev score geometry is invalid')
    from .monitor import summarize,measure
    z=logits.numpy().copy();select=partition['select']
    metrics=measure(dev,z);selected=measure([dev[i] for i in select],z[select])
    for name in ('clean_f1','noisy_f1','weighted_f1'):
        value=selected[name]
        if name in state['best_metrics'] and abs(float(value)-float(state['best_metrics'][name]))>1e-7:
            raise ValueError('Parent best selection does not match saved Dev scores')
    tag='parent_v3187_'+origin['tag']
    baseline=dict(epoch=0,tag=tag,phase='parent_baseline',metrics=metrics,selection_metrics=selected,
        dev=summarize(dev,z),train=None,training={},promoted=True,fusion=model.fusion.describe(),
        origin=origin,score_provenance='Verified parent best logits; identical detector architecture; no extra GPU Dev pass')
    audit=dict(mode='v3187_best',selected=origin['tag'],parent_epoch=origin['epoch'],
        checkpoint_sha256=origin['checkpoint_sha256'],optimizer_loaded=False,
        full_detector_loaded=True,pcl_scale_loaded=True,pcl_scale_frozen=model.pcl_scale.is_frozen,
        parameter_keys=len(values),classifier_input_dim=model.head.aasist.out_layer.in_features)
    return baseline,partial_state(model),torch.from_numpy(z),audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run');args=p.parse_args()
    print(resolve(args.parent_run))
