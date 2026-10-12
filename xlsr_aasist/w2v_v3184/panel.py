"""Small six-condition Dev diagnostic, excluded from checkpoint selection."""
from w2v_aasist.data import read_protocol
from .common import GROUPS,seed_for
from .noise import CATEGORIES


def scene_panel(cfg):
    original=read_protocol(cfg['official_dev_protocol'],cfg['official_dev_root'])
    pool={g:[] for g in GROUPS}
    for row in original:
        if row['domain']=='offline':pool[row['language'],row['label']].append(row)
    n=min(cfg['scene_panel_per_group'],*(len(rows) for rows in pool.values()))
    if n<1:raise ValueError('Scene diagnostics need offline Dev speech in all four groups')
    rows=[];tickets=[]
    for group,values in pool.items():
        selected=sorted(values,key=lambda r:seed_for(cfg['seed'],'v3184-scene-panel',r['id']))[:n]
        for row in selected:
            index=len(rows)
            rows.append(dict(row,condition='offline',source_id=row['id'],group_id=row['id'],split='dev',full_length=True))
            for scene in CATEGORIES:
                tickets.append(dict(index=index,kind='offline',epoch=0,step=0,scene=scene,probe=True,
                    occurrence='scene-panel:'+row['id']+':'+scene))
    return rows,tickets
