"""Large fixed Train probes, full Dev, source-isolated selection, readable curves."""
import csv
import html
from pathlib import Path
from w2v_v317_monitor.metrics import summarize,table_lines,fit_state
from w2v_v317_monitor.render import chart
from w2v_v316_tfcl.metrics import measure
from .common import GROUPS,seed_for,atomic_json
from .augment import PANEL_FAMILIES


def panel_rows(cfg):
    from w2v_aasist.data import read_protocol
    original=read_protocol(cfg['official_dev_protocol'],cfg['official_dev_root'])
    pool={g:[] for g in GROUPS}
    for row in original:
        if row['domain']=='offline':pool[row['language'],row['label']].append(row)
    count=min(cfg['panel_per_group'],*(len(p) for p in pool.values()))
    rows=[];tickets=[]
    for group,values in pool.items():
        for row in sorted(values,key=lambda r:seed_for(cfg['seed'],'mechanism-panel',r['id']))[:count]:
            index=len(rows)
            rows.append(dict(row,condition='offline',source_id=row['id'],group_id=row['id'],split='dev',full_length=True))
            for family in PANEL_FAMILIES:
                tickets.append(dict(index=index,kind='offline',epoch=0,step=0,family=family,probe=True,
                    occurrence='independent-panel:'+row['id']+':'+family))
    return rows,tickets


def print_epoch(entry,state):
    value=entry['metrics'];tag=entry['tag']
    print(f'\n[Dev] V3.18 {tag} FULL Clean={100*value["clean_f1"]:.3f} Noisy={100*value["noisy_f1"]:.3f} Weighted={100*value["weighted_f1"]:.3f}',flush=True)
    for line in table_lines('  Full fixed Dev (raw scores, all rows):',entry['dev']):print(line,flush=True)
    for line in table_lines(f'[Train PROBE] fixed {entry["train"]["count"]} views; no full-Train claim:',entry['train']):print(line,flush=True)
    panel=entry['panel']
    print('[Independent mechanisms] '+ '; '.join(f'{k}: F1={100*v["macro_f1"]:.3f} AUC={100*v["auc"]:.3f}' for k,v in panel['groups'].items())+'; excluded from Weighted/selection',flush=True)
    print(f'[Selection] source-isolated 80% Dev Weighted={100*entry["selection_metrics"]["weighted_f1"]:.3f}; best={state["best_tag"]}; no fallback',flush=True)
    print('[Fit] '+entry['fit']['text'],flush=True)
    print(f'[Epoch] phase={entry["phase"]}; mean objective={entry["training"]["loss"]:.5f}; raw CE={entry["training"]["raw_ce"]:.5f}; compute/wait={entry["training"]["compute_seconds"]:.2f}/{entry["training"]["wait_seconds"]:.2f}s',flush=True)


def render(run,history,steps):
    run=Path(run);out=run/'diagnostics';out.mkdir(exist_ok=True)
    series=lambda field:[(e['epoch'],100*e['metrics'][field]) for e in history]
    body=chart('完整 Dev：历史可比口径',[(k,series(k)) for k in ('clean_f1','noisy_f1','weighted_f1')],percent=True)
    body+=chart('固定 Online：相同分组口径的交叉熵',[(split,[(e['epoch'],e[split]['conditions']['online']['balanced_ce']) for e in history]) for split in ('train','dev')],log=True)
    body+=chart('训练目标与原始 CE（每100次更新汇总）',[(key,[(s['update'],s[key]) for s in steps]) for key in ('loss','raw_ce')],log=True)
    for name in ('online/en','online/zh','seen/en','seen/zh','heldout/en','heldout/zh'):
        body+=chart(name+' Dev Recall',[(label,[(e['epoch'],100*e['dev']['groups'][name]['recall'][c]) for e in history]) for c,label in enumerate(('fake','real'))],percent=True)
    text='<!doctype html><meta charset="utf-8"><title>V3.18 训练监控</title><style>body{font:16px system-ui;background:#f8fafc;color:#172033;max-width:1080px;margin:36px auto}section{background:white;padding:20px;margin:18px 0;border-radius:12px}svg{width:100%}button{cursor:pointer}text{font-size:12px}pre{white-space:pre-wrap}</style><h1>V3.18 训练与泛化</h1><p>Train 是固定分组抽测，Dev 是完整固定清单。原始 Offline 不分类；Noisy 为模拟数据。选模使用来源隔离的 Dev 子集，统一校准在模型固定后完成。</p>'+body
    if history:text+='<p>'+html.escape(history[-1]['fit']['text'])+'</p>'
    text+='<script>document.querySelectorAll("[data-toggle]").forEach(b=>b.onclick=()=>{const g=[...b.closest("section").querySelectorAll("[data-series]")].find(g=>g.dataset.series===b.dataset.toggle);if(!g)return;const hide=g.style.display!=="none";g.style.display=hide?"none":"";b.style.opacity=hide?"0.4":"1";});</script>'
    (out/'curves.html').write_text(text,encoding='utf-8')
    with (out/'epochs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['epoch','phase','clean','noisy','weighted_full','weighted_select','train_online_balanced_ce','dev_online_balanced_ce'])
        for e in history:w.writerow([e['epoch'],e['phase'],*[e['metrics'][k] for k in ('clean_f1','noisy_f1','weighted_f1')],e['selection_metrics']['weighted_f1'],e['train']['conditions']['online']['balanced_ce'],e['dev']['conditions']['online']['balanced_ce']])
    with (out/'groups.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['epoch','split','condition_language','n_fake','n_real','fake_recall','real_recall','F1','AP','AUC','EER','fake_CE','real_CE'])
        for e in history:
            for split in ('train','dev','panel'):
                for name,g in e[split]['groups'].items():w.writerow([e['epoch'],split,name,*g['class_counts'],*g['recall'],*[g[k] for k in ('macro_f1','ap','auc','eer')],*g['class_ce']])
    atomic_json(out/'fitting.json',history[-1]['fit'] if history else {})
