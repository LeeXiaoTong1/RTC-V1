"""Offline HTML monitoring charts plus complete CSV data; no new dependency."""
from collections import defaultdict
import csv
import html
import io
import json
import math
import os
from pathlib import Path
from .history import console_summary

COLORS = ('#2563eb','#e11d48','#059669','#9333ea','#d97706','#0891b2','#475569','#db2777')


def atomic_text(path, value):
    path=Path(path); temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(value, encoding='utf-8');os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def chart(title, series, log=False, percent=False):
    cleaned=[]
    for label, points in series:
        points=[(float(x),float(y)) for x,y in points if y is not None and math.isfinite(y) and (y>0 or not log)]
        if points:cleaned.append((label,points))
    if not cleaned:return '<section><h2>'+html.escape(title)+'</h2><p>等待有效数据</p></section>'
    transform=lambda y: math.log10(y) if log else y
    values=[(x,transform(y)) for _,p in cleaned for x,y in p]
    xmin,xmax=min(x for x,y in values),max(x for x,y in values)
    ymin,ymax=min(y for x,y in values),max(y for x,y in values)
    if xmin==xmax:xmin-=.5;xmax+=.5
    minimum=.02 if log else (1. if percent else max(abs(ymin),abs(ymax),1e-3)*.2)
    span=max(ymax-ymin,minimum);ymin-=.1*span;ymax+=.1*span
    if percent:ymin=max(0.,ymin);ymax=min(100.,ymax)
    def xy(x,y):return 68+760*(x-xmin)/(xmax-xmin),245-215*(transform(y)-ymin)/(ymax-ymin)
    svg=['<svg viewBox="0 0 870 300" role="img" aria-label="'+html.escape(title,quote=True)+'">']
    for i in range(5):
        yy=ymin+(ymax-ymin)*i/4; py=245-215*i/4;v=10**yy if log else yy
        svg.append(f'<line x1="68" x2="828" y1="{py}" y2="{py}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="60" y="{py+4}" text-anchor="end">{v:.3g}</text>')
    for i in range(5):
        x=xmin+(xmax-xmin)*i/4;px=68+760*i/4
        svg.append(f'<text x="{px}" y="267" text-anchor="middle">{x:.5g}</text>')
    legend=[]
    for index,(label,points) in enumerate(cleaned):
        color=COLORS[index%len(COLORS)];coords=' '.join(f'{x:.2f},{y:.2f}' for x,y in (xy(a,b) for a,b in points))
        svg.append(f'<g data-series="{index}"><polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in points:
            px,py=xy(x,y)
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"><title>{html.escape(label)}: x={x:g}, y={y:.6g}</title></circle>')
        svg.append('</g>')
        legend.append(f'<button data-toggle="{index}" style="color:{color}">{html.escape(label)}</button>')
    svg.append('</svg>')
    return '<section><h2>'+html.escape(title)+('</h2><p class="legend">'+''.join(legend)+'</p>')+''.join(svg)+'</section>'


def blocks(steps, width=100):
    groups=defaultdict(list)
    for row in steps:groups[(row['epoch'],(row['step']-1)//width)].append(row)
    result=[]
    for _,rows in sorted(groups.items()):
        cell=dict(cursor=rows[-1]['cursor'],epoch=rows[-1]['epoch'],samples=len(rows))
        for key in ('total_loss','classification_loss','time_loss','structure_loss',
                    'weighted_time_loss','weighted_structure_loss','gradient_norm','compute_seconds','data_wait_seconds'):
            numbers=[r[key] for r in rows if key in r]
            cell[key]=sum(numbers)/len(numbers) if numbers else None
        cell['maximum_example_ce']=max((r['maximum_example_ce'] for r in rows if 'maximum_example_ce' in r),default=None)
        cell['learning_rates']=rows[-1].get('learning_rates',{})
        result.append(cell)
    return result


def write_artifacts(report, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    b=blocks(report['steps']);epochs=report['epochs']
    charts=[chart('训练目标：每 100 次更新均值（对数轴）',[(label,[(r['cursor'],r[key]) for r in b])
        for label,key in [('TOTAL','total_loss'),('CE','classification_loss'),('weighted TFCL-time','weighted_time_loss'),('weighted TFCL-CKA','weighted_structure_loss')]],log=True),
        chart('异常样本与梯度：每 100 次更新的最大单样本 CE / 平均梯度范数',[(label,[(r['cursor'],r[key]) for r in b])
            for label,key in [('max example CE','maximum_example_ce'),('gradient norm','gradient_norm')]],log=True),
        chart('学习率',[(name,[(r['cursor'],r['learning_rates'].get(name)) for r in b])
                       for name in sorted({k for r in b for k in r['learning_rates']})],log=True),
        chart('同口径 Online CE：固定 Train 探针 / 完整 Dev',[(name,[(e['epoch'],e.get(key,{}).get('conditions',{}).get('online',{}).get('balanced_ce')) for e in epochs])
              for name,key in [('Train fixed probe','train'),('Dev Online','dev')]]),
        chart('Dev 成绩（%）',[(label,[(e['epoch'],100*e['metrics'][key]) for e in epochs])
                              for label,key in [('Clean','clean_f1'),('Noisy','noisy_f1'),('Weighted','weighted_f1')]],percent=True)]
    for lang in ('en','zh'):
        series=[]
        for split,condition in (('train','online'),('dev','online'),('dev','seen'),('dev','heldout')):
            for label, index in (('real',1),('fake',0)):
                values=[]
                for e in epochs:
                    v=e.get(split,{}).get('groups',{}).get(condition+'/'+lang,{}).get('recall',[None,None])[index]
                    values.append((e['epoch'],None if v is None else 100*v))
                series.append((f'{split}/{condition}/{label}',values))
        charts.append(chart(f'{lang.upper()} 分组 Recall（%，点击图例切换）',series,percent=True))
    signatures=sorted({e['panel']['panel_signature'] for e in epochs if e.get('panel')})
    for signature in signatures:
        selected=[e for e in epochs if e.get('panel',{}).get('panel_signature')==signature]
        charts.append(chart('全量 Train 固定复测 CE：'+signature[:10],[(c,[(e['epoch'],e['panel']['metrics']['conditions'].get(c,{}).get('balanced_ce')) for e in selected])
                           for c in ('offline','online','noisy_train')]))
        charts.append(chart('全量 Train / Dev Online CE（同一全量面板）',[
            ('Train FULL',[(e['epoch'],e['panel']['metrics']['conditions']['online']['balanced_ce']) for e in selected]),
            ('Dev FULL',[(e['epoch'],e.get('dev',{}).get('conditions',{}).get('online',{}).get('balanced_ce')) for e in selected])]))
    warnings=''.join('<p class="warning">'+html.escape(w)+'</p>' for w in report['warnings'])
    note=('训练 step 的 CE 来自不断变化的增强样本和训练模式；不能直接与 eval 模式的 Dev CE 当作同一口径。'
          '拟合趋势优先使用全量 Online Train/Dev 分组平衡 CE；尚未全量复测的旧轮次只显示原有样本探针。完整 Dev 用于原有选模。'
          'noisy_train 使用固定的 Train 噪声/RTC 配方，不等于 Dev Seen/Heldout。AP 受真假比例影响。'
          '对数图只显示正值，0/缺失不伪造为正数；精确值见 CSV。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3.17 训练诊断</title><style>
body{font:15px system-ui,sans-serif;background:#f1f5f9;color:#0f172a;margin:0;padding:28px;max-width:1300px;margin:auto}
h1{font-size:27px}h2{font-size:17px}section,.intro{background:white;padding:20px;border-radius:12px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.grid section{margin:0}
svg{width:100%;font:12px system-ui,sans-serif}button{border:1px solid #cbd5e1;background:white;border-radius:5px;padding:5px;margin:3px;cursor:pointer}
pre{overflow:auto;line-height:1.55;font-size:12px}p{line-height:1.6}.warning{color:#b45309}a{color:#2563eb}
@media(max-width:650px){body{padding:10px}.grid{grid-template-columns:1fr}}
</style><h1>V3.17 训练诊断</h1>'''
    page+='<div class="intro"><p>'+html.escape(report['run'])+'</p><p>'+html.escape(note)+'</p>'
    page+='<p>已提交更新：'+str(report['committed_cursor'])+'；日志最后更新：'+str(report['steps'][-1]['cursor'] if report['steps'] else 0)+'。超出已提交部分尚未保存 checkpoint。</p>'
    page+='<p><strong>'+html.escape(report['fit']['text'])+'</strong></p>'+warnings+'</div><div class="grid">'+''.join(charts)+'</div>'
    page+='<section><h2>逐轮指标</h2><pre>'+html.escape(console_summary(report,all_epochs=True))+'</pre></section>'
    page+='''<script>document.querySelectorAll('[data-toggle]').forEach(b=>b.onclick=()=>{
const g=b.closest('section').querySelector('g[data-series="'+b.dataset.toggle+'"]');
const off=g.style.display==='none';g.style.display=off?'':'none';b.style.opacity=off?'1':'.35';});</script></html>'''
    atomic_text(out/'curves.html',page)
    atomic_text(out/'summary.txt',console_summary(report,all_epochs=True)+'\n')
    atomic_text(out/'observations.json',json.dumps({k:v for k,v in report.items() if k!='steps'},ensure_ascii=False,indent=2,allow_nan=False))
    stream=io.StringIO(newline='');names=['cursor','epoch','step','total_loss','classification_loss','time_loss','structure_loss',
        'weighted_time_loss','weighted_structure_loss','maximum_example_ce','gradient_norm','compute_seconds','data_wait_seconds']
    writer=csv.DictWriter(stream,fieldnames=names+['lora_lr','head_lr','tfcl_lr','committed'],extrasaction='ignore');writer.writeheader()
    for r in report['steps']:
        lr=r.get('learning_rates',{})
        writer.writerow(dict(r,lora_lr=lr.get('lora'),head_lr=lr.get('detection_head'),tfcl_lr=lr.get('training_only_tfcl'),committed=r['cursor']<=report['committed_cursor']))
    atomic_text(out/'loss_steps.csv',stream.getvalue())
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['epoch','tag','scope','group','fake_count','real_count','fake_recall','real_recall','macro_f1','ap','auc','eer','balanced_ce','fake_ce','real_ce'])
    for e in epochs:
        scopes=[('train_online_probe',e.get('train')),('dev_full',e.get('dev'))]
        if e.get('panel'):scopes.append(('train_panel_'+e['panel']['panel_signature'][:10],e['panel']['metrics']))
        for name,value in scopes:
            if not value:continue
            for group,m in value['groups'].items():
                writer.writerow([e['epoch'],e['tag'],name,group,*m['class_counts'],*m['recall'],m['macro_f1'],m['ap'],m['auc'],m['eer'],m['balanced_ce'],*m['class_ce']])
    atomic_text(out/'group_metrics.csv',stream.getvalue())
    return out/'curves.html'
