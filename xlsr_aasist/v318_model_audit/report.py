"""Self-contained curves and an explicit metrics-only upload allowlist."""
import html
import json
import os
from pathlib import Path
import zipfile
from v318_diagnostics import chart, atomic_text, write_csv

REPORT_FILES = (
    'result.json', 'audit_config.json', 'baseline_metrics.json', 'bn_metrics.json',
    'bn_sampling.json', 'bn_changes.json', 'update_setup.json', 'update_summary.json',
    'updates.jsonl', 'parameters.jsonl', 'adapters.jsonl', 'dev_index.json',
    'baseline_scores.npz', 'bn_scores.npz', 'groups.csv', 'updates.csv',
    'analysis.html', 'summary.txt', 'audit.log', 'audit_code_sha256.json',
)


def read(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def lines(path):
    if not path.is_file(): return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass  # A crashed process can leave a partial final record.
    return rows


def render(out):
    out = Path(out)
    result = read(out/'result.json', {})
    before = read(out/'baseline_metrics.json', {})
    after = read(out/'bn_metrics.json', {})
    updates = lines(out/'updates.jsonl')
    summary = read(out/'update_summary.json', {})
    notes = [f'状态：{result.get("status", "unknown")}；原模型文件未改变：{result.get("original_unchanged", False)}。',
             f'起点：{result.get("source_tag", "未完成验证")}。两项实验分别从原始权重开始。',
             '本报告是诊断，不会替换原 best/last，也没有修改训练设计或学习率。']
    body = ''
    reproduction = result.get('baseline_reproduction')
    if reproduction:
        notes.append(f'原模型复测核对：历史 Weighted={100*reproduction["cached_weighted"]:.3f}，'
                     f'本次复测={100*reproduction["reproduced_weighted"]:.3f}；'
                     f'最大 logit 差={reproduction["max_logit_difference"]:.6g}。')
        if abs(reproduction['cached_weighted']-reproduction['reproduced_weighted']) > .001:
            notes.append('原模型复测与保存成绩相差超过 0.1 分；先核查复现差异，再解释 BN 的效果。')
    if before and after:
        a = 100*before['full']['weighted_f1']; b = 100*after['full']['weighted_f1']
        notes.append(f'完整 Dev Weighted：原模型 {a:.3f} → BN 重估后 {b:.3f}，变化 {b-a:+.3f} 分。')
        a = 100*before['select']['weighted_f1']; b = 100*after['select']['weighted_f1']
        notes.append(f'原选模部分 Weighted：{a:.3f} → {b:.3f}，变化 {b-a:+.3f} 分。')
        notes.append('BN 只使用 Train 的均衡 Online/Noisy 样本重估，Dropout 关闭、权重不更新。'
                     '统计量按每次 BN 调用累计平均；音频/窗口长度不等，结果并非精确总体统计量。'
                     '若多组指标一致改善，支持继续研究 BN 适配；单次无改善不能排除所有 BN 问题。')
        rows = []
        for name, initial in before['groups'].items():
            final = after['groups'][name]
            for variant, item in (('original', initial), ('bn_reestimated', final)):
                rows.append(dict(group=name, variant=variant, n_fake=item['class_counts'][0], n_real=item['class_counts'][1],
                    fake_recall=100*item['recall'][0], real_recall=100*item['recall'][1],
                    F1=100*item['macro_f1'], AUC=100*item['auc'], EER=100*item['eer'],
                    fake_CE=item['class_ce'][0], real_CE=item['class_ce'][1]))
        write_csv(out/'groups.csv', list(rows[0]), rows)
        body += '<section><h2>同一份完整 Dev，固定阈值</h2><table><tr><th>分组</th><th>模型</th><th>Fake recall</th><th>Real recall</th><th>F1</th><th>AUC</th><th>EER</th><th>Real CE</th></tr>'
        for r in rows:
            body += '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
                (r['group'], r['variant'], *(f'{r[k]:.3f}' for k in ('fake_recall','real_recall','F1','AUC','EER','real_CE'))))+'</tr>'
        body += '</table></section>'
    if updates:
        notes.append('短段更新使用原 checkpoint 的 AdamW 动量和结束时学习率，保持不变；没有把余弦曲线延长到第 11 轮。'
                     '它回答“现在实际上更新了多少”，不能单独证明学习率最优，也不能还原第 3 轮的更新。')
        groups = list(updates[0]['updates'])
        rows = []
        for r in updates:
            row = dict(step=r['step'], loss=r['loss'], raw_ce=r['raw_ce'], global_grad_pre=r['global_grad_pre'])
            for g in groups:
                row.update({f'{g}_{k}': r['gradients'][g][k] for k in ('pre','post')})
                row.update({f'{g}_{k}': r['updates'][g][k] for k in ('norm','rms','relative','changed_elements')})
                row[g+'_lr'] = r['lrs'][g]
            rows.append(row)
        write_csv(out/'updates.csv', list(rows[0]), rows)
        def series(getter): return [(r['step'], getter(r)) for r in updates]
        body += chart('训练目标与原始 CE（每步保存）', [(key, series(lambda r, k=key: r[k])) for key in ('loss','raw_ce')], x_label='Update')
        for timing in ('pre','post'):
            body += chart('梯度：'+('裁剪前' if timing=='pre' else '裁剪后'),
                [(g, series(lambda r, g=g, t=timing: r['gradients'][g][t])) for g in groups], log=True, x_label='Update')
        for key, title in (('norm','参数实际变化量'),('relative','参数实际变化量 / 参数量级')):
            body += chart(title, [(g, series(lambda r, g=g, k=key: r['updates'][g][k])) for g in groups], log=True, x_label='Update')
        for g, item in summary.get('groups', {}).items():
            notes.append(f'{g}：LR={item["lr"]:.3g}；每步实际变化量中位数={item["median_update_norm"]:.6g}；'
                         f'完全没有参数变化的步数={item["zero_update_steps"]}。')
        notes.append('梯度较小或相对变化较小没有统一的好坏门槛；AdamW 会使用历史动量。'
                     '不能把裁剪倍数直接当成学习率缩小倍数。应结合各层是否有梯度、是否真的改变和泛化指标决定是否做学习率对照。'
                     'AASIST 中已有部分不参与输出的 BN 参数，缺失梯度会列出，不自动认定为故障。')
    notes.append('LoRA 的 A/B 参数逐步记录；每 10 步另算有效矩阵 B×A 的变化（首尾步也记录）。'
                 '该变化以 FP32 参数计算，不等于 BF16 前向输出的实际差值。')
    notes.append('完整 Dev 和原选模部分都已被查看；结果属于开发诊断，不是新的独立测试成绩。'
                 '两项检查结束后再决定改 BN 策略、做学习率对照，或检查前后端结构。')
    if result.get('errors'):
        notes.append('部分检查失败；不要用不完整结果下结论。错误详情在 result.json 和 audit.log。')
    atomic_text(out/'summary.txt', '\n\n'.join(notes)+'\n')
    doc = '<!doctype html><meta charset="utf-8"><title>V3.18 模型副本检查</title><style>body{max-width:1120px;margin:32px auto;padding:0 20px;font:16px system-ui;background:#f4f6fa;color:#182234}section{background:white;padding:20px;border-radius:12px;margin:20px 0}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}button{border:0;background:transparent;cursor:pointer}p{line-height:1.7}</style><h1>V3.18：BN 与实际更新检查</h1>'
    doc += '<section>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)+'</section>'+body
    doc += '<script>document.querySelectorAll("[data-toggle]").forEach(b=>b.onclick=()=>{const g=b.closest("section").querySelector(`[data-series="${b.dataset.toggle}"]`);g.style.display=g.style.display==="none"?"":"none";});</script>'
    atomic_text(out/'analysis.html', doc)


def package(out):
    out = Path(out); archive = out.parent/(out.name+'_report.zip')
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in REPORT_FILES:
            path = out/name
            if path.is_file() and not path.is_symlink(): z.write(path, name)
    os.replace(temporary, archive)
    return archive
