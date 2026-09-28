"""Inspect CE / real-RTC / simulated-noisy gradients without any parameter update."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
import secrets
import time
import zipfile

import numpy as np
import torch

from audit_w2v_dev import compare_model_configs
from audit_w2v_train import save_csv, upload_report
from recover_w2v_storage import recovery_lock
from start_w2v_en import guard_output, verified_baseline
from w2v_rebuild.core import atomic_json, load_checkpoint, sha256
from w2v_rebuild.model import Detector, forward_chunks
from w2v_rebuild.pair_gradient import loss_terms, gradient_gram, comparison_rows, pair_diagnostics
from w2v_rebuild.pair_gradient_data import AuditTrainData, metadata_fingerprints, recorded_args


ROOT = Path(__file__).resolve().parent
FORMAT = 'w2v_pair_gradient_audit_v1'
EXPORT_FILES = ('manifest.json', 'screen_pairs.csv', 'selected_batches.json', 'pair_scores.csv',
                'gradient_records.jsonl', 'gradient_metrics.csv', 'summary.json', 'report.md', 'completed.json')


def pair_rows(logits, features, labels, metadata, branch, step, phase):
    count = len(labels)//2
    if count != 4 or len(metadata) != count or not torch.equal(labels[:count], labels[count:]):
        raise ValueError('Expected four correctly paired sources per branch')
    p = logits.detach().float().softmax(1)[:, 0].cpu()
    prediction = (p < .5).long()
    y = labels.detach().cpu()
    diagnostics = pair_diagnostics(features[:count], features[count:], labels[:count])
    rows = []
    for i, meta in enumerate(metadata):
        if y[i].item() != meta['label'] or branch != meta['branch']:
            raise ValueError('Pair metadata/order mismatch')
        before, after = bool(prediction[i] == y[i]), bool(prediction[count+i] == y[count+i])
        rows.append(dict(meta, step=step, phase=phase, pair_index=i,
                         reference_pfake=float(p[i]), processed_pfake=float(p[count+i]),
                         reference_correct=before, processed_correct=after, correct_to_wrong=before and not after,
                         **{k: v[i] if isinstance(v, list) else v for k, v in diagnostics.items()}))
    return rows


def screen(model, data, steps, device, microbatch):
    result = []
    model.eval()
    for i, step in enumerate(steps):
        meta = data.metadata(step)
        for branch_index, batch in enumerate(data.pair_batches(step)):
            with torch.no_grad():
                z, h = forward_chunks(model, batch['features'].to(device), batch['mask'].to(device),
                                       microbatch, pad_last=True)
            result += pair_rows(z, h, batch['labels'], meta[branch_index*4:branch_index*4+4],
                                ['rtc', 'noisy'][branch_index], step, 'screen_eval_fp32')
        if i == 0 or (i+1) % 4 == 0 or i+1 == len(steps):
            failures = sum(not r['processed_correct'] for r in result)
            print(f'Prescreen {i+1}/{len(steps)} batches; processed mistakes={failures}', flush=True)
    return result


def select_batches(rows, random_steps, hard_count, seed):
    """Random cohort was drawn BEFORE scoring. Targeted errors are separate."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['step']].append(r)
    pool = [s for s in grouped if s not in random_steps and any(not r['processed_correct'] for r in grouped[s])]
    random.Random(seed).shuffle(pool)
    def rank(step):
        batch = grouped[step]
        return (sum(r['branch'] == 'noisy' and r['correct_to_wrong'] for r in batch),
                sum(r['branch'] == 'noisy' and not r['processed_correct'] for r in batch),
                sum(r['correct_to_wrong'] for r in batch))
    pool.sort(key=rank, reverse=True)
    chosen = [{'step': s, 'cohort': 'random'} for s in sorted(random_steps)]
    chosen += [{'step': s, 'cohort': 'targeted_errors'} for s in pool[:hard_count]]
    for row in chosen:
        batch = grouped[row['step']]
        row['fixed_error_masks'] = {branch+'_error_ce': [not r['processed_correct'] for r in batch if r['branch'] == branch]
                                    for branch in ('rtc', 'noisy')}
        row['screen_noisy_flips'] = sum(r['correct_to_wrong'] and r['branch'] == 'noisy' for r in batch)
        row['screen_noisy_errors'] = sum(not r['processed_correct'] and r['branch'] == 'noisy' for r in batch)
    return chosen


def model_versions(model):
    return {name: (id(p), p._version) for name, p in list(model.named_parameters())+list(model.named_buffers())}


def measure(model, data, selected, device, args, out):
    metrics, scores = [], []
    expected = model_versions(model)
    precision = 'bf16' if data.args.amp == 'bf16' and device.type == 'cuda' else 'fp32'
    modes = [('eval_fp32', False, None)] + [(f'train_{precision}_{i+1}', True, args.seed+100000*(i+1))
                                             for i in range(args.train_repeats)]
    with (out/'gradient_records.jsonl').open('x', encoding='utf-8') as stream:
        for index, selection in enumerate(selected):
            step = selection['step']
            batch, layout = data.logical_batch(step)
            metadata = data.metadata(step)
            inputs, mask, labels = (batch[k].to(device) for k in ('features', 'mask', 'labels'))
            language = batch['language_weights'].to(device) if data.args.language_weighting else None
            for mode, training, seed in modes:
                started = time.monotonic()
                if seed is not None:
                    torch.manual_seed(seed+step)
                model.train(training)
                context = (torch.autocast('cuda', dtype=torch.bfloat16)
                           if training and data.args.amp == 'bf16' and device.type == 'cuda' else nullcontext())
                print(f'Gradients {index+1}/{len(selected)}; {selection["cohort"]}; step={step}; {mode}', flush=True)
                with context:
                    logits, readout = forward_chunks(model, inputs, mask, data.args.microbatch)
                n, r, s = layout
                for branch, offset, take, meta in [('rtc', n, r, metadata[:4]), ('noisy', n+2*r, s, metadata[4:])]:
                    scores += [dict(row, cohort=selection['cohort']) for row in pair_rows(
                        logits[offset:offset+2*take], readout[offset:offset+2*take], labels[offset:offset+2*take],
                        meta, branch, step, mode)]
                terms = loss_terms(logits, readout, labels, layout, data.weights, data.args.real_ce_weight,
                                   language, selection['fixed_error_masks'])
                result = gradient_gram(terms, list(model.named_parameters()))
                context_fields = dict(step=step, cohort=selection['cohort'], mode=mode)
                metrics += [dict(context_fields, **r) for r in comparison_rows(result)]
                record = dict(context_fields, **result, seconds=time.monotonic()-started,
                              fixed_error_masks=selection['fixed_error_masks'], layout=list(layout),
                              source_ids=batch['source_ids'], languages=batch['languages'].tolist(),
                              labels=batch['labels'].tolist(), language_weights=batch['language_weights'].tolist())
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                if model_versions(model) != expected or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Model tensors or .grad unexpectedly changed during read-only audit')
                print(f'  CE={result["losses"]["ce"]:.6g} RTC={result["losses"]["rtc"]:.6g} '
                      f'noisy={result["losses"]["noisy"]:.6g}; {record["seconds"]:.1f}s; no update', flush=True)
                del terms, logits, readout, result
            del inputs, mask, labels, batch
    return metrics, scores


def summaries(metrics, screen_rows, selected):
    grouped = defaultdict(list)
    for row in metrics:
        key = tuple(row[k] for k in ('cohort', 'mode', 'target', 'auxiliary', 'group', 'weight'))
        grouped[key].append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        valid = [r for r in rows if r['cosine'] is not None]
        result = dict(zip(('cohort', 'mode', 'target', 'auxiliary', 'group', 'weight'), key))
        result.update(batches=len(rows), defined_cosines=len(valid),
                      opposing_fraction=sum(r['cosine'] < 0 for r in valid)/len(valid) if valid else None)
        for name in ('cosine', 'norm_ratio', 'opposition_fraction', 'weighted_aux_norm', 'target_norm'):
            values = [r[name] for r in rows if r[name] is not None]
            result[name+'_median'] = float(np.median(values)) if values else None
            result[name+'_p90'] = float(np.quantile(values, .9)) if values else None
        aggregates.append(result)
    groups = defaultdict(list)
    for row in screen_rows:
        groups[(row['branch'], row['language'], row['label'], row['bank'], row['family'], row['band'])].append(row)
    coverage = [dict(zip(('branch', 'language', 'label', 'bank', 'family', 'band'), key), pairs=len(rows),
                     unique_sources=len({r['source_id'] for r in rows}),
                     processed_errors=sum(not r['processed_correct'] for r in rows),
                     correct_to_wrong=sum(r['correct_to_wrong'] for r in rows)) for key, rows in sorted(groups.items())]
    return dict(aggregates=aggregates, screening_coverage=coverage,
                selected_cohorts=dict(Counter(r['cohort'] for r in selected)),
                processed_noisy_errors=sum(r['branch'] == 'noisy' and not r['processed_correct'] for r in screen_rows),
                noisy_error_sources=len({r['source_id'] for r in screen_rows if r['branch'] == 'noisy' and not r['processed_correct']}),
                recommendation='review_gradients_before_any_training_change')


def markdown(summary):
    def fmt(x):
        return 'n/a' if x is None else f'{x:.4g}'
    lines = ['# 配对对比学习梯度检查', '',
             '使用原 best 和记录的 Stage3 训练配置；只读取 Train，参数没有更新。', '',
             f'预筛 noisy 判错视图：{summary["processed_noisy_errors"]}；独立 source ID：{summary["noisy_error_sources"]}。',
             '抽样批次：'+json.dumps(summary['selected_cohorts'], ensure_ascii=False)+'。', '',
             '## 如何解读', '',
             '- cosine < 0：在这批数据、这些参数上，两个目标的原始梯度方向相反。',
             '- norm_ratio：加权对比梯度的大小 / 对应分类梯度的大小；不能只看损失数值。',
             '- opposition_fraction > 0：对比项抵消了分类目标自身梯度下降方向的一部分；负数表示同向帮助。',
             '- shared 排除了对比损失根本不会更新的最终线性分类器，避免稀释方向统计。',
             '- noisy_processed_ce 是所有模拟处理后样本的分类项；noisy_error_ce 只取预筛判错样本，仍保留原分母和权重。',
             '- random 在预测前抽取；targeted_errors 为刻意挑选的错误案例，不能混合估计总体冲突比例。',
             '- eval_fp32 关闭 dropout；train 模式遵循记录的精度及冻结层设置，两个种子分别报告。它们不只存在 dropout 差异，也可能有精度差异。', '',
             '## 共享参数的主要结果', '',
             '| 样本 | 模式 | 分类目标 | 对比项 | 权重 | 批数 | 中位 cosine | 中位 norm_ratio | 反向比例 |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        if row['group'] == 'shared' and row['target'] in ('ce', 'noisy_processed_ce', 'noisy_error_ce'):
            lines.append('| '+' | '.join(str(row[k]) for k in ('cohort', 'mode', 'target', 'auxiliary', 'weight', 'batches'))+
                         ' | '+' | '.join(fmt(row[k]) for k in ('cosine_median', 'norm_ratio_median', 'opposing_fraction'))+' |')
    lines += ['', '## 范围与限制', '',
              '- 测量编码器和 AASIST 的实际可训练参数，每层和共享分类头的细节见 gradient_metrics.csv。',
              '- RTC 权重 0.1；noisy 分别展示 0.05 和 0.1，都是同一组原始梯度的精确缩放，没有额外训练。不是逐步预热权重的重放。',
              '- descent_with/without_noisy 是假设共同学习率下的一阶方向量；没有执行虚拟更新，更不能当作 AdamW 更新或 F1 提升预测。',
              '- 这是同一初始 checkpoint 的小样本局部诊断，不复现已训练 epoch 的 Adam 动量，也不证明最终泛化因果。',
              '- 若错误样本很少，对错误子集只能记为证据不足。来源可能重复，多次 dropout 也不是独立样本。',
              '- pair_scores.csv 保留正负对相似度、FP32/FP64 对比损失与特征梯度；打印为零的损失不一定没有梯度。',
              '- 先查看反向程度、相对大小及多个模式是否一致，再决定是否只移除 noisy 对比。脚本不会自动关闭损失或启动训练。',
              '- ZIP 仅包含元数据、标量与报告，不含音频、模型、特征或梯度向量。', '']
    return '\n'.join(lines)


def package(out, download_dir):
    download = Path(download_dir).expanduser().resolve()
    download.mkdir(parents=True, exist_ok=True)
    archive = download/(out.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for name in EXPORT_FILES:
            path = out/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('Missing/unsafe report file: '+name)
            target.write(path, arcname=name)
    return archive


def run(args, source, config, out):
    started = time.monotonic()
    baseline, digest = verified_baseline(config, source)
    protected = [source, baseline.parent, config['ssl_path'], config['train_data_path'], config['dev_data_path'],
                 config['train_noisy_cache'], *config.get('extra_train_noisy_cache', []),
                 config['dev_noisy_cache'], config['dev_heldout_cache']]
    if config.get('feature_cache'):
        protected.append(config['feature_cache'])
    guard_output(out, protected)
    guard_output(Path(args.download_dir), protected)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    recipe = recorded_args(config, args.device, args.microbatch)
    if device.type == 'cuda' and recipe.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Recorded training requires BF16 support')
    os.environ.update({k: str(v) for k, v in config.get('noise_environment', {}).items()})
    os.environ.update(TOKENIZERS_PARALLELISM='false', RTC_NOISE_CACHE_MB=str(recipe.noise_cache_mb),
                      RTC_B_NOISE_MANIFEST=str(Path(recipe.train_noise_manifest).resolve()))
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    inputs = metadata_fingerprints(config, source)
    print('Building Train-only epoch plan from recorded recipe. No cache generation or Dev inference.', flush=True)
    data = AuditTrainData(recipe)
    rng = random.Random(args.seed)
    screen_steps = sorted(rng.sample(range(data.steps), min(args.screen_batches, data.steps)))
    random_steps = sorted(rng.sample(screen_steps, min(args.random_batches, len(screen_steps))))
    from w2v_rebuild.train import source_hashes
    codes = source_hashes()
    for path in ('audit_w2v_pair_gradients.py', 'audit_w2v_structure.py', 'audit_w2v_train.py',
                 'audit_w2v_dev.py', 'start_w2v_en.py', 'run_w2v_pair_gradients.sh'):
        codes[path] = sha256(ROOT/path)
    manifest = dict(format=FORMAT, baseline_path=str(baseline), baseline_sha256=digest, source_run=str(source),
                    input_sha256=inputs, source_hashes=codes, recipe=vars(recipe), settings=vars(args),
                    train_counts=data.counts.tolist(), ce_weights=data.weights.tolist(),
                    language_budgets=data.language_budgets, epoch1_rotation=data.rotation_summary,
                    screen_steps=screen_steps, random_steps_selected_before_predictions=random_steps,
                    gradient_scope='recorded trainable layers + AASIST; raw gradients, no Adam state or update',
                    torch_version=str(torch.__version__))
    atomic_json(manifest, out/'manifest.json')
    print('Loading original best; configuring recorded trainable layers only.', flush=True)
    checkpoint = load_checkpoint(baseline)
    comparison = compare_model_configs(checkpoint['model_config'], config['model_config'])
    if checkpoint.get('stage') != 3 or not comparison['matched']:
        raise ValueError('Original checkpoint architecture/stage differs after JSON normalization')
    for path in [recipe.train_protocol, str(Path(recipe.ssl_path)/'config.json'), str(Path(recipe.ssl_path)/'preprocessor_config.json')]:
        key = str(Path(path).resolve())
        if checkpoint.get('data_fingerprints', {}).get(key) != inputs[key]:
            raise ValueError('Original best Train protocol/extractor differs: '+key)
    model = Detector.load(recipe.ssl_path, checkpoint['model_config'], checkpointing=not recipe.no_checkpointing)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint; gc.collect()
    model.configure_trainable_layers(recipe.trainable_encoder_layers)
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Audit must not update BatchNorm running state')
    model.to(device)
    manifest['trainable_parameters'] = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    print(f'Trainable encoder layers={recipe.trainable_encoder_layers}; prescreen={len(screen_steps)} batches; '
          f'gradient batches <= {len(random_steps)+args.hard_batches}; NO optimizer is constructed.', flush=True)
    screening = screen(model, data, screen_steps, device, recipe.eval_microbatch)
    save_csv(out/'screen_pairs.csv', screening)
    selected = select_batches(screening, random_steps, args.hard_batches, args.seed+1)
    atomic_json(selected, out/'selected_batches.json')
    metrics, scores = measure(model, data, selected, device, args, out)
    del model; gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('Verifying checkpoint and selected input hashes are unchanged; packaging scalar reports.', flush=True)
    manifest['selected_audio_sha256'] = data.audio_hashes
    atomic_json(manifest, out/'manifest.json')
    for path, expected in {**inputs, **data.audio_hashes, str(baseline): digest}.items():
        if sha256(path) != expected:
            raise ValueError('Read-only audit input changed: '+path)
    summary = summaries(metrics, screening, selected)
    summary.update(format=FORMAT, status='complete', original_best_preserved=True,
                   model_parameters_unchanged=True, optimizer_constructed=False, seconds=time.monotonic()-started,
                   scope=manifest['gradient_scope'])
    save_csv(out/'pair_scores.csv', scores); save_csv(out/'gradient_metrics.csv', metrics)
    atomic_json(summary, out/'summary.json')
    (out/'report.md').write_text(markdown(summary), encoding='utf-8')
    atomic_json(dict(status='complete', original_best_preserved=True,
                     summary_sha256=sha256(out/'summary.json'), manifest_sha256=sha256(out/'manifest.json'),
                     report_hashes={name: sha256(out/name) for name in EXPORT_FILES if name != 'completed.json'}),
                out/'completed.json')
    archive = package(out, args.download_dir)
    print(f'REPORT_DIR={out}\nDOWNLOAD_ZIP={archive}\nPAIR_GRADIENT_AUDIT_COMPLETE=True\nORIGINAL_BEST_PRESERVED=True', flush=True)
    if args.upload_temp:
        try:
            url = upload_report(archive)
            (out/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
            print('TEMP_DOWNLOAD_URL='+url+'\nUPLOAD_COMPLETE=True', flush=True)
        except Exception as exc:
            print(f'UPLOAD_COMPLETE=False\nLocal ZIP is available: {archive}\n{type(exc).__name__}: {exc}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True)
    parser.add_argument('--out')
    parser.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    parser.add_argument('--upload-temp', action='store_true')
    parser.add_argument('--screen-batches', type=int, default=128)
    parser.add_argument('--random-batches', type=int, default=16)
    parser.add_argument('--hard-batches', type=int, default=8)
    parser.add_argument('--train-repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--microbatch', type=int, help='Default: recorded training microbatch')
    args = parser.parse_args(argv)
    if (not 1 <= args.random_batches <= args.screen_batches or args.hard_batches < 0
            or not 1 <= args.train_repeats <= 3 or (args.microbatch is not None and args.microbatch < 1)):
        parser.error('Use positive screen/random counts, random<=screen, hard>=0, 1..3 train repeats and positive microbatch')
    source = Path(args.from_run).expanduser().resolve(strict=True)
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8-sig'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2)
    out = Path(args.out or ROOT/'exp'/('w2v_pair_gradients_'+stamp)).expanduser().resolve()
    print('MODE=read_only_gradient_audit\nREPORT_DIR='+str(out), flush=True)
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        return run(args, source, config, out)


if __name__ == '__main__':
    raise SystemExit(main())
