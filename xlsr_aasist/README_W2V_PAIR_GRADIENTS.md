# 配对对比学习的小规模梯度检查

这个入口只诊断，不启动训练。原始 best 用 `baseline_path` / `init_sha256` 定位并核对；
`--from-run` 只提供数据路径和损失、采样、冻结层等配置，不会读取该 run 的候选权重。
没有优化器、参数更新、checkpoint 保存、音频生成或特征缓存写入。

## 服务器运行

在现有环境中执行，不需要安装新依赖：

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
bash run_w2v_pair_gradients.sh exp/w2v_en_20260928_161650_59ab --upload-temp
```

后台进程的日志：

```bash
tail -n 60 -f "$(cat exp/.latest_pair_gradients_log)"
```

`Ctrl+C` 仅退出上面的日志查看，后台诊断继续运行。完成后必须看到
`PAIR_GRADIENT_AUDIT_COMPLETE=True` 和 `ORIGINAL_BEST_PRESERVED=True`。
ZIP 保存在 `/home/ubuntu/LXT/temp`，上传成功打印 `TEMP_DOWNLOAD_URL=https://temp.sh/...`。
只有显式 `--upload-temp` 才上传；内容是标量、来源路径/ID 和配置，不含音频、checkpoint、特征或梯度向量。
上传失败时本地 ZIP 保留，可用：

```bash
curl --fail --show-error -F "file=@/home/ubuntu/LXT/temp/实际报告名.zip" https://temp.sh/upload
```

## 实际测量什么

1. 使用记录的 epoch 1 完整采样计划，随机抽 128 个逻辑步，只对两个配对分支做 FP32 eval 预筛。
   每步 4 对真实 RTC、4 对模拟 noisy；普通分类支路此时无需前向。
2. 在看预测之前随机选 16 个步骤；额外最多 8 个步骤专门检查处理后错误，优先 clean 正确 / noisy 错误。
   两种样本分开汇总。如果错误不足，不用 Dev 补齐，也不无限扩大检查。
3. 选中的逻辑步恢复原来约 `24 ordinary + 8 RTC views + 8 noisy views` 的真实输入。
   ordinary 仍走原来的增强，配对使用已有缓存。生产采样器、类/语言权重和 CE 分母保持一致。
4. 对同一个前向图分别求 CE、RTC InfoNCE、noisy InfoNCE 以及 noisy 处理后 CE 的梯度。
   有预筛错误时额外测量对应错误 CE；错误集合固定，原分类权重/分母保留。
5. 默认运行 `eval_fp32`、两次不同种子的训练模式。训练模式保留记录的 BF16 / FP32 设置及冻结层策略。
   每一模式内各损失共享同一次 dropout 实现，没有用不同前向的随机扰动比较梯度。
6. 测量所有实际可训练参数，并分组报告 encoder、各编码器层、head_shared、classifier。
   `shared` 不包含对比项不会更新的最后线性分类器。无需更改原有 loss 或模型代码。

默认最多 24 个批次 × 3 种模式的梯度检查，不跑 epoch；需要 GPU，反向会比之前仅推理的结构审计更慢。
梯度副本放 CPU 内存，报告不保存这些向量；实际耗时/显存取决于服务器及记录的可训练层数。
每个预筛阶段和梯度批次都有进度。

直接 Python 入口可以覆盖规模，例如 `--screen-batches 64 --random-batches 8 --hard-batches 4`，
但错误样本可能不足。`--microbatch` 默认沿用配置，不改有效 batch。默认两次训练模式测量不应混合当作独立样本。

## 报告文件

- `report.md`：中文解读与主要表格。
- `summary.json`：分样本来源 / 模式 / 参数组的中位数、90 分位及反向比例。
- `gradient_metrics.csv`：每批的梯度大小、夹角、加权比例和一阶方向量。
- `gradient_records.jsonl`：每项 loss、精确梯度内积矩阵、逻辑 batch ID / 标签 / 语言系数。
- `screen_pairs.csv`、`pair_scores.csv`：真假分数、错误转换、语言/处理流程/强度、正负对相似度；
  FP32 与 FP64 对比损失和特征梯度同时记录，防止把数值舍入为零误认为没有梯度。
- `selected_batches.json`、`manifest.json`、`completed.json`：抽样决策、输入/代码哈希及完整性验证。

## 结论边界

RTC 按权重 0.1 展示；模拟 noisy 按 0.05、0.1 展示，避免只看预热第一个近零权重。
这是同一原始梯度的缩放，不是重跑两个训练实验，也不是重现训练中每一步的权重。

负 cosine 只是这个批次上的局部方向冲突。应同时看加权梯度大小、错误子集，以及多个模式的一致性。
`descent_with_noisy` / `descent_without_noisy` 不包括 Adam 动量、自适应预条件、实际分组学习率、裁剪或权重衰减，
不能直接换算成实际更新或 macro-F1 提升。当前初始权重的结果也不能证明训练后期一定相同。

如果 noisy 对比梯度很小，删除它未必能提分。如果其反向分量稳定且有实际规模，才值得只去掉该 InfoNCE、
保留配对音频 CE 和真实 RTC InfoNCE 的单项试验。脚本不会自动执行这一步。

只支持记录的 `CE + RTC InfoNCE + noisy InfoNCE` 配置；若源 run 开启了预测一致性或局部结构目标，
会明确报错，不会默默丢弃这些项。所有原模型、训练入口和默认行为保持不变。
