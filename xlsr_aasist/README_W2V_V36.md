# V3.6：保留已验证 best，修正最后的真假分类层

目标仍然是官方 Weighted >97；实现和测试通过不代表已达到该成绩。
这一版检验一个具体方向：现有模型的表征是否足以支持更好的分类边界。
它不从 V3.5 的低分权重继续训练，也不重新初始化前后端。

## 起点和改动

- 通过 V3.3 完成记录、提交元数据和 checkpoint SHA256，解析此前默认导出对应的已验证 best。默认来源为 `exp/w2v_v33_20261002_004018_198a`。
- 保留 w2v-BERT 2.0 + MultiConv 的全部表征；只修正最后 `Linear(512,2)` 的 1026 个参数。冻结编码器、MultiConv、池化、分类头第一层及 SELU。
- 按原推理方式，使用完整音频、FP32、相同长度的微批次提取最后线性层的输入。每条输入只做一次完整前向，缓存512维向量和原始 logits。
- 只用官方 Train。每个原始来源的总损失预算固定，原始 Offline/Online 合占50%，已有 noisy_a/noisy_b 各占25%；没有配对 Online 的来源，其 Offline 占50%。同一录音多个处理版本不会重复增加来源权重。
- 比较两个预先确定的分类器：按来源统计的 real/fake 均衡；按来源统计的 EN-real、EN-fake、ZH-real、ZH-fake 四组均衡。
- 两者均使用交叉熵和偏离原分类器的 L2 惩罚，完整批量 L-BFGS 求解，最多200次迭代。没有大模型反向传播、EMA、Adam 状态或逐轮音频再生成。
- λ 候选固定为0.1和1.0，只在按原始音频 SHA 分组的 Train 80/20 划分上选择，然后用全部 Train 重新拟合。相同录音的 Offline/Online/noisy 和重复内容 ID 必须在同组。旧编码器已经见过 Train，因此该留出只用于分类器调参，不是独立域外测试。

完整模型的表征如果已经丢失判别信息，最后一层不能把它恢复；这一版并不承诺单独补齐至97的差距。它提供一次计算和空间开销较小、可以直接提交的优化。

## 输入、评估与默认模型

训练复用 V3.3 已有两个完整 noisy 版本；评估复用 V3.5 固定完整 Dev 缓存。**不会生成新的音频缓存。**

Clean 只统计官方 Online；Noisy 为 Seen/Heldout 两个条件的 pooled Macro-F1 平均；Weighted = 0.3 Clean + 0.7 Noisy。
原 best 和新候选都在本次同一批特征上评估，阈值0.5。
这些是本地模拟指标，不能直接和旧短缓存指标或排行榜数字混算。

默认替换 best 同时要求：

1. Weighted 至少提高0.2个百分点。
2. Noisy 不下降。
3. Clean 下降不超过0.1个百分点。
4. Seen 与 Heldout 的 EN-fake recall 各下降不超过0.5个百分点。

未收敛、分数非有限或不满足条件时保留原 best。报告明确写 `baseline_fallback=True`；这种结果不算新方法取得收益。
两候选在 Dev 上选模仍有选择偏差，最终需官方提交确认。

## 部署、清理和运行

在 Ubuntu 服务器执行：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
python -m w2v_v35.stop --apply &&
bash cleanup_w2v_v36.sh --apply &&
bash setup_w2v_v36.sh &&
bash run_w2v_v36.sh --upload-temp
```

清理复用已部署的保护逻辑：删除已停止 V3.5 的 `last.pt`、旧逐轮 Train 派生音频，以及经现有维护脚本确认可删的过时内容。
保留各 best checkpoint、提交来源和报告、原始数据、V3.3 完整 Train noisy、固定完整 Dev 和特征提取器；后续仍需要这些输入。
不带 `--apply` 可先查看清单。不会自动停止其他版本的进程。

新版本主要时间花在**一次 Train/Dev 特征提取**。随后两种分类器共享这些特征，求解不再跑编码器。
按38,660个来源、最多四种条件估算，Train向量约0.30 GiB；加Dev、索引和日志通常仍在约1 GiB以内，启动时按实际条数检查两组总空间需求。
不再每轮新增约32 GiB音频缓存。最终只保存小型 `best_patch.pt`，它必须与 SHA 绑定的原 best 一起使用，不能删除原 best。
实际服务器耗时、显存峰值和提分需运行后确认。

```bash
# 重新查看；Ctrl+C 只退出查看器，后台工作继续
bash watch_w2v_v36.sh

# 打印当前/完成结果，终端保留所有候选的结果
bash show_w2v_v36.sh

# 中断后接着提取：已提交的向量直接复用，最多重算未提交的尾段
bash run_w2v_v36.sh --resume "$(cat exp/.latest_v36_run)" --upload-temp

# 真正停止 V3.6
python -m w2v_v36.stop --apply
```

每256条向量提交一次游标，避免每个小批次反复重写不断变大的日志。
底层数据传输使用 NumPy，避免之前多进程传递大量 tensor 文件描述符导致的 ancdata 错误。
如果缺少原有完整缓存，程序报出具体缺失信息，不会擅自重建几十GiB缓存。

## 生成官方提交

```bash
bash run_eval_w2v_v36.sh --upload-temp
```

完成后打印 `SUBMISSION_ZIP` 和 `TEMP_DOWNLOAD_URL`，后者是 submission.zip 下载链接。
默认读取 Progress 的无标签协议；不使用 Progress/Eval 标签或音频做训练、调参。
可通过 `V36_RUN` 指定某次V3.6，`EVAL_PROTOCOL` / `EVAL_AUDIO_ROOT` 指定官方无标签 Eval；`SUBMISSION_DIR` 指定输出目录。
导出按协议顺序输出 P(fake)，加载原 best 后应用已选分类层，仍然是单模型完整音频推理。

训练报告ZIP含原始模型及候选逐样本Dev分数、训练来源覆盖、Train调参划分摘要和选择理由，不包含音频、特征矩阵或模型权重。
若导出打印 `SUBMISSION_BASELINE_FALLBACK=True`，提交使用原 best；不要把它解释为 V3.6 提升。

## 验证范围

安装脚本运行本地小型真实 w2v-BERT/MultiConv 测试：固定特征重放、断点恢复、缓存身份验证、同源分组、梯度、平衡损失预算、Dev不参与拟合、退化回退、patch完整性、原模型与新分类层的完整音频导出、协议顺序与分数方向。
CPU测试验证实现和保护机制；不替代 A100 全量运行和官方成绩。
