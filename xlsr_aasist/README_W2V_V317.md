# V3.17：LoRA + 判别出口 TFCL + 简化分类头

本版是独立的新结构训练，最多四轮。以官方原始 w2v-BERT 2.0 权重初始化，原始权重冻结；LoRA 和检测头重新初始化。`--data-run` 只复用 V3.16/V3.16.1 的数据配置、官方配对身份和固定 Dev，不读取该实验的检测器权重或 Adam。旧代码与历史 best 不变。

## 拉取、训练

使用原来的 `sdd` 环境，无新依赖，也不使用 OmniASR 专用环境或 requirements 文件。旧 GPU 任务结束后执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash run_w2v_v317.sh \
  --data-run exp/w2v_v316_tfcl_20261009_020301_4d95 \
  --epochs 4 \
  --upload-temp
```

不写 `--data-run` 时读取 `exp/.latest_v316_tfcl_run`。目录必须保留原配置和 `dev_rows.json`，但不要求旧 `last.pt` 或 Adam。预训练目录默认继承数据配置里的原始 w2v-BERT 路径；需要时用 `--ssl-path /path/to/original/w2v-bert-2.0` 指定。不会自动下载大型模型。

```bash
bash watch_w2v_v317.sh
bash show_w2v_v317.sh
bash run_w2v_v317.sh --resume "$(cat exp/.latest_v317_run)" --upload-temp
```

完整 epoch 后原子保存一次。中途终止时，下次从最近一次完整提交的 epoch 重放未提交部分。恢复使用原始配置，不把 `--epochs` 当追加轮数；不能通过恢复修改架构、采样或学习率。关闭查看窗口不会终止后台任务。

## 验证与 submission

```bash
bash run_validate_w2v_v317.sh --checkpoint best
bash run_eval_w2v_v317.sh --checkpoint best --upload-temp
```

`best` / `best_weighted` 选择本次 V3.17 内完整 Dev Weighted 最高的已训练状态；相同时先比较 Noisy，再比较 Clean。`last` 导出本次最后一轮。不存在 V3.15/V3.16 或未训练初始化回退。

```bash
bash run_eval_w2v_v317.sh --checkpoint last --upload-temp
```

可用 `--run exp/w2v_v317_...` 明确选择实验。默认导出到 `/home/ubuntu/LXT/temp/<实验名>_submission_best`；重做同一导出需指定新的 `--out` 路径，避免覆盖旧文件。默认协议和音频目录与旧版本相同，可显式设置 `--protocol`、`--audio-root`。评分仍为 P(fake)，阈值 0.5，完整录音推理。`submission_meta.json` 记录实际 epoch、checkpoint 哈希、初始化来源与无历史回退信息。

## 模型实际变化

1. 冻结全部原始编码器参数，仅在最后八层的 Q、V 投影加入 rank=8、alpha=16 的 LoRA，LoRA dropout=0.05。不使用量化或外部 PEFT 库。原始编码器仍完整参与特征提取。
2. 所有 SSL 层经过共享归一化、128 维投影和门控，用全局 softmax 层权重融合。保留四个 MultiConv 块及原来的多核时间建模，块 dropout=0.1。
3. 四块输出拼接后经过共享投影及 LayerNorm，形成 128 维逐帧判别特征。这份特征既供 TFCL 比较，也供分类，没有旧特征旁路。
4. 整段掩码注意力均值/标准差池化形成 256 维表示，接 dropout=0.2 和 Linear(256,2)。移除旧残差适配器与 512 维隐藏分类层。最终输出整条录音的真假，不引入局部伪造标签。
5. 双向八头时间软对齐和跨视图通道 CKA 移到 MultiConv 输出。两侧均参与梯度；完整有效轨迹用于时间分支，仅结构分支池化到 201 格；仍排除填充与已知缺失帧。训练分支在推理时移除。

这与 V3.15 的“MultiConv 输入处约束”、V3.16.1 的“最后一层 SSL 约束”不同。TFCL 梯度现在直接覆盖融合层、MultiConv 和 LoRA，但这本身不证明说话人或语言依赖已消除。

按公开 w2v-BERT 结构计算：LoRA 262,144 参数，检测头 2,185,115 参数；检测器合计 2,447,259，可训练 TFCL 分支另有 106,650。原来的约 1.964 亿可训练检测器参数降为约 245 万。推理仍调用完整编码器，速度和显存不按这个比例缩减。

## 采样与训练

每次更新仍是 16 个来源、每个最多三种完整视图。来源配额依据四组来源数量的平方根分配，再用逆配额权重维持每组 CE 贡献 25%。按当前 Train 分布，每批 EN-fake/EN-real/ZH-fake/ZH-real 为 4/2/7/3。EN-real 每轮平均曝光约从 6.45 降至 3.22 次，同时增加多数来源覆盖；同组期望监督预算不降低。这是曝光与覆盖调整，不保证单独解决过拟合。

每个来源内 Offline/官方 Online/模拟 Noisy 的 CE 比例仍为 10/50/40；缺少 Online 为 20/80。TFCL 按相同来源权重分配两条各半的边；缺失或无效边贡献为零，不放大另一条边。RTC 与噪声生成分布沿用 V3.16，避免这次同时更换增强机制。

默认 LoRA、检测头、TFCL 学习率均 1e-4；全新 AdamW，weight decay=0.01；5% warmup 后余弦衰减至 10% 下限；TFCL 两项权重各 0.15，前半轮逐渐升至全量。新的参数化和初始化需要新的学习率，不能与旧完整微调数值直接等同。默认最多四轮，不声称四轮必然达到最优。

## 速度、磁盘与指标

- 自动用中位、较长和最长 Train 来源试跑物理微批及 activation checkpointing，选满足显存余量的较快设置；试跑后恢复权重、Adam、随机状态。16 个来源的逻辑更新预算不变。默认微批上限 24、帧预算 14,400；更长录音完整单独处理，不裁掉后段。
- 原始冻结权重只引用现有目录；只保存 LoRA、检测头、辅助分支、Adam、RNG 和一个 best。默认预算包含原子替换，估计新增峰值约 0.59 GiB，加保留空间要求约 10.59 GiB 空闲。实际大小以 `storage_budget.json` 为准。不会建立新波形或帧缓存，不会删除历史 best。
- 每轮只打印完整 Dev 的 Clean/Noisy/Weighted 和各组 AP、AUC、EER、Recall、F1，再打印固定 Train/Dev Online 的分组平衡 CE、Recall。逐步细节进入 `training_steps.jsonl` / `details.log`。
- 固定 Train 探针最多每组 64 条官方 Online，确实来自参与训练的数据，用于观察记忆程度；不是独立测试集。Dev 保持原样，`generalization_epoch_*.json` 保存各组 CE、Recall 和正确类别概率。探针评估恢复随机状态，不改变后续训练随机序列。
- 只使用官方 Train 学习、Dev 验证和选择。Progress/最终测试只在单独 submission 命令中推理，不参与训练、阈值拟合或架构选择。

## 检查

```bash
python -m unittest discover -s w2v_v317 -t . -v
```

包含：LoRA 初始等价与冻结权重不变、TFCL 到后端和 LoRA 的梯度、无分类旁路、填充/完整长度一致性、checkpoint 重计算、不同微批的带权目标一致性、缺失 Online 权重、平滑采样、试跑回滚、真实 WAV/特征提取/小型 SSL 的中断恢复与 best/last 导出。Linux 原生 APM 边界在本地集成测试中替代；A100 的 BF16/显存/速度和官方数据最终分数仍需服务器运行确认。
