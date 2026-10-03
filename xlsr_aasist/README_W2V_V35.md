# V3.5：以 Online 部署为目标的训练版本

本版本保留 w2v-BERT 2.0 + MultiConv，改变初始化、损失分配、处理条件轮换与选模方式。目标是改善未知通信条件下的真假判别；目前的工程测试不构成提分或达到官方 Weighted 97 的证据。

## 训练方案

| 项目 | 实现 |
| --- | --- |
| 新模型起点 | 通用 `facebook/w2v-bert-2.0` 预训练编码器 + 新初始化的 MultiConv。严格校验公开模型权重 SHA256，不从 V3.4 或历史反伪造 checkpoint 继续训练 |
| 受保护对照 | 根据 V3.3 已完成实验与 submission 元数据，解析产生 93.3941 提交的实际 checkpoint；只用于对照和默认导出的兜底 |
| 训练来源 | 仅官方 Train；每个独立来源每轮恰好一次，保留缺失 Online 的来源 |
| 语言与真假 | EN-fake、EN-real、ZH-fake、ZH-real 四组整轮累计分类损失系数相等，不重复叠加类别权重或重复过采样 |
| 条件预算 | 每源 Offline 10%、官方 Online 30%、Noisy 60%；缺失 Online 时将剩余 10:60 重新归一化 |
| Noisy 监督 | 两个版本采用 `0.5 × mean(CE) + 0.5 × max(CE)`；即较难版本占 75%、另一版本 25%，相等时各 50% |
| 输入 | 完整原始录音、完整 Online、两个完整 Noisy；取消额外短片段副本、CKA 和旧配对一致性项 |
| 阶段 1 | 冻结编码器，训练新后端 1 轮，最大学习率 `1e-4` |
| 阶段 2 | 更新全部 24 个编码器块及后端，最多 5 轮；最上层 `1e-6`、向下逐层乘 `0.9`，后端 `1e-5`；输入投影等非编码器块参数仍冻结 |
| 调度 | 每阶段前 5% 步数预热，再余弦下降至最大值的 10%；连续 2 个完整 joint epoch 无新候选最佳时早停 |
| EMA | 训练中维护参数移动平均。每轮仅验证一次 EMA，不把原始权重和 EMA 各推理一遍 |

“四组相等”指损失的系数预算，不代表四组实际梯度范数必然相等。稀有 EN-real 来源没有因同源多视图而被当成新增独立录音。

训练 Noisy 先混入环境噪声，再进行通信处理。每源两个版本覆盖 FFmpeg/WebRTC 两家族，在旁路通信处理、仅编码、仅降噪、仅增益、完整链之间轮换；旁路仍包含环境干扰。条件分布按语言×真假分层，避免某种处理方式成为标签线索。每轮更换噪声片段、SNR 与处理参数。

噪声按当前录音长度选择，不再用最长语音过滤整个噪声池。当前源没有足够长的噪声时，仅对噪声进行交叉淡化循环，不截短或重复语音。

## 效率与缓存

- 每源最多 4 个完整视图，取消旧版最多 8 个 full/short 视图。一个视图只参与一次显式编码器前向，沿用梯度检查点的反向重算以控制显存。
- 逻辑 batch 默认 16 个来源；每 4 个来源释放一次反向图，累计后统一裁剪梯度并更新一次。物理微批默认最多 4 条、总帧预算 2400；长录音保持完整，允许单条超过帧预算。
- 6 个 CPU DataLoader worker 懒生成 Noisy，预取深度 1，与 GPU 训练重叠；使用已有的三存储传输方式，避免大量共享内存句柄。
- GPU 优先保留激活，默认激活预算 16 GiB、预留 8 GiB，必要时卸载到 CPU。EMA 放在模型设备，只跟踪当前可训练参数；冻结阶段不逐步复制整个编码器。
- 每轮只做一次完整 Dev；初次额外测一次受保护对照。不对随机后端做一次无意义的全 Dev 验证。
- Train 缓存位于 `data/rtc_v35/train`。存储修复版在下一轮开始、尚无读取器时，先淘汰已经完成的上一轮派生缓存，只保留即将执行轮次已生成的文件。`last.pt` 保存的是整轮结束状态，恢复下一轮无需上一轮的 Noisy 文件。断点重放复用当前轮缓存。
- 每轮训练前检查完整 Noisy、原子替换 `last.pt`、可能同时晋升的两个 EMA 权重文件的磁盘峰值；旧 checkpoint 仍保留。空间不足时提前停止并写出 `storage_budget.json`。
- 固定完整 Dev 位于 `data/rtc_v35/dev`，首次生成后复用。旧训练缓存、旧短 Dev、官方音频均保留；新版本不会擅自把仍可用于复现旧 best 的缓存删除。
- 每步保存 `performance.jsonl`：数据等待、计算耗时、来源吞吐、显存峰值和卸载量，便于区分处理生成与模型计算瓶颈。

这减少了重复视图和验证开销，但**不能在未测量 A100 的情况下承诺每轮耗时**。每轮重新生成两份完整 Noisy、全 24 块反向以及首次完整 Dev 准备均有成本。完整训练最多 6 轮，总时长也不能直接与旧的一轮适配比较。

## 验证和默认导出

固定验证使用官方 Dev Online 作为 Clean；每个 Dev Offline 来源生成一个完整 Seen、一个完整 Heldout。两者使用同一噪声混合与 SNR：Seen 含 FFmpeg/WebRTC，Heldout 使用训练排除的 `anlmdn`。Train/Dev 环境噪声来源分离，四个 SNR 区间按组分配，不构造大规模笛卡尔积。Progress/Eval 不参与训练、选模或阈值调整。

选模阈值固定 0.5，`Noisy = (Seen Macro-F1 + Heldout Macro-F1) / 2`，`Weighted = 0.3 × Clean Online + 0.7 × Noisy`。每个条件内对全部样本计算 Macro-F1；这是新的完整音频代理指标，**不能与旧短 Dev 数值直接横比，也不是官方排行榜分数**。对照与新候选在同一新 Dev 上比较。不再用 Offline EN-real recall 否决更好的 Online Weighted。

- `best_model.pt`：新 Dev Weighted 的最佳模型，包含受保护对照。新训练没有胜出时，明确导出 `reference`。
- `best_candidate.pt`：新训练产生的最佳 EMA，便于审查，即使尚未超过对照。
- `last.pt`：整轮边界的原始模型、Adam、EMA、调度进度和 RNG。中途停止后恢复会重放尚未保存的这一轮。
- `report.md` 和 `epoch_N.json`：指标与选择记录；`completed.json` 绑定默认导出模型。
- 推理只使用一个模型，不做分数集成。导出是完整录音、FP32、官方协议顺序的 `P(fake)`。

## 部署与启动

在已有 V3.3/V3.4 环境 `sdd` 中运行。若 V3.4 仍在训练，先停止它；停止命令只匹配当前目录的 V3.4 进程，没有匹配时直接返回。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
python -m w2v_v34.stop --apply &&
bash setup_w2v_v35.sh &&
python -m w2v_v35.maintenance --apply &&
bash run_w2v_v35.sh --upload-temp
```

默认对照实验为 `exp/w2v_v33_20261002_004018_198a`，默认提交身份记录为 `/home/ubuntu/LXT/temp/w2v_v33_20261002_004018_198a_submission/submission_meta.json`。路径不同时用 `--source-run` 和 `--submission-meta` 指定。

启动会优先复用本地、SHA 匹配的通用预训练权重；没有时下载固定公开版本的配置、特征提取器和约 2.32 GB 权重。可用 `--pretrained-path /path/to/generic-w2v-bert-2.0` 指定已有完整目录。微调模型不能冒充通用初始化。固定版本：`6b1c0b3a98343376f0cffd7cfbbe3ce4db45d459`，权重 SHA256：`eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043`。

## 清理范围

先只看清单：

```bash
python -m w2v_v35.maintenance
```

确认过的旧实验完成/停止后，`--apply` 才删除明确命名的非最佳中间 checkpoint。保留所有 best、候选最佳、提交和配置引用的权重、各版本最新恢复点、所有 V3.5 实验以及无法证明可淘汰的文件。保护原始 91.68 权重和 93.3941 提交对应权重的 SHA256。

四个未修改的 V3.1/V3.2 旧启动/安装脚本先归档到 `exp/maintenance_archive` 再移除；脚本被用户改过则保留。不会删除旧运行时 Python 包，因为当前版本仍复用其中组件；旧 best 的推理入口保留。实际清单、删除结果与归档路径记在 `exp/maintenance_v35_*.json`。

任何相关训练、评估或缓存任务仍活跃时拒绝清理。旧版本若仅被异常终止、缺少可证明的完成/停止记录，其恢复权重可能继续保留；这是预期行为。实际释放量由打印清单决定，逻辑大小不等同于有硬链接情况下的物理释放量。

## 看进度、恢复与导出

```bash
# 同一行进度条；已完成的 Dev 结果留在上方
bash watch_w2v_v35.sh

# 单独打印所有已完成验证
bash show_w2v_v35.sh

# 后台训练保留；Ctrl+C 只退出查看器
# 主动结束 V3.5
python -m w2v_v35.stop --apply

# 恢复最近 V3.5，配置和数据身份须与保存时一致
bash run_w2v_v35.sh --resume "$(cat exp/.latest_v35_run)" --upload-temp

# 完成后导出默认选中模型并打印下载链接
bash run_eval_w2v_v35.sh --upload-temp
```

`--upload-temp` 只上传报告 ZIP 或 submission ZIP，不上传模型权重、训练音频或数据集。报告下载链接和 submission 下载链接是两种不同产物。

## 保存时磁盘空间不足的恢复

例如 `Insufficient checkpoint space: need 9.19 GiB free` 表示保存检查要求该卷至少有 9.19 GiB 空闲（包括安全余量），不表示还差 9.19 GiB。完整恢复状态含模型、Adam 与 EMA；原子保存时旧 `last.pt` 仍占空间。这是磁盘问题，不是 GPU 显存问题。

旧版在大 checkpoint 保存后才删除上一轮缓存、写 Dev 汇总，导致第三轮可能完成训练和验证却没有提交新的恢复点。修复版提前回收不再需要的派生缓存，并在保存权重前写 `epoch_N_pending.json` 和打印未提交指标；`epoch_N.json` 仍仅表示已提交结果。

针对 `w2v_v35_20261003_161443_4fb5`，训练进程已经失败退出后运行：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
python recover_w2v_v35.py --run exp/w2v_v35_20261003_161443_4fb5 --apply
```

恢复工具先验证已保存状态和元数据。若 `epoch_3_scores.jsonl` 完整，就从逐样本 logits 还原第三轮指标，无需模型推理；如果第三轮 EMA 权重已经先保存，使用同卷硬链接保留它，避免复制大文件。随后只删除已完成轮次、恢复不再读取的 V3.5 派生缓存，不删除 best、`last.pt`、`.previous`、当前轮缓存、固定 Dev 或官方音频。默认不加 `--apply` 时只检查和打印。

当输出 `V35_STORAGE_RECOVERY_READY=True` 后，可以按原配置恢复：

```bash
bash run_w2v_v35.sh --resume exp/w2v_v35_20261003_161443_4fb5 --upload-temp
```

若最后成功提交的是 epoch 2，恢复会重跑 epoch 3；已保存的第三轮 EMA 可供评估，但不能替代缺失的第三轮原始参数和 Adam 状态。是否值得继续训练应先看恢复出的第三轮指标。若空间预检仍失败，先按 `storage_budget.json` 处理剩余磁盘容量，不要删除 `last.pt` 或 `.previous`。

原 `4e32fd6` 版本的 checkpoint 可以通过明确的代码哈希白名单恢复：仅允许本次存储顺序、空间检查和指标落盘改动，配置、数据身份、训练目标及其余代码仍须一致。迁移记录为 `storage_resume_compat.json`。
