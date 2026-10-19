# V4：Omni 多尺度适配、真实来源扩充与 Online/Noisy 梯度协调

从 **V3.18.13 已完成或安全暂停的 best** 新建实验，保留 Omni 3B、全层融合、重叠窗口、统计汇聚、AASIST、现有动态增强、单源单次 Noisy+RTC、XEUS 配对和官方原始 Offline↔Online PCL。

## 默认配置

| 项目 | V4 |
| --- | --- |
| 初始化 | V3.18.13 best；新 optimizer 和学习率调度 |
| MultiConvAdapter | Omni 后 16 层 MHSA 输出处；瓶颈 64；核 3/7/15/23；混合卷积核 3；输出投影零初始化 |
| 继承 LoRA | 保留其前向作用，全程冻结参数并关闭 LoRA dropout |
| 可训练部分 | 新 Adapter、原融合、AASIST、汇聚与分类器 |
| Dev EN-real | 按完整 canonical source 约 75% 并入 Train、25% 保留验证；固定 split seed 4001 |
| 逻辑 batch | 沿用父配置，通常 24 Online＋24 Noisy 分类来源；不扩大 batch |
| 微批次 | 4，帧预算 6000；只有累积完逻辑 batch 才协调与更新 |
| 每轮曝光 | 沿用父配置 source_draws_per_stream；默认每流 EN-Fake 9668、EN-Real 8000、ZH-Fake 8000、ZH-Real 8000 |
| 新增训练 | 10 轮，无早停；第 1 轮 CE＋PCGrad，第 2–3 轮逐渐加入 XEUS/PCL，第 4–10 轮完整权重 |
| 学习率 | Adapter 1e-4；原融合及后端为父配置初始率的 0.5 倍；200 update 预热、余弦衰减到 0.1 倍 |
| 精度 | 沿用父配置：通常 Omni BF16、AASIST FP16，主参数/梯度 FP32，GradScaler |
| workers | Train 4、Dev 2；可用命令行调整 |
| 部署 | 一个 detector、一个最终分类输出，P(fake) 固定阈值 0.5，无教师或 PCL 依赖 |

8000 是每轮来源曝光次数，不是独立录音数。新增来源进入既有轮换池；不是额外增加 8000 次曝光。实际来源数量、迁移清单和曝光预算写入 report。

## 三项实现的边界

**MultiConvAdapter** 在注意力后增加多尺度时间卷积残差，保持原 Q/K/V/O、LoRA 的 checkpoint 名称不变。每个时间卷积之间都屏蔽 padding，避免变长 batch 的填充污染有效帧。后 16 层仍使用 activation checkpointing；没有把早期冻结编码器加入反向传播。零初始化只保证初始推理等价，不保证训练后性能不下降。

**来源扩充** 使用父实验已经按官方配对、字节 SHA256 与解码 PCM 合并的来源登记表，重新核验官方 Dev 原始音频。只迁移 EN-real 的完整来源及其 Offline/Online，Offline 进入原有动态增强→加噪→RTC；没有 Online 的官方 Offline 仍保留。剩余 Select/Audit 的来源归属、面板条件和噪声材料保持固定。迁移后持久化新的唯一登记表，采样器、PCL、XEUS 与验证全部绑定它。原始清单、录音和父实验不被覆盖。

25% 分别从父 Select/Audit 内按来源和有无 Online 分层保留，并保证非空，所以小组实际比例存在整数取整。划分只使用固定哈希顺序，不使用模型错误或分数。历史 Dev 仍不是全新的外部测试集。

由于剩余 EN-real 数量改变，V4 SelectS 先在各面板内对 EN/ZH 的 Macro-F1 等权，再使用 0.30 Online＋0.35 Regular＋0.35 Heldout。旧的合并语言分数仍保存为参考，**不能将 V4 SelectS 与旧 SelectS 直接比较**。父模型会在完全相同的 V4 留出来源上重新评估；没有新 epoch 提升时，best 保留该父模型（Adapter 仍为零输出）。

**梯度协调** 分别累积完整逻辑 batch 的 Online、Noisy 八组均衡 CE 梯度；内积为负时，使用原始两侧梯度做对称 PCGrad，然后加入单独累积的 XEUS/PCL 梯度，最后统一 unscale、裁剪并做一次 AdamW 更新。没有恢复 MLDG 临时内层更新，也没有把特征强行做 Clean/Noisy MSE 对齐。

Online 是官方 Online 流，不是“从未经过传输的原始 Offline”。两个训练流仍保持不同来源；因此这是两域目标协调，不是原论文的逐条同源双输入复现。启动时另做最多 8 条、语言/真假均衡的**同源 Train 梯度探针**，以官方 Online 对照同一 Offline 的模拟 Noisy；只记录结果，不更新参数或使用 Dev。

PCGrad 只协调 CE。辅助梯度和 AdamW 的预处理仍可能改变最终更新方向，因此报告不会承诺最终更新绝对无冲突，也不会用 Train loss 的下降代替未知 Noisy 泛化验证。

## 更新并启动

在 V3.18.13 完成或安全暂停后执行。不要终止正在写 checkpoint 的进程。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch --track origin/v4-noisy-generalization

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist

bash run_w2v_v4.sh \
  --parent-run "$(cat exp/.latest_v31813_run)" \
  --epochs 10 \
  --submit-best \
  --upload-submission \
  --upload-temp
```

默认 Progress protocol 和音频根目录与 V3.18.13 一致。其他路径通过 `--submission-protocol`、`--submission-audio-root` 指定。

准备过程复用现有私有 XEUS/音素环境和模型下载缓存。旧 Train 的音素记录通过哈希校验后复制；XEUS 保留旧 Train 码本并复用旧来源直方图，仅对新增来源运行教师，再重算扩充训练集的近邻。新缓存放在 `assets/v4/<identity>/`，不覆盖父缓存。相同命令重跑会复用完整缓存；中断的准备保留已完成记录/分片。可以先只准备：

```bash
bash run_w2v_v4.sh --parent-run "$(cat exp/.latest_v31813_run)" --prepare-only
```

训练中查看、停止、恢复：

```bash
bash watch_w2v_v4.sh
bash stop_w2v_v4.sh
# 等待训练终端打印 PAUSED_RUN，再执行恢复。
bash run_w2v_v4.sh --resume "$(cat exp/.latest_v4_run)" --upload-temp
```

Ctrl+C 在训练阶段会请求完成当前 optimizer update 后保存。恢复采用原配置，不接受修改 epochs、数据划分、学习率或 batch 的参数。已记录的 submission 请求也会在续训完成后执行。

单独生成并上传 submission：

```bash
bash submit_w2v_v4.sh \
  --run "$(cat exp/.latest_v4_run)" \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

## 日志与验证

每一步保存 CE、XEUS margin、PCL、辅助权重 ramp、两侧 CE、梯度余弦/范数、PCGrad 是否触发、修正幅度、辅助梯度与协调后 CE 的夹角、GradScaler 状态和 CUDA 峰值分配量。终端每轮保留完整 Dev 表；`coverage_epoch_*.json` 保留配对覆盖率。

`paired_gradient_probe.json` 是小规模 Train 诊断；`parent_selection.json` 和 `selection_epoch_*.json` 用于同一来源上的父/新模型比较；`locked_audit.json` 只在完成后评估选定 best。每个分组保留 RealR、FakeR、F1、AUC 等指标以及预测文件；不要仅根据 EN-real recall 判断成功。

可用 `--gradient-coordination none`、`--adapter-layers 0`、`--dev-real-train-fraction 0` 关闭对应变化做对照。所有 V4 模式均冻结继承 LoRA；`--mode baseline` 关闭 XEUS margin但保留 PCL，与 V3.18.13 的该选项一致。改变数据比例会改变验证集合，不能直接比较不同划分的 SelectS。

本地测试覆盖真实 AASIST＋小编码器的初始等价、Adapter 反传、LoRA 冻结、checkpoint 重计算、padding 隔离、梯度公式与微批次等价、实际 WAV/PCM 来源隔离、实际采样曝光、新来源音素/XEUS 增量准备、暂停续训及单模型导出。没有在本地运行完整 Omni 3B GPU 训练；速度、峰值显存和泛化收益仍需要服务器实测。新增梯度缓冲与 Adapter 会增加开销，保持逻辑 batch 与微批次预算不扩大。

方法参考：[MultiConvAdapter](https://arxiv.org/html/2510.24852v1)、[DPDA 梯度协调](https://arxiv.org/html/2509.20682v1)。本版是在现有 Omni RTC 检测器上的适配，并非对论文数据集和训练设置的原样复现。
