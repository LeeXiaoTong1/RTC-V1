# V5.2：Online 为主的分类与同源跨处理对比

V5.2 保留 V5 的 OmniASR、多层融合、窗口 AASIST 和 384 维分类表示，按本次确认调整训练目标。仍从 V3.18.9 的 verified best 初始化，保留所有已有 LoRA 权重，只训练最后 16 层 LoRA、融合与后端。不会载入旧优化器。

## 已固定的变化

| 项目 | V5 | V5.2 |
|---|---:|---:|
| 官方 Online 分类权重 | 25% | **60%** |
| 模拟 Clean-RTC 分类权重 | 25% | **20%** |
| 模拟 Noisy-RTC 分类权重 | 50% | **20%** |
| Clean-RTC ↔ Noisy-RTC PCL | 继承父版本系数 | **原系数 × 0.2** |
| 官方 Online ↔ Clean-RTC PCL | 继承父版本系数 | **保持不变** |
| 同源官方 Online ↔ Noisy-RTC 对比 | 无 | **新增** |

这些百分比是分类损失权重，不是音频数量比例。每路内部仍对 EN/ZH × Real/Fake 四组等权，来源抽样、增强配方及来源隔离保持 V5 的规则。PCL 系数从父配置读取，不假设每个父实验都相同；例如原 noise=0.15、platform=0.045，则新系数为 0.03、0.045。noise 分支的方差正则也乘 0.2，platform 方差正则不被间接放大。

## 对比学习的具体定义

正对是同一 canonical source 的 **官方 Online 与 Offline 生成的 Noisy-RTC**，不拿另一段相似录音冒充同源。两端均反向传播。使用分类器前、最后一次 dropout 前的真实 384 维表示，L2 归一化后计算双向 InfoNCE；未启用 temporal_stats 的兼容父模型为 320 维。没有另加推理网络或独立检测器。

负样本须满足三个条件：不同 canonical source、同语言、相反真假标签。同类不同来源不作为负样本，以避免训练目标变成区分说话人或录音身份。标签仅使用本阶段允许训练的数据标签。

每个 A 侧逻辑 batch 每组最多选择 3 个来源，优先复用已安排的官方 PCL 参照。B 侧仍只有分类及继承的置信度项；这里的 A/B 是来源隔离分组，不包含临时参数更新或元学习。缺失官方 Online 的来源继续参与分类和可用 PCL，不补造正对，也不替换来源。

长度分桶可能使一个物理小批次没有异类负样本。因此保留容量受限的训练特征队列：每个「语言 × 标签 × 视图」最多 32 个不同来源，总计最多 256 条表示。历史表示停止梯度，当前正对两端保留梯度；排除同源历史记录。当前小批次的合格异类表示也可作为负样本。队列只在一次完整优化成功后更新，失败重试不推进队列。

默认 `contrastive_weight=0.01`、温度 `0.2`，前 100 次成功更新线性预热。它们是尚待实验验证的初始设置，不代表已标定的最优值。四组按已计划的来源预算归一化；缺失配对不提高其他来源权重。继承 A/B 各占一半的总目标，因此对比项在总损失中的实际贡献还有 A 侧的 0.5 因子。日志中的 `CL` 已包含这个因子。

所有分类与辅助目标复用每个视图的一次编码。官方参照现在还要经过分类后端；没有可复用参照的来源需要额外编码，因此不能据此承诺训练速度提高。队列只保存小型表示，不保存音频或大模型计算图；GPU 上的队列每步合并传输一次。推理和权重平均完全不使用队列。

## 训练流程与数据边界

默认依旧是 **1 轮 Train 筛选训练 + 1 轮 Train+全部官方 Dev 最终拟合**，另有输入方案及权重方案的评估时间。两阶段均从相同 V3.18.9 best 开始，最终阶段采用已经锁定的输入与权重平均规则。不会先跑完筛选再连续追加一轮到筛选权重上。

筛选阶段保留未训练 Dev 用于选择及一次独立 Audit；最终阶段 Dev 加入 Train 后，不再用它选择 best。PCL 仍只使用父 Train 已验证的音素资产，新增 Dev 来源无需新音素标注即可参与对比。Progress/Eval 不进入训练、配对或负样本队列。

初始化用 V3.18.9 best；`--resume` 只能传入本版 V5.2 run，不能用它接着训练旧 V5 目录。模型、优化器、随机状态、队列及尾部权重平均一同恢复；已完成的阶段不会重训。默认每 100 步保存，正常 Ctrl+C 会在完成当前更新后保存；强行杀进程只能恢复至最近保存点。

## 更新与启动

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch v5.2-online-contrastive
git pull --ff-only origin v5.2-online-contrastive

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v52.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v3189_20261016_183410_a437 \
  --submit-best --upload-submission
```

此命令沿用父配置中的提交路径。若原实验未保存提交路径，添加 `--submission-protocol` 和 `--submission-audio-root`。不需要提交文件时可省略最后两个开关。

只想先完成筛选阶段，在新运行命令中加 `--phase screen`。之后用同一个 V5.2 run 执行 `--resume /absolute/path/to/w2v_v52_run --phase final`，开始既定最终拟合。断点继续默认 `--phase all`：

```bash
bash run_w2v_v52.sh --resume /absolute/path/to/w2v_v52_run
bash watch_w2v_v52.sh
```

`--contrastive-weight 0` 可在**新运行**中关闭对比作为对照；其余分类/PCL设置不变。恢复运行不允许修改损失、温度、队列容量等训练参数。

## 如何检查是否生效

终端显示 `CE`、`PCL-n`、`PCL-p`、`CL`、`pairs` 和 `neg`。后三个损失值均为实际计入总损失的加权值；`pairs` 是产生对比项的正对数/送入该分支的正对数，`neg` 是本步负样本比较次数。

首步队列为空且小批次没有合格负样本时，`CL=0` 可以出现；有合格历史负样本后应参与计算。`steps.jsonl` 同时记录原始对比损失、有效系数、队列数量、无负样本/无官方配对/无效处理对等跳过原因，以及间隔抽查的融合层梯度范数。不能只凭某个损失的小数位判断它是否有效。

`screen/epoch_*.json` 与 `final/epoch_*.json` 保存整轮平均辅助损失、有效正对及负样本数量。最终是否提高 Noisy 指标，需要看相同评测条件下的结果，当前代码验证不能替代服务器完整训练和竞赛成绩。

## 本地验证

```bash
python -m unittest discover -s w2v_v52 -t . -p 'test_*.py'
```

覆盖独立公式对照的 CE 梯度、PCL 及方差分支缩放、双向对比及负样本筛选、实际 AASIST 384 维特征回传、来源隔离及缺失配对、无音素资产的 Dev 配对、重复编码计数、AMP 失败回滚、暂停/崩溃恢复、队列保存、最终 Dev 边界和单模型提交导出。使用 CPU 测试与微型编码器；尚未运行服务器 3B 模型完整训练。
