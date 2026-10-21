# V5.2：Online 为主的分类与同源跨处理对比

V5.2 保留 V5 的 OmniASR、多层融合、窗口 AASIST 和 384 维分类表示，按本次确认调整训练目标。当前默认起点改为用户报告 Progress weighted **94.61** 的 V5.1 提交模型：`w2v_v5_20261021_075307_d991/final.pt`。V5.1 的文件仍使用 V5 的版本格式。该分数是用户提供的结果，程序不读取 Progress 标签或用其选择训练参数。

初始化读取已经完成权重平均、实际供 submission 使用的 **final.pt**，核验完成记录、配置、输入规则、权重规则和校验和；不读取 `final/last.pt`、筛选 checkpoint 或 V3.18.9 祖先权重。保留所有已有 LoRA 权重，只训练最后 16 层 LoRA、融合与后端。优化器与对比队列重新建立；学习率从 V5 最终阶段的结束值继续，按本轮既定余弦规则衰减，不重启分类头预热。

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

从这次 V5.1 提交模型开始，默认**总共只增量训练 1 轮**。父模型已经训练过全部官方 Train+Dev，因此直接复用已核验的最终训练清单、来源关系和全来源覆盖预算。不会重复加入 Dev，也不重新执行输入筛选、权重筛选或 Dev Audit。`--epochs 2` 表示直接增量训练两轮，不是两个阶段各两轮。

继承父模型锁定的输入处理、RTC 上下文长度和最终权重规则，在训练前写入 `recipe_lock.json`。如果父规则是 `anchor_tail_50`，本轮最终模型为 **50% 这次 94.61 对应的 final.pt + 50% 本轮尾段平均权重**；不会再次回到 V3.18.9 做混合。如果父规则是 `tail_mean` 或 `last`，则按相应固定规则导出，不在已训练 Dev 上重新比较候选。

PCL 仍只使用原始 Train 已验证的音素资产，Dev 来源无需新音素标注即可参与分类和对比。Progress/Eval 不进入训练、配对或负样本队列。已经训练过的 Dev 不能提供本轮独立泛化评估，因此不会打印一个用于选模的 Dev best。

2026-10-21 修复：每个来源显式保留采样器安排的 PCL 分支，官方 Online 参照因对比学习加入后，不会被误当作平台 PCL 配对。旧实现会让没有音素资产的 Dev 来源进入平台 PCL，并在来源预算为 0 时触发 `Positive PCL source budget required`；未选入平台 PCL 的 Train 来源也会被错误计入其覆盖率。修复保留 Dev Offline 的 Clean-RTC／Noisy-RTC 分类与官方 Online 对比，不增加 Dev 音素依赖，不改变 60/20/20 分类权重及 PCL 系数。已安排但对齐失败的 PCL 来源仍记录零有效配对，不重新分配其损失权重。

`--parent-run` 指向旧 V5/V5.1 的完整 run 目录；`--resume` 只能指向新建的 V5.2 run。断点恢复同时恢复模型、优化器、随机状态、队列及尾部平均，已完成的阶段不会重训。默认每 100 步保存，正常 Ctrl+C 会在完成当前更新后保存；强行杀进程只能恢复至最近保存点。

兼容路线：如果**显式**传入 V3.18.9 的 run，才沿用原先的 1 轮 Train 筛选＋1 轮 Train+Dev 最终拟合；两阶段分别从该 verified best 初始化。V5.1 父模型禁止 `--phase screen`，可使用默认 `all` 或直接 `final`。

## 更新与启动

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin v5.2-online-contrastive
if git show-ref --verify --quiet refs/heads/v5.2-online-contrastive; then
  git switch v5.2-online-contrastive
else
  git switch -c v5.2-online-contrastive FETCH_HEAD
fi
git pull --ff-only origin v5.2-online-contrastive

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v52.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v5_20261021_075307_d991 \
  --epochs 1 \
  --submit-best --upload-submission
```

默认提交路径仍是 `/home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt` 与该数据集的 `wav/progress`，也可显式传入 `--submission-protocol` 和 `--submission-audio-root`。不需要提交文件时可省略最后两个开关。启动日志会打印父 checkpoint 的完整路径、tag、SHA256，以及总轮数和继承的导出规则。

对于首个训练更新就因上述零 PCL 预算错误退出的运行，更新代码后重新执行上面的启动指令，从同一个 V5.1 `final.pt` 新建 V5.2 run 即可，不用重训 V5.1。不要对这个旧失败目录直接使用 `--resume`：代码校验和已变化，恢复保护会拒绝；也不要手动改写旧配置的校验和。旧目录保留用于核查。

断点继续使用本轮生成的 V5.2 run：

```bash
bash run_w2v_v52.sh --resume /absolute/path/to/w2v_v52_run
bash watch_w2v_v52.sh
```

V5.1 截图中的 `artifact delivery incomplete` / `curl exit 22` 发生在上传阶段。只要 `completed.json`、`final.pt`、`final_config.json`、`recipe_lock.json` 和最终数据清单通过验证，即可作为父模型；上传失败不会要求重训。保留原目录，不把 submission.zip 当成 checkpoint。

`--contrastive-weight 0` 可在**新运行**中关闭对比作为对照；其余分类/PCL设置不变。恢复运行不允许修改损失、温度、队列容量等训练参数。

## 如何检查是否生效

终端显示 `CE`、`PCL-n`、`PCL-p`、`CL`、`pairs` 和 `neg`。后三个损失值均为实际计入总损失的加权值；`pairs` 是产生对比项的正对数/送入该分支的正对数，`neg` 是本步负样本比较次数。

首步队列为空且小批次没有合格负样本时，`CL=0` 可以出现；有合格历史负样本后应参与计算。`steps.jsonl` 同时记录原始对比损失、有效系数、队列数量、无负样本/无官方配对/无效处理对等跳过原因，以及间隔抽查的融合层梯度范数。不能只凭某个损失的小数位判断它是否有效。

`final/epoch_*.json` 保存整轮平均辅助损失、有效正对及负样本数量；仅显式使用 V3.18.9 父模型时还会生成 `screen/epoch_*.json`。最终是否提高 Noisy 指标，需要看相同评测条件下的结果，当前代码验证不能替代服务器完整训练和竞赛成绩。

## 本地验证

```bash
python -m unittest discover -s w2v_v52 -t . -p 'test_*.py'
```

覆盖独立公式对照的 CE 梯度、PCL 及方差分支缩放、双向对比及负样本筛选、实际 AASIST 384 维特征回传、来源隔离及缺失配对、无音素资产的 Dev 配对、重复编码计数、AMP 失败回滚、暂停/崩溃恢复、队列保存、最终 Dev 边界和单模型提交导出。新增提交模型精确加载及预测不变、上传失败兼容、完整 LoRA 继承、最终学习率、原 Train 音素作用域、单阶段预算、继承权重规则与父文件变化拒绝等测试。使用 CPU 测试与微型编码器；尚未运行服务器 3B 模型完整训练。
