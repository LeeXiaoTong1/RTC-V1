# V3.18.10：来源隔离、条件轮换的一阶元域训练

本版实现项目定稿方案，默认主实验为 **E2，15 轮，从官方 Omni 3B 重新训练检测器**。它是受 [MLDG](https://arxiv.org/abs/1710.03463) 与[一阶元学习](https://arxiv.org/abs/1803.02999)启发的项目配方，不是对某篇论文全部配置的逐项复刻，也没有预先验证能提高 Progress 分数。

## 模型与数据边界

只复用旧实验的数据清单、官方预训练资产、增强与噪声素材。**不读取旧检测器 checkpoint，不继承旧优化器、旧 best 或分数校准。** 官方 Omni 基座加载预训练权重；全 60 层 LoRA、多层融合、AASIST、汇聚和分类器重新初始化。

保留完整音频、极短音频末端镜像补至 3920 样本、128 维融合、128 帧窗口／64 帧步长、GN/LN，以及 384 维读出（窗口均值／标准差 320 维，加窗口内部时序变化 64 维）。双 PCL、方差约束、参考编码和 EN-real 专门置信度惩罚均不参与本版训练。

官方 Online 保留原始比例及温和动态增强。Offline 只用于“动态基础增强 → 场景噪声／回声 → RTC”生成一条 Noisy，不直接分类。动态变体轮换继续启用，不新建增强音频磁盘库。

每条流每轮曝光预算按 EN-Fake、EN-Real、ZH-Fake、ZH-Real 顺序为 **9668／8000／8000／8000**，Online 与 Offline 模拟流各使用一份预算。这里是来源访问次数，不是独立录音数。每轮 1403 次正式更新，普通任务 48 条分类音频，最后一批 40 条，15 轮共 21045 次更新。没有为了元学习额外提高 EN-real 曝光。

## 条件任务与损失

一个普通任务包含 24 条 Online、24 条 Noisy，分为 A/B 各 24 条；每侧各含 12 Online、12 Noisy，并覆盖两条流 × 两种语言 × 真假的八个单元。整数余数跨任务轮换；每侧损失是八个单元的平均 CE 再平均，标签平滑 0.02。物理小批次使用整个侧的单元计数归一化。

依据官方 Offline/Online 配对、`source_id/group_id` 和已知内容标识合并原始来源，普通任务全部 48 条不能重复原始录音。通过重排原有来源／变体访问清单消除冲突，不丢弃或追加采样。现有清单不能证明说话人／生成器隔离；报告单独列出身份元数据覆盖，不把文件名或语言冒充来源域。

任务等比例轮换：换干扰、换 RTC、共同变化。处理族为 WebRTC／FFmpeg／Light，干扰为办公室、咖啡厅、雨声、脚步、键盘、回声。X 使用非留出条件，Y 使用留出条件；A=X/B=Y 与反方向轮换。完整条件轮换后再次翻转方向，防止某种留出条件永久绑定同一方向。SNR、增益、码率等非考察因素按任务／位置使用相同采样规则，不读取真假、语言、长度或损失；回声不解释为加性噪声的 SNR。

三种处理族仍可能共享编解码环节，条件差异只是局部泛化代理。官方 Online 在两侧持续提供真实接收端分类监督。

## 一阶更新的准确含义

1. 原参数上计算 `gA = grad(LA)`，缓存可训练参数和原始 gA。
2. 临时执行无动量、无衰减 SGD：`theta_tmp = theta - alpha * clip(gA, 1)`。
3. 临时参数上计算 B 的分类梯度 gB。
4. **先恢复原参数**，再将 `(gA + beta*gB)/(1+beta)` 裁剪至 1，正式执行一次 AdamW。

不建立二阶图、不复制 3B 基座、不在临时参数上继续 AdamW。临时阶段异常会恢复参数、缓冲和随机状态，主优化器矩从未被临时步骤修改。正式 AdamW 提交点内的致命设备异常仍需从磁盘 checkpoint 恢复，不能声称 GPU 更新绝对原子。

**E0 的零步长退化关系是“相同 A/B 划分、相同 beta 权重的普通分类训练”。** beta 不等于 1 时，不等于无条件对全部 48 条平均。这一差异不能藏在对照里。第 1 轮预热则对完整任务使用八单元均衡的普通 CE。

角色轮换的局部解释：忽略裁剪、随机层和自适应优化器，`gB(theta-alpha*gA) ≈ gB-alpha*HB*gA`。平均两个方向后，更新包含 `-alpha*beta/[2(1+beta)] * (HB*gA + HA*gB)`，括号对应两侧梯度内积的梯度。这只说明小步长下梯度协调的动机，实际带裁剪、随机层及 AdamW 的训练不保证每步跨条件改善。

## 固定训练配方

| 阶段 | 行为 |
| --- | --- |
| 第 1 轮 | 冻结 LoRA；新融合、后端与分类器预热，完整 48 条 CE |
| 第 2—3 轮 | 开启所有 LoRA；beta 从 0 线性到 0.5 |
| 第 4—15 轮 | beta=0.5；继续联合余弦衰减 |

LoRA 前／中／后 20 层峰值学习率 `5e-6 / 1e-5 / 2e-5`，其余检测模块 `1e-4`。第 2 轮最初 200 次更新预热 LoRA 学习率；第 2—15 轮连续余弦衰减，最终为峰值的 10%，不会在第 4 轮突然切换一个新学习率。AdamW 权重衰减 `1e-4`，一维参数及偏置不衰减。

第 1 轮后，用固定的 12 个 Train 任务（每类四个）校准内层 alpha。候选 `0.001 / 0.003 / 0.01 / 0.03 / 0.1` 从同一预热状态测试，只看 A：全部有限、至少 9/12 损失不增、相对下降中位数在 0.5%—5%、最坏单任务增长不超过 20%。选通过的最小值，余下训练固定。没有候选通过会明确失败，保留已完成预热 checkpoint，不暗中退成零步长。

一次生成一个校准任务，五候选复用同一音频和随机状态；不同时缓存十二个完整波形任务。仅有损失缩放溢出时可有限降低局部缩放并重试相同 A，退出时恢复主缩放器状态，不改变 alpha 门槛。校准不是泛化评估，不读取 B、Dev 或 Progress 来选 alpha。

## 验证与对照

新建并持久化 `validation_protocol.json`。Dev 按原始来源分为约 80% Select 和 20% Audit；语音原始关系、实际文件哈希与解码 PCM 用于防泄漏。Noisy 从真实 Dev Offline 合成，不把历史 Noisy 或 Online 当成干净 Offline。噪声按原始录音、组合素材关系和 PCM 连通分量隔离 Train／Select／Audit，不能把同一录音的切片分到两边。

固定选模分数：`S = 0.30*Online Macro-F1 + 0.35*Regular Noisy Macro-F1 + 0.35*anlmdn Noisy Macro-F1`。三个面板同一次验证加载，按来源轮换场景／处理族，每条来源不展开为所有条件的笛卡尔积。分语言保存 Real 误报和 Fake 漏检。全部使用未经临时适配的主模型，原始 softmax 和 0.5 阈值。

训练结束才加载选出的 best，在独立来源／噪声素材的 Audit 上评估一次，留出处理用 `g711_mulaw`；保存结果与 checkpoint 和协议哈希绑定。重复导出 Audit 只读取已完成的审计，不重新跑目标适配。Audit 不提供可选模的标量，也不参与选 alpha、beta、阈值。

这不是旧版 Dev Weighted，不能直接与此前的 97 分比较。历史 Dev 已参与项目开发，重新切分不会使它变成从未见过的外部测试。[模型选择协议会影响域泛化结论。](https://arxiv.org/abs/2007.01434)

| 实验 | alpha | 条件组织 |
| --- | --- | --- |
| E0 | 0 | 与 E2 相同任务、权重与角色轮换 |
| E1 | 校准值 | 与 E2 完全相同波形／配方，仅在八单元内重新分配 A/B，混合条件 |
| E2（默认） | 校准值 | 条件留出＋角色轮换 |

三者使用相同初始化种子、曝光、15 轮预算和固定选模协议。E2 相对 E0 的改善才支持元更新有价值；再优于 E1 才更支持条件组织有效。单次种子结果也不能证明稳定提升。

## 效率、内存与诊断

- 正常任务只编码这 48 条分类音频，各侧前后向顺序执行；无 PCL 参考、第二个基座或二阶图。每 200 次更新做一次额外的匹配随机状态诊断，记录 A 前后损失、delta_B、模块梯度及临时／正式更新量，不每步重复诊断前向。
- 完整波形按长度装载，默认物理上限 12 条、帧预算 9000、窗口批次 16。小批次不会改变侧损失权重；保留重叠窗口，未裁掉录音以省显存。
- Omni BF16，AASIST FP16 autocast，参数主副本、汇聚及 CE 保持 FP32。组合梯度统一反缩放后裁剪。溢出会恢复并重试同一音频任务，最多三次重试；持续失败明确退出，不偷偷消耗曝光／正式更新预算。
- CPU 默认使用可用核数一半，最多 8 个 worker，并按可见 RAM 下调。每 worker 预取 1 批、原有内存缓存总预算分摊、内部单线程，保留持久 worker。三个验证面板共享一个临时验证池。
- 每轮保存计算／供数等待时间、等待占比和 CUDA 峰值；`steps.jsonl` 保存分单元分类表现及稀疏迁移诊断，`diagnostics/curves.html` 保存 loss／选模／delta_B 曲线。终端保留每轮结果，不被下一轮覆盖。

首次启动需要检查语音和噪声文件完整性、构造固定协议，可能暂时显示 Startup。恢复时验证已存文件哈希，不重新解码 PCM。只生成清单、日志、checkpoint 和报告，不新建大量增强波形缓存。计算量仍受序列长度、全层 LoRA、检查点重算和顺序 A/B 影响；本地 CPU 测试不能替代 A100 的显存与速度测量。

## 更新与启动 E2

沿用现有 `sdd-v318` 环境和已经准备好的完整噪声库，不需要重新准备音素。不自动停止旧训练。以下命令按顺序执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
DATA_RUN=$(python -m w2v_v31810.source) && bash run_w2v_v31810.sh --data-run "$DATA_RUN" --experiment E2 --epochs 15 --submit-best --upload-submission --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

`--data-run` 优先读取已有 V3.18.9 数据路径，只读取其清单／资产。也可以明确填写该实验绝对路径。不要用 `--resume` 指向旧版本。启动会先做服务器原生小模型接口、FP16 和 MetaDG 反向检查，然后加载正式 3B。仅检查初始化可先运行 `--preflight-only`；之后使用新实验自身的 `--resume` 开始正式训练。

E0/E1 分别把 `--experiment E2` 改为 `E0`／`E1`，保持其他参数不变，顺序运行并分别记录实验路径；不要同时启动三个 3B 作业。

查看、暂停、恢复：

```bash
bash watch_w2v_v31810.sh
```

Ctrl+C 只退出只读查看器。需要暂停训练时再执行：

```bash
RUN=$(cat exp/.latest_v31810_run)
bash stop_w2v_v31810.sh --run "$RUN" --timeout 60
```

看到 `PAUSED` 后才可关机。`STOP_PENDING` 代表等待当前任务／验证完成并保存；不要强杀。恢复继续读取保存的配置、优化器、随机状态、缩放器、alpha 和任务游标：

```bash
RUN=$(cat exp/.latest_v31810_run)
bash run_w2v_v31810.sh --resume "$RUN"
```

## 评估与交付

训练完成自动导出 best，生成并上传 submission.zip 与 report.zip。导出进度单独显示音频数量。查看两个链接：

```bash
RUN=$(cat exp/.latest_v31810_run)
cat "$RUN/download_links.txt"
```

独立生成 best 提交并上传两个压缩包：

```bash
RUN=$(cat exp/.latest_v31810_run)
bash submit_w2v_v31810.sh --run "$RUN" --checkpoint best --upload-temp --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

网络失败时复用已生成文件重试，不重复 GPU 推理：

```bash
bash upload_w2v_v31810.sh --run "$RUN"
```

只重评 Select 或读取最终已锁定 Audit：

```bash
bash submit_w2v_v31810.sh --run "$RUN" --checkpoint best --dev
bash submit_w2v_v31810.sh --run "$RUN" --checkpoint best --audit
```

report.zip 不打包模型权重或音频，包含配置、来源／曝光审计、校准记录、各轮指标、训练日志、性能、loss 曲线及最终 Audit。临时下载链接会过期，必要时用上传命令加 `--force` 重新上传。

## 验收范围

`python -m unittest discover -s w2v_v31810 -t . -v` 检查数值退化关系、真实微批权重、事务回滚、AMP、同来源隔离、条件与方向轮换、精确曝光、动态增强、真实 FFmpeg/Opus、来源／素材防泄漏、固定验证、暂停恢复和交付失败重试。测试使用 CPU 与小模型；服务器原生 smoke 检查也只是接口验收，完整 Omni 3B/A100 吞吐、显存峰值和真实泛化收益必须由正式实验确认。
