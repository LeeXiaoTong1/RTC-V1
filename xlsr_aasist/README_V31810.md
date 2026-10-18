# V3.18.10：来源隔离、条件轮换的一阶元域训练

本版实现项目定稿方案，默认主实验为 **E2，15 轮，从官方 Omni 3B 重新训练检测器**。它是受 [MLDG](https://arxiv.org/abs/1710.03463) 与[一阶元学习](https://arxiv.org/abs/1803.02999)启发的项目配方，不是对某篇论文全部配置的逐项复刻，也没有预先验证能提高 Progress 分数。

## 开训前优化与正确性修复

本次优化仍为 V3.18.10：**读取全部 60 层特征，只在最后 16 层（第 45—60 层）训练 LoRA**。矩阵参数的 AdamW 权重衰减恢复为 `0.01`；一维参数和偏置不衰减。保持损失、曝光预算、模块峰值学习率、15 轮配方和默认 FP16 后端。校准继续使用匹配随机状态的训练模式；固定选模分数完全相同时仍保留较早轮次。

默认 LoRA 参数由 15,728,640 个减少到 4,194,304 个。前 44 层保持冻结，其输出仍参与可学习融合，但编码器不建立这些层的反向和重算路径。所有 60 层的前向与融合仍有计算和显存成本，不能把参数减少比例直接当成提速或显存节约比例。

启动阶段对实际 Train 音频计算文件和解码 PCM 身份，并将其与已知来源、官方配对统一归并。最终来源登记表 `source_registry.json` 传给正式采样和校准任务，随实验保存并校验；不再由未补全的原清单另建一套来源身份。同内容、不同 WAV 封装的录音也不能进入同一个 A/B 任务。

官方配对表的 `online_id` 允许为空：部分 Offline 的传输结果被过滤，未保留 Online。这类 Offline 是合法独立来源，继续进入模拟 Noisy 的来源池；不要求每条 Offline 都有 Online，也不补造官方 Online 或删除原始 Offline。有效配对仍合并同一来源。非空 Online ID 只有一端能找到、缺失 Offline ID、损坏 CSV 或标签／跨集合冲突仍报错。注册表及验证报告分别记录有效配对数和空 Online 记录数。

Dev 的完整来源组件保持隔离。原有固定哈希划分若因 Offline-only 来源导致某个语言／类别在一侧没有真实 Online，会按固定顺序交换完整来源组件，保持 Audit 来源数量，并记录交换。两侧必须各有真实 Online 和 Offline；不会用模拟音频填充官方 Online 指标。已有完整配对数据的正常划分不变。

稀疏诊断同时区分计划步长与实际 FP32 参数变化，并复用已有匹配前向记录 logits 变化。临时参数状态会阻止常规验证，恢复主参数后解除。FP32 参数发生变化本身仍不保证某个 BF16 LoRA 分支的计算输出改变；整体 logits 变化也不能单独归因于 LoRA。

每轮 Select 的 `selection_scores_epoch_*_step_*.npz` 和最终一次 Audit 的 `locked_audit_scores.npz` 保存已有推理结果的压缩逐条预测，包含来源身份、语言、标签、面板、协议哈希及 checkpoint 标记；补充 AUC、EER 和混淆计数，并纳入 report.zip。没有增加验证前向，也不把这些诊断另作选模目标。

**更新后请新建实验。** 内部训练配方更新为 v2，旧配方的 V3.18.10 checkpoint 不会被静默裁成 16 层后续训；恢复入口会明确拒绝。新配方创建的实验仍按下方 `--resume` 命令恢复。未保存统一来源登记表的更早实验也不能精确续训。这里修复的是实现漏洞，未据此认定实际数据已经发生重复泄漏。

## 模型与数据边界

只复用旧实验的数据清单、官方预训练资产、增强与噪声素材。**不读取旧检测器 checkpoint，不继承旧优化器、旧 best 或分数校准。** 官方 Omni 基座加载预训练权重；后 16 层 LoRA、全 60 层融合、AASIST、汇聚和分类器重新初始化。

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
| 第 2—3 轮 | 开启已配置的后 16 层 LoRA；beta 从 0 线性到 0.5 |
| 第 4—15 轮 | beta=0.5；继续联合余弦衰减 |

默认后 16 层 LoRA 全部属于编码器后段，峰值学习率为 `2e-5`；融合、后端、汇聚和分类器等其余检测模块为 `1e-4`。第 2 轮最初 200 次更新预热 LoRA 学习率；第 2—15 轮连续余弦衰减，最终为峰值的 10%，不会在第 4 轮突然切换一个新学习率。AdamW 权重衰减 `0.01`，一维参数及偏置不衰减。`parameter_inventory.json` 记录实际适配层、参数数量及各优化器组的学习率和衰减。

可用 `--lora-layers 60` 新建深度对照，此时前／中／后 20 层的 LoRA 峰值学习率分别为 `5e-6 / 1e-5 / 2e-5`，其他设置不变，包括新的 `0.01` 衰减。16 层与 60 层的构建过程消耗不同的随机数，因此即使使用相同种子，也不声称两者具有逐张量相同的后端与共有 LoRA 初值；单次深度比较不能单独证明因果。E0/E1/E2 对照必须保持相同的 `--lora-layers`，才沿用相同构建与初始化。

第 1 轮后，用固定的 12 个 Train 任务（每类四个）校准内层 alpha。候选 `0.00001 / 0.00003 / 0.0001 / 0.0003 / 0.001 / 0.003 / 0.01 / 0.03 / 0.1` 从同一预热状态测试，只看 A：全部有限、至少 9/12 损失不增、相对下降中位数在 0.5%—5%、最坏单任务增长不超过 20%。选通过的最小值，余下训练固定。实际运行表明，旧候选最小值 `0.001` 已可能超过 5% 的下降上限；补充较小候选，保留全部原门槛。没有候选通过仍明确失败，保留已完成预热 checkpoint，不暗中退成零步长。候选以配置中保存的列表为准，拒绝非正数、重复或乱序列表；报告逐个记录拒绝原因。

一次生成一个校准任务，九候选复用同一音频、A 梯度和随机状态；不同时缓存十二个完整波形任务。新增候选只增加一次性校准前向，不增加正式训练每步的编码量。仅有损失缩放溢出时可有限降低局部缩放并重试相同 A，退出时恢复主缩放器状态，不改变 alpha 门槛。校准不是泛化评估，不读取 B、Dev 或 Progress 来选 alpha。

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

首次启动需要检查语音和噪声文件完整性、构造固定协议，可能暂时显示 Startup。恢复时验证已存文件哈希，不重新解码 PCM。只生成清单、日志、checkpoint 和报告，不新建大量增强波形缓存。计算量仍受序列长度、LoRA 深度、检查点重算和顺序 A/B 影响；本地 CPU 测试不能替代 A100 的显存与速度测量。

## 更新与启动 E2

沿用现有 `sdd-v318` 环境和已经准备好的完整噪声库，不需要重新准备音素。不自动停止旧训练。以下命令按顺序执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
DATA_RUN=$(python -m w2v_v31810.source) && bash run_w2v_v31810.sh --data-run "$DATA_RUN" --experiment E2 --lora-layers 16 --epochs 15 --submit-best --upload-submission --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
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

### 旧五候选校准失败的恢复

如果旧版完成第 1 轮预热后停在 `No inner-step candidate passed`，更新仓库后使用专用恢复入口。它只接受完整预热、`alpha=None` 且尚未做任何联合更新的 checkpoint，核对原发布代码哈希和来源清单；不允许修改模型、数据、学习率等其他设置。新建相邻实验目录，复制第一轮模型、AdamW 状态、随机状态、缩放器、历史和验证协议，只更新候选列表与代码指纹；原实验完全保留。额外占用一份检测器 checkpoint 和清单的空间，不复制 Omni 或音频，也不重新跑第一轮。新目录继续原来的总轮数，并保留 submission/report 上传要求。

```bash
FAILED_RUN=$(cat exp/.latest_v31810_run) &&
RUN=$(python -m w2v_v31810.recover_calibration --run "$FAILED_RUN") &&
bash run_w2v_v31810.sh --resume "$RUN"
```

恢复后仍需要由服务器实际验证是否存在通过门槛的较小候选。已经校准或已联合训练的旧实验不能使用此入口，也不能通过改写配置文件绕过代码指纹。

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
