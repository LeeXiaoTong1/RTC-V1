# V3.18：OmniASR W2V 3B + SSL-AASIST

V3.18 从公开 Omni SSL 权重、全新 LoRA 和全新检测后端开始。`--data-run` 只复用官方音频、协议、噪声和完整 Dev 清单，不加载 V3.15–V3.17 的检测参数、优化器或特征缓存。旧版本文件不修改。

默认方案 C3：完整音频编码一次 → 共享 AASIST 提取区域证据 → 限幅汇聚 → 整条音频二分类；官方 Online 独立分类，Offline 仅用于生成两种含噪通信视图。同源双视图使用风险分类损失，不使用 TFCL、语言对抗、音素或重建损失。

## 对 V3.17 结果与输入分布的判断

截图第三轮 Weighted 从 92.492 提高到 93.366，不能据此判断第三轮整体性能已经下降。但固定 Train Online 接近 100%，Train CE 从 0.0008 降到 0.0002，而 Dev Online CE 从 0.3410 升到 0.3648，提示置信度与泛化之间出现问题。EN-real 仍弱；这些现象不能单独证明 fake 在训练损失中权重过高。

本地官方 Train 协议及配对表审计如下；服务器启动后会再次核验其实际清单，保存 `input_audit.json`。

| 原始条件 | EN fake | EN real | ZH fake | ZH real |
|---|---:|---:|---:|---:|
| Offline | 7,499 | 1,500 | 23,698 | 5,963 |
| Online | 7,133 | 1,422 | 23,102 | 5,468 |

V3.17 按来源数量平方根分配 16 个来源：EN-fake 4、EN-real 2、ZH-fake 7、ZH-real 3。但 CE 已经给四个交叉组各 25%，即 fake/real 各 50%。抽样次数不等于损失权重。缺少官方 Online 配对会改变条件参与比例；按完整清单估计 Offline/Online/Noisy CE 约 10.52%/47.39%/42.09%，并非服务器某一轮的实测统计。

V3.18 每次逻辑更新：

| 数据流 | 来源数 | 每来源版本数 | 音频视图数 | 损失预算 |
|---|---:|---:|---:|---:|
| 官方 Online | 16 | 1 | 16 | 50% |
| Offline 生成的 Noisy | 16 | 2 | 32 | 50% |

每条流中 EN-fake、EN-real、ZH-fake、ZH-real 各 4 个来源。因此一轮更新共 48 个完整音频视图，fake/real 各 24；语言也等量。32 个抽样槽位可能包含相关的 Offline–Online 录音，不把它们虚报为 32 个独立说话人或独立内容。各交叉组在各流的损失预算为 12.5%。Online 不因缺少配对而被丢弃；不直接分类原始 Offline。

组内采用连续循环洗牌：每组遍历完再洗牌，轮次边界不重置小组。完全相同的音频文件按哈希去重，冲突标签直接报错。默认约 2417 次更新/轮；均衡抽样仍会重复较少的 EN-real，不声称它创造了新的真实说话人。每轮保存实际覆盖、重复次数、视图数、各组 CE/正确数及处理机制比例，便于识别重复学习问题。协议中的 en/zh 只是分组标签，不保证每条录音是纯单语言。

## 模型与防过拟合设计

- 默认前端使用 **omniASR-W2V-3B SSL**，60 层、2048 维，取最后一层；不是 CTC/LLM 版。输入为 16 kHz 完整波形，逐条录音进行 layer normalization。CNN 前端按真实长度执行，Transformer 使用长度掩码，后端只接收有效帧。保留 `--omni-size 1b` 选项，显式选择时使用 48 层、1280 维。
- 原始编码器全部冻结。最后 16 层（3B 中第 45–60 层）Q/K/V/output 四个投影增加 LoRA，rank=16、alpha=32、dropout=0.05，共 4,194,304 个可训练参数。冻结参数不会进入 Adam。
- 后端取竞赛官方 SSL-AASIST，3B 对应 2048→128 投影、原卷积/图注意力/BN/读出结构；保留原公式，不替换为 GN/LN。暴露 dropout 前的 160 维表示供汇聚。后端和汇聚共 583,499 个参数，总可训练量 **4,777,803**。选择 1B 时输入投影自动使用 1280→128。
- 完整 SSL 序列分成 128 帧、步长 64 帧的重叠窗口，尾部补最后一个完整窗口，短录音直接使用其有效长度。所有窗口共享 AASIST，SSL 不因窗口重复计算。窗口只提取证据，不单独贴 fake/real 标签。
- 根据帧覆盖次数修正窗口基础权重；贡献网络输出 `.5 + sigmoid(...)`，末层零初始化，因此初始等价于覆盖修正后的平均汇聚。倍率限制在 0.5–1.5，归一化后相对基础权重的理论范围为 1/3–3，不能误称归一化后的范围仍是 0.5–1.5。
- 默认第 1–2 轮仅训练后端；第 3–10 轮训练后端、LoRA、贡献网络。联合阶段固定所有 AASIST BN 的 running mean/variance，保留 affine 可训练；BN 缓冲区随 checkpoint 保存。
- 原始 SSL 的 dropout 始终关闭；联合阶段只开启 LoRA dropout 及 AASIST 自身 dropout。使用 AdamW、weight decay=0.01（不施加到 bias/一维归一化参数）、梯度裁剪=1、label smoothing=0.02。日志同时输出未经平滑的 CE，不能用目标损失下限误判拟合状态。
- 预热后端学习率 1e-4；联合阶段后端 3e-5、LoRA 1e-5、贡献网络 1e-4，独立预热与余弦下降。前两轮关闭风险加权，联合第一轮从 0 平滑升到 0.25。

同源 Noisy 两个损失为 L1、L2，最终使用 `0.75 * mean(L1,L2) + 0.25 * max(L1,L2)`。较难版本占 62.5%，较容易版本仍占 37.5%；不只追逐最差版本，不要求不同说话人或语言特征被强行抹平。相等时两边梯度各半。

AASIST 和较少可训练参数并不保证不发生过拟合。因此仍记录 Train/Dev 同口径指标，保存最佳与最后状态，并提供可选早停。

## 增强与泛化边界

同一 Offline 来源每次生成两个完整视图，保证处理家族不同：FFmpeg DSP、WebRTC NS/AGC、轻量滤波、旁路四选二。两个视图分别抽取噪声、信噪比、处理参数；不同随机种子不保证永不抽到同一噪声文件。

连续噪声/短事件/混合约 60%/20%/20%。连续噪声保留自然长片段，只有原噪声长度不足时拼接并做短交叉淡化，不把所有背景都切成短事件。预热 SNR 10–25 dB，联合阶段 5–25 dB；Opus/无编解码约各 50%，与处理强度独立采样。每条录音创建新的处理状态，检查输出长度与有限数值。

另外在 Dev Offline 上固定抽取每组 64 个来源，使用训练中未使用的 G.711 μ-law 与 ANLMDN 做诊断，最多 512 视图。它不计入 Weighted，也不参与自动选模。真实测试集的通信与噪声分布仍未知；更广增强不等于复现测试集。

## 评估、选择与空间

- 每轮保留历史完整 Dev Online/Seen/Heldout 清单，逐条完整音频评估，不抽样 Dev。打印 Clean/Noisy/Weighted，以及各组 AP/AUC/EER/fake Recall/real Recall/F1。
- Train 改为固定较大抽测：每条流、每个交叉组最多 512 个来源。默认 2048 Online + 4096 Noisy = **6144 视图**，固定样本和增强参数，明确标为 PROBE，不冒充全量 Train。可以用 `--train-probe-per-group` 调整。
- 在官方配对表与重复音频归并后，按来源隔离 Dev 的 80% 选模 / 20% 校准。每轮 full Dev 只用于历史可比展示；`best` 使用 80% 来源的原始 Weighted，Noisy/Clean 用于同分排序。完整来源及其 Online、Noisy 版本不会跨两部分。
- 训练结束、checkpoint 固定之后，分别为 best/last 在保留的 20% Dev 上拟合一个所有语言/条件共享的正斜率仿射分数校准。它调整概率与边界，不改善排序，不使用 Progress。导出默认校准；`--raw` 可导出原始分数。校准后的全 Dev 含拟合样本，报告明确分开 select/calibration 指标，不将其宣称为独立验证成绩。
- 默认完整训练 10 轮，2 轮预热 + 8 轮联合，`--patience 0`，不会因早停自动缩短。`best` 始终保留本次选模最佳状态；可选 `--patience 3`：联合阶段连续 3 轮没有超过 0.02 个百分点的选模改善则早停，且至少完成 3 轮联合。较小改善仍能更新 best，早停计数与 best 身份分别记录。
- 只保存一份 `last.pt` 事务文件，内部包括当前/最佳的 LoRA、后端、全部 BN 缓冲区、Adam、随机状态。冻结的公开 SSL 权重只引用。中断从最后完整轮次继续，未提交轮次会重跑。原始逐步日志可能保留重跑记录，图表按已提交状态生成。模型规格、维度和哈希同时校验，1B 与 3B 检测 checkpoint 不可混用。
- 不生成增强音频或特征磁盘缓存。公开 3B 文件为 12,256,910,184 字节（约 **11.42 GiB / 12.26 GB**），校验官方固定 SHA256，支持断点续传；已存在时直接引用。环境已安装时，下载前建议至少 **25 GiB 空闲**，下载器要求剩余下载大小加 12 GiB 预留空间。新环境和依赖安装还需另外预留空间。部分权重/Adam 通常远小于 1 GiB，保存前保留 10 GiB 空闲；不要删除既有 Dev 波形清单所依赖的音频。
- BF16 SSL、FP32 AASIST，冻结前端、最后 16 层重计算、持久 CPU 增强 workers、有限 RAM LRU、流式梯度累积。逻辑48视图与物理 microbatch 分开。完整长音频仍可能增加耗时/显存，未给出未经 A100 实测的速度保证。

## 第一次安装与运行

已有训练任务运行时先等它结束。新版不会自动杀掉旧训练。以下命令在服务器执行；独立环境避免改变旧 `sdd`。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist

conda create -n sdd-v318 python=3.11 -y
conda activate sdd-v318
bash setup_w2v_v318.sh
bash prepare_omni3b_v318.sh
python -m w2v_v318.preflight --omni-size 3b --weights
```

`sdd-v318` 若已经存在，跳过创建，直接激活。环境固定 torch/torchaudio 2.8.0 cu126、fairseq2 0.6.0、omnilingual-asr 0.2.0；机器需要兼容 CUDA 驱动、FFmpeg 和 WebRTC 包的构建环境。安装脚本检查实际原生处理库；`--weights` 额外加载所选的真实前端、运行正反向、确认 LoRA 梯度并打印显存。该预检查使用短合成音频，不能代替真实长音频下的显存与速度测试。任何预检查失败应先修复，避免启动长训练。

如果公开 3B 已下载，避免重复保存：

```bash
bash prepare_omni3b_v318.sh --checkpoint /absolute/path/omniASR-W2V-3B.pt
```

复用已存在 V3.16 的数据身份，先审计、再训练：

```bash
bash audit_w2v_v318.sh --data-run exp/w2v_v316_tfcl_20261009_020301_4d95
bash run_w2v_v318.sh \
  --data-run exp/w2v_v316_tfcl_20261009_020301_4d95 \
  --omni-size 3b \
  --epochs 10 \
  --upload-temp
```

不指定 `--data-run` 时尝试最近的 V3.17、V3.16 数据配置。官方 Dev 配对表通常自动找到；特殊路径加 `--dev-pairs /absolute/path/dev_offline_online_pairs.csv`。3B 默认物理 microbatch=4、frame budget=2400、eval batch=8、workers=4；1B 为 8/4800/16。可以在第一次启动前调整 `--microbatch`、`--frame-budget`、`--eval-batch`、`--workers`，保持逻辑来源数量和损失预算不变。联合阶段 BN 固定；预热阶段改变物理 batch 仍可能改变 BN 统计，不能保证不同物理配置完全等价。3B 冻结参数仍参与前向和部分反向计算，不承诺与 1B 相同的速度或更高成绩。

若显式使用旧的 1B 方案，执行 `bash prepare_omni1b_v318.sh`、`python -m w2v_v318.preflight --omni-size 1b --weights`，启动训练时加 `--omni-size 1b`。切换到 3B 必须开始新 run，不能通过 `--resume` 把已有 1B 检测器升级成 3B；公开权重和后端维度均不相同。

查看与恢复：

```bash
bash watch_w2v_v318.sh
bash show_w2v_v318.sh
bash run_w2v_v318.sh --resume "$(cat exp/.latest_v318_run)" --upload-temp
```

恢复使用记录的配置，恢复命令中的其他模型超参数不会覆盖它。每轮日志集中显示 Dev、Train PROBE、诊断面板；逐批详情存 `steps.jsonl`，图表在 run 内 `diagnostics/curves.html`，另有 CSV。报告 zip 包含诊断及清单，不含权重或音频。上传失败保留本地文件。

完成后生成本次 V3.18 的 submission，绝不退回旧模型：

```bash
bash run_validate_w2v_v318.sh --checkpoint best
bash run_eval_w2v_v318.sh --checkpoint best --upload-temp
# 原始分数对照：加 --raw；最后一轮：--checkpoint last
```

`submission_meta.json` 记录 V3.18 的实际轮次、checkpoint SHA256、公开前端与校准参数。`best` 是本次选模子集上的最好，并非承诺排行榜最好。

## 预置消融与验证范围

`--variant C0` 整段 AASIST + 双视图均值；C1 区域汇聚 + 均值；C2 整段 + 风险；默认 C3 区域汇聚 + 风险。提供可比开关，不自动启动四次训练。所有变体均使用独立 Online 流和相同四组预算。

CPU 测试使用实际 WAV 文件（合成信号）、实际 AASIST 和小型可训练编码器，原生库边界有明确替代：检查 LoRA 梯度、冻结参数、覆盖、抽样预算、microbatch 梯度、BN 保存、断点恢复、源隔离、跨进程评估、best/last 及 submission 导出。新增检查覆盖 3B 选择、权重与规格错配、下载复用及剩余空间、60 层中最后 16 层的 LoRA 形状、2048 维 AASIST 的真实正反向、不同前端 checkpoint 混用拒绝。执行 `python -m unittest w2v_v318.test_core w2v_v318.test_assets -v`。完整公开 3B、Linux fairseq2/WebRTC、A100 显存与性能需上述服务器 preflight 及正式实验确认，不能把 CPU 测试当作完整训练验证。

来源：[Meta OmniASR](https://github.com/facebookresearch/omnilingual-asr)、[3B 模型卡](https://huggingface.co/facebook/omniASR-W2V-3B)、[3B 固定权重身份](https://huggingface.co/facebook/omniASR-W2V-3B/blob/b34b7fba5ac95adbffd9e60813a4425cb0fc6242/omniASR-W2V-3B.pt)、[RTC 官方后端](https://github.com/JunXue-tech/RTC-SDD/blob/main/xlsr_aasist/model/model.py)、[SSL-AASIST 上游](https://github.com/TakHemlata/SSL_Anti-spoofing)。后端许可见 `THIRD_PARTY_LICENSES/SSL_AASIST_V318.txt`。
