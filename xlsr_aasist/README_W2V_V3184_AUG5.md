# V3.18.4：Online / Offline 五视图增强、均衡监督与注意力统计池化

本次完整入口是 `run_w2v_v3184_aug5.sh`，模型版本仍为 **V3.18.4**，配方身份为 `v3184_aug5_asp_v1`。它继承已保存的 **V3.18.3-R2 best**，保留 Omni 全 60 层融合、最后 16 层 LoRA、AASIST 和 C3。新配方、运行指针与 checkpoint schema 独立；既有 R2 和旧 V3.18.4 文件没有被覆盖。

## 数据扩充与比例

Online、Offline 的每个原始录音分别定义 5 个可复现视图，真假及中英文使用同一增强策略：

| 槽位 | 处理 | 设置 |
|---|---|---|
| 0 | 原始波形 | 保留原始分布 |
| 1 | 轻微减速 | 0.970–0.990 倍；有理重采样，音高随速度一起变化 |
| 2 | 轻微加速 | 1.010–1.030 倍；不使用声码器重合成 |
| 3 | 温和频响变化 | 低／高频分量各 ±1–3 dB |
| 4 | 时间上下文 | 保留 85%–95%，至少 2 秒；过短或裁掉过多能量时改为动态增益包络 |

这些是本项目的起始参数，尚非 GPU 实验验证的最佳值。只做整体音量缩放会大部分被当前输入 LayerNorm 抵消，因此视图采用速度、频响和时变处理。裁剪保留至少 65% 原始能量；它不能保证所有局部真假线索都保留，需要后续消融验证。

按现有来源审计，real 的可用视图如下：

| 原始流 | 原始录音数 | 原始 + 4 个增强视图 |
|---|---:|---:|
| EN-real Online | 1,422 | 7,110 |
| EN-real Offline | 1,500 | 7,500 |
| ZH-real Online | 5,468 | 27,340 |
| ZH-real Offline | 5,963 | 29,815 |

这是 **5 倍虚拟训练视图**，不代表增加了 5 倍说话人或独立来源。Online 和 Offline 还有同源关系，不能相加当成独立录音来源。所有视图在内存按需生成，使用原始来源 ID、流类型和槽位决定随机种子，不预生成海量 WAV。实际池会按原始音频指纹去重，最终计数以运行中的 `input_audit.json` 为准。

采样改为 **EN-fake / EN-real / ZH-fake / ZH-real = 4 / 3 / 5 / 4**，Online 和 Offline 两条流采用相同比例。每步仍有 16 条 Online + 16 个 Offline 来源各两份 Noisy，共 48 条视图；两条流独立抽样，可能抽到同源 Online/Offline，日志另计实际独立来源数。

- 中英文抽样位置为 7:9，即 43.75% / 56.25%；real 的中英文位置为 3:4。
- 四组分类损失仍各 25%，Online / Noisy 各 50%；C3 两份 Noisy 的风险系数仍为 0.25。
- 先轮转原始来源，再轮转同一来源的 5 个视图；组内一个逻辑批次不重复同一原始录音。不会把 5 个视图当作 5 个独立来源同时凑一个批次。
- 每轮仍按原始来源确定步数，当前为 2,417 步，不增加到 5 倍。EN-real Online 每轮 7,251 次抽取，原始来源平均 5.10 次，但优先使用不同视图；原来 4/2/7/3 为每轮 4,834 次抽取。
- 6 轮预算足以覆盖两种语言、两条流的 real 全部 5 个槽位。ZH-fake 池更大，6 轮不保证遍历其全部 5 个槽位；每轮 `coverage_epoch_*.json` 如实记录覆盖率。

因此，增强后仍然需要分组采样和损失权重：扩充视图并没有消除独立来源的不平衡。这次保留均衡监督，同时把更多物理位置分给 real。

## Noisy 构造和模型

处理流程为：

```text
Online -> one of 5 base views -> Omni all-layer fusion -> AASIST windows -> ASP -> classifier
Offline -> one of 5 base views -> scene A -> local RTC -> detector
                              -> scene B -> local RTC -> detector
```

同一 Offline 的两路 Noisy 共用一次基础增强结果，分别加入不同场景。原始 Offline 不额外输入检测器；Online 不再叠加一次完整 RTC 链。

噪声沿用 [RNNoise 完整来源方案](README_W2V_V3184_RNNOISE.md)：办公室、咖啡店使用经过筛查的原始 RNNoise 来源，雨声、脚步、键盘使用 ESC-50，回声使用延迟衰减副本。两份 Noisy 都启用六场景；先加噪再共同经过 RTC。RTC 比例为 WebRTC NS/AGC 50%、FFmpeg 30%、light 20%，随后使用 16/24/32 kbps Opus。默认无 bypass、不额外叠加旧 stress。第 1 轮 SNR 10–25 dB，随后 5–25 dB；回声以延迟和衰减描述。

六场景和处理顺序来自公开说明；具体 SNR、比例、录音 ID 和平台内部参数不是官方完整复现。参赛规则允许本地 RTC 模拟，禁止官方数据经过真实平台重传以及引入外部语音训练数据。[官方规则与场景说明](https://www.junxue.tech/rtc-sdd-challenge/)

模型读取每个 AASIST 窗口的 160 维表示，加入整句上下文计算注意力，再汇聚 **160 维加权均值 + 160 维稳定化标准差**，由 320→2 分类层输出分数。保留窗口重叠校正，避免重叠区域重复计数；单窗口时标准差为零，不声称它提供额外跨窗口信息。

迁移时复制 R2 的 LoRA、融合、后端及 BN 缓冲；分类层的均值权重直接继承，标准差权重置零，新注意力末层置零。初始化的评估输出因此保持原模型函数，随后注意力自由学习，没有旧残差幅度上限。零初始化只用于迁移，标准差分支第一步即可得到梯度。

方法依据：[ASP](https://arxiv.org/abs/1803.10963) 研究加权均值与标准差，但其说话人验证结果不能当成本项目伪造检测增益；[RawBoost](https://arxiv.org/abs/2111.04433) 支持研究传输和设备失真增强，本实现的温和频响变换不是完整 RawBoost 复现。速度扰动参考 [Interspeech 2015](https://doi.org/10.21437/Interspeech.2015-711)，这里采用更小范围，其跨任务收益仍需实测。

## 学习率与效率

从成熟 R2 best 增量适应，重新建立优化器；默认 1 轮冻结 LoRA 的适应训练 + 5 轮联合训练。

| 模块 | 适应轮最高 LR | 联合阶段最高 LR |
|---|---:|---:|
| LoRA | 0，保留已有前向贡献 | 1e-5 |
| AASIST 主体 | 3e-5 | 3e-5 |
| 全层融合 | 1e-4 | 1e-4 |
| 层混合权重 | 3e-4 | 3e-4 |
| 原窗口贡献模块 | 1e-4 | 1e-4 |
| 新注意力 | 3e-4 | 3e-4 |
| 均值＋标准差分类层 | 1e-4 | 1e-4 |

适应轮前 200 步渐入；联合阶段 LoRA 前 200 步渐入，其余模块持续训练，联合阶段余弦衰减至 0.1 倍。成熟模块使用较小步长，新池化较快适应。BN 继承并使用原动量，只从训练波形更新。每 100 步保留模块梯度和实际更新审计。

没有新增第二次 Omni 前向、整句对比重算或永久特征缓存。Offline 基础视图只处理一次，之后共享给两路 Noisy。视图轮转和来源轮转共享缓存的索引计划。实际每步时间仍需服务器测量，不能仅凭 CPU 测试保证 GPU 耗时完全不变。

基础增强不增加 WAV 磁盘缓存。此前完整噪声库准备结果约 14.8 GiB，包含下载包、选中的原始素材和规范化音频；这部分并非零占用。默认全部放在数据盘的项目目录。首次准备建议数据盘至少剩余 30 GiB，训练仍保留其自身空间检查。封存父模型额外保存一份小型适配器／后端 checkpoint，不复制 Omni 公共大模型。

旧固定 Dev、固定 Train probe、选择和校准分区保持不变；新增六场景诊断不参与历史 Weighted、选模和校准。实际选择是来源隔离的 80% Dev 子集，不能和完整 Dev Weighted 混称。Progress/Eval 仅生成提交，不参与调参。

## 服务器操作

以下命令块全部为 ASCII。先确认数据盘已挂载到 `/home/ubuntu/LXT`。每一块成功后再执行下一块。

拉取代码、激活原环境：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
export TMPDIR="$PWD/.tmp_v3184_aug5"
mkdir -p "$TMPDIR"
```

停止当前 R2 并封存已保存 best：

```bash
bash seal_v3183r2_for_v3184.sh --stop-running
```

该操作先验证最新 R2 的 PID、进程组、启动脚本、运行目录和日志，再发送 TERM；只认可 `last.pt` 中完整提交的 epoch。会新建带 `_sealed_e...` 后缀的快照并写入 `exp/.latest_v3184_parent_run`，保留原运行目录可恢复。如果 R2 已停止，仍可以执行；如果已有完成标记，核验后直接使用完成的运行。中途尚未保存的训练进度会丢弃，不把它写成完成的 epoch。返回成功后再开始下一步。

下载并准备官方论文所用来源的噪声；已有完成素材会复用并核验：

```bash
bash prepare_noise_v3184_rnnoise.sh --jobs 4
```

开启真正的 V3.18.4 AUG5，结束后自动生成 best submission 并上传分数 ZIP：

```bash
PARENT_RUN=$(cat exp/.latest_v3184_parent_run)
bash run_w2v_v3184_aug5.sh \
  --parent-run "$PARENT_RUN" \
  --epochs 6 \
  --warm-epochs 1 \
  --submit-best \
  --upload-submission \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

初始化会运行安装环境中的小型 fairseq2 原生接口检查，包括多窗口标准差梯度，再加载 3B 参数。日志出现 `[Train] V3.18.4 epoch 1/6` 后才进入训练；查看器读取每步日志，Ctrl+C 只关闭查看器。

```bash
bash watch_w2v_v3184_aug5.sh
```

完成后读取实际 temp.sh 链接：

```bash
RUN=$(cat exp/.latest_v3184_aug5_run)
OUT="/home/ubuntu/LXT/temp/$(basename "$RUN")_submission_best"
cat "$OUT/temp_download_url.txt"
```

上传失败时，以下命令仅重试已有分数上传，不重新执行 GPU 推理：

```bash
bash upload_w2v_v3184_aug5.sh --out "$OUT"
```

重启后挂载数据盘、激活同一环境，在该目录恢复：

```bash
bash run_w2v_v3184_aug5.sh --resume "$(cat exp/.latest_v3184_aug5_run)"
```

恢复使用已保存参数、视图轮转、随机状态、优化器和提交请求，从最后完整 epoch 接续，不允许夹带新的学习率或增强参数。独立导出入口是 `export_w2v_v3184_aug5.sh --run RUN --checkpoint best --upload-temp`，默认使用同一 Progress 路径；训练自动导出流程更适合常规使用。

## 对照与验收

默认启用 ASP；从同一个父 best 新建运行时，可用 `--pooling mean` 比较增强＋采样本身的贡献。另可用 `--view-factor 1` 关闭基础五视图增强；两者都不是对运行中配置的覆盖。先跑主实验，再按实际分语言、分场景结果决定对照，不要求额外完整训练才能启动。

本次本地测试覆盖初始 R2 输出对照、短录音/单窗口、标准差与注意力梯度、原始来源轮转、四组损失质量、增强在 RTC 之前且共享 Offline 基础波形、固定探针、逐张量一致的断点恢复、best 导出和上传失败重试，以及提前结束父运行的校验与封存。真实噪声＋FFmpeg/Opus 检查单独记录于开发审计。

本地为 Windows CPU、小型编码器及真实 AASIST；没有新的完整 Omni-3B GPU 分数，也未在本机执行 Linux 原生 WebRTC。程序通过测试说明实现和流程得到检查，不等于已经证明官方 Noisy 泛化提升。
