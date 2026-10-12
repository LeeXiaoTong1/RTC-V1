# V3.18.4：按论文噪声来源构建的方案

本入口用 **RNNoise 原始 `office` / `coffee` 录音、ESC-50 的 rain / footsteps / keyboard_typing，以及延迟回声**替代上一版的小型办公室／咖啡近似素材。它复用已发布 V3.18.4 的模型、训练、恢复和 best submission 流程；原有训练代码与小型素材入口均保持原文件内容，已开始的旧运行可以按原配置恢复。

此前“小型方案”说明中“RNNoise 缺少场景标签”的判断需要纠正：公开下载目录没有逐条标签清单，但原始贡献包的文件名包含 `-office.raw` 和 `-coffee.raw`。本方案直接按这些原始贡献者标签选取，不通过关键词模型猜测类别，也不使用咖啡机／杯碟组合代替整个类别。

## 公开证据与实现范围

| 项目 | 已核实的公开信息 | 本实现 |
|---|---|---|
| 办公室、咖啡 | RTCFake 论文引用 RNNoise 数据来源 | 2025-01 更新的原始贡献包；仅选 `office` / `coffee` 标签 |
| 雨声、脚步、键盘 | RTCFake 论文引用 ESC-50，并列出这三类 | 固定 ESC-50 commit 的 120 个文件；原始录音来源隔离 |
| 回声 | 论文引用 CLAD；其实现为延迟、衰减的语音副本 | 沿用 V3.18.4 相同的回声形式，保留 40–150 ms、0.1–0.5 增益范围 |
| 顺序 | 离线语音 → 噪声模拟 → RTC 传输 | 同样先加噪，再共同输入本地 RTC 处理 |
| RTC | 举办方通过真实平台传输；参赛规则禁止将官方数据经 RTC 平台重传用于增强 | 使用本地 WebRTC NS/AGC、FFmpeg、轻量链和 Opus 模拟 |

这是**按论文来源与处理顺序构建的本地模拟**。论文及官方基线没有给出完整噪声生成脚本、确切录音 ID、SNR 分布或平台内部参数，因此不能称为官方生成过程的逐项复现。CLAD 示例的延迟／增益组合也不能据引用关系直接当成 RTCFake 的参数。本实现没有引入真实平台重传、AEC 参考信号链或网络 PLC。

证据：

- [RTCFake §3.1–3.2 与附录](https://arxiv.org/html/2604.23742v1)
- [竞赛规则](https://www.junxue.tech/rtc-sdd-challenge/)
- [RNNoise 原始数据与发布方 SHA256](https://media.xiph.org/rnnoise/)
- [RNNoise 当前说明：更新过的贡献包](https://github.com/xiph/rnnoise/blob/main/README)
- [CLAD 原始实现](https://github.com/CLAD23/CLAD/blob/518efcb80c8dafa602cd74f52628337373fbc3be/DatasetUtils.py)
- [ESC-50 固定版本](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)

## 数据准备与质量控制

首次下载 RNNoise 包 **6,756,960,430 字节，约 6.3 GiB**，另下载 ESC-50 120 个 WAV 和约 2.3 MB 的公开 Silero VAD 权重。只在准备阶段使用 CPU 进行解包、重采样及语音筛查，训练过程中不运行 VAD。已有小型素材库中的 ESC-50 原始下载可以复用，不下载外部语音库。

原始包与 VAD 权重均固定 SHA256；下载按 32 MiB 分段，断网后保留已完成段。校验完整压缩包后才读取归档，逐项选取有界的普通文件，不解包无关条目或软链接。准备阶段建议有约 **16 GiB 可用空间**供分段合并及音频处理使用；完成后只保留完整下载包、选中原始录音及规范化音频，分段文件自动清理。

素材处理规则固定并写入清单：

1. 使用原始 `office` / `coffee` 标签；排除发布者仓库中已知清晰语音报告及少数明确报告严重失真的文件。不会因录音原始音量大就将它排除，混合时以 SNR 统一控制。
2. 原始 RAW 缺少采样率元数据。只接受长度能够唯一对应约一分钟、44.1 或 48 kHz 的录音；其采样率明确标为**按时长推断**。不满足这个条件的录音列入排除报告，避免悄悄按错误采样率处理。
3. 用固定版本的 Silero VAD 检测语音，去除连续高语音概率区间并留出前后各 250 ms；仅保留至少 3 秒的连续非语音候选片段。严重削波或近乎无声的素材也排除。VAD 的标签不是人工无语音保证，`preview.html` 可以试听，报告保存每条录音的筛查结果。
4. 同一原始录音的全部片段留在同一分区；相同浏览器信息、相邻上传时间组成的会话，以及完全相同的原始／PCM 内容，也合并后分区。约 20% 来源组用作本地诊断。原始包没有录音者和房间 ID，因此只声称录音／会话分组与精确内容隔离，不声称已经实现录音者或房间隔离。
5. ESC-50 延续 folds 1–4 / 5 初始分区，并检查原始 Freesound ID 和 PCM 内容不跨训练／验证。所有片段统一保存为 16 kHz 单声道 PCM16。

`manifest.json` 最后写入，是准备完成的标志。中断后重复准备命令即可复用完成的下载、归档索引与筛查结果。完成后的目录会校验素材、代码、运行库身份；修改准备方式应使用新的输出目录，不在已训练模型依赖的目录内覆盖文件。

来源筛查依据：[清晰语音报告](https://github.com/xiph/rnnoise/issues/251)、[失真与音量讨论](https://github.com/xiph/rnnoise/issues/252)、[Silero VAD](https://github.com/snakers4/silero-vad/tree/1e261b036686cd0017d500ee96acd1c4ba572a9d)。筛查阈值、分区比例和采样率推断是本项目的数据准备设置，非赛事官方参数。

## 本次完整素材验证

本地已完整下载并通过发布方 SHA256 校验的 6.3 GiB 原始包。按标签共找到 1,179 条办公室、77 条咖啡录音；先排除 3 条已有明确失真报告的录音，再对 1,253 条进行筛查，最终 1,170 条原始录音产生 1,599 个片段。另有 48 条因质量／采样率条件排除、35 条没有足够长的可用非语音区间。

| 类别 | 训练片段 | 诊断片段 | 训练来源组 | 诊断来源组 |
|---|---:|---:|---:|---:|
| 办公室 | 1,216 | 293 | 723 | 183 |
| 咖啡店 | 69 | 21 | 36 | 9 |
| 雨声 | 32 | 8 | 32 | 8 |
| 脚步 | 32 | 8 | 24 | 8 |
| 键盘 | 32 | 8 | 30 | 7 |

办公室片段来自训练 903 条／诊断 217 条原始录音；咖啡片段来自训练 40 条／诊断 10 条原始录音。片段数不是独立来源数，来源组也不是已核实的独立房间数。回声由当前语音动态生成，无外部录音计数。不同运行库版本可能使自动筛查边界略有差别，应以服务器生成的清单为准。

验证结果：

- 1,719 个最终 WAV 全部通过文件校验、16 kHz 单声道、非空／有限值检查，以及清单的来源隔离检查。
- 360 次六场景混合覆盖训练／诊断分区与不同音频长度，加噪阶段实测 SNR 误差低于 0.001 dB。
- 12 次真实 FFmpeg／light＋Opus 调用覆盖六场景，长度和数值检查通过。
- 10 项新准备流程测试与 14 项既有噪声、增强、训练／恢复／best 导出兼容测试全部通过；两份新脚本通过语法和参数转发检查。
- 实际 Silero 权重的分批推理与逐条推理对照保留相同筛查区间；分批只合并独立录音，不切断同一录音内部的时序状态。

上述执行环境为 Windows CPU；没有在本机运行 Linux 原生 WebRTC 或完整 3B GPU 训练。后两者沿用已有实现，仍需服务器启动检查和训练结果验证。

## 训练配方

新入口默认使用此完整来源库，**两路 Noisy 的场景概率均为 100%**，六类等概率；同一来源的两路使用不同场景。原始 Offline 仅用于生成 Noisy，不增加额外分类视图。每次更新仍为 16 条官方 Online 加 16 个来源各两路 Noisy，总共 48 条视图；4/2/7/3 采样、四组各 25% 监督、Online/Noisy 各 50% 以及 C3 目标保持原实现。

完整处理顺序：原始语音 → 一种场景噪声或延迟回声 → 共同增益与峰值保护 → 本地 RTC → Opus → 检测器。

新入口启用现成的 `three-profile`：两路分别以 50% / 30% / 20% 选择 WebRTC、FFmpeg、light，然后使用 16 / 24 / 32 kbps Opus。无 bypass；关闭另外叠加的 Noisy B stress。此比例用于覆盖多种处理响应，不能解释为官方平台的占比。

SNR 保持第 1 轮 10–25 dB，后续 5–25 dB；实际共同有效片段 SNR、噪声来源和处理链进入日志。回声用延迟／增益描述，不伪造一个加性噪声 SNR。以上范围均为实验选择。

模型仍为 Omni＋全层融合＋AASIST，优先从**训练完成的 V3.18.3-R2 best**新建运行，继承全部 LoRA、融合、后端和 BN 缓冲，重建优化器。保留 1 轮头部／融合适应、5 轮联合训练及 R2 的分模块学习率。R2 尚在训练时可以先准备噪声，完成后再启动本入口。

原固定 Dev Weighted 的数据、算法和 best 选择方法不变；六场景 panel 使用新的隔离噪声，作为额外诊断，不参与 Weighted、选模和校准。应同时查看新场景诊断与旧 Dev，不能只凭其中一项断言官方 Noisy 泛化已经提升。Progress/Eval 不用于调参。

## 服务器命令

先挂载 `/home/ubuntu/LXT` 数据盘，再拉取代码。下面命令块只包含 ASCII。数据准备不使用 GPU，可以在 R2 运行期间进行；网络下载线程数由 `--jobs` 控制，语音筛查固定单 CPU 线程。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash prepare_noise_v3184_rnnoise.sh --jobs 4
```

准备成功后输出 `NOISE_MANIFEST=.../assets/noise_v3184_rnnoise/manifest.json` 和各类来源组／时长。下载或筛查中断后重复最后一行。可用以下命令只核验已经准备好的数据：

```bash
bash prepare_noise_v3184_rnnoise.sh --verify-only
```

R2 完成后，用该次 R2 best 新建完整来源实验。最近指针若不是目标 R2，显式填写正确的运行目录。

```bash
PARENT_RUN=$(cat exp/.latest_v3183r2_run)
printf 'PARENT_RUN=%s\n' "$PARENT_RUN"
bash run_w2v_v3184_rnnoise.sh \
  --parent-run "$PARENT_RUN" \
  --epochs 6 \
  --warm-epochs 1 \
  --submit-best \
  --upload-submission \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

这仍是 V3.18.4 训练程序，因此运行指针、日志查看、恢复和导出沿用原脚本。`config.json` 中的噪声 manifest 与 provenance 明确标识本方案。仅拉取代码不会改变已经启动的训练；要换素材需新建运行，不能给旧运行的 `--resume` 改配方。

```bash
bash watch_w2v_v3184.sh
```

训练完成后会自动从本次 best 生成 `scores.txt`、`submission.zip`，并上传只有分数的 ZIP。读取 temp.sh 下载链接：

```bash
RUN=$(cat exp/.latest_v3184_run)
OUT="/home/ubuntu/LXT/temp/$(basename "$RUN")_submission_best"
cat "$OUT/temp_download_url.txt"
```

上传失败时只重试上传：

```bash
bash upload_w2v_v3184.sh --out "$OUT"
```

重启后，挂载数据盘并激活原环境，再从最后保存的完整 epoch 继续：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
RUN=$(cat exp/.latest_v3184_run)
bash run_w2v_v3184.sh --resume "$RUN"
```

独立导出本次 best：

```bash
bash run_eval_w2v_v3184.sh \
  --run "$RUN" \
  --checkpoint best \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

现有小型噪声库可以保留用于对照；本方案的命令读取不同素材目录。尚未完成完整 3B GPU 对照实验，代码与素材检查不能替代泛化性能实测。
