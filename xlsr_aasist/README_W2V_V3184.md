# V3.18.4：小型场景噪声库与 R2 best 继续适应

本版针对环境噪声覆盖，保留 Omni-W2V-3B、全 60 层投影/SwiGLU 融合、AASIST、窗口贡献汇聚和 C3 目标。默认等待正在训练的 V3.18.3-R2 完成后，从其 **best** 新建训练运行；完整继承 LoRA、融合、后端、贡献模块和 BN 缓冲，只重建优化器与学习率日程。V3.18.3-R2 的代码与运行文件不被修改。

这不是已经验证的性能提升。代码与数据流程验证通过，完整 3B GPU 训练、Progress/Eval 表现仍需实际实验。

## 小型素材方案

在线下载固定清单的 **128 个文件，58,666,616 字节（约 59 MB / 55.9 MiB）**，不需要用户上传文件、Freesound 账户或下载无标签大包。素材一次性转成 16 kHz 单声道 PCM16，训练时从磁盘读取并使用有界内存缓存，不重复下载、MP3 解码或重采样。

| 场景 | 素材 | 训练文件／来源组 | 验证文件／来源组 |
|---|---|---:|---:|
| 办公室 | 电话、键盘、打印、设备底噪；公开小型场景素材 | 2 / 2 | 1 / 1 |
| 咖啡 | 咖啡机、水流、搅拌；验证为无对白的杯碟／蒸汽合成场景 | 4 / 2 | 1 / 1 |
| 雨声 | ESC-50 rain | 32 / 32 | 8 / 8 |
| 脚步 | ESC-50 footsteps | 32 / 24 | 8 / 8 |
| 键盘 | ESC-50 keyboard_typing | 32 / 30 | 8 / 7 |
| 回声 | 当前语音的延迟、衰减副本 | 在线生成 | 固定诊断配方 |

同一原始录音的不同片段按一个来源组计数；同一咖啡机素材包按一个录音组保守分组。训练先均匀抽场景，再均匀抽该场景的来源组，最后抽文件，避免雨声或长录音凭数量占据主要概率。ESC-50 以 folds 1–4 / 5 为初始划分，再按原始 Freesound ID 和组合素材的组件关系检查隔离；同时检查规范化 PCM 哈希，禁止跨训练／验证复用同一音频。

办公室、咖啡只是明确标注的**小规模场景近似**。它们没有大量独立房间、设备或录音者，不能把随机裁剪当作新来源，也不能视为复现举办方的真实文件。素材选择依据发布者描述，未对全部音频逐条人工试听；`preview.html` 提供本地逐条预览，`ATTRIBUTION.txt` 与 `preparation_report.json` 保存来源和许可。咖啡组合素材的已知底层组件也列入归属信息。

来源：

- [RTCFake 论文](https://arxiv.org/html/2604.23742v1)：将办公室、咖啡、回声、雨声、脚步、键盘作为环境噪声场景。本版只是据此构建可控本地近似，不能据类别推定官方的具体 SNR、处理链或选取文件。
- [ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)：仅下载三个类别的 120 个 WAV，固定 commit 和每个文件的 SHA256；保留数据集 CC-BY-NC-3.0 归属与原始录音链接。
- [Busy Office No People Loop](https://freesound.org/people/Fupicat/sounds/534123/)、[Office Typing](https://freesound.org/people/BeaconStudio/sounds/862604/)、[Quiet Office Room Tone](https://freesound.org/people/mzui/sounds/135097/)：CC0 公共预览。
- [Coffee Shop, No Voices](https://freesound.org/people/Thimblerig/sounds/702710/)：CC-BY-4.0 公共预览；[Coffee machine](https://freesound.org/people/portwain/sounds/261121/)：CC0；[Coffee Machine Noises](https://freesound.org/people/MattRuthSound/packs/29527/)：三个 CC-BY-4.0 公共预览。

Freesound 使用发布页面公开提供的 HQ MP3 预览链接，不请求登录后的原文件。所有下载有固定长度、SHA256 和重试；支持服务器允许时的 Range 续传，不支持 Range 时从该文件开头重取。失败不会产生完成标志；已完成下载和转换可复用。准备锁由操作系统释放，进程退出或重启后不会留下阻塞工作的死锁。不要混用不同清单的输出目录。

## 噪声与 RTC 顺序

数据仍是每步 16 条官方 Online，加 16 个原始来源各生成 Noisy A/B，共 48 条波形。原始 Offline 用于产生 Noisy，不作为额外分类视图，也不增加对比学习。

两个 Noisy 视图各以 80% 概率选择六类目标场景，20% 沿用原噪声配方；两路都被选中时使用不同场景。六类等概率；回声属于其中一类，不对每条带噪语音额外叠加回声。每个来源、出现次数及视图有固定种子，重启可重现配方。

处理顺序为：**原始语音 → 混入场景噪声／生成延迟回声 → 共同输入增益和峰值保护 → RTC 处理 → 可选 Opus → 检测器**。不在官方 Online 上再叠加此噪声。雨、脚步、键盘保留自然事件序列；长度不足时衔接完整片段并做短交叉淡化，不把脚步切成几十毫秒的随机噪声碎片。

加性噪声以语音和噪声共同有效片段计算 SNR，首轮 10–25 dB，之后 5–25 dB；共同缩放保护峰值，保持相对 SNR。不使用全句静音稀释事件噪声。连续八次采样均无法得到有效共同片段时明确报错，避免把未加噪样本计为成功增强。回声为 40–150 ms 延迟、0.1–0.5 倍幅度，不伪称其为某个加性 SNR；这是简化回声失真，不是完整 AEC、房间脉冲响应或双讲模拟。

两种 RTC 设置均已实现：

- **默认 `--rtc-policy legacy`**：保持 R2 的 FFmpeg/WebRTC NS+AGC/light/bypass、可选 Opus，以及 Noisy B 的受控压力增强；先验证新增场景噪声的作用。
- **独立对照 `--rtc-policy three-profile`**：Noisy A/B 各按 50% WebRTC、30% FFmpeg、20% light 抽处理器，并使用 16/24/32 kbps Opus。两路独立抽样以保持各自比例；关闭额外 Noisy B stress，避免多次堆叠破坏。这里的比例是实验设计，不是官方 RTC 处理概率。

建议第一次先跑默认设置。RTC 对照应另建运行并指定**同一个已完成 R2 parent**，仅改 `--rtc-policy three-profile`，方便解释收益来源。当前实现不声称包含 AEC、真实网络丢包/PLC 或任何外部平台重传。

## 参数、监测与效率

默认 6 轮：第 1 轮冻结 LoRA 更新（继承的 LoRA 仍参与前向），让融合／后端适应新输入；后 5 轮联合训练。它是新噪声适应阶段，融合层没有重新随机初始化。BN 保留并以父 R2 动量继续更新。

| 模块 | 默认峰值学习率 |
|---|---:|
| LoRA | 2e-5（首轮为 0） |
| AASIST、窗口贡献模块 | 各 1e-4 |
| 投影／SwiGLU | 3e-4 |
| 层混合 logits | 1e-3 |

学习率沿用发布版 R2 的默认峰值，200 update 预热和余弦衰减，末端为峰值 10%；此次不再同时提高学习率。显式自定义过 R2 学习率的用户，需在新运行命令中再次指定想保留的值。两流配额均为 4/2/7/3、监督各组 25%、Online/Noisy 各 50%；C3 为 `0.75*mean(CE_A,CE_B)+0.25*max(CE_A,CE_B)`，标签平滑 0.02。

旧固定 Dev、Train probe、独立机制 panel、selection/calibration 分区完全继承。**历史 Weighted 的数据和算法不变**。best 仍从本次已完成训练轮次的原始 selection Weighted 中选择，不回退历史 best。

新增六场景 Dev 诊断：默认四个语言／类别组各 32 个原始来源，每个来源生成六种条件，共 768 条视图；使用隔离的验证噪声。开始训练前评估一次，之后每轮评估一次，记录 F1、AUC、EER、real/fake recall、固定 fake recall 下 real recall。它不参与 Weighted、选模或校准。图中 epoch 0 是新运行初始化在这些场景上的实际结果。素材和语音来源隔离不意味着涵盖未知 RTC 机制；此小诊断也不是无偏官方泛化分数。

本版不增加训练模型分支、第二次编码前向或句级对比重算。新增成本是 CPU 场景混音、每个 worker 最多 32 MiB 的新噪声缓存，以及每轮 768 条诊断推理；总耗时仍需看服务器日志。`steps.jsonl`、`sampling_and_loss_*.json` 记录每视图场景曝光与既有模块梯度／更新量。`scene_baseline.json`、`scene_scores_*.npz`、`diagnostics/curves.html` 和 `groups.csv` 保存诊断。父 best 旧 Dev 分数使用已保存预测作为参照，不重复完整旧 Dev 推理；R2 参数及缓冲逐项一致性另做检查。

## 服务器操作

以下命令块只含 ASCII。先确认 `/home/ubuntu/LXT` 的数据盘已挂载。当前 R2 可以继续训练；这次 Git 发布只增加 V3.18.4 文件。准备噪声不使用 GPU，R2 运行期间可先用两个下载／转换 worker。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash prepare_noise_v3184.sh --jobs 2
```

看到 `NOISE_MANIFEST=.../assets/noise_v3184/manifest.json` 即准备成功。断网可重复最后一条命令，已校验文件复用；`--list-only` 只看清单。无需安装新包，不要在 R2 正运行时重建 Python 环境。

**R2 完成后再启动训练**。以下命令读取最近一次 R2 运行，启动器会拒绝未完成 parent，也会拒绝与现有训练／推理并发占用 GPU。若最近指针并非你要的 R2，请把 `PARENT_RUN` 赋值为实际已完成运行目录。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
PARENT_RUN=$(cat exp/.latest_v3183r2_run)
printf 'PARENT_RUN=%s\n' "$PARENT_RUN"
bash run_w2v_v3184.sh \
  --parent-run "$PARENT_RUN" \
  --noise-manifest "$PWD/assets/noise_v3184/manifest.json" \
  --epochs 6 \
  --warm-epochs 1 \
  --rtc-policy legacy \
  --submit-best \
  --upload-submission \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

启动时有代码／数据／大模型文件校验和小型原生 Omni 接口测试，随后加载父 best、运行初始六场景诊断，再进入第一轮。看到 `[Train] V3.18.4 epoch 1/6` 才是开始参数更新。服务器启动检查失败会明确报错，不会静默退回旧版本。Ctrl+C 只关闭日志查看器，训练在后台继续。

```bash
bash watch_w2v_v3184.sh
```

训练全部完成后，自动使用本次 **best**（不是 last）生成 `scores.txt`、`submission.zip` 和元数据，再上传只有分数的 ZIP 到 temp.sh。输出目录与链接位置：

```bash
RUN=$(cat exp/.latest_v3184_run)
OUT="/home/ubuntu/LXT/temp/$(basename "$RUN")_submission_best"
cat "$OUT/temp_download_url.txt"
```

若上传失败，分数和 ZIP 保留，执行下面命令只重试上传，不重新训练或推理：

```bash
bash upload_w2v_v3184.sh --out "$OUT"
```

如需训练完成后另行导出 best：

```bash
bash run_eval_w2v_v3184.sh \
  --run "$RUN" \
  --checkpoint best \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

重启机器后先挂载数据盘、激活同一环境；再从最后完整保存的 epoch 恢复，未完成的一轮会重跑，已经保存的提交请求自动恢复：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
RUN=$(cat exp/.latest_v3184_run)
bash run_w2v_v3184.sh --resume "$RUN"
```

`--resume` 使用原配置，不能同时改变学习率、噪声清单或 RTC 策略。新实验需用 `--parent-run`；显式传入旧 V3.18.3（约 87 分的有限残差版本）也受支持，但会按 R2 迁移规则重建融合并重置 BN，只能称结构迁移，不能称完整继承。优先用训练完成的 R2 best。

## 验证与已知边界

```bash
bash test_w2v_v3184.sh
```

测试覆盖固定素材完整性、原始来源和 PCM 隔离、Range/非 Range/校验失败、准备锁与转换复用；真实 FFmpeg 与 Opus 的长度和数值；两视图采样概率、回声和 SNR；R2 所有可训练张量及 BN 一致继承；真实 AASIST 加小型 CPU 编码器的训练、断点重放、best 选择、导出及上传失败重试。上传测试使用模拟网络，不上传合成分数。完整 3B CUDA 和原生 Linux WebRTC 的执行依赖服务器已有运行环境，不能用本地 CPU 通过替代 GPU 验收。

一次局部 Dev 上涨仍不足以证明 noisy 泛化突破。先比较 R2 与本版同口径 raw Dev、六场景 epoch 0/后续曲线和各语言排序指标，再考虑同一 parent 的 RTC 对照。Progress/Eval 不用于调参、校准或选模。
