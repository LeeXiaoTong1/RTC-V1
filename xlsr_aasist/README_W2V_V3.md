# V3：w2v-BERT 2.0 + MultiConv / 整段原始语音与整段 noisy

V3 是独立训练入口，保留原 AASIST 和旧 MultiConv 的实验目录、最佳权重及回退代码。默认从原平台 Weighted 91.68 checkpoint **只导入完整 w2v-BERT 编码器**，重新初始化 MultiConv 后端；不会自动使用旧 MultiConv 第一轮权重。没有新增官方 Train 之外的语音，沿用已有 Train 噪声库。Dev 只验证与选模。

该实现针对已经观察到的英文 real 召回回落和 noisy 表现不足，提供完整覆盖、阶段最佳恢复、低学习率适配和多指标保留。它不能承诺训练一定达到性能上限，实际 Dev 和平台结果仍需服务器训练验证。

## 部署与启动

先等当前训练结束。不要在训练或缓存生成进程仍运行时更新代码；新入口也会拒绝与这些任务同时运行。使用现有 `sdd` CUDA 环境：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v3.sh &&
bash run_w2v_v3.sh --upload-temp
```

安装脚本保留已有 CUDA PyTorch，只安装已有固定版本的适配依赖并运行小型功能测试。不会加载生产数据或修改 checkpoint。训练启动脚本后台执行完整流程：检查原 best → 生成/复用两个整段 noisy 版本 → 实际读取器验证完整覆盖和 Train/Dev 隔离 → 清理替代缓存 → V3 训练 → 导出报告。

旧数据路径默认从已有 Stage3 配置读取。必要时传 `--source-config /完整路径/stage3/config.json`；原 FFmpeg 无法自动找到时传 `--ffmpeg /原FFmpeg可执行文件`。必须与固定 Dev 缓存的 FFmpeg 版本一致。

原编码器起点固定校验 SHA256：

```text
/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt
db3f8167742bf2fe41cfad028dec962d56f6c61295442870620421d7f3a9bbee
```

## 查看进度与结果

同一行实时进度条：

```bash
bash watch_w2v_v3.sh
```

显示当前阶段、实际完成数量、百分比、训练 loss 和当前阶段 ETA。准备、验证、保存和上传会切换状态；未知总量的工作不伪造进度。Ctrl+C 只结束查看，后台训练继续。训练详细日志每 100 步输出，Dev 每 200 批输出；实时状态独立保存在 `<日志路径>.progress.json`，最多每秒原子更新一次。

查看历史指标、异常和报告下载链接：

```bash
tail -n 80 -f "$(cat exp/.latest_v3_log)"
```

查看已完成的半轮/整轮评估表：

```bash
cat "$(cat exp/.latest_v3_run)/report.md"
```

`epoch_1.json` 是第一整轮汇总；`epoch_1_step_*.json` 是半轮评估。每次评估均保存逐样本分数。训练退出后 ZIP 位于 `/home/ubuntu/LXT/temp/`；`--upload-temp` 会打印 `TEMP_DOWNLOAD_URL=https://temp.sh/...`。诊断 ZIP 仅包含报告、配置、日志、分数，不包含音频或权重。

## 两个整段 noisy 版本及缓存清理

每条官方 **Offline Train 原始录音**生成两个等长版本；Online Train 保留在普通训练分支，不再给已有 RTC 版本重复生成缓存。

| 版本 | 条件 |
|---|---|
| v0 | 5–15 dB；一段覆盖全录音的连续 Train 噪声＋一套固定处理设置 |
| v1 | 15–25 dB；另一套固定处理设置＋独立噪声抽样 |

四个 SNR 子档与处理设置在语言、真假各组内分配；所有组使用相同可用噪声库。每个版本从头到尾处理，不裁成四秒，不接回干净尾段，不拼接不同语音，也不混合真假录音。每条源录音的两种版本使用不同处理设置；噪声抽样不保证两个版本来自不同噪声文件。

处理使用已有 FFmpeg `afftdn + dynaudnorm + Opus` 模拟链，并非真实 WebRTC。为提供完整连续背景，噪声池仅保留足够覆盖最长源录音的噪声文件；实际池大小会输出，不能把完整时长理解成无限噪声多样性。

缓存路径复用已实现的 `data/rtc_noisy_full2_v1/train`，不改变其生成配方。生成可中断后复用已完成且哈希匹配的文件。按已审计 38,660 条 Offline Train、71.73 小时估计，两个 FLOAT WAV 版本约 **30.8 GiB**；生成阶段建议预留 **34–36 GiB**，随后训练还会按实际模型/优化器计算 checkpoint 余量。

完整生成、哈希/头信息检查和 V3 实际读取器验证通过后，默认清理：

- 原配置指定的 `dataset/rtc_noisy_cache_v2/train_g*`、本项目 `data/rtc_noisy_improved_v1/train_g*` 中清单所属的旧短 Train WAV/sidecar。
- 本项目 `data/w2v_feature_cache` 中符合旧命名和特征形状的固定长度 NPY。

保留原始语音、Train/Dev 协议、固定 Dev Seen/Heldout、新整段缓存和**所有 checkpoint**。旧缓存配置/清单保留，并归档清理记录；未知文件保留。需要暂缓清理可加 `--keep-old-caches`。只准备缓存可加 `--prepare-only`。已清理旧 Train 缓存后，运行旧实验需重新生成其缓存。

固定 noisy Dev **仍为原来的短缓存**，便于与历史结果比较；不声称已经在完整 noisy Dev 上验证。最终评估输入使用完整音频。

## 架构与适配

```text
16 kHz mono full waveform
 -> official per-utterance fbank / CMVN, 160 dimensions
 -> w2v-BERT 2.0: input representation + 24 encoder layers
 -> shared 1024-to-128 projection / SwiGLU / layer fusion
 -> 4 MultiConv blocks, multi-scale temporal kernels
 -> correctly aligned channel concatenation
 -> attentive mean/std pooling -> classifier -> P(fake)
```

保持旧 MultiConv 参数键和架构兼容。前端不对不同长度语音补齐到同一批长度；仅相同有效帧数才能合并微批。后端保留有效帧掩码、不使用 BatchNorm；极短音频为提取特征重复至最低长度，其他录音不裁短。

保留两项适配修正：按正确时间/通道顺序拼接四个 block；CKA 使用完整计算图，不能经 `.item()` 或重新构造 Tensor 断开梯度。CKA 在**完整逻辑 batch**计算，不能退化为每条或两条微批独立计算。本项目是 w2v-BERT 适配实现，不宣称复现官方论文全部设置。[MultiConv 论文](https://arxiv.org/abs/2509.03409)、[官方代码](https://github.com/hoanmyTran/dissimilarity_deepfake_detection)，许可证见 `THIRD_PARTY_LICENSES/MultiConv.txt`。

## 训练配方

| 项目 | V3 默认设置 |
|---|---|
| 语音覆盖 | 每完整 epoch：全部普通 Train 一次；每条 Offline Train 的两个 noisy 版本各一次 |
| 逻辑批次 | 目标 16 ordinary＋16 noisy；按两个分支实际数量均匀分配，末批不丢弃、不重复补样 |
| 真假/语言 | 按语言、真假、noisy 版本分层穿插；保持真实总数，不额外放大英文或 real |
| 类别损失 | ordinary 与 noisy 分别计算 N/(2×类样本数)，不叠加均衡重采样 |
| 普通增强 | 每条每轮确定性抽样：约 50% 不加 RawBoost，约 50% 用原 RawBoost5；同一概率适用于所有语言/真假 |
| Noisy 增强 | 使用完整缓存，读取后不再加噪、不再拼接 |
| 分支预算 | 第一个 epoch noisy CE 权重从 0.30 升至 0.50；ordinary 取剩余权重 |
| CKA | 权重前 200 步逐渐升至 0.01；不恢复旧 RTC/noisy 配对对比损失 |
| 后端阶段 | 最多 1 epoch，冻结整个前端；head 学习率峰值 1e-4，warmup 后余弦下降 |
| 联合阶段 | 从后端阶段选出的最佳点恢复权重及匹配 Adam；解冻最后 4 层，前 20 层及特征投影冻结 |
| 联合学习率 | encoder 1e-7、head 1e-5；warmup＋余弦下降，并结合验证停滞减半 |
| 联合预算 | 最多 5 epoch；根据验证停滞提前停止，最多两次自适应降学习率 |
| 验证 | 每半 epoch 执行相同完整 Dev；保存半轮和整轮候选，固定阈值 0.5 |
| 优化器 | AdamW，weight decay 1e-4，梯度裁剪 1.0 |
| 精度 | 训练 BF16；验证与 submission FP32 |

每轮覆盖计划记录真实曝光数；每条录音两种 noisy 版本不会因为 rare-real 重采样而挤掉大部分 fake 覆盖。分层只改变顺序，不改变样本数。中文指标仍保存，英文四种条件的真假召回逐次输出。

## 保留第一轮优势与选模

后端阶段的半轮和整轮都参与选优。联合阶段从选定的后端最佳点开始，建立两项保护参考：四种条件 EN-real recall 的平均值，以及 Seen/Heldout EN-fake recall 的平均值。默认允许相对参考下降最多 1.0 和 0.5 个百分点；只有通过保护且 Weighted 更好的候选才替换部署默认 best。被接受的候选可以抬高参考，不能逐次下调参考。

这两项是聚合保护，并不保证每个语言、每个条件或每个 SNR 档都不下降；完整分组结果保留供审查。Weighted 仍为 `0.3 Online F1 + 0.35 Seen F1 + 0.35 Heldout F1`，noisy F1 沿用四档平均口径，不能与平台 91.68 混为一谈。

连续两次评估没有有效改善时，恢复选定最佳点及匹配 Adam 并降低学习率；明显召回退化可以提前触发一次恢复。降学习率之后至少保留两次评估机会，预算不够时不会假装“降学习率后再立即停止”。不会只恢复权重而继续沿用退化点的优化器动量。

| 文件 | 用途 |
|---|---|
| `best_model.pt` | 通过两项保护的 Weighted 最佳；默认提交使用 |
| `best_weighted.pt` | 不受保护约束的最高 Weighted，保留用于比较 |
| `best_noisy.pt` | 最高 noisy Macro-F1，保留用于比较 |
| `control_best.pt` | 选定最佳点及配套优化器，用于阶段切换和恢复 |
| `last.pt` | 最近验证边界的完整恢复状态，包括采样位置、RNG、指标累计和控制器 |

保存采用临时文件/原子替换；恢复还会核对候选文件与 `last.pt` 的提交状态。切勿在运行中手动删除这些文件。需要恢复：

```bash
bash run_w2v_v3.sh --resume "$(cat exp/.latest_v3_run)" --upload-temp
```

要求代码、配置、依赖和输入元数据一致；从最近保存的半轮/整轮验证边界续训，边界之后尚未保存的步骤会重跑。恢复不会再次生成或清理缓存。

## 计算与内存

相比旧实现为逻辑 batch CKA 显式重跑整网，V3 每个微批只执行一次完整编码器前向，保留完整目标的梯度。需要反向使用的 CUDA 中间激活暂存 CPU，设定可用主机内存预算，避免无界累积；参数存储不重复拷贝。最后可训练层的梯度检查点仍会在反向时进行必要局部重计算。

每步检查非有限输出和梯度，确认有效后才推进优化器。内存不足会明确报错，不偷偷裁短音频或改变 CKA/类别目标。`--activation-budget-gib` 可在确有 RAM 余量时指定预算；`--microbatch 1` 减少微批峰值显存，但不能消除整段超长语音自身成本或整个逻辑 batch 的激活总量。`--keep-activations-on-gpu` 仅适用于显存足够的设备。

磁盘检查按真实模型和优化器规模保守预留候选切换、恢复和临时保存空间，可能需约数十 GiB，具体以启动输出为准。代码测试不能代替服务器完整 600M 模型的显存、CPU RAM 和性能验证。

## 生成 submission.zip

训练结束后，默认使用已存在的 Progress 数据：

```bash
bash run_eval_w2v_v3.sh --upload-temp
```

输出仅含官方 ID 顺序的 `scores.txt`，分数为 P(fake)。结束打印 submission.zip 的 temp.sh 链接。改用 noisy 最佳候选：

```bash
V3_CHECKPOINT="$(cat exp/.latest_v3_run)/best_noisy.pt" bash run_eval_w2v_v3.sh --upload-temp
```

指定另一套官方评估输入可设置 `EVAL_PROTOCOL` 和 `EVAL_AUDIO_ROOT`。不使用评估标签调参、挑阈值或训练。

## 验证范围

本地检查包含真实小型 HF 编码器与 MultiConv 的前向/反向、完整逻辑 batch 梯度等价、冻结层、掩码/帧顺序、原模型和旧 MultiConv 权重兼容、两个 full-noisy 版本完整覆盖、原始语音保留、阶段切换与 Adam 配对恢复、半轮/整轮中断恢复、旧缓存保护、报告及 submission 导出。服务器的完整训练尚未执行；CUDA 激活卸载硬件测试在无 GPU 的本地环境跳过，在服务器 setup 时执行。
