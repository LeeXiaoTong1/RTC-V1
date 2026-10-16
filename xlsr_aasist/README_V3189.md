# V3.18.9: activity-conditioned RTC attenuation

V3.18.9 从已停止的 **V3.18.8 best** 新建实验，默认新增 **6 轮、0 轮预热**。保留全部模型结构、重叠窗口、统计汇聚、双分支 PCL、单 Noisy、动态基础增强和来源预算。本次主要改动是模拟短暂停顿时背景衰减、语音恢复及近噪声底弱语音受抑制的情况。

## RTC 模拟

处理顺序：Offline 基础增强 → 环境噪声/回声 → 原降噪与增益链 → 新的动态衰减 → 一次 Opus 编解码。真实 Online、固定 Dev 与提交音频不会再次经过新处理。

按来源访问确定性抽样：25% 不增加动态衰减、45% 温和衰减、30% 较强衰减。保留原来的 WebRTC / FFmpeg / Light 分布。所有语言、Real/Fake 使用相同的参数分布；处理器不读取标签、音素、干净参照或 Progress 音频。

新处理使用 20 ms 能量窗口、10 ms 步长及双阈值、保持时间、渐变关闭/恢复。温和分支衰减下限抽样为 -16 至 -26 dB，较强分支为 -30 至 -45 dB。保持时间分别为 30–70 ms、10–40 ms；恢复时间分别为 4–12 ms、8–25 ms；关闭时间分别为 20–45 ms、8–25 ms。强度是关闭状态的目标，不是整句降音量。

能量参考由当前录音自身的分位数初始化，状态随帧更新；这是一种离线数据增强代理，**不是严格因果的语音 VAD，也不声称复现了官方平台、DTX、AEC 或丢包补偿**。没有足够能量变化的录音、全静音或过短输入会跳过衰减，不强制制造静音。不会删除、插入或重复内部片段，也不随机把波形置零。旧版降噪器在一句内的连续状态仍保留。

`--rtc-activity off` 可建立关闭新衰减的对照；只能新建实验时指定，恢复训练使用保存的配置。

## PCL 与可观测性

- 模拟干净 RTC 与模拟 Noisy RTC 使用相同参数，各自依据本端音频独立计算动态状态。不会把干净音频的活动标签作为 Noisy 的答案。
- 对新增衰减达到 12 dB 的区间记录位置，并扩展 20 ms 编解码保护区。PCL 再保护 400 样本前端感受野的半宽；两端无效区间合并后保留仍可对齐的子区间。过短的剩余区间按原最小帧数要求剔除。
- 抗噪分支检查模拟干净与 Noisy 两端；跨平台分支检查模拟干净端，真实 Online 继续使用已有对齐与有效性检查。
- 这些区间表示“明显受抑制”，不等于证明信息全部擦除。原能量/相关性筛选继续保留。没有有效配对的参考不进入编码器。
- `steps.jsonl`、`sampling_and_loss_*.json` 与 `training_history.json` 记录处理分支、受影响样本数、门控次数和 PCL 屏蔽覆盖。报告的 `diagnostics/curves.html` 增加衰减比例曲线。终端仍只显示进度/loss、每轮指标和下载链接。

## 训练预算、继承和效率

每次更新保持 24 条 Online 来源曝光与 24 条 Offline 来源曝光；每条 Offline 只生成一条 Noisy。分类视图仍为 48，实际编码小批次上限 12，帧预算 9000，窗口批次 16。双 PCL 引用计入这些限制。噪声素材和音素缓存直接复用，不新建增强波形缓存。

Opus 不重复编码；新增处理是 CPU 上的能量统计和包络乘法，不增加 GPU 模型或编码视图。一次本地 Windows CPU 微基准中，12 秒音频的新处理耗时中位数约 10.6 ms（100 次）；这不代表服务器整步耗时或训练加速比例。

完整加载父实验 `last.pt` 中的 `best_model`，不截断分类器。新建优化器，起始学习率按 V3.18.8 best 轮次实际结束水平继承，再在新增轮次中衰减；来源访问轮次累计继承。若 V3.18.8 的 best 仍是 epoch 0 父基线，也能正确继承其权重、初始学习率与来源偏移。新实验保留父 best 为 epoch 0 基线；因为后端计算精度改变，首次启动会重新评估一次父 best，随后与新训练使用同一精度比较。恢复训练不重复该评估。若新增训练没有超越它，导出会明确标记父基线获胜。

## FP16 与 CPU 并行

AASIST 的卷积、线性层和矩阵乘法默认采用 FP16 autocast，Omni 保持 BF16；参数及优化器主副本仍是 FP32。归一化、统计汇聚、分类损失和 PCL 保留 FP32。逻辑批次的各小批次统一缩放损失，全部反向完成后只反缩放一次，再裁剪梯度、更新参数。溢出时跳过整次更新并降低缩放倍率；缩放器状态随 checkpoint 保存、恢复，报告记录跳过次数。遵循 [PyTorch AMP 梯度累积规则](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html#gradient-accumulation)。`--backend-precision fp32` 可在新实验中建立精度对照。

默认 worker 数按可用 CPU 核数的一半选择，最多 8 个，并按可见内存余量进一步下调；评估默认最多 4 个。每个 worker 只预取 1 批，启用持久 worker，Torch、BLAS 和 FFmpeg 各使用单线程，避免进程与内部线程叠加。原始音频、噪声及场景缓存按父实验总预算分摊，不因 worker 增多成倍扩张。只增加内存中的并行处理，不创建增强音频磁盘缓存。

如需明确指定，可在启动参数添加 `--workers 8 --eval-workers 4 --prefetch-factor 1`。增加 worker 只会缓解读取和模拟造成的供数等待，不能保证 GPU 计算更快。`worker_resource_audit.json` 记录选择依据；每轮报告保留 CPU 等待时间、计算时间和等待占比。内存估算包含队列与缓存但不是系统内存硬上限，资源紧张时可用 `--workers 4` 或更小值新建实验。

本地验证包含真实 FFmpeg/Opus、CPU FP16 运算、梯度缩放与恢复、跨 worker 确定性；本机没有 A100 或 native WebRTC，无法预先确认服务器加速比例和最终显存峰值。启动器会先在服务器运行小模型原生接口、FP16 输出类型及反向传播检查，再加载正式 3B 模型。

## 部署与训练

先关闭只读查看器，再暂停 V3.18.8。以下命令全部使用 ASCII。暂停超时会返回 `STOP_PENDING`；等它完成后再开始新训练。已经停止的会显示 `ALREADY_STOPPED`。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
OLD_RUN=$(cat exp/.latest_v3188_run)
bash stop_w2v_v3188.sh --run "$OLD_RUN" --timeout 60
```

暂停完成后更新并启动：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
PARENT_RUN=$(python -m w2v_v3189.parent) && bash run_w2v_v3189.sh --parent-run "$PARENT_RUN" --epochs 6 --submit-best --upload-submission
```

父路径默认读取 `exp/.latest_v3188_run`；可以通过 `--parent-run /absolute/path/to/run` 显式指定。不要用 `--resume` 指向 V3.18.8。新代码只新增文件，不覆盖旧版本，也不会自动停止已有服务器任务。

查看器中的 Ctrl+C 只关闭查看器。再次查看：

```bash
bash watch_w2v_v3189.sh
```

完成后自动生成并上传 submission.zip 与 report.zip，打印两个 temp.sh 链接：

```bash
RUN=$(cat exp/.latest_v3189_run)
cat "$RUN/download_links.txt"
```

需要单独导出已完成或安全暂停实验的 best：

```bash
RUN=$(cat exp/.latest_v3189_run)
bash submit_w2v_v3189.sh --run "$RUN" --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress --report-dir /home/ubuntu/LXT/temp
```

只重试上传、不重新推理：

```bash
RUN=$(cat exp/.latest_v3189_run)
bash submit_w2v_v3189.sh --retry --run "$RUN"
```

暂停与恢复：

```bash
RUN=$(cat exp/.latest_v3189_run)
bash stop_w2v_v3189.sh --run "$RUN" --timeout 60
```

```bash
RUN=$(cat exp/.latest_v3189_run)
bash run_w2v_v3189.sh --resume "$RUN"
```

## 试听与验证范围

训练开始后可关闭查看器，执行 CPU 试听导出。每组包含原始 Offline、官方同源 Online、加噪但未 RTC、V3.18.9 Noisy Online，以及同一混合输入关闭新处理的 RTC 对照。04/05 在相同播放音量下比较。只读取 Train，不加载模型或修改训练状态。导出的 fresh draws 不代表正在训练的那个 batch。

```bash
bash export_audio_examples_v3189.sh --per-group 2 --upload
```

要明确试听较强分支，可加 `--activity-profile strong`；这是诊断覆盖，不是训练中的随机混合分布。文件仍写在数据盘 `/home/ubuntu/LXT/temp`。

本地验证覆盖合成短暂停顿/弱边界/恢复、双端 PCL 屏蔽、完整 best 继承、断点恢复、预算、日志、提交与报告。原生 FFmpeg/Opus 路径和用户既有 Train 导出中的 5 个可本地运行案例已检查。未使用 Progress 音频拟合配方。Windows 本地没有原生 WebRTC 模块，也没有实际 Omni 3B/A100 训练验证；Noisy 泛化收益需要新实验评估。
