# V3.3：同源配对与通信处理多样性

V3.3 保留 w2v-BERT 2.0 + MultiConv，以 V3 `epoch_2_step_2417` 的完整模型权重开始新实验。该 checkpoint 的既有本地 Weighted 为 95.766%；这不是官方平台成绩，也不是 V3.3 已达到的结果。准备时同时核验 checkpoint 的版本、tag 和 SHA256，原 91.68 AASIST checkpoint 继续保护。

## 先停止 V3.2，再部署

退出进度查看用 Ctrl+C；这只关闭查看器。停止后台训练须运行：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist &&
python -m w2v_v32.stop --version v32 --apply
```

确认出现 `TRAINING_STOPPED=True` 后执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v33.sh &&
bash run_w2v_v33.sh --upload-temp
```

停训保留已保存的 best / last，不保存尚未到验证边界的更新。新流程检测到旧任务仍在运行会停止准备，不修改缓存。

setup 保留现有 Torch/Transformers，仅在必要时安装 `webrtc-audio-processing==0.1.3`，并运行测试。Linux 编译该依赖需要 C++ 工具链、Python 开发头文件和 SWIG；既有版本已安装时无需重装。FFmpeg 必须支持 libopus，并匹配固定 Dev 缓存记录的版本。CUDA、BF16、真实 WebRTC/Opus 检查在服务器执行，本地 CPU 检查不能替代它们。

默认先准备一次新缓存，然后依次运行 control、candidate，各 1 轮、半轮和整轮验证。两组从同一个 V3 权重独立开始，使用相同录音、顺序、随机种子和更新次数。两组不是连续训练两轮，也不融合模型。

## 数据与预算

- 只用官方 Train 语音及已隔离的 Train 噪声清单；Dev 只验证，Progress 不参与训练或选模。
- 通过官方 `train_offline_online_pairs.csv` 建立身份映射，不按相似编号猜测对应。自动定位失败时可显式传 `--train-pair-manifest /实际路径/train_offline_online_pairs.csv`。
- 每个 Offline 独立录音作为一个 source，关联可用的官方 Online 与两条完整 noisy。缺少 Online 的 source 仍保留。
- 每轮每个 source 仅访问一次，按语言和真假比例分层组织；16 个 source 构成一次更新。类别权重按独立 source 数计算，避免同源扩展造成重复加权。
- 分类损失：一半预算给可用 Offline/Online 的平均损失，另一半给两条 noisy 的平均损失。每个条件再按 70% 完整音频、30% 随机 3–6 秒片段分配；片段与完整音频相同时合并。
- 完整配对视图不再叠加 RawBoost 或局部静音。普通短视图以 25% 概率 RawBoost；noisy 短视图保留 5% 概率的短局部静音，限制持续时间、占比和有效能量。

## 两条完整 noisy

`noisy_a` 复用旧 full2 缓存中的一条 FFmpeg 处理音频，按语言、真假及旧强度分组均衡选择 v0/v1；`noisy_b` 从完整 Offline 生成：环境噪声 → 实际 WebRTC NS/AGC → Opus 编解码。每种处理族覆盖既有四个 SNR 区间；处理条件不按真假设定不同分布。这里是本地 DSP 模拟，不是真实平台采集。

新缓存默认 `data/rtc_noisy_v33/train`。A 优先硬链接到独立的新目录，跨文件系统则复制；B 新生成。固定种子、处理配方、源文件身份、音频 SHA256 和完成记录均落盘；中断可复用已校验文件。缓存完成需通过完整时长、有限数值、文件 hash 与实际训练读取器检查。

新缓存成功后，默认只删除旧 `data/rtc_noisy_full2_v1/train` 中 manifest 明确拥有的已替代 WAV/sidecar，并备份旧元数据。未知文件、所有 checkpoint、原始 Train、Dev 缓存保留。A 硬链接仍占用对应音频数据，删除旧文件名不等于释放整份空间。重新恢复旧 V3/V3.1/V3.2 训练可能需要重新生成旧缓存；若要保留旧缓存，在首次启动加 `--keep-old-caches`。磁盘准备峰值仍需容纳新 B；跨盘复制 A 会额外占空间。

## 配对目标与训练

control 的配对系数为 0；candidate 在前 10% source 曝光中将系数从 0 逐渐升到 0.02，其余配置相同。

配对使用同一 source 的 Offline 与可用 Online/noisy 完整视图，读取已有最后一个 MultiConv block 的帧特征。最多池化为 256 个有效 token，做带弱相对时间先验的双向软对齐，再约束对齐后的时间特征及 token 关系结构。padding 和低能量无效区域不参与配对；分类监督仍覆盖所有视图。

这是受跨处理一致性思路启发的实现，**不是 TFCL 论文频率 CKA 的逐项复现**。没有教师网络、可训练对齐分支或额外编码器前向。特征可一致但不可判别，因此所有视图继续接受真实标签 CE；报告按语言/真假记录有效配对、跳过和塌缩统计，并在指定更新记录 CE/配对的共享特征梯度比例。没有把低一致性损失等同于已获得鲁棒性。

原 MultiConv block 多样性 CKA 保留 0.01，每个 source 只作用于 Offline 完整视图。冻结前 20 层编码器，微调后 4 层与后端；默认编码器 LR `5e-8`、后端 LR `2e-6`，前 100 次更新预热。新目标建立新的 AdamW；后续恢复或降低 LR 时，模型和对应 Adam 状态一起恢复。

默认每组只训练 1 轮。可在首次启动显式指定 `--epochs 2`，不建议未看对照结果就延长。继承按长度组织微批、BF16、有限显存驻留/CPU 卸载策略；整段音频不截短以换取速度。

## 选模与报告

固定 Dev、整段原始音频、既有短 noisy Dev、统一 0.5 阈值和 FP32 验证。`best_model.pt` 只接受满足固定基线底线的 Weighted 改善，否则保留起点：

- 四组 EN-real recall 分别最多下降 0.5 个百分点。
- Seen/Heldout EN-fake recall 分别最多下降 0.3 个百分点。
- Clean 最多下降 0.2 个百分点；Noisy 不下降；Weighted 达到最小改善幅度。

另行标记“达到本次实际目标”：Noisy 提升至少 0.3 个百分点、Seen/Heldout EN-real 平均提升至少 2 个百分点，并同时通过上述保护。保存一点小提升不等于验证了新方案。两组报告展示 candidate 相对 control 的差值；结果可能没有提升，不能承诺未知平台提分。

```bash
# 动态进度；每次验证结果保留在终端。Ctrl+C 只关闭查看器。
bash watch_w2v_v33.sh
# 打印已有两组指标和完成后的比较报告
bash show_w2v_v33.sh
```

完整记录在 `exp/w2v_v33_日期_随机码/`，下含 `control/`、`candidate/`。根目录的 `comparison.json` 和 `report.md` 保存最终选择；各组保留逐次验证、训练诊断、输入和代码 hash。结束时报告 ZIP 默认保存到 `/home/ubuntu/LXT/temp`，`--upload-temp` 会打印 `TEMP_DOWNLOAD_URL=https://temp.sh/...`，只上传诊断文件，不含音频、权重或优化器状态。

恢复使用实验根目录，不能传单个 arm；完成的 arm 自动跳过，未完成 arm 从最后验证边界恢复：

```bash
bash run_w2v_v33.sh --resume "$(cat exp/.latest_v33_run)" --upload-temp
```

准备缓存时中断可重新运行原启动命令，缓存有独立完成校验。仅准备可用 `--prepare-only`；它打印的 `RUN=...` 可作为之后 `--resume` 的路径。恢复采用保存的配方，不接纳临时更换的训练参数。

停止 V3.3：

```bash
python -m w2v_v33.stop --version v33 --apply
```

两组结束并审阅报告后，可手动导出默认 Progress 提交：

```bash
bash run_eval_w2v_v33.sh --upload-temp
```

导出要求请求的组全部完成，默认选择通过保护的单模型 winner。若两组都没有合格收益，会使用起点，报告明确标记 fallback；不会把无条件指标最优但违反召回底线的权重自动用于提交。

## 数据加载传输错误后的恢复

若日志先出现 `received 0 items of ancdata`，再出现 `Pin memory thread exited unexpectedly`，需要更新本次数据加载传输修复，然后恢复原实验。修复将 worker 间传输的 CPU tensor 合并为少量共享存储，并在 Linux worker 中采用文件名共享方式；预取限制为每 worker 一个完整 source batch。训练 batch、视图、损失预算、优化器和样本顺序保持不变。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v33.sh &&
bash run_w2v_v33.sh --resume "$(cat exp/.latest_v33_run)" --upload-temp
```

恢复复用已经完成并通过核验的新缓存，从该 arm 最后保存的验证边界继续；边界后的未保存更新会重新执行。若错误发生在第一次更新前，使用已保存的 baseline 状态，无需重新生成缓存或丢弃已有基线评估。

兼容仅接受发布 `01c495c` 的全部已知 V3.3 源码 hash 到本次明确的 I/O 修复；配置、数据 hash、其他版本源码和依赖版本仍必须完全相同。不会通过关闭校验来接受未知变动。恢复时打印 `V33_TRANSPORT_RESUME_COMPAT=True`，并在对应 arm 下保存 `resume_transport_compat_*.json`，记录原 `last.pt` 的 SHA256 及逐文件差异；兼容步骤本身不改写 checkpoint。之后正常训练按原有保存事务更新 `last.pt`，新版本的后续恢复继续执行完整精确校验。
