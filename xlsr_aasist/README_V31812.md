# V3.18.12：全局证据残差修正实验

本版先验证假设，不从头训练检测器。使用一个已完成或安全暂停的 **V3.18.10 best**：冻结 Omni 3B、原 LoRA、多层融合、AASIST、窗口汇聚和原分类器，仅训练新增残差。没有新增元学习更新、PCL、XEUS 或分数集成。父 checkpoint 已有的训练成果完整保留。

## 模型

同一次 Omni 前向得到完整有效时序 `H[T,128]`。全局统计为 `g=[mean(H),std(H)]`（256 维，population std）；保留原 AASIST 重叠窗口和统计汇聚得到 `l`（384 维）。唯一预测路径为：

```text
g, l -> Train-fit 标准化 -> Linear(640,128) -> SiLU -> Dropout(0.1)
     -> Linear(128,384) -> Δl
l + Δl -> 原有且冻结的 Linear(384,2) -> P(Fake)，阈值 0.5
```

最后一层权重和偏置为零，初始输出复现父模型；没有截断任何分类器权重。标准化只影响残差输入，统计量仅来自残差 Fit 数据，按八组等权。全局统计在窗口之前计算，长音频不裁切、不重复计算重叠区。极短语音沿用原镜像补长供局部后端使用；全局统计只取原始内容覆盖的有效帧，少于 400 个原始采样点时全局统计为零。Transformer 的极短输入上下文仍含镜像部分，这不是完整消除补长影响。

该实验实现用户给出的残差假设。[ICASSP 2025 的相关论文](https://ieeexplore.ieee.org/document/10889337/)研究 STCA 与 LMDC 的全局/局部互补，本版没有声称复现其结构或性能。V3.18.11 的小样本观察是待复核线索，不能代替本流程的全量结果。

## 三个步骤与判断条件

1. **全量特征复核**：按已验证 PCM 来源登记表，将旧 Train 的来源按语言、类别和是否成对分层，固定留出约 10%。同源 Offline、Online 和所有增强必须同属 Fit 或 Inner。Fit 全部来源提取特征，分别训练 256 维全局线性探针和 384 维局部线性探针。两者使用同一八组 CE、相同曝光预算、相同 Train-inner 选轮方式，输出完整 Inner 和原 Dev Select 指标及逐条预测。修订后的条件是：全局探针在 Inner 上不增加 Online EN-real 误报，在 Dev Select 上严格减少误报；两边 EN AUC 相对父模型下降都不超过 0.01。这是筛查条件，不是有效性的证明。旧版要求 Inner 也严格改善，会使父模型已经零误报的实验必然停止；本次明确修复这个上限问题。
2. **冻结模型训练残差**：分别运行三个固定随机种子，每个最多 20 个轻量特征训练轮次，连续 5 轮没有更好的 Train-inner 候选则停止。每个种子的轮次，以及最终种子的选择，全部只用 Train-inner；不用 Dev 选择学习率、阈值或最佳种子。先比较是否满足保护条件，再比较原有 30% Online + 35% Regular + 35% Heldout Macro-F1。即便没有满足保护条件，仍保存诊断候选并报告失败，不能当成有效新模型。
3. **证据门槛**：至少 2/3 种子同时通过 Inner、Dev Select 的保护条件，且由 Inner 选出的候选也通过，才对这个唯一候选测原 Audit。Inner 要求 Online EN-real 误报不增加，允许维持零错误；Dev Select/Audit 则要求 EN-real 误报严格下降。三个面板的 Macro-F1 均不能下降，Online 和 Noisy 的 EN/ZH Real 误报、Fake 漏检均不能增加，避免靠新增 Online Fake 漏检换取 Real 召回。Audit 还给出 500 次按来源成组、按语言/类别分层的配对 bootstrap。只有 EN-real 改善的区间上界小于零，三个面板 Macro-F1 的区间下界不小于零、各类错误率变化的区间上界不大于零，才标记 `joint_eligible=true`。这些是逐项描述性 95% 区间，不是多重检验校正后的性能保证。

**本版不会自动开展第三阶段联合微调。** 先拿真实实验结果判断路线；通过后再审阅要开放哪些参数及训练预算。没有证据时不会通过新增门控或损失强行继续。提交脚本也会拒绝未通过本次验证门槛的残差。

请区分两种泛化：Train-inner 只对新残差/探针是未见来源，父 checkpoint 已训练过旧 Train。原 Dev Select/Audit 与 Train 来源隔离，但 Dev 在历史版本中已用于开发，不能称为全新盲测。原 Dev 划分完全不变，也不把 Dev EN-real 放入训练。Heldout Noisy 沿用父协议的模拟处理，不等同于真实未见 RTC 平台。

## 数据与效率

- 原音频读取、轻度增强、加噪→RTC、活动抑制、短音频处理均直接调用 V3.18.10/V3.18.9 的现有函数和父配置，不新增 DSP 配方。覆盖提取按来源遍历，并调用父采样器的 `view_spec` 和条件控制；它不是逐张票据复演父元训练任务。
- 默认每个 Fit 来源/流缓存 **2 次访问**，每次 Offline 访问只生成 **1 条 Noisy Online**。训练时轮换已有缓存。同一轮的每条流曝光仍为 EN-fake 9668、EN-real 8000、ZH-fake 8000、ZH-real 8000（读取实际父配置）。缓存覆盖全部来源，不等于每轮训练全部 ZH-fake。
- 使用原有八组等权 CE 和父配置的 label smoothing；无额外先验损失。残差 AdamW 初始学习率 `3e-4`、weight decay `1e-4`，原分类器冻结。默认特征 batch 为 256，这只是小型缓存向量，不是把 256 条波形送进 Omni。
- **有限缓存不是无限动态增强。** 2 次访问不能覆盖每条源的全部增强族；长期约 20% 原始、80% 增强的访问规则仍由父采样器提供，但两次访问不保证每条源都见到原始版本。需要更完整的第一轮覆盖可在新实验启动前用 `--cache-views 5`，代价是更长的特征提取。不能根据 Dev 结果反复调缓存、学习率再把同一 Dev 当盲测。
- 编码和后端各一次，缓存仅包含 FP32 的 256+384 维特征和原 logits，不保存波形、长时序或新 3B 模型。每个视图约 2.57 KB 数组空间，另有少量来源元数据；所有缓存设置写入身份校验。提取完成后释放 GPU 中的父模型，残差训练使用小特征。只在需要最终 Audit 时再加载父模型。
- 提取默认 `microbatch=2`、`frame_budget=4000`、4 个 CPU worker、预取 1；避免用大型波形 batch 换速度。每 64 条缓存成一块，可中断续提。完整输入超过父模型规定的上限会明确报错，不悄悄裁切。
- 本地 CPU 已测真实 AASIST/Tiny encoder 的数值与流程；服务器上 3B 权重、真实数据吞吐和 CUDA FP16 行为仍需实际运行确认。

## 更新与启动

### 已完成旧版特征复核：直接继续残差训练

若旧版在 `stop_global_hypothesis_not_supported` 停止，且已经生成两个探针的完整报告，用下面的入口。它识别已发布的 v1 代码哈希并校验原报告、父快照和缓存；在新目录训练残差，原目录及其报告保持不变。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin v3.18.12-global-residual
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v31812.sh --continue-from /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v31812_20261019_144138 --upload-report
```

**不会重提 `train_0`、`train_1`、`train_inner` 或 `dev_select`，不会重新训练 global/local 探针。** 这里只读取校验后的缓存，以及父快照中原分类器的 384→2 权重；无需在残差训练前加载 3B 编码器。接下来是三个独立种子的残差训练，每个最多 20 轮，5 轮无改善早停，默认学习率仍为 `3e-4`。只有满足最终验证条件、需要首次 Audit 时才重新加载父模型。

新目录名以 `_continued` 结尾，启动打印 `RUN=...` 并更新 `exp/.latest_v31812_run`。数据和模型快照在同盘上以只读用途的硬链接复用，不能硬链接时复制；特征缓存直接读取原目录，**请保留原 run 的 `cache` 目录**。源文件、票据、模拟参数或分片哈希不匹配会明确报错，不会悄悄重算或接受另一版特征。训练参数可以在创建新 continuation 时指定；缓存与增强参数继承旧实验。

这个入口只执行一次。中断新实验后使用下面已有的 `--resume "$RUN"` 恢复新目录；不要再次 `--continue-from` 创建另一份训练。旧 v1 的 `--resume` 不会绕过代码身份校验。

`--upload-report` 在结束时上传轻量报告并打印 `REPORT_URL=...`，同时写入 `report_url.txt`。上传失败会保留训练结果与本地 `report.zip`。日志现在明确区分探针诊断和实际 `inner_guard`，每个候选还会打印完整 Dev Select 的各组 FakeR、RealR、F1、AUC。

### 从头建立本版特征缓存

先确保 V3.18.10 已完成或使用其停止脚本安全暂停。只关闭日志查看不等于停止训练。父 run 必须保留 `last.pt`、`completed.json` 或 `paused.json`、配置、来源登记表及验证协议。首次运行会验证父文件、实际素材、预训练权重与固定运行环境，然后把冻结的检测器 partial 权重复制进新 run；不会复制多 GB 官方 Omni 基座。

首次切换新分支：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch --track origin/v3.18.12-global-residual
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v31812.sh
```

默认读取 `exp/.latest_v31810_run` 中的 best。如需指定另一份已完成的 V3.18.10：

```bash
bash run_w2v_v31812.sh --parent-run /完整路径/w2v_v31810_实际目录
```

这条脚本是前台训练，Ctrl+C 会中断本次工作。希望断开终端也继续运行时，使用：

```bash
mkdir -p exp
LOG="$PWD/exp/v31812_launch_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_w2v_v31812.sh > "$LOG" 2>&1 < /dev/null &
printf '%s\n' "$LOG" > exp/.latest_v31812_log
tail -f "$LOG"
```

这里的 `tail -f` 只是查看，Ctrl+C 只退出查看。停止后台实验：

```bash
bash stop_w2v_v31812.sh
```

确认日志出现 `Paused` 后，可从最后完整特征块/残差轮次续跑：

```bash
RUN=$(cat exp/.latest_v31812_run)
rm -f -- "$RUN/stop_requested"
bash run_w2v_v31812.sh --resume "$RUN"
```

恢复使用保存的全部配置；新传入的调参选项不用于覆盖既有实验。需要改变配置，建新 run。父模型、DSP 代码或缓存身份变化会拒绝复用；不使用不兼容缓存掩盖差异。

## 结果与提交

`training.log` 保留每一轮；`feature_review.json` 是全量探针对照，`residual_results.json` 是各随机种子的分类与同源错误变化，`summary.json` 给出最终结论，`report.zip` 收集轻量诊断文件和带来源 ID 的逐条预测，不含音频、特征缓存或权重。

`both_wrong` 统计原 Online 和 Noisy 都错的来源；`online_correct_noisy_wrong` 统计原 Online 对而 Noisy 新增错误的来源。分别记录纠正和新引入的错误，并按语言、真假及面板拆分。

通过门槛之后，可以对同一个共享编码器、一个最终分类器生成 submission：

```bash
RUN=$(cat exp/.latest_v31812_run)
OUT="/home/ubuntu/LXT/temp/$(basename "$RUN")_submission_$(date +%Y%m%d_%H%M%S)"
bash submit_w2v_v31812.sh --run "$RUN" --out "$OUT" \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

没有使用 Progress 拟合阈值、更新权重或选择残差。原 parent partial、原官方权重和最终残差均受校验；不会对多个随机种子的输出进行平均。此架构说明不替代竞赛方对具体提交的最终认定。

## 本地检查

```bash
python -m unittest discover -s w2v_v31812 -t . -p 'test_*.py' -v
```

测试覆盖原模型零残差等价、训练后在线/缓存路径一致、短输入与窗口统计、冻结参数、八组 CE、曝光数量、PCM 来源隔离、真实音频缓存与续提、缓存损坏拒绝、精确续训、零误报上限、Online Fake 保护、配对来源 bootstrap、阶段停止与单一候选 Audit。新增的完整 continuation 回归测试确认原实验内容不变、父编码器和音频提取函数不会被调用、两个探针不重训、三个残差种子实际完成 CPU 更新，并拒绝缺失分片或改变的提取代码。
