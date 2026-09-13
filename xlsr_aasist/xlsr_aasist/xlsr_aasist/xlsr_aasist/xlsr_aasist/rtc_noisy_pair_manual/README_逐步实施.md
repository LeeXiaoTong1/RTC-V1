# RTC 配对训练的含噪通信扩展：手动添加代码

这一版落实你选定的三个补充：加入 Offline 先加噪、再通信仿真的样本；保留当前真假分类和真实通信配对约束，将同类判别约束扩展到含噪通信；加入固定含噪 Dev，兼顾真实 Online 与含噪条件选模。

代码参考当前 RTC-V1 的 RTC 配对版本，采用新增文件的方式交付。没有修改、提交或推送你的 Git 仓库。之前提供的恢复前端代码包不参与这次训练。

## 一、这次模型实际学什么

模型仍是同一个 XLS-R＋AASIST。每步把三部分数据拼接起来，进行一次批量前向计算：

| 数据部分 | 默认数量 | 输入处理 | 监督 |
|---|---:|---|---|
| 原有普通训练样本 | 24 条 | 保留 V1 的 RawBoost→随机 MUSAN | 真假分类 |
| 真实通信配对 | 4 对，即 8 条 | 官方对应的 Offline 与 Online，不额外增强 | 真假分类＋现有 RTC 配对对比损失 |
| 含噪通信配对 | 4 对，即 8 条 | Offline 原音频与其“先加噪、再通信仿真”版本 | 真假分类＋新增含噪配对对比损失 |

合计 **40 条波形**。没有额外恢复网络、教师模型或独立检测器。推理仍只输入一条待检测音频。

两种配对流可以选到不同的原始语音，依靠共享模型联合训练，并不要求每步都凑出同一条语音的严格三元组。这样能够保留现有真实配对采样方式，也让没有成功传输得到 Online 的 Offline 参与含噪配对。

所有原始 Train 样本仍通过普通训练流每轮遍历一次。官方真实配对继续用已经生成的 `dataset/train_rtc_pairs.jsonl`，不会根据相似文件名、说话人或文本重新配对。

## 二、损失定义和新增部分

```python
loss = ce + 0.1 * rtc_real + noisy_weight * rtc_noisy
```

- `ce`：对上述所有波形进行真假分类，沿用 fake=0、real=1 和原来的类别权重 `[0.1, 0.9]`。
- `rtc_real`：直接复用现有 `utils/rtc_loss.py`，作用于官方 Offline/Online 对。
- `rtc_noisy`：复用同一个损失函数，作用于 Offline/其含噪通信仿真版本。
- `noisy_weight`：前两轮按训练步线性从接近 0 增长到 0.1，之后固定。现有真实 RTC 项始终保持 0.1。

两种配对损失都把同源样本当作正配对，把相反真假类别当作负样本；其它同标签语音不强行作为正样本。真实配对、含噪配对分别维持真假数量平衡，保证每步能找到相反类别。

约束仍放在已有模型分类前、最后一次 Dropout 之前的 160 维表示上。本次不改 `model/model.py`，不改参数键名，也不把原来的对比损失替换成纯余弦拉近。

## 三、含噪通信样本如何生成

每条 Offline 音频生成四个版本：

| 版本 | 加噪前设置的目标信噪比 |
|---|---|
| 0 | 5–10 dB 内随机取值 |
| 1 | 10–15 dB 内随机取值 |
| 2 | 15–20 dB 内随机取值 |
| 3 | 20–25 dB 内随机取值 |

实际顺序是：

**读取整条 Offline → 加入 MUSAN 环境噪声 → 本地降噪 → 动态音量调整 → Opus 编解码 → 按基线裁取前 64600 点／循环补齐。**

本次采用前一个代码包中已经跑通的 **FFmpeg `afftdn`＋`dynaudnorm`＋`libopus`**。这是本地通信效应的近似模拟，不是 WebRTC，也不能当作微信、Zoom 等真实平台的复现。与较早草案中的 WebRTC NS/AGC 相比，这是明确的工程实现差异；本次没有附带一个未经本地验证的 WebRTC 后端。

| 处理项 | 本代码实际参数 |
|---|---|
| 输入、输出 | 16 kHz、单声道 |
| `afftdn` 降噪强度 | 6、12、18 中随机选一个，`nf=-35`，开启噪声跟踪 |
| `dynaudnorm` 最大增益因子 | 2、4、8 中随机选一个，峰值目标为 0.9 |
| Opus | VoIP，20 ms 帧，可变码率；16、24、32 kbps 中随机选一个 |
| 处理状态 | 一整条音频连续处理，每条新音频重新初始化 |
| 输入裁剪 | 保持旧版首段裁剪；没有新增未经核验的随机时间对齐 |
| 额外 RawBoost | 新的含噪通信分支不叠加；原有普通训练分支保留 |

噪声、信噪比和通信配置分别抽样，不把某个信噪比固定绑定某一套降噪和编码配置。处理之后不再强行调整回原信噪比。

MUSAN 混音沿用 V1 的有效能量估计和整体增益缩放，避免靠硬截断峰值处理混音溢出。这里的有效能量估计不是精确语音活动标注。

Opus 使用 Ogg 封装中的预跳过与尾部裁剪信息处理编码长度。代码核对解码长度，不重复手动平移波形。这不意味着真实 Online 的时间轴已经与 Offline 逐采样对齐，因此本版继续使用整段表示约束。

四个版本一次生成并缓存，训练读取时随机选其中一个；Dev 的全部四个版本每轮保持固定。真实语音和伪造语音采用相同处理，标签始终继承原始音频。

## 四、向项目添加文件

解压本包后，把下列文件按原有相对路径逐个添加到：

```text
/home/ubuntu/LXT/RTC/xlsr_aasist
```

| 顺序 | 新文件 | 作用 |
|---|---|---|
| 1 | `rtc_noisy/__init__.py` | 新的 Python 包 |
| 2 | `rtc_noisy/common.py` | 音频格式、信噪比分组和噪声来源核对 |
| 3 | `rtc_noisy/simulator.py` | 本地通信仿真 |
| 4 | `prepare_rtc_noisy.py` | 生成 Train/Dev 含噪通信缓存 |
| 5 | `rtc_noisy/data.py` | 缓存验证、含噪配对读取 |
| 6 | `rtc_noisy/training.py` | 两类配对损失及分信噪比验证 |
| 7 | `main_train_rtc_noisy.py` | 新的训练入口 |
| 8 | `run_train_rtc_noisy.sh` | 路径和默认参数 |
| 9 | `check_rtc_noisy.py` | 本地处理检查、可选真实模型梯度检查 |
| 10 | `main_eval_rtc_noisy.py` | 兼容原模型的评分与提交打包 |
| 11 | `run_eval_rtc_noisy.sh` | Progress 评分命令 |
| 可选 | `tests/test_rtc_noisy.py` | 小规模自动测试 |

`rtc_noisy` 是新增目录，不要把外层 `rtc_noisy_pair_manual` 当作运行目录。向已有 `tests` 目录只添加新测试文件，保留之前的文件。

依赖的是你已经应用并跑通的 RTC 配对代码，包括 `main_train_rtc.py`、`utils/rtc_data.py`、`utils/rtc_loss.py`、`utils/rtc_pairs.py` 和 V1 的 `utils/env_noise.py`。本包没有覆盖这些文件。

## 五、先检查运行环境

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC/xlsr_aasist

export XLSR_PRETRAINED="/home/ubuntu/LXT/RTC/pretrained/xlsr2_300m.pt"

python check_rtc_noisy.py --rtc
```

若提示缺少 FFmpeg，再执行：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
python check_rtc_noisy.py --rtc
```

沿用现有 sdd 环境中的 PyTorch、librosa、NumPy、SciPy、soundfile 和 fairseq，无需重新安装整个 requirements 或升级 pip。

如需先检查你真实 XLS-R＋AASIST 的输出与对比梯度，可执行：

```bash
CHECKPOINT="./exp/xlsr_aasist_V1_RTC_pair_epoch100_bs40_20260910215821754634/ckpt/best_model.pth"

python check_rtc_noisy.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0
```

这个检查使用合成输入，不更新模型，验证 2 类输出、160 维表示以及对比损失回传。它不是性能测试。初始化使用的是你已经完成的 RTC 配对模型，不再回到最初的 V1/B 模型。

## 六、生成训练缓存

```bash
python prepare_rtc_noisy.py \
  --dataset_root ./dataset \
  --split train \
  --noise_manifest /home/ubuntu/LXT/RTC/external_noise/noise_split/train.jsonl \
  --output ./dataset/rtc_noisy_cache/train \
  --workers 4
```

该命令读取 `dataset/train_label.txt` 和 `dataset/wav/train`，只为其中的 Offline 生成缓存。所有 Offline 都可使用，不要求在官方配对表中有 Online。

程序沿用已有 MUSAN 噪声清单中的 `path`、`original_recording`、`sha256`、`split`。只使用 `noise` 子集，不使用 `speech` 或 `music`。缓存生成前会核对噪声文件是否与清单中的哈希一致。

缓存使用 FLOAT WAV，保留处理后的浮点输出。每个版本固定保存 64600 个采样点，约 0.258 MB。四个版本共约 1.034 MB/条 Offline：

- 假设训练 Offline 约 3.9 万条，训练缓存约 40 GB。
- 7739 条 Dev Offline 的四个版本约 8 GB。
- 请以程序打印的实际条数和空间估算为准。

处理发生在 CPU 上，训练时不会逐批调用 FFmpeg。同一命令中断后可以重跑，会复用已完成的同源版本；改变配置时使用新输出目录。

## 七、生成固定 Dev 缓存

```bash
python prepare_rtc_noisy.py \
  --dataset_root ./dataset \
  --split dev \
  --noise_manifest /home/ubuntu/LXT/RTC/external_noise/noise_split/dev.jsonl \
  --output ./dataset/rtc_noisy_cache/dev \
  --workers 4
```

读取 `dataset/dev_label.txt` 与 `dataset/wav/dev` 中的 Offline，生成相同四个信噪比区间的版本。与 Train 使用不同的完整噪声录音；训练入口会同时核对原始录音标识和文件哈希，防止两边复用同一录音。

通信参数从同一组候选值独立抽取，因此这个 Dev 检查的是保留语音、保留噪声上的模拟泛化，不能声称它模拟了一个真正未知的平台。真实 Online Dev 另行保留并评估。

`--limit` 仅用于在另一个目录创建小型试运行缓存；不完整缓存不能拿来正式训练或选模型。

## 八、检查数据，再正式训练

```bash
CHECKPOINT="./exp/xlsr_aasist_V1_RTC_pair_epoch100_bs40_20260910215821754634/ckpt/best_model.pth"

bash run_train_rtc_noisy.sh "$CHECKPOINT" --check_data
```

程序会核对：真实配对表、Train/Dev 缓存是否完整、真假标签是否对应、两侧噪声录音是否重叠，以及普通增强与训练缓存是否使用同一个 Train 噪声清单。它还会打印缺少 Online 配对但仍能用于新分支的 Offline 数量。

检查通过后运行：

```bash
bash run_train_rtc_noisy.sh "$CHECKPOINT"
```

默认参数为：

| 参数 | 设置 |
|---|---|
| 普通样本数 | 24 条/步 |
| 真实通信配对 | 4 对/步，真假各 2 对 |
| 含噪通信配对 | 4 对/步，真假各 2 对 |
| 实际波形批次 | 40 条/步 |
| 优化器 | Adam，学习率 `1e-6`，权重衰减 `1e-4` |
| 参数更新 | 沿用现有模型的可训练参数，不新增整体冻结操作 |
| 真实 RTC 损失系数 | `0.1` |
| 含噪 RTC 损失系数 | 前两轮按步升到 `0.1`，之后固定 |
| 混合精度 | 网络 BF16，分类与对比损失 FP32 |
| 普通训练增强 | RawBoost 算法 5＋MUSAN 概率 0.5、SNR 10–30 dB |
| 新增通信增强 | 四个 5–25 dB 分组，不额外叠加 RawBoost |
| 最大轮数 | 30 |
| 早停 | 本地选模指标连续 10 轮未改善 |
| 推理裁剪 | 保持基线首段 64600 点／循环补齐 |

这些参数是本次固定的实施起点，尚未在你的服务器上验证为最优值。原有真实配对和新含噪配对分别采样，同一个 Offline 可以有多个真实 Online 对应，不受一一对应假设限制。

每轮仍遍历全部普通训练样本。但普通批次从旧版默认 32 调到 24 后，75785 条音频每轮约 3158 步；相比旧版 2369 步增加约三分之一。验证还新增了四份含噪 Offline，因此总耗时会增加，不能因为波形批次仍为 40 就认为每轮一样快。

若显存不足，先减少普通样本数。例如总批次变为 32 条：

```bash
ORDINARY_BATCH_SIZE=16 bash run_train_rtc_noisy.sh "$CHECKPOINT"
```

如果只想运行 10 轮：

```bash
NUM_EPOCHS=10 bash run_train_rtc_noisy.sh "$CHECKPOINT"
```

不要把 `ORDINARY_BATCH_SIZE=40` 理解为总共 40 条；两种配对波形会额外加入。

## 九、如何看日志和选中的模型

开始训练前先对初始化模型运行 `Initial dev`。每轮日志显示：

| 名称 | 含义 |
|---|---|
| `CE` | 所有训练输入的真假分类损失，按优化步取平均 |
| `RTCReal` | 官方 Offline/Online 配对对比损失 |
| `RTCNoisy` | Offline/含噪通信仿真配对对比损失 |
| `NoisyWeight` | 本轮新增损失系数的平均值；前两轮不是最终 0.1 |
| `TrainAcc` | 本轮全部输入波形的训练准确率 |
| `CleanOnlineF1` | 真实 Online Dev 的 Macro-F1 |
| `NoisySimF1` | 四个信噪比区间各自 Macro-F1 的平均值 |
| `DevProxy` | 本地选模指标，不是官方 weighted |
| `TrainMinutes` | 本轮训练时间，不含验证 |

选模指标：

```text
NoisySimF1 = (F1_5–10 + F1_10–15 + F1_15–20 + F1_20–25) / 4
DevProxy = 0.3 × CleanOnlineF1 + 0.7 × NoisySimF1
```

这里先计算各区间 F1 再平均，与把四组混在一起算一次 F1 不同。评分阈值固定为 0.5，伪造概率等于 0.5 时判为伪造。指标在终端显示为百分数，在 JSON 中保存为 0–1。

每轮同时记录 Clean Dev 的 Offline、Online、整体统计，以及四个含噪区间的混淆矩阵、准确率和损失。检查 CleanOnline 是否退化、低信噪比是否改善，不要只看一个代理总分。

新实验目录保存：

- `config.json`：初始化权重哈希、缓存来源、噪声环境变量和参数。
- `metrics.jsonl`：初始化结果及逐轮结果、输入数量和信噪比抽样数量。
- `ckpt/best_model.pth`：按本地代理指标选中的权重，格式仍兼容旧评估入口。
- `ckpt/best_model.meta.json`：这个 best checkpoint 对应第几轮、各项 Dev 指标。
- `ckpt/last_model.pth` 和对应 `.meta.json`：最后完成一轮的模型。
- `ckpt/epoch_N_proxy_....pth`：刷新最佳时保留的权重。

初始化权重也是候选。如果训练没有超过初始指标，best 可以仍对应第 0 轮；不把退步后的模型自动当作改进结果。各轮权重用于推理或重新开始微调，本版没有宣称保存优化器状态后精确续训。

## 十、评估和生成提交文件

读取你本次实际生成的 best 路径，然后评分：

```bash
read -r -p "请输入新实验 best_model.pth 的完整路径：" NEW_CHECKPOINT
bash run_eval_rtc_noisy.sh "$NEW_CHECKPOINT" ./exp/eval/RTC_noisy_pair cuda:0
```

生成：

```text
./exp/eval/RTC_noisy_pair/progress_scores.txt
./exp/eval/RTC_noisy_pair/submission.zip
```

ZIP 根目录只有 `scores.txt`，内容为官方 ID 与第 0 类伪造概率；检查 ID 顺序、数量及分数是否有限。这里仅生成文件，不会自动上传提交。

你也可以继续使用原来的 `run_eval_rtc.sh` 评估新 checkpoint。新的评估入口沿用同一套首段裁剪和单模型输出，没有增加三个窗口聚合或测试时通信处理，以便与上一版直接比较。

## 十一、可选消融开关

本版直接训练完整方案。若后续需要区分新样本与新损失的贡献，保留完全相同的数据流，只关闭新增配对项：

```bash
NOISY_WEIGHT=0 TRACK=xlsr_aasist_RTC_noisy_data_control \
  bash run_train_rtc_noisy.sh "$CHECKPOINT"
```

它仍包含新增含噪通信样本的分类监督，以及原来的真实 RTC 配对损失，所以不是旧版 RTC 模型的原样重跑。

## 十二、验证范围与出处

交付前完成了 8 项 CPU 测试，覆盖实际 FFmpeg 处理、完整音频处理后裁剪、缓存断点复用、缺失 Online 配对的保留、标签对应、Train/Dev 噪声隔离、四个区间 F1 取平均、0.5 阈值、两种配对损失的梯度、原格式 checkpoint 兼容和提交 ZIP 内容。

其中一项使用现有完整 AASIST 后端，验证两种损失均能反向传播并保持参数键名；该测试用小型可求导模块替代了 XLS-R。另一项跑通小型合成数据的缓存、训练、保存、评估全过程。没有在这里使用你的真实 XLS-R 权重和 A100 完成训练，因此不提供新的比赛成绩或速度保证。

添加文件后如需运行同一套测试：

```bash
python -m unittest discover -s tests -p 'test_rtc_noisy.py' -v
```

方法方向参考了对处理失真的研究，但本版是现有代码的扩展，不是 PCL 或 TFCL 的完整复现。[TFCL 原文](https://arxiv.org/html/2607.17761v1)采用了时间与频率结构约束，不能把其论文收益直接视为本实现的预期收益。

仿真调用参考 FFmpeg 官方文档：[afftdn](https://ffmpeg.org/ffmpeg-filters.html#afftdn)、[dynaudnorm](https://ffmpeg.org/ffmpeg-filters.html#dynaudnorm)、[libopus](https://ffmpeg.org/ffmpeg-codecs.html#libopus)。

外部噪声、训练数据构造与单模型评分遵循本次核验的[官方数据流程与规则](https://www.junxue.tech/rtc-sdd-challenge/#rules)：不使用外部语音训练，不向真实 RTC 平台重传官方数据，不使用 Progress/Eval 训练或选模型。本地含噪 Dev 始终只是代理验证条件。
