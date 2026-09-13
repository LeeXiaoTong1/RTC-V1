# RTC 含噪配对训练 V2：手动更新说明

本包实现四项优化，继续使用现有 XLS-R＋AASIST、MUSAN 和 FFmpeg 处理链。每次访问一条 Offline，仍然只取一个含噪通信版本。默认每批总计 40 条波形，采用原有 BF16 自动混合精度训练。

本包是在已有 `RTC_noisy_pair_manual.zip` 和真实 RTC 配对训练代码之上增加的文件。请将它放进你已经能够运行上一版的项目；它不是一个独立的全新仓库。

## 1. 四项修改是什么

| 修改 | V2 的实现 |
|---|---|
| 类别权重 | 普通训练分支保留 `fake=0.1, real=0.9`；真假均衡的真实 RTC 配对、含噪配对全部使用等权 CE |
| 含噪分类监督 | 含噪通信音频的平均分类损失单独占 `0.3`；其他输入的分类损失合计占 `0.7` |
| 四版本轮换 | 每条 Offline 的四个版本打乱后逐个使用，取完再重新打乱；轮换状态跨 epoch 保留 |
| 泛化验证 | FFmpeg 的 27 个参数组合分为 21 个训练组合和 6 个留出组合；增加固定的留出组合 Dev |

分类类别顺序沿用原项目：伪造为 0，真实为 1。以下均为工程起始配置，没有预先承诺分数改善。

### 分类损失

默认批次顺序如下：

| 输入组 | 波形数 | 组内平均方式 |
|---|---:|---|
| 普通训练音频 | 24 | 保留原来的加权 CE，除以本组样本权重之和 |
| 真实 Offline/Online 配对 | 8（4 对） | 等权 CE |
| 含噪配对中的 Offline 参照 | 4 | 等权 CE |
| 先加噪、再经 FFmpeg 处理的音频 | 4 | 等权 CE |

具体计算为：

```text
CE_other = (24 × CE_ordinary + 8 × CE_real_pair + 4 × CE_reference) / 36
CE_total = 0.7 × CE_other + 0.3 × CE_noisy_processed
Loss     = CE_total + RTCWeight × RTCReal + NoisyWeight × RTCNoisy
```

四个 CE 项均先在各自组内求平均。最后一个不完整普通批次按实际人数计算 `CE_other`，含噪分类项仍占 0.3。

`NOISY_CE_WEIGHT=0.3` 控制含噪音频的**真假分类**；`NOISY_WEIGHT=0.1` 继续控制含噪配对的**对比约束**，两者分开。真实 RTC 对比系数仍为 0.1，含噪对比系数仍在前两轮逐渐升到 0.1。这里的 30% 是损失组合系数，不代表梯度范数必然占 30%。

普通分支继续使用原有 RawBoost 和 MUSAN 配置；两个配对分支不再额外叠加随机增强。模型结构、同源配对关系和推理方式保持兼容。

### 四版本轮换

四个版本对应加噪前设定的四个 SNR 区间：`5–10`、`10–15`、`15–20`、`20–25 dB`。每个版本还独立抽取一段噪声和一个允许的 FFmpeg 参数组合。它们的含噪波形预生成后固定，并不是每次读取重新做通信处理。

例如某条 Offline 的访问顺序可以是：

```text
第 1～4 次访问：版本 2、0、3、1
第 5～8 次访问：版本 1、3、0、2
```

每组四次访问覆盖四个版本。组与组交界处允许相同版本连续出现；这不影响每个完整四次周期的覆盖。它不是“任意滑动的四次访问都不重复”。

轮换按**每条 Offline 实际被抽样的次数**推进，跨 epoch 保留。一个 epoch 不保证每条 Offline 都被访问四次，更不保证遍历它的全部版本。

实现中由主进程预先生成 `(source, bank, band)` 取样指令，DataLoader 工作者只按指令读取音频。因此不会让每个 worker 各维护一套轮换状态。预取不提前提交访问计数；训练完成整轮后才提交。记录保存在 checkpoint 旁的 `.meta.json` 中。

## 2. 上传与复制

将 `RTC_noisy_pair_v2_manual.zip` 上传到：

```text
/home/ubuntu/LXT/RTC/xlsr_aasist/
```

在服务器运行下面的命令。代码块仅包含需要执行的命令。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC/xlsr_aasist
unzip -q RTC_noisy_pair_v2_manual.zip
cp -r RTC_noisy_pair_v2_manual/rtc_noisy_v2 ./
cp RTC_noisy_pair_v2_manual/main_train_rtc_noisy_v2.py ./
cp RTC_noisy_pair_v2_manual/prepare_rtc_noisy_v2.py ./
cp RTC_noisy_pair_v2_manual/check_rtc_noisy_v2.py ./
cp RTC_noisy_pair_v2_manual/run_prepare_rtc_noisy_v2.sh ./
cp RTC_noisy_pair_v2_manual/run_train_rtc_noisy_v2.sh ./
mkdir -p tests
cp RTC_noisy_pair_v2_manual/tests/test_rtc_noisy_v2_*.py ./tests/
cp RTC_noisy_pair_v2_manual/README_RTC_NOISY_V2.md ./
```

所有新增文件使用 `v2` 名称，不覆盖原训练入口、模型文件或旧缓存。现有 `rtc_noisy/`、`utils/rtc_*.py`、`main_train_rtc_noisy.py` 等文件需要继续保留。

先检查环境与实现：

```bash
python check_rtc_noisy_v2.py --tests --rtc
```

该命令在 CPU 上检查损失梯度、一个批次一次前向、优化器更新、checkpoint 兼容性、真实 DataLoader 多进程轮换，并检查 27 个 FFmpeg 组合的输出长度。它不加载 XLS-R 权重、不启动正式训练。

沿用已经跑通的 `sdd` 环境；无需重新安装 `requirements.txt`。本包依赖旧版已经使用的 PyTorch、NumPy、SciPy、librosa、soundfile、tqdm、fairseq 和支持 `afftdn/dynaudnorm/libopus` 的 FFmpeg。如果 FFmpeg 仍出现长度变化，先解决环境问题；程序不会靠补零或截断掩盖处理链输出长度错误。

## 3. 生成 V2 缓存

为了真正把 6 个组合从这一轮训练数据中排除，需要重新生成 V2 缓存。旧版在全部 27 个组合中抽样，不能直接改文件夹名当作 V2 缓存。

默认路径：

| 内容 | 路径，相对于项目根目录 |
|---|---|
| 官方音频 | `dataset/wav/train`、`dataset/wav/dev` |
| 官方标签 | `dataset/train_label.txt`、`dataset/dev_label.txt` |
| 训练噪声清单 | `../external_noise/noise_split/train.jsonl` |
| Dev 噪声清单 | `../external_noise/noise_split/dev.jsonl` |
| 训练缓存 | `dataset/rtc_noisy_cache_v2/train_g0` |
| 训练组合 Dev | `dataset/rtc_noisy_cache_v2/dev_seen` |
| 留出组合 Dev | `dataset/rtc_noisy_cache_v2/dev_heldout` |

生成命令：

```bash
bash run_prepare_rtc_noisy_v2.sh
```

也可以只调整预处理并发数：

```bash
CACHE_WORKERS=8 bash run_prepare_rtc_noisy_v2.sh
```

缓存生成在 CPU 上执行。生成中断后，用相同参数重跑会复用已完成的文件。参数、种子或数据发生变化时，请使用新目录。不要在训练读取某套缓存时重建那套缓存。

每个缓存波形为 64,600 点、单声道、Float32，音频负载约 258 KB。总计生成：

```text
4 × Train Offline 数量 + 8 × Dev Offline 数量
```

条音频。按当前数据规模，新增缓存大致需要数十 GB（约 56 GB，具体以生成脚本打印为准），还不包含旧缓存和 checkpoint。两个 Dev 都保留每条 Offline 的四个版本；训练时仍然每次只读一个。

### 21/6 个 FFmpeg 组合如何划分

继续使用同一条处理链：`afftdn → dynaudnorm → Opus`。三个可变参数分别为：

- `noise_reduction`：6、12、18。
- `max_gain`：2、4、8。
- `bitrate`：16,000、24,000、32,000 bit/s。

以下六个**组合**仅用于 `dev_heldout`，其余 21 个用于 Train 和 `dev_seen`：

| noise_reduction | max_gain | bitrate |
|---:|---:|---:|
| 6 | 2 | 16,000 |
| 12 | 4 | 24,000 |
| 18 | 8 | 32,000 |
| 6 | 4 | 32,000 |
| 12 | 8 | 16,000 |
| 18 | 2 | 24,000 |

每个单独的参数取值都在训练中出现；留出的是它们的组合。两边每个参数取值的边际频数也相同，避免留出集只是系统性地使用更强或更弱的某一个参数。

`dev_seen` 和 `dev_heldout` 对同一原音频、同一 SNR 档使用完全相同的噪声混合结果，只有通信参数组合不同。脚本记录混合波形哈希，启动训练时核对配对。Train 与 Dev 的噪声按完整原始录音隔离。

这是同一个 FFmpeg 模拟器内的组合泛化检查，不能把它解释为已经覆盖了未知真实 RTC 平台。

## 4. 初始化与启动训练

若要明确检查“训练没见过的通信组合”，建议从你之前的**真实 RTC 配对模型、尚未加入 FFmpeg 模拟的 checkpoint**开始：

```bash
CHECKPOINT="./exp/xlsr_aasist_V1_RTC_pair_epoch100_bs40_20260910215821754634/ckpt/best_model.pth"
bash run_train_rtc_noisy_v2.sh "$CHECKPOINT" --check_data
```

数据检查通过后：

```bash
bash run_train_rtc_noisy_v2.sh "$CHECKPOINT"
```

这是一次从该权重开始的 V2 微调，不会覆盖旧实验。

你也可以将 `CHECKPOINT` 换成当前 Noisy=84.0525 的 FFmpeg 模型继续微调，代码支持。但它此前可能已经见过这六个组合，此时 `dev_heldout` 只能称为“本次训练阶段留出的条件”，不能宣称对整个训练历史都未见。脚本会读取初始化实验的配置并记录这一点；配置缺失时标记为未知，不靠目录名推断。

默认训练参数：

| 参数 | 默认值 |
|---|---|
| 普通输入 / 真实 RTC 对 / 含噪对 | 24 / 4 / 4，共 40 条波形 |
| 含噪分类系数 | 0.3 |
| 真实 RTC / 含噪对比系数 | 0.1 / 0.1；含噪项前两轮升到目标值 |
| 普通分支 MUSAN | 概率 0.5，SNR 10–30 dB |
| 学习率 / 权重衰减 | `1e-6` / `1e-4` |
| 优化器 | Adam |
| 精度 | BF16 自动混合精度；CE/对比损失用 FP32 计算 |
| 最长训练 / 早停耐心 | 30 轮 / 连续 10 轮指标不改善 |
| 输入长度 | 沿用 64,600 点；首段裁剪、短音频循环补齐 |

BF16 是 16 位格式，FP32 是 32 位格式。模型参数保持常规 FP32，由 autocast 选择适合的计算精度，并不是把 FP32 称为半精度。

如需指定本版缓存位置，请使用新的变量，避免继承上一版的路径：

```bash
RTC_V2_CACHE_ROOT="./dataset/rtc_noisy_cache_v2" bash run_train_rtc_noisy_v2.sh "$CHECKPOINT"
```

独立路径也可通过 `RTC_V2_TRAIN_CACHE`、`RTC_V2_DEV_SEEN_CACHE`、`RTC_V2_DEV_HELDOUT_CACHE` 设置。

每次 `--model_path` 都代表开始一次新微调，重建优化器并从新的轮换周期开始。本包保存轮换历史以便核查，但没有实现含优化器和全部随机状态的精确断点续训；不要将再次传入权重误认为完全恢复中断的训练。

## 5. 如何看指标与选 checkpoint

新增日志会分别显示：

- `CEOrd`、`CEReal`、`CERef`、`CENoisy`：四组分类损失。
- `RTCReal`、`RTCNoisy`：两类配对约束。
- `NoisyCEWeight`、`NoisyWeight`：分类与对比的不同系数。
- `CleanOnlineF1`：官方 Dev 中真实 Online 条件的 Macro-F1。
- `NoisySeenF1`：训练组合 Dev 的四档 Macro-F1 均值。
- `NoisyHeldoutF1`：留出组合 Dev 的四档 Macro-F1 均值。
- `FakeAsReal`：伪造被判成真实的比例，分母为伪造样本数。
- `RealAsFake`：真实被判成伪造的比例，分母为真实样本数。

百分号形式用于终端显示，`metrics.jsonl` 内 F1 和错误率通常为 0～1 小数，训练准确率 `acc` 为百分数。混淆矩阵行是真实标签、列是预测标签，顺序均为 `[fake, real]`。

本版在训练开始前固定使用：

```text
DevRobustProxy = 0.30 × CleanOnlineF1
               + 0.35 × NoisySeenF1
               + 0.35 × NoisyHeldoutF1
```

选模。含噪部分合计仍占 70%，两种条件各占一半。原公式 `0.3 × CleanOnlineF1 + 0.7 × NoisySeenF1` 也会作为 `DevProxyV1` 记录；因为缓存重建，其数值不能直接当作旧版同一验证集上的对照。

留出组合 Dev 参与选模，所以它是开发集，不是未经选择的最终测试集。Progress/Eval 音频或提交分数不参与本代码的调参、抽样或选模。

最初加载的 checkpoint 也会先在新 Dev 上评估，作为第 0 轮候选。如果后续训练没有更好结果，`best_model.pth` 保留第 0 轮，避免默认用最后一轮覆盖更好的初始化。查看它对应哪轮：

```bash
python -m json.tool "你的新实验目录/ckpt/best_model.meta.json"
```

每轮保存 `last_model.pth`，指标提升时更新 `best_model.pth`。模型仍是普通 `state_dict`，使用原有评估入口：

```bash
bash run_eval_rtc_noisy.sh "你的新实验目录/ckpt/best_model.pth" "./exp/eval/rtc_noisy_v2" "cuda:0"
```

推理仍然是一条输入经过一个检测器，不加入通信模拟、不做多模型融合，也不使用四个缓存版本。

## 6. 可选：以后增加新缓存素材

默认只有 `train_g0`，每条 Offline 四个版本。若后续需要换新噪声裁剪和处理组合，可以预先生成第二个训练缓存库；Dev 保持不变：

```bash
python prepare_rtc_noisy_v2.py \
  --dataset_root ./dataset \
  --role train \
  --noise_manifest ../external_noise/noise_split/train.jsonl \
  --output ./dataset/rtc_noisy_cache_v2/train_g1 \
  --seed 1234 \
  --generation 1 \
  --workers 4
```

在一次新的微调开始时加入：

```bash
bash run_train_rtc_noisy_v2.sh "$CHECKPOINT" \
  --extra_train_noisy_cache ./dataset/rtc_noisy_cache_v2/train_g1
```

每条 Offline 在当前缓存库的四次周期完成后切换到下一个库，仍然每次只读取一个含噪版本。不要在训练运行中途替换缓存文件。本功能默认关闭，不是启动本版必须执行的一步。

## 7. 验证范围

交付端已运行 8 项不依赖 PyTorch 的检查，覆盖轮换跨 epoch/状态恢复/预取计划、损失组合系数、通信组合隔离、Dev 噪声配对及泄漏检查。已在 FFmpeg 6.1.1 上实际检查全部 27 个组合：输入输出均为 70,321 点，重复处理结果一致。Python 语法与 Bash 语法也经过检查。

交付端未安装 PyTorch、librosa、soundfile，依赖下载未获执行环境放行，因此未在这里运行 PyTorch 反向传播、多进程 DataLoader 和完整缓存/训练验证。这些 PyTorch 检查已经附在 `tests/test_rtc_noisy_v2_torch.py` 中，请在现有 `sdd` 环境执行上面的 `--tests`；它们不需要 GPU。实际缓存/数据兼容性由 `run_prepare_rtc_noisy_v2.sh` 和 `--check_data` 核查。

本包没有在你的数据上完成正式训练，不能提前判断四项修改各自贡献多少，也不能承诺超过当前 Noisy 分数。

参考接口：`RTC-V1` 的 `codex/rtc-paired-training` 分支（核对版本 `8de73be4926e6ccea5e289b04a8c4c2dd6304925`），以及此前交付的 `RTC_noisy_pair_manual.zip`。参考仓库未被本次工作修改。
