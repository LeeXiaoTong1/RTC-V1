# V1 上的 RTC 配对判别训练

保留 V1 的 XLS-R＋AASIST、RawBoost、MUSAN、真假分类损失与已有 checkpoint。
新增训练入口 `main_train_rtc.py`，从官方对应表读取同源 Offline／Online 样本，
在 AASIST 分类层前的 160 维特征上加入配对判别约束。评估仍是单条音频、单个模型。
本实现没有降噪前端、教师或音素识别模型。

## 1. 实际训练方式

默认每步一次前向，输入按顺序拼接为：

- 32 条原 V1 训练样本，仍使用 RawBoost → MUSAN → 裁剪/补齐。
- 4 条官方配对的 Offline 音频和对应的 4 条 Online 音频，均不额外增强。
- 合计 **40 条音频**。4 对中真假各 2 对，为对比损失提供异类负样本。

所有 40 条音频计算原有真假分类损失（类别权重仍为 fake=0.1、real=0.9）。
只有后 8 条原始配对音频计算 RTC 损失。每个锚点的正样本是同源对侧音频，
负样本只包含对侧不同真假标签的音频。同类但不同源的音频不参与该锚点的分母。
特征经 L2 归一化，使用温度缩放余弦相似度，双向取平均：

`总损失 = 全部输入的分类损失 + rtc_weight × RTC配对损失`

初始参数：`rtc_weight=0.1`、`rtc_temperature=0.1`、学习率 `1e-6`，训练 5 轮。
这些是起始实验配置，尚无真实训练结果证明其最优。
网络保留 BF16 混合精度，分类及 RTC 损失在 FP32 计算。

配对音频沿用 V1 的 16 kHz、前 64600 采样点裁剪和短音频重复补齐。
这里是同源语句/输入片段级约束，不做采样点或音素对齐；不能据此声称消除了通信延时。
所有未配对训练音频仍保留在普通分类数据流中。

**实验解释：**额外的干净配对样本会改变整体干净/含噪比例及类别比例。
MUSAN 的 0.5 概率只作用于普通的 32 条样本，全部 40 条中预期约 40% 加 MUSAN。
因此 `rtc_weight=0` 的成对采样对照是必须的，不能仅与历史 V1 分数比较便把收益归因于 RTC 损失。
每轮会读完原训练列表，另加入配对样本；总训练输入量多约 25%，实际耗时需在你的 GPU 上测量。

## 2. 获取真实配对文件

官方提供：

- [train_offline_online_pairs.csv](https://huggingface.co/datasets/JunXueTech/RTCFake/blob/main/data/meta/train_offline_online_pairs.csv)
- [官方 meta 目录](https://huggingface.co/datasets/JunXueTech/RTCFake/tree/main/data/meta)

需要你的 Hugging Face 账号已获官方数据集授权。仓库不包含数据，开发环境对 CSV 内容访问返回 401，
所以未臆造表头或生成任何真实配对结果。先在本地数据目录查找该文件；若没有，从获授权的官方页面下载，放到：

`/home/ubuntu/LXT/RTC/xlsr_aasist/dataset/meta/train_offline_online_pairs.csv`

在已有 sdd 环境内执行（无需重新安装 requirements.txt）：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC/xlsr_aasist
python prepare_rtc_pairs.py \
  --pairs_csv ./dataset/meta/train_offline_online_pairs.csv \
  --train_protocol ./dataset/train_label.txt \
  --train_data_path ./dataset/wav/train \
  --output ./dataset/train_rtc_pairs.jsonl
```

程序按官方 CSV 每一行读取关系，校验两端都属于训练协议、同真假标签、文件存在、
没有重复配对或同一 Online 对应多个不同 Offline。允许一个 Offline 对应多个 Online。
程序不会用编号、说话人或相同标签自行猜配对。输出旁的 `.meta.json` 记录来源哈希、配对数量和覆盖率。

官方列名 `offline_id` / `online_id` 可自动识别。传输失败产生的空对应字段会被跳过，
并计入 `skipped_unpaired_rows` 和 `skipped_unpaired_by_reason`；未配对音频仍留在原训练协议中，
照常参与分类训练。非空但无效的路径、标签冲突仍报错，不会被当作传输失败跳过。

常见且唯一的 `offline_path`/`online_path` 等列名可自动识别。若实际列名不同，程序会列出表头并停止。
这时给同一命令补充 `--offline_column` 和 `--online_column`，值必须是官方 CSV 实际列名。
不得将不明字段随意指定为路径列。

## 3. 从现有 B/V1 checkpoint 开始

以下使用你此前提供的 B 第 7 轮 checkpoint；如你实际保留的是其他 V1 权重，替换这一处。

```bash
CHECKPOINT="./exp/xlsr_aasist_B_noise_probe_epoch10_bs40_20260909193749/ckpt/epoch_7_dev_loss_0.157348.pth"

# 数据预检：不加载 XLS-R、不占用 GPU。
bash run_train_rtc.sh "$CHECKPOINT" ./dataset/train_rtc_pairs.jsonl --check_data

# 启用 RTC 配对损失。
bash run_train_rtc.sh "$CHECKPOINT" ./dataset/train_rtc_pairs.jsonl
```

脚本使用当前 Python 环境，不覆盖 PATH。默认数据目录相对于脚本所在目录，MUSAN 位于
`../external_noise/noise_split/train.jsonl`，SNR 10–30 dB。启动时会打印 MUSAN 配置、配对数量与实际音频批大小。
如果数据放在别处，可设置 `DATA_ROOT`、`DATASET_ROOT`、`RTC_B_NOISE_MANIFEST`。
XLS-R 默认保留 `/home/ubuntu/LXT/RTC/pretrained/xlsr2_300m.pt`；可通过 `XLSR_PRETRAINED` 环境变量
覆盖，或训练时传入 `--ssl_path`。评估时设置同一 `XLSR_PRETRAINED` 路径。

CUDA 不可用时新训练入口会明确停止，防止再次误用 CPU 长时间训练。
不支持 BF16 的设备可显式设置 `AMP=none`。A100 可继续用默认 BF16。

## 4. 必做的分类对照

从**同一个初始 checkpoint**重新运行，保持随机种子、步数、普通采样和配对采样一致：

```bash
RTC_WEIGHT=0 TRACK=xlsr_aasist_V1_RTC_pair_CE_control \
  bash run_train_rtc.sh "$CHECKPOINT" ./dataset/train_rtc_pairs.jsonl
```

这组保留同样的原始配对样本和分类训练，仅关闭 RTC 损失的梯度贡献；日志仍显示 RTC 损失以便比较。
不要从已经完成 RTC 微调的权重启动对照。
若需要严格固定相同的 5 轮训练预算，可对两组都设置 `EARLYSTOP_EPOCH=6`。

## 5. 验证与输出

开发集使用原始音频、不加增强。分别记录 Offline、Online、全部数据的混淆矩阵和 Macro-F1。
阈值为伪造分数 0.5，fake=0、real=1，伪造分数仍取 `softmax[:, 0]`。
默认按 **Dev Online Macro-F1 最大**选择 checkpoint；相同时选择 DevLoss 更小者。
可对两组统一传入 `--selection_metric dev_loss` 以采用损失选择。
程序会先验证初始 V1 权重，只有验证指标超过它才更新最佳权重。

每次实验目录包含：

- `config.json`：训练参数、初始权重哈希、协议和配对表哈希、MUSAN 设置。
- `metrics.jsonl`：第 0 轮初始验证，以及每轮 CE、RTC、有效锚点数、训练耗时、开发集各项指标。
- `ckpt/best_model.pth`：最佳模型参数；若所有微调轮次都不如初始模型，该文件保留初始 V1。
- `ckpt/last_model.pth`：最后完成的一轮模型参数。
- `ckpt/epoch_*_dev_loss_*.pth`：取得新最佳指标的轮次快照。

`.pth` 仍是原有 `state_dict`，不新增参数键；是模型权重，不包含优化器断点状态。
原 `main_eval.py` 可以直接加载。新增 `run_eval_rtc.sh` 只修正路径默认值，调用同一评估程序：

```bash
bash run_eval_rtc.sh "实际实验目录/ckpt/best_model.pth" ./exp/eval/rtc_pair
```

输出为 `progress_scores.txt`。模型选择与调参使用 Dev，Progress/Eval 不参与训练、选模或人工分析。
此版本没有实现第二个 idea 中的恢复前端或本地 RTC 模拟。

## 6. 本地验证范围

```bash
python -m unittest discover -s tests -p 'test_rtc_*.py' -v
```

测试涵盖 CSV/协议配对检查、真实小型合成 WAV 的读取、配对采样、对比损失、梯度、
实际 AASIST 后端的旧/新 checkpoint 兼容性和微型训练循环。
模型测试以一个可训练的微型 SSL 替身代替 XLS-R；它不能证明真实 XLS-R GPU 训练速度或竞赛性能。
本开发环境没有你的正式音频、官方受限 CSV、预训练大权重或 A100，完整训练需在你的服务器运行。
