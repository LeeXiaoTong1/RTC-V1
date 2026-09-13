# RTC-SDD：B 步骤（环境噪声增强）

本包针对官方 XLS-R + AASIST 的 `xlsr_aasist` 目录。B 的目的是检验：在既有基线上补充环境噪声训练，能否改善噪声条件下的检测表现。本包没有经过竞赛数据上的模型训练，不能承诺分数提高。

## 具体行为

训练中，每条官方 Train 音频仍按原方法读取并进行 RawBoost；之后以 50% 概率加入一条公开非语音环境录音，再按原代码裁剪/补齐为 64,600 个采样点。加噪概率与真假标签无关。Offline 和 Online 训练样本均按这一规则处理。每次读取可以抽到不同的噪声、片段和强度，原 WAV 不会被改写。

初始 SNR 范围为 10–30 dB，均匀随机采样。这是预先设定的实验起点，不是官方测试噪声的参数。50% 未额外加环境噪声的样本仍然保留基线原有 RawBoost，不能称为完全没有增强的原始样本。

波形混合为 x + alpha*n。alpha 按语音和噪声共同活跃帧上的能量计算。活跃帧用 20 ms 能量和相对最高帧 -30 dB 的门限估计；它不是语音识别或精确的语音活动标注。对短噪声事件只随机插入一次，不循环铺满整段，避免事件变成周期噪声。混合后如振幅过大，对整条混合波形共同缩放，避免硬裁剪并保持信噪比。

B 属于普通加噪对照。对 Online 样本，其顺序是 RTC 后加噪；它没有重建“加噪后再经过 RTC”的官方生成过程。后者属于后续 C 步骤。B 也不包含额外降噪器、回声生成器、双视图训练或一致性损失。

## 1. 上传并解压代码包

把下载的 `RTC_B_noise.zip` 上传到服务器 `/home/ubuntu/LXT/RTC/`，然后：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC
unzip RTC_B_noise.zip
python rtc_b_noise/smoke_test.py
```

代码需要 NumPy、SciPy；这两个包一般已随官方 requirements 安装。验证模型时使用你已跑通的 PyTorch、librosa、fairseq 环境。本包不要求升级 pip、PyTorch 或 fairseq。

## 2. 准备非语音噪声素材

第一版使用 MUSAN 的 noise 子集。OpenSLR 官方完整压缩包约 11 GB，其中含 speech、music、noise 三部分；下面只解压 noise，下载仍需传输整个压缩包。

```bash
mkdir -p /home/ubuntu/LXT/RTC/external_noise
cd /home/ubuntu/LXT/RTC/external_noise
wget -c 'https://openslr.magicdatatech.com/resources/17/musan.tar.gz'
tar -xzf musan.tar.gz musan/noise
```

如果该镜像连接失败，可以在 OpenSLR 官方页面选择另一个镜像：https://www.openslr.org/17/

MUSAN 原论文说明 noise 子集排除了可理解的语音，但有少量带模糊人声的人群噪声。准备脚本会根据文件名及 ANNOTATIONS/LICENSE 中对应条目的 crowd、speech、voice、talk 等关键词排除明显人声素材。该过滤不等同于音频内容检测。请对剩余含糊来源进行核查；不要将 speech、music 目录或 Progress/Eval 文件加入噪声池。不要自行先切碎原噪声录音后再划分。

划分时以原 WAV 文件为单位，固定种子，80% 用于训练、20% 用于验证；字节完全相同的文件归到同一组。输出仅为索引，不另复制音频。如果多个不同 WAV 来自同一原始长录音，需要先按原录音分组，不能直接当作独立文件交给本脚本。

```bash
python /home/ubuntu/LXT/RTC/rtc_b_noise/prepare_noise.py \
  --noise-root /home/ubuntu/LXT/RTC/external_noise/musan/noise \
  --out /home/ubuntu/LXT/RTC/external_noise/noise_split
```

输出：`train.jsonl`、`dev.jsonl`、`split_report.json`。查看 report 中排除记录。可用 `--exclude-list 文件路径` 追加手工排除名单，每行填写相对于 `musan/noise` 的 WAV 路径，再指定新的 `--out` 重新生成划分。脚本不会覆盖已有非空划分目录。

## 3. 安装补丁并开始训练

```bash
cd /home/ubuntu/LXT/RTC/xlsr_aasist
python ../rtc_b_noise/install_b.py --repo .
```

安装器仅向 `utils/` 添加 `env_noise.py`，并对 `utils/data_utils.py` 插入导入、初始化和调用。它会自动备份原 data_utils.py；如果关键代码位置不匹配，会报错且不修改文件。它会保留你对 XLS-R 权重路径、数据路径和训练参数的已有配置。

打开增强，然后运行已经成功复现基线时的同一训练命令。若原命令就是官方 `bash run_train.sh`：

```bash
export RTC_B_NOISE_MANIFEST=/home/ubuntu/LXT/RTC/external_noise/noise_split/train.jsonl
export RTC_B_NOISE_PROB=0.5
export RTC_B_SNR_MIN=10
export RTC_B_SNR_MAX=30
export DATA_ROOT=/home/ubuntu/LXT/RTC/RTCFake/data/wav
bash run_train.sh
```

训练启动时应出现 `[RTC-B]` 信息，包括噪声文件数、manifest 摘要、概率和 SNR。没有这行说明需要检查是否设置变量、是否通过训练数据集读取。

第一轮建议沿用 A 的初始化方式、随机种子、学习率、批大小、训练预算和 Dev 最佳权重选择方式。官方脚本未指定 `--model_path` 时，会从原有 XLS-R 预训练权重及新初始化的 AASIST 开始训练。不能把“B 从 A 的最佳检测权重继续训练”与“A 从头训练”直接当作严格增强消融；如果需要节省成本做继续训练试验，还要给 A 设置相同的额外训练预算。

模型结构、分类损失、标签方向、原始 Dev 验证逻辑和推理代码均保持原流程。每条输入仍只有一次模型前向；新增代价主要在 CPU 音频读取与混合。官方保存目录按时间区分实验，请记录 B 对应的新目录。

关闭 B：

```bash
unset RTC_B_NOISE_MANIFEST RTC_B_NOISE_PROB RTC_B_SNR_MIN RTC_B_SNR_MAX
```

关闭后补丁不会调用加噪，也不会额外消耗随机数；如需完全恢复文件，可以使用安装器打印的准确备份路径恢复 data_utils.py。

## 4. 使用同一组固定 Dev 样本比较 A 和 B

`evaluate_b.py` 专门用于有真假标签的官方 Dev。它计算原始 Dev 和 Dev 加噪后的 Macro-F1，默认分别测试 10、20、30 dB。对同一语音，噪声文件和起始位置在不同权重、不同 SNR 测试中固定。它只读 `dev.jsonl`，不会用训练噪声替代验证噪声，也不会因训练环境变量仍然设置而修改原始 Dev。

在下面命令中，把 `你的权重文件绝对路径` 替换为真实路径。基线、B 各执行一次，并分别保存为 A_dev.json、B_dev.json。当前项目的真实权重路径只在你的服务器上，包内不会猜测。

```bash
python /home/ubuntu/LXT/RTC/rtc_b_noise/evaluate_b.py \
  --repo /home/ubuntu/LXT/RTC/xlsr_aasist \
  --checkpoint '你的权重文件绝对路径' \
  --dev-data /home/ubuntu/LXT/RTC/RTCFake/data/wav/dev \
  --dev-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/dev_label.txt \
  --noise-manifest /home/ubuntu/LXT/RTC/external_noise/noise_split/dev.jsonl \
  --out /home/ubuntu/LXT/RTC/B_dev.json
```

这里沿用官方模型的标签映射：0=spoof，1=real；取 softmax 的第 0 列作为伪造分数，并在 0.5 阈值下计算 Macro-F1。输出指标均为百分数。JSON 记录每个 SNR 的两类 F1、混淆矩阵、全部噪声版本合并后的 Macro-F1，以及 0.3*原始 + 0.7*噪声 的代理分数。

默认评价传入 Dev 协议的全部样本；若该协议含 Offline 和 Online，结果也包含两者。本脚本没有本地 RTC 模拟，生成的 noisy Dev 不等价于官方 noisy Online。代理分数仅用于 B 的开发对照，不能和 Progress 分数直接作数值比较，也不能据此宣称达到排行榜水平。

第一轮至少比较原始 Dev、10 dB、20 dB、30 dB 四项的 A/B 差异，同时看真实、伪造两类 F1。若模拟噪声指标增加但干净表现明显下降，需在 Dev 上检查增强强度；如果完全没有收益，先确认 `[RTC-B]` 已启用及素材有效，再判断方法。后续需要选择对噪声更稳健的检查点时，应给 A/B 使用同一套新的 Dev 选择规则，避免把选择规则改变混入增强收益。

## 包内检查与范围

`smoke_test.py` 检查实际混合 SNR、峰值限制、静音、短事件、固定噪声随机性、F1 计算和补丁兼容保护。开发时还对当日下载的官方 data_utils.py 做了补丁兼容检查。没有在你的服务器、完整 XLS-R/fairseq 模型或竞赛数据上执行训练，也没有自动审核所有 MUSAN 音频。新方法的检测效果需要你按上面的 A/B 实验确认。

## 来源（2026-09-08 核对）

- 竞赛规则：https://www.junxue.tech/rtc-sdd-challenge/
- 官方基线：https://github.com/JunXue-tech/RTC-SDD
- MUSAN 官方数据页：https://www.openslr.org/17/
- MUSAN 原论文：https://arxiv.org/abs/1510.08484

本包脚本是为本次 B 实验编写的辅助实现，不是竞赛官方发布的改进模型。
