# V3.18.13：XEUS 声学近邻判别＋官方原始 Offline/Online 音素一致性

本版检验一个假设：在声学结构相近、来源不同的真假样本之间增加弱判别约束，能否减少检测器对有限训练来源的依赖。没有预先证明它能解决 EN-real；XEUS 近邻也不等于相同文本、音素或说话人。

## 起点和保持的设置

V3.18.11 是诊断版本。它实际分析的是 V3.18.10 的 `epoch_15_step_1403`，来自：

```text
/home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v31810_20261018_145711_recal_732feb
```

默认从该实验的已记录 best 完整加载检测器权重，新建 V3.18.13 实验、优化器和调度器。允许 `--parent-run` 显式指定另一个已完成或安全暂停、最后16层 LoRA 的 V3.18.10；不接受把 V3.18.11 诊断目录当 checkpoint，也不迁移 V3.18.12 残差模块。

- 保留官方 Omni 3B、最后16层 LoRA、60层可学习融合、128维完整时序、重叠窗口、AASIST、384维统计分类表示。
- 保留原始长度输入、极短镜像补长、GroupNorm、SSL BF16／AASIST FP16 混合计算；损失使用 FP32。
- 保留 V3.18.10 的源身份登记表、Train/Select/Audit 划分、动态增强和单个 Noisy+RTC 视图。Dev、Progress 不进入准备、聚类、配对或损失标定。
- 本版使用普通八组均衡 CE，不运行 MLDG 内外层更新，不加入说话人对抗或 XEUS 特征模仿。
- 提交仍为一个共享编码器、一个最终分类器的 `P(fake)`；不做独立模型分数集成，也不拟合提交阈值。

## 两个辅助目标各约束什么

**XEUS 跨来源判别。** 只在准备阶段加载冻结的官方通用 XEUS，取第10个编码器块的1024维输出。从 Train 每个归并来源选原始 Offline 代表，平衡抽取 EN/ZH × Real/Fake 的帧，拟合128个声学单元；聚合成来源级平方根频率直方图。采用余弦相似度，限定同语言、不同 canonical source、相反真假标签、原始时长比在 `[0.5,2]` 内，保留相似度至少0.8的 Top16 候选。这些是固定的工程初值，不是论文已证明的最优值。

每轮最多在每个 `Online/Noisy × EN/ZH` 单元安排一个 Real/Fake 对，每个来源参与辅助判别最多2次／轮。通过交换已有 Fake 票据来配对，严格保持本轮来源曝光和动态增强版本的多重集合；不补抽样本。找不到可用近邻的样本照常做 CE。Noisy 对共用 RTC family、噪声场景、SNR 及处理强度控制，噪声素材／偏移仍由各来源独立确定。Online 使用官方传输音频，不能声称其真实平台强度被精确匹配。

定义 `d = logit_fake - logit_real`，配对损失为：

```text
L_pair = softplus(2m + d_real - d_fake),  m = 0.5
```

这里对原提案做了必要修正：`softplus(m+d_real)+softplus(m-d_fake)` 是两项可分离的单样本目标，交换配对关系不会改变它们。本版的损失真正依赖两位搭档；两者同时反传，不 detach 其中一方。CE 负责绝对的零 logit 差／0.5 概率边界，配对项负责相对真假间隔。

**官方同源 PCL。** 只对原始 Offline 与其官方保留的 Online 计算融合特征的音素池化 MSE，两侧都接收梯度。要求 Train、相同 canonical source、相同语言、相同标签；用原有固定 CTC 识别器和无歧义音素匹配元数据。原始 Offline 只作参照，不进入真假 CE。每个逻辑更新最多从 EN/ZH × Real/Fake 各选1个有效来源，共最多8条额外编码参考音频。没有官方 Online 的 Offline 仍参加 Noisy 分类，但不强造 PCL 对。

PCL 不约束模拟噪声，不跨 Real/Fake 拉近特征。按照 Omni 的400样本感受野、320步长，仅使用真实原始区域，排除静音支持和短音频镜像补出的部分。CTC 时间边界是近似边界，不等同于人工标注。

## 训练预算、学习率与效率

默认新增15轮：第1轮只做 CE、冻结 LoRA；第2—3轮开放 LoRA，逐步加入两个辅助项；第4—15轮固定弱辅助权重、余弦衰减。第一轮之后只使用 Train 的6个固定任务，在共享融合层上测量梯度尺度；XEUS/PCL 的目标比例分别为 CE 的10%/5%，系数上限0.1/0.01。无可用梯度或非有限值会明确停止，保留已保存 checkpoint，不偷偷把损失置零。

采用温和学习率重启：V3.18.10 原始初始学习率的四分之一。默认末段 LoRA `5e-6`，融合和检测头 `2.5e-5`；这是新优化器的新调度，不声称无缝恢复旧 Adam 动量。新调度最低为初始值的0.1。

每条流每轮来源曝光：

| 组 | 官方 Online 流 | Offline→Noisy 流 |
|---|---:|---:|
| EN-Fake | 9668 | 9668 |
| EN-Real | 8000 | 8000 |
| ZH-Fake | 8000 | 8000 |
| ZH-Real | 8000 | 8000 |

每轮67336条分类视图、1403个逻辑更新。通常一个逻辑 batch 是24 Online＋24 Noisy，来源不重复；尾 batch 更小。上述是曝光次数，不是新增独立录音。八组 CE 先组内平均再等权平均，重复曝光不会自动提高某类的损失权重。

默认物理 microbatch=4、padding frame budget=6000；一个 XEUS 对作为不可拆分的2条音频同时前向。梯度累积仍覆盖完整逻辑 batch，最后只更新一次。PCL 独立按2条参考音频编码，不把整个 batch 的计算图留在显存中。完整音频超过原有3000帧上限会报错，绝不静默裁剪。默认4个训练 worker、2个评估 worker、prefetch=1，可用 `--workers 8` 做受控吞吐对比。

XEUS 不参加15轮正式训练。准备时使用单音频，避免官方波形归一化受不同 padding/batch 影响；长录音分20秒核心段＋1秒上下文，仅统计各核心归属帧。截断到第10块前，会用一条短音频核对完整19块前向的第10块输出。帧缓存只用于小规模聚类采样，其余立即汇成128维直方图，64来源一分片，可中断重跑。不保存全量高维时序特征。极少量拟合码本的来源会在全量直方图阶段再计算一次。

首次准备包括 CTC 元数据核验／补齐、XEUS 依赖和模型下载、码本与近邻索引构建，可能较耗时；后续复用缓存。正式训练相对 MLDG 减少了内层重复前后向，但新增最多8条 PCL 参考编码，最终提速和显存峰值需服务器实测。

## 更新和启动

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch --track origin/v3.18.13-xeus-pcl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v31813.sh --epochs 15 --submit-best --upload-submission --upload-temp
```

若已在该分支，更新用 `git pull --ff-only origin v3.18.13-xeus-pcl`。启动脚本只启动这一个训练，不会自动额外跑对照实验。默认 parent 指向上文的诊断来源；若服务器路径不同，追加 `--parent-run /实际/V3.18.10/run`。

默认自动准备缺失的资源。原有 `assets/phonemes_v3186/manifest.json` 可兼容时直接使用；不兼容的旧缓存不会被覆盖，需要明确指定新的 `--phoneme-manifest assets/phonemes_v31813/manifest.json`。XEUS 索引在 `assets/xeus_v31813/manifest.json`，大模型缓存及独立环境在 `/home/ubuntu/LXT/v318-runtime/`。模型权重不会写入代码仓库。网络无法访问 Hugging Face 时可追加 `--hf-endpoint https://hf-mirror.com`，仍锁定同一提交。

单独准备／预检入口：

```bash
python -m w2v_v31813.prepare_phoneme_environment
python -m w2v_v31813.prepare_teacher
```

XEUS 固定为 `espnet/xeus@b651261280bf792df4632ccb2d1ba4eeaabeaee2` 的 `model/xeus_checkpoint_new.pth` 与配套配置。代码使用官方 ESPnet `v.202506` 的固定提交 `7d395a8bb223f6bddd479d4d1dd6feddf49f82f4`，调用未掩码的 `inference_encode`。模型卡旧 fork 的 `encode` API 与这份新版配置不兼容，本版不混用它们。独立环境固定 NumPy/Scipy 等兼容依赖，继承已有 Torch/CUDA；启动前核对导入、Torch 版本及环境归属。训练环境不做 pip 升级。

## 停止、恢复、日志和提交

安全停止，在完成当前优化器更新后保存：

```bash
bash stop_w2v_v31813.sh
```

等待 `PAUSED_RUN=...`。前台训练收到 Ctrl+C 也会请求安全暂停；只是在 `watch` 查看日志时 Ctrl+C 只退出查看。日志和恢复指令：

```bash
bash watch_w2v_v31813.sh
# 恢复使用已保存参数，不能顺便改变 batch、轮数或损失设置：
bash run_w2v_v31813.sh --resume "$(cat exp/.latest_v31813_run)"
```

`last.pt` 原子保存当前权重、最佳权重、优化器、AMP scaler、随机状态及采样进度；每200次更新、每轮末和安全暂停时保存。没有新增更好模型时保留 parent 作为 best。遇到 OOM 等异常，恢复上一次成功保存的边界；不会谎称保存了失败更新。

独立导出 best：

```bash
bash submit_w2v_v31813.sh \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

自动导出使用相同默认路径，可在训练启动时通过 `--submission-protocol`、`--submission-audio-root` 指定。导出是一个模型的原始 softmax；无需 XEUS 模型和音素识别模型。上传失败时再次运行相同导出命令，可复用校验通过的 scores，不重复 GPU 推理。

## 如何判断是否值得继续

每轮保留完整 Select 指标，最终仅对已选 best 执行 Audit。选模仍为原来的 `0.30 Online + 0.35 Regular + 0.35 Heldout`，固定0.5；因此需要同时检查分组结果，不能把 SelectS 改善直接称为 EN-real 改善。

- `prior_quality.json`：各语言近邻覆盖、最近邻与随机相似度、平衡候选池中的真假标签纯度。高标签纯度可能提示类别／录音条件捷径；没有说话人和设备标注时，无法声称排除了身份影响。
- `coverage_epoch_*.json`：实际 XEUS 配对量、每来源次数上限、PCL 来源覆盖。
- `auxiliary_calibration.json`：实际梯度范数、确定的辅助系数。
- `selection_epoch_*.json`、`diagnostic_epoch_*.json`：Real/Fake 召回、AUC、Macro-F1，以及逐来源纠正／新增错误；额外报告不超过 parent Fake 漏检数时的 Real 误判，仅作诊断，不用于选模或提交阈值。
- `locked_audit.json`：最终独立 Audit；`report.zip` 含日志、指标和小型预测，不含音频、教师模型或 checkpoint。

可选择另起对照：`--mode baseline` 为同一起点、PCL、CE、预算和学习率，但去掉 XEUS 配对及其间隔项；`--mode random` 则保留相同配对上限、同语言／时长匹配与间隔项，用随机跨来源 Fake 候选替代 XEUS。三者各自标定弱辅助系数，且只运行用户实际启动的那一个。对照的实际配对覆盖需要一起比较，避免把覆盖率差异误认作教师收益。

如果仍只改善 Real、同时增加 Noisy Fake 漏检，就不支持本设计的主要假设。PCL 恢复与继续训练本身也可能带来变化，需要与对照区分。

## 验证与来源

CPU 测试覆盖配对梯度、真假方向、八组累积、原始 PCL／短音频屏蔽、来源预算、真实 AASIST/LoRA 接线、训练暂停恢复、teacher 第10块接口及提交导出。完整 Omni＋XEUS CUDA 下载、端到端准备和全量服务器训练仍需实际运行，不能用小模型测试代替性能验证。

```bash
python -m unittest discover -s w2v_v31813 -t . -p 'test_*.py' -v
```

方法依据：[XEUS 官方模型](https://huggingface.co/espnet/xeus)、[XEUS 原论文](https://aclanthology.org/2024.emnlp-main.570/)、[固定 ESPnet 源码](https://github.com/espnet/espnet/tree/7d395a8bb223f6bddd479d4d1dd6feddf49f82f4)。本版的近邻判别＋官方 PCL 组合是当前任务的实验设计，不能由通用语音任务结果推导出 RTC 深伪检测收益。
