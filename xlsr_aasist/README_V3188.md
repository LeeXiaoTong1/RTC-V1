# V3.18.8: single-Noisy continuation of V3.18.7 best

V3.18.8 从已停止的 V3.18.7 v2 实验中读取 **best**，新增训练默认 6 轮，头部预热为 0。每次访问一条 Offline 源，只合成一个 Noisy Online 分类样本。保留原模型、双分支 PCL、动态基础增强和每轮来源曝光预算，并降低实际 GPU 批次与窗口处理上限。

## 变化和预算

| 项目 | V3.18.7 v2 默认 | V3.18.8 默认 |
| --- | ---: | ---: |
| 每次更新的 Online 来源曝光 | 24 | 24 |
| 每次更新的 Offline 来源曝光 | 24 | 24 |
| 每条 Offline 源生成的 Noisy 分类视图 | 2 | 1 |
| 每次更新的分类音频数 | 72 | 48 |
| 含有效 PCL 参考时的最大编码音频数 | 108 | 84 |
| 实际 GPU 小批次上限，包含 PCL 参考 | 16 | 12 |
| 小批次填充后总帧数上限 | 12000 | 9000 |
| AASIST 窗口分批上限 | 32 | 16 |
| 推理批次上限 | 24 | 12 |

这里的 48 是一次优化器更新累计的分类音频数，不是一次前向必须装下 48 条。Online 与 Offline 两条流独立抽样，各自去重；同源录音仍可能同时出现在两条流，因此不能把 48 说成保证互不重复的原始来源数。

普通批次的 PCL 参考最多包含 24 条匹配的模拟干净 RTC 和 12 条官方 Online。参考音频不参与分类损失。同一源的 Noisy、干净 RTC、官方 Online 最多三条，保持在同一个小批次；12 条上限可以容纳四个三视图来源，保留原方差正则最少四来源的机会。长录音可能令实际小批次更小，此时方差项按覆盖条件跳过并记录原因。

默认继承每条流的 EN-Fake / EN-Real / ZH-Fake / ZH-Real 曝光数 `[9668, 8000, 8000, 8000]`，每流每轮 33668 次，共 1403 次更新，最后一次每流 20 条。每轮分类视图从 101004 条降到 67336 条。Real 的 8000 是来源曝光量，包含重复访问和动态增强，不代表 8000 条独立原始录音。若父实验使用了其他合法来源预算，则沿用父配置并重新计算步数。

分类损失仍按 Online 50% / Noisy 50% 分配，四个语言类别组各占 25%，不会因为减少一个 Noisy 视图而把其监督权重降到原来的一半。原来的双 Noisy 风险聚合项随双视图一起移除，改为单视图平滑交叉熵。EN-Real 置信度正则、双 PCL 权重及训练期尺度、方差正则继承父实验。

## 增强、模型和效率

- Offline 的增强仍在“加噪 → RTC”之前；真实 Online 保留原始分支与温和动态增强。只生成一个 Noisy，不先生成两个再丢弃一个。
- 单 Noisy 在来源访问周期中轮换原来的两类噪声配方入口，并继续动态更新增强参数。这样保留跨访问的条件变化，但不能保证与每次同时生成两种噪声效果完全相同。
- 来源访问偏移及 RTC 压力阶段从父 best 所在轮延续，不重新从第一轮弱增强阶段开始；新版本有自己的确定性随机序列。采样与恢复可复现，但不是 V3.18.7 原优化轨迹的逐步延续。
- 完整波形输入、极短反射补长及 PCL 有效区域屏蔽保持不变。保留重叠窗口和注意力统计汇聚，分类器仍为 384 维输入。
- 保留全部编码层 LoRA、融合、GN/LN AASIST 和 PCL；完整加载 best 的检测器及冻结的 PCL 尺度，不截取分类器权重。
- 同一物理批次的分类与 PCL 复用编码；无有效音素对的参考不会送入编码器。全层梯度检查点与融合检查点开启。
- 复用父实验的噪声素材和音素文件，不重新下载，也不建立增强波形副本缓存。仍会保存新实验自己的日志、指标和 checkpoint。

分类视图减少约三分之一，最多编码视图减少约 22.2%；这不是实际训练时间或显存下降比例。小批次、填充长度、梯度检查点和 RTC 处理都会影响耗时。资源预检检查完整录音是否能满足帧数预算，不能代替 A100 实测，也不能保证所有输入都不再 OOM。不会通过裁剪录音绕过资源限制。

若仍发生训练 OOM，终端会显示具体异常，并保存 `oom_context.json` 和 CUDA 显存摘要；失败的更新不会覆盖最后一个安全 checkpoint。不要直接编辑实验配置或跳过指纹校验来恢复。

## 继承与 best 选择

从父实验 `last.pt` 内的 `best_model` 读取完整 best，而不是把 last 权重误当 best。校验父配置、代码指纹、数据清单、checkpoint 和 best 对应的 Dev 分数。

父 checkpoint 只有 last 时刻的优化器状态，可能与 best 不在同一轮，因此新建 AdamW 优化器，不混用该状态。每个模块的起始学习率按父 best 所在轮的实际结束水平计算，在新增 6 轮中继续余弦衰减；不重复头部预热、LoRA 学习率爬升或 PCL 爬升。优化器动量重置，因此这属于从 best 开启的新实验。

父 best 已保存的 Dev logits 被记录为明确的 epoch 0 基线，不额外跑一遍 3B 模型评估。新实验仍按原固定、来源隔离的 80% Dev 选择模型，其余部分用于原有校准流程。

**若新增轮次没有超过父基线，最终 best 仍是父 V3.18.7 best。** 终端及导出元数据会明确标记 `origin=v3187_parent` 和 `baseline_fallback=true`；这不能被算作 V3.18.8 训练取得提升。新增轮次获胜则标记 `origin=v3188_trained`。每轮表格保留，进度显示当前 loss 和进度。

## 部署与训练

以下命令在 Linux 服务器执行，代码块只使用 ASCII。V3.18.7 已停止后，无需再次准备噪声或音素。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
PARENT_RUN=$(python -m w2v_v3188.parent) && bash run_w2v_v3188.sh --parent-run "$PARENT_RUN" --epochs 6 --submit-best --upload-submission
```

父路径默认从 `exp/.latest_v3187_run` 解析，随后严格校验；路径错误时不会启动训练。也可以用 `--parent-run /absolute/path/to/run` 显式指定已经停止的 V3.18.7 v2 实验。不要用 `--resume` 指向旧版本实验。

默认提交集为 `/home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt` 及其 `wav/progress` 目录。使用其他位置时，在启动命令上添加 `--submission-protocol` 和 `--submission-audio-root`。

脚本启动后台训练并打开只读日志查看器，自动设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（已有自定义值时保留）。查看器中的 Ctrl+C 只关闭查看器。再次查看：

```bash
bash watch_w2v_v3188.sh
```

训练完成后自动生成并上传 submission.zip 和 report.zip，显示两个独立 temp.sh 链接。报告不包含录音和模型权重。读取链接：

```bash
RUN=$(cat exp/.latest_v3188_run)
cat "$RUN/download_links.txt"
```

## 暂停、继续、单独提交

关闭查看器后，以下命令请求在优化器更新边界保存并暂停。超时只表示暂停仍在等待，不应当把超时当作已经停好。

```bash
RUN=$(cat exp/.latest_v3188_run)
bash stop_w2v_v3188.sh --run "$RUN" --timeout 60
```

确认显示暂停完成后，可以关闭服务器。重新启动、挂载数据盘并激活同一环境后，续训使用保存的配方、优化器、随机状态和提交请求；不再附加新的 epochs 或 batch 参数：

```bash
RUN=$(cat exp/.latest_v3188_run)
bash run_w2v_v3188.sh --resume "$RUN"
```

已完成或安全暂停的实验，可手动生成 best 提交和两份下载链接：

```bash
RUN=$(cat exp/.latest_v3188_run)
bash submit_w2v_v3188.sh --run "$RUN" --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress --report-dir /home/ubuntu/LXT/temp
cat "$RUN/download_links.txt"
```

上传失败且分数已生成时，只重试交付，不重复 GPU 推理：

```bash
RUN=$(cat exp/.latest_v3188_run)
bash submit_w2v_v3188.sh --retry --run "$RUN"
```

本版本保留 V3.18.7 源码、实验和 checkpoint；新实验写入独立的 `w2v_v3188_*` 目录。验证以本地 CPU 合成数据回归、脚本语法和版本边界为主，未在本地宣称完成真实 3B/A100 训练验证。
