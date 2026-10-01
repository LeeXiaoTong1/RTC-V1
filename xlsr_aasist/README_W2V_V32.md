# V3.2：处理条件覆盖与 GPU 训练效率

架构继续使用 w2v-BERT 2.0 + MultiConv。复用 V3/V3.1 已验证 best 和两份整段 noisy Train 缓存；新运行写入独立目录。原 91.68 checkpoint、旧实验与全部数据缓存保持不变。本版不需要生成或删除音频缓存，也不添加外部语音。

## 训练目标

完整遍历原始 Train 和每个 Offline Train 来源的两个整段 noisy 版本。保留 V3.1 的整段/短段监督：默认每条来源的分类损失预算为整段 70%、短段 30%，无法产生不同短段时仅保留整段。先在波形上裁剪，再分别计算官方特征及归一化；不截取已归一化特征冒充短录音。

在读取 noisy 时，约 50% 保持不变、35% 施加一种轻处理、10% 施加两种不同的轻处理、5% 单独尝试局部静音：

| 处理 | 默认范围 | 目的 |
|---|---|---|
| 频率响应变化 | 温和带通，高端截止 3.4–7.2 kHz，低端 60–160 Hz | 减少对单一通道频谱的依赖 |
| 平滑动态音量 | ±3 dB，0.5–1.5 秒平滑变化 | 覆盖时间变化的增益，避免只改变被归一化抵消的全局音量 |
| 局部衰减 | 3–9 dB，80–240 ms，最多覆盖录音的 10% | 局部线索变弱时仍利用其余证据，不直接把语音置零 |
| 局部完全静音 | 40–160 ms，含两端各 5 ms 平滑过渡；中央真正置零 | 覆盖短时信号完全缺失，单独使用，不与其他处理叠加 |

局部静音新增限制：受影响的整个区间（包括渐变）不得超过整段和实际短片段各自长度的 5%；按最短视图的 5% 保守限长，低于 40 ms 则跳过。完整和短视图都必须保留至少 90% 的原信号能量，否则跳过，以免清除安静录音中唯一有声片段。这是信号能量保护，并不是语音识别或 VAD。原本已静音的区域不算成功增强。因此 5% 是抽中该方案的概率，实际应用比例可能更低；epoch JSON 会分别记录应用和各类跳过次数。

同一整段只处理一次，短视图截取同一结果。内部静音时间仍标记为有效帧，不能与合批 padding 混淆。Noisy/ordinary 的样本数、full/short 损失预算、学习率和两份缓存均不因此改变。

这些是通用信号扰动，并不声称复现某个 RTC 平台或真实 codec。范围没有按 Progress 样本或其预测结果拟合。处理方案只取决于录音身份、版本、epoch 和种子，与真假和语言标签无关。先处理整段，再从同一结果生成短段，不增加第三、第四个模型输入。普通 Train 沿用既有原始/RawBoost 分配；Dev 不施加新处理。每个条件按语言、类别的实际次数写入 epoch JSON。

学习率仍为前端 5e-8、后端 2e-6，只训练后 4 层和分类头；最多 2 轮，每半轮评估。沿用源样本归一化 CE、full-only 全逻辑 batch CKA 和逆频率类别权重，不额外加大英文或 real 权重。

## 加速方式与比较边界

1. 训练时在同一逻辑 batch 内按长度排序，把接近长度的完整音频合批。使用正确的右侧 padding 掩码，不截断录音，不把 padding 当有效静音。每个物理 batch 最多 4 条、1600 帧预算（沿用来源配置），最长/最短长度比不超过 1.5。超长单条仍完整处理。
2. 前向保留的中间结果优先留在 GPU，默认最多 18 GiB，另留 8 GiB 给临时计算等；超出预算时才转存 CPU。自动考虑当时可用显存。它不是“总显存固定 18 GiB”，也不能对任意长度音频保证不 OOM。
3. 将 25 层特征的投影分组计算，保留相同权重和层求和次序；减少 CPU/GPU 同步和逐条分数传输。加载使用 6 个 worker、预取与固定内存。
4. 逻辑 batch、每来源损失预算和每步一次优化器更新不变；不因为物理合批增大而提高学习率。仍保留梯度检查点节约瞬时显存。

本次追加执行优化：官方特征提取器分组处理波形，但每条仍独立归一化；在加载 worker 内完成原有长度分组及 padding，再固定最终传输张量的内存，避免主线程重新合批后丢失 pinned-memory 属性。GPU 用独立传输流预取下一物理 batch，只有一批前瞻，不提前计算模型。显存检查从每个保存张量查询驱动改为每个物理 batch 刷新一次，期间仍检查实时 allocator 用量。冻结且输入无需梯度的编码器层直接执行，最后可训练层仍使用梯度检查点。减少已在 CPU 验证过的掩码在 GPU 上重复检查造成的同步。

这些追加改动保持此前 V3.2 的物理分组与计算次序。混合长度的官方特征逐位一致；带 dropout 的小型真实 w2v-BERT 上，输出、梯度、Adam 更新和随机数状态逐位一致。CUDA 的异步传输和激活存储检查会在服务器 `setup_w2v_v32.sh` 中执行；本地 CPU 环境只能跳过这些 CUDA 检查。它们不改变采样、增强、精度、学习率或监督目标，但实际训练成绩和加速幅度仍需运行结果确认。

不同 GEMM 组织和 dropout 随机数分配可能改变具体训练轨迹，不能宣称逐位相同。无 dropout 的有效帧输出、损失与梯度有对照测试；验证和导出仍使用原有精确长度路径和原始融合，保持比较条件。实际 A100 加速倍数和精度以服务器记录为准。

## 启动

已有 V3.1 在训练时，应先到合适的保存点，或接受丢弃最近保存点之后的未保存步数，再停止它。不要同时启动两轮训练。

如果已启动旧代码的 V3.2，应在拉取更新**之前**执行 `python -m w2v_v32.stop --version v32 --apply`。更新后开一个新运行，不用 `--resume` 接续旧代码；源码指纹变化会拒绝旧断点恢复。旧 checkpoint 文件保留，必要时可用 `--source-run 旧运行目录` 将其中已验证的 best 作为新起点。尚未启动 V3.2 则正常更新、启动即可。单独更新 `v32_console.py`、`watch_w2v_v32.sh` 不改变训练指纹，可以供仍运行的旧 V3.2 查看进度；要使用训练加速改动则必须重新启动。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist &&
python -m w2v_v32.stop --version v31 --apply &&
python -m w2v_v32.stop --version v32 --apply &&
cd .. &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v32.sh &&
bash run_w2v_v32.sh --upload-temp
```

停止工具只终止本目录下指定版本的训练进程及其子进程，不删除文件。去掉 `--apply` 可只查看。若旧进程尚未退出，新启动会拒绝并行运行。

默认读取 `exp/.latest_v31_run` 中已有 `best_model.pt` 的运行；未找到则回到 V3 的 `exp/w2v_v3_20260930_181111_5eca`。V3.1 没有达标提升时，它的 best 就是启动前保存的 V3 权重。也可显式指定 `--source-run exp/某个运行`。只接收通过验证保存的 best 权重，不用 `last.pt` 作为新目标起点；新目标使用新 AdamW。

启动时重新验证所选起点并先保存 baseline。最终 best 的晋升要求：Weighted 达到最小增益，Noisy 不下降，四个条件的 EN-real、两个 noisy EN-fake 及 Clean 均通过各自保护线。保护线固定在启动模型，不随候选逐步下降。未达标则 best 保留启动模型。

## 看进度、指标与速度

在交互终端运行启动脚本后，默认直接进入实时查看器。流程消息和每次已保存的 baseline/半轮/整轮 Dev 指标依次保留在终端上方，下面仅一行进度持续刷新。指标包含 Clean、Seen、Heldout、Noisy、Weighted，以及四个条件下英文真假 recall、选模决定和学习率。再次进入会从已有报告补回以前结果，不重新评估。普通日志仍落盘；查看器不逐条展示每 100 步的进度行。

Ctrl+C 仅退出查看器，后台继续训练。只启动后台而不自动查看可加 `--no-watch`；从非交互环境启动时也不自动进入。报告上传结束后 temp.sh 链接会打印在终端并保留。

```bash
bash watch_w2v_v32.sh
# Ctrl+C 只关闭查看器；下面显示已完成的 Dev 评估
bash show_w2v_v32.sh
```

每步性能保存为运行目录内 `performance.jsonl`，最近一步为 `performance_latest.json`。它们独立于用于恢复的指标，记录数据等待、计算墙钟时间、平均物理 batch、GPU 峰值、中间结果 GPU 保留量与 CPU 转存量，以及固定内存传输批数和显存驱动查询次数。启动前几步包含预热，比较稳定速度时查看几十步后的记录。进度条 ETA 对应当前两次验证之间的半轮；日志 `ETA_min` 对应本轮剩余训练步数，都不含后续 Dev 验证时间。

```bash
cat "$(cat exp/.latest_v32_run)/performance_latest.json"
```

可选参数：`--gpu-activation-gib 12` 降低中间结果驻留上限；`--workers 4` 减少 CPU worker；`--condition-probability 0` 关闭新增处理以运行相同 V3.1 训练目标的加速路径；`--silence-probability 0` 仅关闭局部静音并恢复原 50%/40%/10% 条件分配。默认静音占总处理概率的 10%（即 5%），从单种轻处理份额中划出，不增加总增强概率。高级参数 `--no-gradient-checkpointing` 减少重计算但增加显存，不作为默认。上述参数只用于新运行；恢复必须使用原配置。

```bash
bash run_w2v_v32.sh --resume exp/你的V32目录 --upload-temp
bash run_eval_w2v_v32.sh --upload-temp
```

训练结束导出报告并打印 temp.sh 链接；评估命令生成 `submission.zip`。报告只含记录和分数，不上传音频或 checkpoint。

实现依据：当前固定版本 Transformers 4.38.2 的 w2v-BERT attention/convolution mask；[PyTorch 中间结果保存与卸载](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)；[PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。这些支持执行机制，不是本任务提分的保证。
