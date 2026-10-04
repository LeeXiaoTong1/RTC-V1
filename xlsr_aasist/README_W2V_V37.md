# V3.7：冻结已提交 best，学习语言偏差修正

起点是此前已提交、分数为 **93.3941** 的 V3.3 best，通过完成记录、提交元数据和 SHA256 解析并核验；不从 V3.5 重训模型或未经确认的 V3.6 候选接着训练。目标仍是官方 Weighted >97，但本地测试通过不代表已达到这个目标，实际全量耗时、显存和成绩必须在服务器运行及官方提交后确认。

## 模型和训练边界

- 原 w2v-BERT 2.0、MultiConv、池化及最后分类器之前的512维表征全部冻结；不做全模型微调。
- 复用 V3.3 的两个完整 Train noisy 视图和 V3.5 的固定完整 Dev 缓存。原 Offline/Online 合计占每个来源损失预算的50%，noisy_a/noisy_b 各25%。同一 source_id 的各条件共享一个来源预算；重复内容按原始音频 SHA 绑定在同一调参划分。
- 完整音频只做一次冻结前向，保存512维表征和原始 logits。可直接复用身份、输入和数值格式均兼容的 V3.6/V3.7 已完成特征；不复制大矩阵，不生成新的音频缓存。
- 冻结的通用 ECAPA 语言识别模型只在官方 Train 音频上提取256维教师向量。V3.7 用同一 detector 的512维表征训练 `512 → 64 → 256` 的小型 tanh 语言分支；分支输出经过归一化。教师不接触 Dev/Progress/Eval，也不参与提交推理。
- 只在 Train 的真实语音上拟合中心化 ridge 映射，预测并按 α∈{0.25, 0.5, 0.75} 的强度从512维表征中减去语言相关分量，再拟合锚定原分类器的二分类线性层。去偏是可被关闭的候选，并非预设一定有效。
- 最终仍是一个真假模型：一次冻结主干前向、一个内部小分支、一个真假分类器。没有多个独立真假模型的预测加权、平均或投票。该实现是语言去偏的近似方案，不声称逐项复现独立语言编码器的原论文。

## 调参、对照和默认回退

先为完整官方 Train 准备冻结向量，再比较一个仅训练最后分类层的 `head_only_control` 对照和一个 `language_debias` 候选。两者均使用 EN-real、EN-fake、ZH-real、ZH-fake 四组的来源均衡损失。按原始音频 SHA 分组进行 Train 80/20 调参：Offline/Online/noisy_a/noisy_b 及重复内容必须在同一组。语言分支、中心化统计、ridge 映射及分类器均只在该划分的拟合部分学习，以留出部分的分组/来源均衡交叉熵选择超参数，然后使用完整 Train 重拟合。这是一次完整任务中的调参步骤，不是等待确认后才扩大的小样本试验。

Dev 不参与拟合或超参数搜索。原主干此前见过 Train，因此这个来源留出是调参验证，并非独立域外测试。最终恰好两个候选与原 best 使用同一固定完整 Dev；Dev 选模仍有选择偏差，收益需官方提交确认。若仅分类头对照胜出，报告不会把收益归因于语言去偏。

Clean 仅统计官方 Online；Noisy 为 Seen/Heldout 的 pooled Macro-F1 平均；Weighted = 0.3 Clean + 0.7 Noisy。统一阈值0.5。这些本地完整 Dev 指标不和旧短缓存指标或排行榜分数混算。

新模型只有满足完整保护条件才替换原 best：Weighted 至少增加0.2个百分点、Noisy 不降、Clean 最多下降0.1个百分点；Online/Seen/Heldout 的 EN-fake recall 各最多下降0.5个百分点；同三种条件的 EN-real recall 平均至少增加0.5个百分点，且每种条件的 EN-real 和 ZH-real recall 各最多下降0.5个百分点。避免只把错误从一个语言组转移到另一个组。未收敛、非有限数值或不满足保护条件时保留原 best，报告明确标记 `baseline_fallback=True`。回退不能算作 V3.7 提升。

## 安装和完整运行

在 Ubuntu 的现有 CUDA 环境中运行：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
bash setup_w2v_v37.sh
bash run_w2v_v37.sh --upload-temp
```

默认就是完整官方 Train；不需要 `--smoke`，也不需要另设小样本开关。安装脚本保留现有 CUDA PyTorch，不停止已有任务，不删除已有缓存；启动脚本发现仍有训练、评估或缓存写入任务时会拒绝重叠运行。

若已有完整 V3.6 特征，可显式复用：

```bash
bash run_w2v_v37.sh --feature-run "$(cat exp/.latest_v36_run)" --upload-temp
```

`--feature-run` 只接受通过完整性和身份校验的已完成切分。缺少所需原缓存或发现不兼容输入会给出具体错误，不会擅自重建几十 GiB 的音频。`--language-model-dir PATH` 指定训练用教师包位置；首次训练准备固定版本的教师权重，后续复用本地文件。

原始512维 Train 向量按38,660来源、每来源最多4条件估算约0.30 GiB，256维教师向量约0.15 GiB，另外需要 Dev 特征、索引、日志和约81 MiB教师权重；实际按记录条数和可用空间检查。不会逐轮新增约32 GiB音频缓存。拟合主要在小矩阵上完成，原主干没有反向传播、优化器状态或 EMA。语言教师准备和 Train 向量提取先执行，下载或加载失败会在检测器长时间提取之前暴露；已完成语言向量恢复时不重复构造教师。

```bash
# 重新打开查看器；Ctrl+C 只退出查看器，后台训练继续
bash watch_w2v_v37.sh

# 打印当前或完成后的完整结果
bash show_w2v_v37.sh

# 中断后复用已提交的特征游标继续
bash run_w2v_v37.sh --resume "$(cat exp/.latest_v37_run)" --upload-temp

# 预览要停止的本目录 V3.7 进程
python -m w2v_v37.stop

# 确实停止 V3.7，保留已保存的结果和特征
python -m w2v_v37.stop --apply
```

查看器始终把原模型、所有候选的完整指标和下载链接保留在动态进度行上方，不清屏覆盖历史结果。无交互终端时可使用 `bash watch_w2v_v37.sh --once`。

## 可选清理

清理与安装、训练分开。只有显式执行以下第二条命令才会停止本目录的 V3.6 工作流及经身份确认的子进程，并删除保护清单允许的旧中间 checkpoint、归档过时启动脚本：

```bash
bash cleanup_w2v_v37.sh
bash cleanup_w2v_v37.sh --apply
```

清理复用现有 best/SHA/引用/运行锁保护，不删除原始音频、两个完整 Train noisy 视图、固定完整 Dev、512维特征、教师向量、教师权重、原 best 或选中 patch。它不会停止 V3.7 或其他版本；若其他任务仍活跃，维护操作拒绝删除。V3.6 仍在运行时，预览只列出拟停止进程，后续清理计划会因活跃任务而拒绝执行。

## 官方提交

```bash
bash run_eval_w2v_v37.sh --upload-temp
```

默认读取无标签 Progress 协议，可通过 `V37_RUN` 指定运行目录，`EVAL_PROTOCOL` / `EVAL_AUDIO_ROOT` 指定无标签 Eval，`SUBMISSION_DIR` 指定输出位置。按协议原顺序导出 P(fake)，完成后输出 `SUBMISSION_ZIP` 和 `TEMP_DOWNLOAD_URL`。不使用 Progress/Eval 标签或音频训练、调参。

最终 `best_patch.pt` 与 SHA 绑定的原 best 配合使用，不能单独替代原 checkpoint。只有选中语言去偏时 patch 内才保存语言分支和映射；原 best 回退或仅分类头对照不需要这些分支参数。**提交推理不需要独立 ECAPA 教师、教师向量或教师软件包**，仍需原 detector 使用的特征提取器文件。

若打印 `SUBMISSION_BASELINE_FALLBACK=True`，提交的是原 best。报告 ZIP 保存比较指标、逐条 Dev 分数、来源覆盖、分组调参结果和选择理由，不携带音频、大型特征矩阵或模型权重。

## 验证范围

离线小模型测试覆盖特征复用、来源分组、Train-only 拟合、去偏映射、单模型重放、保护回退、patch 身份、完整音频导出、分数方向和协议顺序。控制台测试确认所有指标在进度上方保留，退出查看器不向训练进程发送信号。Shell 入口做语法检查。固定版本的真实语言教师权重已通过本地 CPU 加载及完整分段推理检查。最终选模前还会在服务器实际设备上用部署模块重放候选 Dev 向量并核对误差，随后按部署模块的分数选模。它们验证实现和保护机制，不替代服务器完整运行或官方成绩。

方法参考：[Language Orthogonalization, 2026](https://arxiv.org/html/2609.16458v1)。这里使用训练期教师蒸馏、中心化、可调强度和原分类器锚定，与论文的直接语言编码器推理不同。学习到的语言向量还可能包含来源和通信条件信息，不能把收益自动解释为已经剥离了纯语言因素。
