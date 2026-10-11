# V3.18.2：同源三视图、平滑来源采样、分类约束的对比学习

本版本在独立的 `w2v_v3182` 包和 `exp/w2v_v3182_*` 目录工作。沿用 V3.18 的 Omni W2V 3B、SSL-AASIST、最后层特征、128/64 帧局部汇聚、最后16层 rank16 LoRA，以及原有噪声与 RTC 实现。无需重新下载已有且校验通过的 Omni 权重。

训练目标改为：每个来源的 Offline、官方配对 Online、当次生成 Noisy 均参与分类，并在最终分类前的160维特征上增加同源对比。仅推理时输入一条录音。

## 从 Git 更新服务器

继续使用原来的 `LeeXiaoTong1/RTC-V1` 仓库和 `w2vbert2-balanced-robust-fast` 分支。在当前服务器执行：

```bash
(
set -e
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
git pull --ff-only origin w2vbert2-balanced-robust-fast
bash audit_w2v_v3182.sh --weights
bash run_w2v_v3182.sh --workers 2
)
```

真实环境检查通过后才会启动训练。这里沿用服务器之前跑通的两个数据加载进程；其余使用本版本默认配置。无需从本地上传压缩包。

## 训练入口

在已有 V3.18 的 Linux 训练环境中进入 `xlsr_aasist`，先执行：

```bash
bash test_w2v_v3182.sh
bash audit_w2v_v3182.sh --weights
```

第一条使用合成音频、小型编码器和实际 AASIST 完成 CPU 回归；第二条检查原生处理链、真实 Omni 3B 前后向、三视图梯度重算和 LoRA 更新，并报告 GPU 峰值内存。CPU 回归不代表真实 3B 或真实数据训练已完成。`--weights` 会加载约11.42 GiB的3B权重，需要服务器GPU。

已有 V3.18 环境不需要重装。首次安装可使用 `bash setup_w2v_v3182.sh --cuda 11.8`，或按原环境选择 `12.6`；安装器复用 V3.18 的固定依赖与环境检查。1B资产仅在明确指定 `--omni-size 1b` 时使用。

直接开始主方案 B：

```bash
bash run_w2v_v3182.sh
```

默认按 `.latest_v318_run`、`.latest_v317_run`、`.latest_v316_tfcl_run` 顺序寻找数据来源。也可明确指定一个现有训练目录：

```bash
bash run_w2v_v3182.sh --data-run /absolute/path/to/existing_run
```

仅复用经过校验的 Train/Dev 清单、配对身份及噪声资源。新训练使用官方预训练编码器、全新检测头和 LoRA；不会续训旧检测器。历史数据清单及其来源文件仍须保留。

启动后自动后台运行。关闭查看器不会停止训练：

```bash
bash watch_w2v_v3182.sh
bash show_w2v_v3182.sh
```

`--no-watch` 只关闭自动查看；`--upload-temp` 才上传本次诊断报告。默认不上传。报告不包含模型权重和原始音频。

## A/B 对照和共享预热

主方案每次在预热结束时保存不可变 `warmup.pt`、`warmup.json`。主方案 B 完成后，保存其完整路径，再从同一预热状态运行 A：

```bash
V3182_TREATMENT_RUN="$(cat exp/.latest_v3182_run)"
bash run_w2v_v3182.sh --init-warmup "$V3182_TREATMENT_RUN" --ce-only
```

等 A 完成后：

```bash
V3182_CONTROL_RUN="$(cat exp/.latest_v3182_run)"
bash compare_w2v_v3182.sh \
  --control "$V3182_CONTROL_RUN" \
  --treatment "$V3182_TREATMENT_RUN" \
  --out /absolute/path/to/new_v3182_comparison.json
```

比较器核对架构、采样、清单、代码、训练预算及共享预热哈希。A的对比权重为0，B大于0；报告分别给出选模子集与完整Dev的差值，以及六组 real@95%fake、real@99%fake 的变化。比较目录必须在两个训练目录之外。

也可先运行 `--warmup-only`，然后分别从该目录使用 `--init-warmup` 和 `--init-warmup ... --ce-only` 启动 B/A。每次仅启动一个训练任务。共享预热继承父实验的模型、学习率、批次、随机种子、优化器、BN统计和数据；禁止顺便调整这些设置。它只支持本版本预热状态。

## 固定初始配置

| 项目 | 默认值 |
|---|---|
| 每次更新的来源数 | 16，三视图共48条波形 |
| 来源采样 | 各组来源数开平方，再分配整数配额 |
| 当前清单配额 | EN-fake/EN-real/ZH-fake/ZH-real = 4/2/7/3 |
| 分类与对比锚点权重 | 四组分别求均值，每组25% |
| 完整配对的视图CE权重 | Offline/Online/Noisy = 10%/45%/45% |
| 缺少Online | Offline/Noisy-A/Noisy-B = 10%/45%/45% |
| 对比温度 | 0.1 |
| 对比权重 | 联合训练第1轮从0渐增至0.05，之后固定 |
| 对比负样本有效总质量 | 4，EN/ZH各半；记录实际来源数 |
| 训练预算 | 2轮分类预热＋8轮联合训练 |
| 联合学习率 | LoRA 1e-5，AASIST 3e-5，汇聚权重1e-4 |
| BN | 预热更新运行统计，联合阶段固定统计、学习仿射参数 |
| 旧的两份Noisy最大损失项 | 关闭 |

每轮更新数为 `ceil(去重来源数/16)`；现有38,660来源对应2,417次更新。EN-real为1,500个来源，每轮4,834次抽样，平均3.22次。组内完整遍历后才重新打乱，跨轮延续游标，并处理洗牌边界以保证同一逻辑批次不重复来源。有多个经过验证的Online版本时按来源循环选择；没有Online的来源不会被删除或伪造配对。

波形数量接近旧版不意味着计算时间相同；对比阶段重算前向会增加耗时。减少来源曝光但保持组监督总权重，也不保证消除来源过拟合。三视图共享来源，不算三个新说话人。

## 对比目标与梯度实现

每个来源有四条有向正对：Offline→Online、Online→Offline、Offline→Noisy、Noisy→Offline。缺少Online时，使用Offline与两份Noisy的双向正对。

负样本仅来自其他来源的相反标签；相同标签不进入分母负样本。每个负样本使用与正样本对应的视图槽，避免把另一个来源的三份视图误算为三个独立来源。EN、ZH负样本分别求指数相似度均值，再各占固定负样本质量的一半。该目标是加权InfoNCE变体，不是未经修改的SupCon。

四组对比损失各占25%，每个来源内四条有向正对取平均。无置信度删样本，不把Offline设为停止梯度的教师，也无独立投影头。

整个逻辑批次都参与对比。物理微批次可拆开三视图，以使用显存预算。启用对比时：

1. 无图前向得到全部小批次的logits与160维特征，记录每个小批次的随机状态。
2. 在完整逻辑批次计算CE＋对比目标，缓存对logits与特征的梯度。
3. 逐小批次恢复随机状态并重算，注入两种梯度，最后统一裁剪与更新。

重算核对输出误差；发生不一致时明确失败。联合阶段BN运行统计必须固定。重算结束恢复第一遍结束时的随机状态，避免额外消耗Dropout随机数。CE-only不执行额外前向。

方法参考：[Gradient Cache原论文](https://aclanthology.org/2021.repl4nlp-1.31/)。本项目实现专门处理双输出梯度、Dropout和BN状态，仍需真实GPU检查确认本机原生后端行为。

## 恢复、选模和导出

```bash
bash run_w2v_v3182.sh --resume /absolute/path/to/v3182_run
bash run_validate_w2v_v3182.sh --run /absolute/path/to/v3182_run
bash run_eval_w2v_v3182.sh --run /absolute/path/to/v3182_run
```

按完整提交的轮次恢复；中断轮会重放。恢复模型、LoRA、优化器、BN、随机状态及来源游标，并清理未提交的步级日志。`warmup_ready` 表示只完成预热，不能导出为训练完成模型。恢复使用保存配置，不能借 `--resume` 改学习率、采样或对比参数。

继续使用同一份固定Dev，按来源将20%保留给校准拟合、80%用于选模；官方Online和同源Noisy保持同一分区。选模指标仍为原始分数的 Weighted F1：0.3×Online Macro-F1＋0.35×Seen Macro-F1＋0.35×Heldout Macro-F1，各条件按实际样本池计算。

模型选定后，只在校准来源上拟合全局正仿射变换，在选模子集比较原始分数与该变换。变换无收益则保存恒等变换。`--raw` 强制导出原始分数；默认导出已经比较选定的方案，不保证总是使用仿射校准。没有按语言或Dev分组分别部署阈值。

选模子集也参与了原始/校准方案选择，因此该成绩不是全新独立测试。Noisy Dev仍是既有模拟数据，不等同于官方未知通信平台的测试分布。

## 诊断与验收

每轮保存：

- `coverage_epoch_*.json`：独立来源、重复次数、配额、配对覆盖、实际条件权重。
- `steps.jsonl`：加权CE、对比损失、对比系数、正负相似度、候选来源数、重算误差、梯度范数和耗时。
- `training_history.json`、`dev_scores_*.npz`：完整逐轮结果及原始分数。
- `diagnostics/curves.html`、`groups.csv`：真假precision/recall/F1、AUC/EER、real@95%fake与real@99%fake。
- `calibration_best.json`：校准拟合信息、原始与校准结果、最终采用的变换。

默认Train探针为四组各512个可配对来源的三视图，最多6,144条波形；它不是全量Train，也不是独立测试。其Noisy视图组成不同于旧版探针，因此不要把两版探针当成完全相同的清单。

验收关注：相同fake recall下的real recall是否改善，来源曝光是否符合计划，Train/Dev差距是否缩小，以及B相对共享预热A的改善。不能单凭对比loss下降或固定阈值real recall上升认定有效；本版不承诺回到历史分数。
