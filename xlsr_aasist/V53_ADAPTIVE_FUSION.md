# V5.3：从 V5.1 继续训练的输入自适应层融合

V5.2 的 Progress weighted 为用户报告的 94.52，略低于此前 V5.1。因此按最新决定，V5.3 改为 **V5.1 配方＋输入自适应层融合**，从 V5.1 已提交的 `final.pt` 增量训练，默认 **1 轮**。新模型的实际收益尚未验证，不能由实现测试推断提分。

## 起点与目标

默认 parent 固定为：

```text
xlsr_aasist/exp/w2v_v5_20261021_075307_d991/final.pt
```

这是此前 94.61 分对应的 V5/V5.1 运行。配置版本号在文件中仍为 `5`。启动检查完成记录、最终配置、锁定权重规则、checkpoint SHA256 和最终训练清单，不使用 `.latest_v52_run`，不回退到 `last.pt` 或旧祖先模型。上传失败不影响已完成模型的使用。

| 项目 | 当前 V5.3 |
|---|---|
| 分类 CE | 官方 Online 25% / 模拟 Clean-RTC 25% / 模拟 Noisy-RTC 50% |
| 组均衡 | 各视图内 EN/ZH × Real/Fake 等权，沿用 V5.1 |
| PCL | 直接继承 V5.1 的真实系数、冻结尺度、方差与其余辅助设置，不再乘 0.2 |
| PCL 配对 | A 侧 Clean-RTC ↔ Noisy-RTC，以及有有效音素对应的官方 Online ↔ Clean-RTC |
| 对比学习 | 不启用，不安排专用参考视图，不创建、搬运或保存队列 |
| 数据 | 继承完整 Train+Dev 清单及来源预算，Dev Offline 继续参与 Clean/Noisy RTC 模拟 |
| Dev 音素 | 不为 Dev 凭空生成音素标注，PCL 资格仍限定原 Train 资产 |
| 前后端 | Omni 3B、多层投影/SwiGLU、128 维输出、AASIST 和窗口统计保持原样 |
| LoRA | 保留 parent 全部适配器贡献，仅最后 16 层可训练 |
| A/B | 两侧各半权重的普通梯度累加，没有虚拟元更新或 PCGrad |
| 输入 | 沿用 parent 已锁定的输入处理、增强和 RTC 上下文长度 |
| 选模 | 已训练过完整 Dev，不再用 Dev 选模，继承 parent 已锁定的参数平均规则 |

这次从 V5.1 final 开始新建增量实验。优化器重新建立，学习率从 parent 末期组学习率开始，1 轮内衰减到其一半。`--resume` 才恢复当前 V5.3 的优化器、随机状态与进度。步数由继承的数据清单和来源预算决定，不硬编码为 2229。

## 唯一的结构变化

原模型的所有录音共用 `softmax(layer_logits)`，现在改为每条输入的：

```text
softmax(原 layer_logits + 条件 MLP(当前录音的浅层统计))
```

条件统计直接取自同一次冻结 Omni frontend 输出的有效帧均值和总体标准差，排除 padding。MLP 为 `LayerNorm(2D) → Linear(2D,128) → SiLU → Linear(128,L)`，在当前 Omni 3B 上 D=2048、L=60。它不额外执行 Omni，不保存 60 层完整特征，仍逐层投影并流式汇总。

末层零初始化，初始预测与 V5.1 一致。所有新增参数使用融合组学习率，由原有 CE/PCL 训练。权重平均时使用初始零输出的门控作为 parent anchor，不把训练后的门控错当作 parent。续训与导出要求门控参数完整。

## 已去除的冗余

| 位置 | 调整 | 保留的行为 |
|---|---|---|
| 普通训练步 | 不再每步复制全部可训练参数，仅每 200 步更新量诊断时备份 | 提交前的 buffer、模式与 RNG 回滚，AMP 失败重试仍保留 |
| 逐层融合重计算 | 确定性的 LN/Linear/SwiGLU 不保存冗余 RNG 状态 | 60 层融合、激活重计算保留；Omni/LoRA 的 dropout RNG 保护不变 |
| 波形进入 GPU | 在 CPU 检查输入，再按同长度组一次堆叠传输 | 不改变归一化位置、有效长度、padding 或输出几何 |
| 进度文件 | 每 10 步和轮末刷新，与控制台同步 | 每个已提交更新仍写详细 steps.jsonl，保存边界与续训语义不变 |
| V5.2 对比学习路径 | 取消其专用编码、后端计算与队列操作 | V5.1 分类视图、有效 PCL 参考仍全部保留 |

最后一项是配方回退带来的计算减少，不能称为对 V5.2 的等价提速。其余工程优化以相同 V5.1 目标做数值回归。不通过裁剪音频、减少窗口、减少层数或删掉有效配对来加速。

Omni 完整编码、最后 16 层反向及必要激活重计算仍是主要开销。本地只有 CPU，尚无服务器 CUDA 吞吐量和峰值显存实测，不承诺具体提速百分比。`steps.jsonl` 的 `execution.parameter_snapshot_bytes` 可确认普通步为零，诊断步保留快照。

## 更新并训练

在 Linux 训练服务器执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin v5.3-adaptive-layer-fusion
git switch v5.3-adaptive-layer-fusion
git pull --ff-only origin v5.3-adaptive-layer-fusion

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v53.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v5_20261021_075307_d991 \
  --epochs 1 --submit-best --upload-submission --upload-temp
```

省略 `--parent-run` 也使用上面的固定目录。目录迁移后可以显式指定新的完整 V5.1 运行目录。传运行目录，不是 `final.pt` 文件或 `final/` 子目录。启动日志会显示 `inherited V5.1 CE Online25/CleanRTC25/Noisy50`、`contrast=OFF` 和实际 parent 的路径、tag、SHA256。

**不要用旧 V5.2 或基于 V5.2 的旧 V5.3 运行做 `--resume`。** 本次 recipe/schema 已独立更新，会明确拒绝把旧配方误当作新配方继续跑。历史 V5.1/V5.2 文件和模块不修改。

```bash
# 查看日志
bash watch_w2v_v53.sh

# 请求在已提交更新边界保存，Ctrl+C 也使用该保存路径
bash stop_w2v_v53.sh

# 等待保存完成后，恢复本次新配方 V5.3
bash run_w2v_v53.sh --resume "$(cat exp/.latest_v53_run)" \
  --submit-best --upload-submission --upload-temp

# 上传失败后重试，不重新训练、不重复评分已存在的 scores
bash upload_w2v_v53.sh --run "$(cat exp/.latest_v53_run)"
```

## 中断续训与困难样本记录

一次 Ctrl+C 请求在当前更新提交后保存。等 `V5.3_PAUSED=...` 或返回提示符后，再更新本分支并执行上面的 `--resume` 命令；不要重新指定 `--parent-run`，也不要额外增加 `--epochs`。日志中的 `next_step` 以磁盘 checkpoint 为准，可能略晚于终端最后打印的第 461 步。若强制杀进程，恢复到最近一次已保存的更新，未保存部分会重做。

启动器现在加载独立的 `w2v_v53_observe` 只读观察器。原 `w2v_v53` 模型、训练、损失代码以及已有 config、recipe lock 和 checkpoint 不改写，因此当前 V5.3 可以直接续训。观察器自身的代码指纹保存在阶段目录的 `hard_examples_policy.json`，并校验它所依赖的分类输出顺序。历史版本和 V5.2 的训练不受影响。

观察器读取本来已经计算的分类 logits，不增加 Omni/AASIST 前向或反向，不改变数据抽样、dropout、梯度、学习率、优化器或随机状态。每步只额外搬运很小的分类结果到 CPU；日志和汇总仍有少量开销，CUDA 实际耗时尚未测量。

每个已提交更新记录所有分类视图的分组统计，并按 **加权 CE 超出标签平滑理论下限的贡献** 排序，保存最多 8 条“判错或单样本 CE 超出下限至少 0.05”的视图。包括：

- 来源 ID、原音频路径、Train/Dev 来源、语言、真假标签、视图类型和时长；
- P(fake)、正确类概率、未平滑 CE、平滑 CE、该样本在整步 CE 中的权重及超额贡献；
- 基础增强、Noisy/RTC 配方及可用的种子、SNR、编解码和活动门控摘要；
- 对同源模拟配对，同时记录 Clean-RTC/Noisy 的预测，区分只 Noisy 错、只 Clean 错、两者都错、两者都正确；
- 区分错分、判对但置信度不足、判对但过度自信受到标签平滑惩罚。

这些是训练时已有 dropout 下的观察，不是重新执行一次确定性评估。Online 独立分类来源没有凭空补出同源 Noisy；纯 PCL 参考没有分类输出，不计入分类困难样本。模拟视图的路径指向原始 Offline 文件，实际处理条件另存于配方；不保存或上传音频。

输出位于当前运行的 `final/`（历史 screening 路线则为 `screen/`）：

| 文件 | 内容 |
|---|---|
| `hard_examples.jsonl` | 每步完整观察；超过 8 个困难候选时明确记录省略数量及已记录样本覆盖的超额 CE 比例 |
| `hard_examples_summary.json` | 全部已观察分类视图的组统计、最差更新、最多 100 个来源/视图组及其最差实例，定期和停止时刷新 |
| `hard_examples_policy.json` | 观察器代码、开始位置、阈值与恢复位置 |

汇总纳入原报告打包；完整 JSONL 若超过原报告大小限制，会列为跳过项，但保留在服务器。来源出现次数与配对结果只针对实际记录的困难候选，不能当成全量来源错误率。

**从本次更新后才开始记录具体样本，不能追溯之前第 121、381 等步的具体音频。** 中断续训会按 checkpoint 游标清除未保存的尾部观察，AMP 重试不重复记为训练步。

另开一个终端可以查看当前最困难的 10 个来源/视图组：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
python -m w2v_v53_observe.summary --run "$(cat exp/.latest_v53_run)" --top 10
```

终端的 `[V5.3 hard]` 行直接提示高损失更新的错误数、最主要贡献样本及 Clean/Noisy 配对情况。统计同时核对整步 CE 与每个分类组的损失/正确数；若对应关系不一致，明确报错并隐藏样本身份，避免错误归因。

## 诊断与对照

每 200 步记录门控权重的跨输入标准差、相对共享权重的变化和参数更新量。第一步前向中权重差异为零是预期结果，参数更新后才会出现输入差异。权重变化不能证明学到了噪声适配，更不能代替实际 Noisy 指标。

若做来源隔离的结构对照，显式从尚未用 Dev 训练的 V3.18.9 best 开始，分别运行 `--phase screen --fusion-mode static` 和 `--phase screen --fusion-mode dynamic`，其余参数、种子和预算相同。两组都采用当前 V5.1 目标。已用完整 Train+Dev 训练的 V5.1 final 只能走固定预算竞赛增量路线，不能再把原 Dev 当作未见来源。

## 本地验证

```bash
python -m unittest discover -s w2v_v53 -t . -p 'test_*.py'
python -m unittest discover -s w2v_v5 -t . -p 'test_*.py'
python -m unittest discover -s w2v_v53_observe -t . -p 'test_*.py'
```

覆盖真实 AASIST 与小型 Omni 形状编码器、零初始化迁移、60 层流式融合、启用 dropout 的重计算梯度与随机状态一致性、与 V5.1 完整一步优化的损失/参数/Adam 状态一致性、AMP 失败回滚、暂停续训、最终权重平均、导出和上传重试。完整 3B CUDA 训练与竞赛效果需在服务器验证。
