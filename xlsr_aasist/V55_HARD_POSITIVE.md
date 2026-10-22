# V5.5：从 V5.1 回退基线接入 Hard Positive 训练

V5.5 从已经完成并提交的 **V5.1 `final.pt`** 继续训练。模型、增强、输入处理和基础损失回到 V5.1；新增的是论文的困难真实样本分组、Guidance 与 Triplet 两项训练目标。

默认父运行：`exp/w2v_v5_20261021_075307_d991`。V5.1 在磁盘上的版本号是 `5`，代码位于 `w2v_v5`。新入口拒绝 V5.2、V5.3、V5.4 和未经核验的 checkpoint，不会自动改用 `last.pt`。

## 1. 哪些内容保留，哪些内容新增

| 部分 | V5.5 的实际行为 |
|---|---|
| 模型结构 | 原 V5.1 OmniASR、静态多层融合、AASIST 窗口证据及统计池化。没有新增推理模块。 |
| 前端微调 | 保留父模型所有已学习的 LoRA，仅更新最后 16 层；较早层的适配器继续参与前向，但冻结。 |
| 分类权重 | 官方 Online 25%、模拟 Clean-RTC 25%、模拟 Noisy-RTC 50%。保留类别/语言平衡和原标签平滑。 |
| 原有损失 | 保留模拟配对 PCL、官方平台 PCL、原 PCL 方差项及已有的置信度正则；沿用父运行的开关、参数和 Train 音素资产范围。 |
| 数据与增强 | 复用 V5.1 的 Train＋Dev 来源、覆盖预算、基础增强、接收端处理和已锁定输入策略。Dev 来源继续参与分类和新增句级损失；没有音素资产的 Dev 不参与 PCL。 |
| 优化器 | 沿用 V5.1 的普通联合 AdamW 更新，学习率从父运行结束时的设置继续；新运行不加载父模型的 Adam 历史。 |
| 新增内容 | 同源 Real 的两个处理版本、困难正样本选择、Guidance、全部 2×3 配对的 Triplet。 |
| 新损失的位置 | 原分类器最后 dropout **之前的 384 维句级表示**，包含窗口证据均值、标准差和窗口内时间变化。不是帧级 128 维融合特征，也不是分类概率。 |
| 最终权重 | 沿用父运行锁定的权重平均规则；若为 `anchor_tail_50`，使用 V5.1 final 权重 50%＋本轮尾段平均权重 50%。单个模型推理。 |

V5.2 的对比记忆、V5.3 的动态层权重、V5.4 的修复分支和 KD 均不在依赖路径中。历史版本文件保持原样，新实现放在独立的 `w2v_v55` 中。

`v51_baseline.json` 固定了 204 个继承的运行时 Python 文件；这些文件与 V5.1 的实现提交 `a218dc1` 相同。启动时核对内容，允许 Windows/Linux 换行差异；后续续训另由本次运行的完整代码指纹约束。

## 2. 论文机制及项目适配

参考：Jiwon Seo 等，*Hard Positive-targeted Training for Robust Audio Deepfake Detection under Neural Codec Processing*，Interspeech 2026，DOI: 10.21437/Interspeech.2026-2167。[作者论文](https://www.isca-archive.org/interspeech_2026/seo26b_interspeech.pdf)

原文每组含一个 Clean Real 参照、两个经神经编解码处理的 Real、三个 Fake（两个未处理、一个处理后）。先选择离参照最远的 Real，再同时约束困难 Real 与 Fake、参照与 Fake 之间的距离。这里沿用原文公式（1）至（7），不引入独立教师、输出蒸馏、重建网络或类别原型。

项目中的六个表示：

| 符号 | 从当前 V5.1 数据流中取得什么 |
|---|---|
| `a` | 某条 Offline Real 的现有 Clean-RTC 版本。 |
| `p1` | 同一来源原有的 Noisy-RTC 版本，与 `a` 使用原 V5.1 的配对生成流程。 |
| `p2` | 同一来源、同一基础增强波形的第二个 Noisy-RTC 版本，使用另一接收端配方随机分支，独立初始化处理状态。它不要求与 `a` 使用完全相同的 RTC 参数。 |
| `n1,n2` | 两个不同 Fake 来源的 Clean-RTC 版本；当前分组的 Offline Fake 不足时，使用已有官方 Online Fake。 |
| `n3` | 第三个不同 Fake 来源的 Noisy-RTC 版本。 |

六条都来自同一语言，且都接受原分类监督。`p2` 的生成只运行需要的 Noisy 处理，不额外生成随后丢弃的 Clean 版本。每组四个不同来源；三个 Fake 不重复。

**与论文的区别需要明确：**处理类型从神经编解码换成项目已有的环境噪声＋RTC；参照从未处理 Real 换成 Clean-RTC；模型保留 OmniASR＋原 AASIST；两个正样本明确使用同一来源。这是迁移论文的训练机制，不是复现其数据集、前端或实验成绩。

### Guidance：困难 Real 与 Fake 要分开

令 `d(u,v)=sum((u-v)^2)`，使用原始句级表示的平方欧氏距离，FP32 计算。没有另外做 L2 归一化或增加投影头。

```text
p_h = argmax_{p1,p2} d(a,p)
L_G = mean_{n1,n2,n3} softplus(d(a,p_h) - d(p_h,n) + m)
```

困难 Real 离参照较远、同时又接近 Fake 时，这个损失会要求调整它们的相对位置。选择索引不反传，但选中的向量、参照和三个 Fake 都参与反传；不存在停止梯度的教师。

### Triplet：参照也要保住真假间隔

```text
L_T = mean_{p in {p1,p2}, n in {n1,n2,n3}}
      max(0, d(a,p) - d(a,n) + alpha)
```

全部六个正负组合参与计算。该项补充 Guidance：参照与 Fake 的距离应比参照与处理后 Real 的距离更大。某个组合满足间隔后，其 hinge 损失为零是正常行为。

### 与 V5.1 的目标如何合并

```text
L_V55 = L_V51_CE_PCL_regularizers + lambda_G * L_G + lambda_T * L_T
```

默认 `m=0.2`、`alpha=0.3` 是原文测试过的设置。`lambda_G=lambda_T=0.01`，前 200 个更新线性增加到目标权重。**这两个系数及预热长度是项目的首轮配置，不是论文给出的最优参数**；原文没有给出可直接移植的损失权重。没有增加一段只训练新模块的阶段。

两个逻辑侧 A/B 各自按计划分组数平均新损失，再各乘 1/2 合并。归一化不依赖拆成了多少个物理小批次。

## 3. 覆盖与计算预算

每个更新在 A/B 两侧、EN/ZH 两种语言中各轮换选一个 Offline Real 参照，通常为 **4 组**。其余 Real 仍接受原分类和 PCL 监督，并非每一步都进入新损失。这是明确的计算预算选择；选取依据是确定性种子、更新位置和来源，能够重放和续训。

只有组内两个正样本的“困难选择”使用当前表示距离；没有对整个数据集额外评分或维护历史表示库。最后不足完整分组的小批次保留全部 CE/PCL，并在日志中记录无法凑齐三个不同 Fake 的原因。

原来的每个 Noisy 来源分类权重，在被选中的 Real 上平分给 `p1` 和 `p2`。例如原来该来源的 Noisy CE 权重是 `w`，现在两项各为 `w/2`；全局仍是 Online/Clean-RTC/Noisy = 25/25/50。原置信度正则和 PCL 仍使用原主视图。

每步最多增加 4 次音频编码。CE、PCL 和新损失共享这次编码及分类前向。六条用于度量的表示保留在同一个物理批次的计算图中，不做跨批次 detach、重算或漏项。

默认 `--microbatch 16 --frame-budget 9000`。一个完整度量分组连同原配对分类和官方 PCL 参照，最多需要 13 个视图；程序提前检查全部训练 epoch 的分组尺寸。若过长则报告所需预算，不静默截断音频或拆散分组。帧预算只是资源上界检查，不能代替实际 GPU 显存测试。分组长度和反向开销决定真实速度，不承诺只增加某个固定百分比。

## 4. 启动、暂停、续训、导出

推荐通过 Git 更新。分支 `v5.5-v5.1-hard-positive` 直接从 V5.1 提交 `a218dc1` 建立，再加入 V5.5 文件：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch v5.5-v5.1-hard-positive
git pull --ff-only origin v5.5-v5.1-hard-positive
```

也可以使用增量包：包内部以 `xlsr_aasist/` 为根，在上述仓库目录下解压。原有 V5.1 模型、数据和运行目录必须存在；缺失或改变会由启动检查指出。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
bash run_w2v_v55.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v5_20261021_075307_d991 \
  --epochs 1 \
  --submit-best --upload-submission --upload-temp
```

本项目原训练服务器有两个同名的 `sdd-v318` 环境，必须使用上述完整路径。`conda activate sdd-v318` 可能进入系统盘的 `/home/ubuntu/.conda/envs/sdd-v318`，其 PyTorch/CUDA 构建与原驱动不兼容。终端前缀都显示 `(sdd-v318)`，不能据此判断环境正确。其他服务器应使用其已验证可运行 V5.1 的环境完整路径。

启动前可核对：

```bash
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

原服务器的已固定配置为 Python 位于 `/home/ubuntu/LXT/v318-runtime/envs/sdd-v318/`、PyTorch `2.6.0+cu118`、CUDA build `11.8`；`CUDA available` 和 `BF16` 均应为 `True`。若这里失败，先检查实际环境，不要直接重装依赖或升级驱动。此检查在创建训练运行、读取父 checkpoint 和更新模型之前执行；此处退出可修正环境后重新运行启动命令，无需 `--resume`。

输出的新目录为 `exp/w2v_v55_时间_随机后缀`。默认起点就是上述 V5.1 运行，也可以显式指定另一个通过完整性核验的 V5.1 已完成运行。

```bash
# 请求在完成当前更新并保存后暂停；等待 V5.5_PAUSED=... 再续训
bash stop_w2v_v55.sh

# 使用刚输出的 V5.5 目录，不能填写 V5.1 父目录
bash run_w2v_v55.sh --resume /完整路径/exp/w2v_v55_时间_后缀 \
  --submit-best --upload-submission --upload-temp

# 查看当前日志
bash watch_w2v_v55.sh

# 上传失败后重试，复用已经生成的文件
bash upload_w2v_v55.sh --run /完整路径/exp/w2v_v55_时间_后缀
```

Ctrl+C 同样请求保存后暂停。正常续训恢复模型、优化器、随机状态、损失累计和尾段快照；崩溃时从最近已保存的更新恢复，未提交的日志会被裁去。续训不能改变损失权重、分组策略或 epoch 数。

若需要公平的损失对照，另起一轮并传 `--hp-mode control`：使用相同的第二个正样本、分组、分类权重和训练预算，但把 Guidance、Triplet 权重置零。它与原 V5.1 直接续训仍有数据视图差异，不能混称一个对照。

## 5. 如何确认新约束在做什么

终端示例字段：`CE=... G=... T=... HP=4 encoded=...`。

| 记录 | 应该看什么 |
|---|---|
| `final/steps.jsonl` | 原始/加权 G、T，组内距离，选中第几个正样本，六条预测的正确性与真实标签概率；`p_true` 始终是真实标签概率。 |
| `final/gradient_audit_*.json` | 每 200 步及每个 epoch 首步抽查；分别记录两项损失到句级表示和融合投影的梯度，以及与 CE 的梯度方向关系。后者是代表性物理批次、侧权重合并前的局部检查。 |
| `final/hard_positive_summary.json`、`final/epoch_*.json` | 间隔违反比例、平均距离/特征范数、Clean 正确而 Noisy 错误次数、Real/Fake 及各条件错误率。选中组的统计不代表所有训练来源。 |
| `final/coverage_epoch_*.json` | 原来源覆盖、分组数和跳过原因。 |
| `final/parameter_inventory.json` | 保留和可训练的 LoRA 层，确认只有最后 16 层参与更新。 |

`T=0` 不自动意味着失效：当六个组合都满足间隔时本来就应为零。应结合间隔违反比例与梯度看。`G` 使用 softplus，充分满足间隔时也可能极小；不为追求日志非零而增加常数或虚构梯度。

V5.1 已经使用全部 Train＋Dev，不能把同批 Dev 再当未见来源来证明泛化。最终效果须检查官方 Clean/Noisy 分项，尤其同时看 Real 误报和 Fake 漏检。新约束针对表征间隔，不能保证真实 Noisy 指标必然提高。

## 6. 本地验证范围

CPU 测试覆盖原文两项公式、六个配对、困难选择、梯度、真实 AASIST 多窗口读出、原 V5.1 关闭新分组时的数值/梯度一致性、CE 权重守恒、增强复用、来源覆盖、资源边界、暂停/崩溃恢复、父 checkpoint 核验、最终权重平均和提交文件重试。

2026-10-22 验证：31 项 V5.5 测试和 23 项 V5.1 相关回归测试全部通过；5 个 Shell 操作脚本逐一通过 Bash 语法检查。

测试使用小型编码器替身连接实际 AASIST 后端。完整 3B OmniASR、真实 RTC 数据和 BF16 GPU 训练尚需在训练服务器运行；本地测试不构成性能提升或显存充足的证明。
