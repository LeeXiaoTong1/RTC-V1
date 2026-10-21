# V5.3：输入自适应层融合

V5.3 在 **V5.2 完整配方**上，只改变 Omni 多层融合权重的生成方式。默认从已完成的 V5.2 `final.pt` 开始，继续训练 **1 轮**。这不是已经验证提分的结论；本地 CPU 检查验证实现和数值行为，正式效果仍需服务器训练与同口径评分。

## 唯一的结构改动

原模型对所有音频使用同一组 `softmax(layer_logits)`。现在每条输入使用：

```text
16 kHz 接收端语音
  → 现有冻结 Omni frontend（卷积/位置声学特征）
      → 去除 padding 的时间均值 + 总体标准差
      → LayerNorm → Linear(2D,128) → SiLU → Linear(128,L)
      → 层权重调整 Δ(x)
  → 原 Omni Transformer + 保留的全部 LoRA
      → 原逐层 LN / 128 维投影 / SwiGLU
      → softmax(原 layer_logits + Δ(x)) 加权融合
  → 原 AASIST / 窗口统计 / 单一真假分类器
```

`D` 和 `L` 继承 parent 的编码器配置。条件来自 Transformer 前、同一次计算中已得到的冻结 frontend 输出，因此各层仍然可以流式汇总；无需第二遍 Omni，也不保存所有层的完整特征。条件统计按录音分别计算，排除批次 padding。它是浅层声学摘要，不是额外教师或显式噪声标签；没有引入 Clean/Noisy 条件分类损失。

新增 MLP 的最后一层权重和偏置为零，起始融合保持 parent 行为。共享基础层权重仍然可训练。新参数使用原 `fusion_lr`，随同真假分类、既有 PCL 和对比学习目标联合更新。冻结 frontend 不因条件分支解冻。

## 原配方保留哪些内容

| 内容 | V5.3 行为 |
|---|---|
| 分类 CE | 官方 Online 60% / 同源模拟 Clean-RTC 20% / Noisy-RTC 20% |
| PCL | 原 V5.2 实际系数与冻结尺度原样继承；**不会再乘一次 0.2** |
| 对比学习 | 保留 V5.2 A 侧同源官方 Online / 模拟 Noisy，继承温度、权重、预热及队列容量 |
| LoRA | 保留 parent 全部已学习适配器的贡献，仅最后 16 层可训练 |
| 后端 | AASIST、128 维融合输出、窗口统计和分类器不变 |
| 增强与输入 | 沿用 parent 的输入处理、模拟配方及上下文长度 |
| 数据 | 复用 parent 最终 Train+Dev 清单；其中 Offline 继续参与 Clean/Noisy RTC 模拟 |
| Dev 的 PCL | 不凭空为 Dev 创建音素标注；仍按 parent 原 Train 资格安排 PCL |
| 优化器 / 对比队列 | 从 final.pt 开始的是新一次增量实验，重新建立；`--resume` 才恢复 V5.3 自身优化器、队列和 RNG |
| 最终权重规则 | 继承 parent 已锁定的规则，如 `anchor_tail_50`；单一模型参数平均，不融合预测分数 |

PCL 的 `v52_parent_pcl_weights` 仍记录 V5.2 调整前系数，实际 `pcl_noise_weight` 继承调整后的值，保留方差辅助项原有的相对尺度。原有其他辅助设置也沿用配置，没有新加入 PCGrad、元学习或 XEUS。

新模块初始参数使用独立、可重复的 CPU 随机状态创建，不改变原训练的随机流。权重平均时，把 V5.2 parent 扩展成带**初始零输出门控**的同结构模型；绝不把训练后的门控当作 parent。V5.3 续训和导出则要求全部门控参数齐全，不能忽略缺失键。

## 服务器更新与启动

先等待当前 V5.2 完成。以下命令在原 Linux 训练服务器执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin v5.3-adaptive-layer-fusion
git switch v5.3-adaptive-layer-fusion

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v53.sh --epochs 1 --submit-best --upload-submission --upload-temp
```

不传 `--parent-run` 时，读取 `xlsr_aasist/exp/.latest_v52_run` 中记录的目录，要求它确实是**已完成 V5.2**。会验证 `completed.json`、`final.pt`、最终配置、锁定配方和数据清单。若记录缺失、指向未完成运行或内容不匹配，明确停止；不会静默退回 `last.pt`、V5.1 或其他最近目录。历史上传失败不等于训练失败，只要最终模型的完成记录与校验正确即可继承。

若服务器有多个 V5.2 运行，显式指定希望使用的那个完成目录：

```bash
bash run_w2v_v53.sh \
  --parent-run /完整路径/已完成的V5.2运行目录 \
  --epochs 1 --submit-best --upload-submission --upload-temp
```

传运行目录，而不是 `final.pt` 文件或 `final/` 子目录。启动日志会打印实际 parent 路径、tag 和 SHA256，方便确认。显式的已完成 V5/V5.1 final 也兼容，但默认只读取 V5.2。

每轮步数由继承的数据清单与来源覆盖预算决定，不硬编码 2229。延续原最终训练的学习率衰减策略，从 parent 训练末期的组学习率开始，1 轮内衰减到其一半；新门控与融合组使用同一学习率。没有额外 warmup epoch。

## 中断、继续和上传重试

```bash
# 查看当前日志
bash watch_w2v_v53.sh

# 请求在已提交更新边界保存；Ctrl+C 也走同一保存路径
bash stop_w2v_v53.sh

# 等待保存完成后恢复同一次 V5.3
bash run_w2v_v53.sh --resume "$(cat exp/.latest_v53_run)" \
  --submit-best --upload-submission --upload-temp

# 网络上传失败后重试；不重新训练、不重新评分已有 scores
bash upload_w2v_v53.sh --run "$(cat exp/.latest_v53_run)"
```

不带 `--resume` 的再次启动会创建**新实验**，从 V5.2 parent 重新开始。`--resume` 使用已保存设置，拒绝变更融合模式、损失参数或轮数。提交 zip 和报告先保存在 `--download-dir`（默认 `/home/ubuntu/LXT/temp`），上传失败时文件保留。

## 如何判断门控有没有工作

每 200 个训练步骤采样一次：

```text
[V5.3 fusion] mode=dynamic input_std=... delta=... gate_update=...
```

- `input_std`：本次编码视图之间，各层权重的平均标准差。
- `delta`：动态权重相对当前共享基础权重的平均绝对变化。
- `gate_update`：本步门控参数实际更新量的范数。

第一步前向的 `input_std` 和 `delta` 为零是预期行为；第一步 `gate_update` 应能非零，后续才开始出现输入差异。静态对照下门控冻结，这些值应为零。统计覆盖本步全部编码视图，包括辅助参照，**不是独立的 Online/Noisy 分组成绩，也不能证明门控学到了噪声而非语言或说话人**。

报告保留 `run_summary.json`、`parameter_inventory.json` 和 `fusion_diagnostics.jsonl`。体积过大的步骤日志仍按报告上限省略，并在 `REPORT_MANIFEST.json` 标明；关键设置和门控采样优先收录。`parameter_inventory.json` 中 `weights` 是共享基础权重，逐输入权重另看采样日志。

## 可选的结构对照

V5.2 final 已见过全部 Train+Dev，不能再把这些来源当作“未见验证集”。默认竞赛增量路线因此没有 Dev 选模。下面的来源隔离结构诊断只能显式从未用 Dev 训练的 V3.18.9 best 开始，两次采用相同输入、数据、损失、预算和种子：

```bash
# 第一组：共享层权重；先完成后再启动第二组
bash run_w2v_v53.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v3189_20261016_183410_a437 \
  --phase screen --fusion-mode static --epochs 1

# 第二组：输入自适应层权重
bash run_w2v_v53.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v3189_20261016_183410_a437 \
  --phase screen --fusion-mode dynamic --epochs 1
```

这两个对照采用 V5.2 目标，但不加入 Dev 训练。不搜索新的输入处理或最终权重规则，统一使用 `anchor_tail_50`。动态组还在配方锁定后，使用首个均衡 **Train** batch 的平均动态权重做一次固定权重 Audit；计算均值不读取 Dev 或 Progress。它是诊断，不参与重新选模。比较两份报告的 `screen_completed.json` 与各组 Recall/Macro-F1。

历史 Dev 已用于此前研究判断，所以这里是训练来源隔离的内部对照，不是全新盲测。静态训练对照检验整体结构收益；同一动态模型的固定权重 Audit 辅助检查收益是否依赖输入变化。两者都不能用“权重变化了”代替真实提分证据。

## 本地验证

```bash
python -m unittest discover -s w2v_v53 -t . -p 'test_*.py'
python -m unittest discover -s w2v_v52 -t . -p 'test_*.py'
```

检查使用小型 Omni 形状编码器和实际 AASIST，以及真实优化、配对 CE/PCL/对比损失代码。覆盖零初始化等价、60 层流式/重计算、梯度、静态对照、parent 继承、AMP 失败回滚、续训与权重平均、导出和上传重试。没有在本地执行完整 3B CUDA 训练或竞赛评测。
