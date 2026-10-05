# V3.11：在保护真假判别的前提下学习语言去偏

V3.10 的候选出现了有价值的信号：示例截图 Weighted=96.499，语言探针识别率从 91.1% 降到 89.8%。但该候选仍 `eligible=False`、`Selected=baseline`：中文真音召回、部分 fake 检测和英文排序能力未保住。候选分数不是已经获准替换 best 的成绩，也不是官方 Progress/Eval 成绩。精确增益必须与本次运行重新测量的 baseline 比较，不能与旧缓存跨数值执行策略相减。

本版把重点放在训练过程中的冲突约束，而不是只在训练结束后拒绝坏结果。这是待验证的工程假设，不保证达到官方 Weighted 97。

## 动机与做法

1. **去掉语言差异，也可能碰到真假判别所需的信息。** 保留 real-only、同处理条件内的中英文对抗训练。在最终 512 维检测特征处，判断语言去偏梯度与真假分类方向是否冲突，只移除冲突分量，保留其余更新。语言分类器仍正常学习。最后 8 层编码器、整个 MultiConv、特征适配器继续训练。
2. **改善 EN-real 时，需要保住原本正确的 fake 和中文真音。** 从已核验的 Train 缓存分数读取原模型的正确类别优势；只有与真实标签一致且有把握的样本参与保护。当前正确类别优势低于目标时增加惩罚，达到目标则不限制它。原模型判错或犹豫的样本只接受真实标签监督，不被要求复制旧错误。
3. **分类器尚未适应时，过早施加去偏压力可能不稳定。** 首轮前 10% 步数训练真假任务与保护约束，让语言分类器先学习；之后逐步增加去偏强度，到首轮末达到 0.05。真假训练每步都在进行。
4. **把“还没通过 best 验收”与“训练不再进步”分开。** 默认最多 4 轮。正常情况下至少观察 2 轮，随后以候选 Weighted 连续 4 次验证没有改善为早停依据；比原基线下降超过 2 个百分点时立即早停。晋升 best 的原条件全部保留，不提高容忍度来制造赢家。

投影借鉴 [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)。本实现采用**特征层、逐样本、单向保护**，不是论文完整的参数梯度 PCGrad，也不保证共享网络和 Adam 更新后每个样本都不下降。因此仍需正确样本保护、固定 Dev 和独立语言探针共同验证。语言识别率下降只算辅助证据，不能代替检测性能，也不是因果消融结论。

## 默认设置与开销

| 项目 | 默认设置 |
|---|---|
| 训练轮数 | 最多 4；正常至少观察 2；灾难性下降例外 |
| 实际微调 | 编码器最后 8 层 + 全部 MultiConv + 特征适配器 |
| 学习率 | encoder 最大 2e-7，层衰减 0.8；MultiConv 2e-6；适配器/语言分支 1e-4 |
| 批次 | 16 个 source；语言 × 真假均衡；microbatch=4，frame budget=2400 |
| 输入 | 完整音频；复用普通/通信/噪声视图；不新增增强种类 |
| 去偏 | 4 个处理条件的 real-only 语言分类器；最大 GRL=0.05 |
| 正确样本保护 | 权重 0.2；原正确类别 logit 优势至少 1；留 0.5 余量；目标最高 4 |
| 验证 | 每半轮固定 Online Clean/Seen/Heldout Dev + 独立 Train 语言探针 |
| 数据划分 | 延续 V3.10 seed=31001 和采样规则，探针 source 与本轮训练隔离 |
| 新音频/帧特征缓存 | 0；使用原 Train logits，只在内存增加每行一个标量 |
| 大模型计算 | 每个 microbatch 一次检测器前向和反向；无第二个教师编码器 |
| 权重文件 | 滚动 last.pt、最终 best.pt；冻结前缀引用原 best |

新增运算发生在 512 维向量、二分类分数与小语言分支上，主模型计算预算与 V3.10 相同。本地 CPU 测试不能保证 A100 速度；报告记录真实 `seconds/update`。本版允许有进展的候选继续训练，总时长可能超过此前首轮早停的运行。

实际模型最坏 checkpoint 原子写入预算约 7 GiB，启动会按真实参数重新估计；这是新旧状态并存时的需求，不是每轮新增 7 GiB。已有音频、借用特征缓存和原 best 仍是依赖。

## 更新与启动

等 V3.10 打印 `complete`、保存完成后执行。V3.11 是独立运行，旧版文件保持原样。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash setup_w2v_v311.sh
bash run_w2v_v311.sh \
  --source-run "$(cat exp/.latest_v310_run)" \
  --upload-temp
```

V3.10 来源须已完成并选中 baseline。程序核验 best、原始来源和数据，重新从**受保护原 best**初始化，不使用未通过验收的 last.pt 候选。若 V3.10 后来晋升了新 best，程序会拒绝丢弃这个改进，避免意外退回旧模型。

也支持原有明确来源（同一受保护 checkpoint）：

```bash
bash run_w2v_v311.sh \
  --source-run exp/w2v_v39_20261005_101913_96bd \
  --upload-temp
```

启动以相同 FP32、相同 batch 规则核对零适配器与原模型，重新测量本次 baseline。历史 Dev 缓存只用于身份核验和数值偏差诊断，不充当本轮增益基准。恢复复用已核验的本次 baseline。

setup 检查现有环境并运行测试，不更换 torch/transformers，不下载新模型；有 CUDA 时执行 BF16 更新与原模型重放测试。启动拒绝与已有训练进程同时运行。

## 终端与结果

```bash
bash watch_w2v_v311.sh
bash show_w2v_v311.sh
```

保留每次验证结果；滚动进度显示当前训练。新增 `retention`、`conflict_rows` 和候选停滞次数。报告汇总整个验证区间的冲突梯度比例、移除的梯度能量、保护损失、每步耗时，而非只记录最后一个 batch。冲突比例用于诊断，高低都不自动意味着好坏。

验收仍看 Weighted、Noisy、Clean、中英文 real/fake recall、AUC 和匹配 fake recall 后的 real recall。语言探针使用本次微调未训练的官方 Train 真音来源，分别重训线性与非线性读出器；这不是全新语料测试，因为原模型曾见过 Train。

提交部署检测器和特征适配器，无需语言标签或语言分类器：

```bash
bash run_eval_w2v_v311.sh --upload-temp
```

输出仍是只含协议原顺序 scores.txt 的 ZIP，分数为 P(fake)。best.pt 为部分权重，必须保留引用的原 best。

## 停止、恢复与节省磁盘

Ctrl+C 只关闭显示器。停止训练：

```bash
python -m w2v_v311.stop --apply
```

恢复指定实际目录；配置不变，从最后成功保存的验证边界继续，此后未保存的步骤会重跑：

```bash
bash run_w2v_v311.sh --resume exp/w2v_v311_实际运行目录 --upload-temp
```

若 V3.10 已完成且不再需要续训，可回收其优化器状态。清理核验 best 和依赖，只删除这个已完成目录的 last.pt 及其临时文件，保留 best、报告、数据和旧代码：

```bash
python -m w2v_v310.cleanup --run "$(cat exp/.latest_v310_run)" --apply
```

V3.11 完成后同理：

```bash
python -m w2v_v311.cleanup --run "$(cat exp/.latest_v311_run)" --apply
```

需要消融时：`--adv-weight 0` 关闭对检测器的语言对抗更新；`--no-gradient-protection` 恢复未投影的语言梯度；`--retention-weight 0` 关闭正确样本保护。默认不用调整，不会自动多跑对照组。
