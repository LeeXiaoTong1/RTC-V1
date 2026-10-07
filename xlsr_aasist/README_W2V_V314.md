# V3.14：先适配最终分类层，再做简化的监督微调

起点固定为实际提交过的 **V3.12 `last.pt`**：
`exp/w2v_v312_20261006_004830_7d2f/last.pt`。
用户反馈该模型官方 Weighted 为 **93.558**；它的本地固定 Dev Weighted 约为 **96.2945**。
这两个分数对应不同数据，不能混用。V3.12 的 `best.pt` 曾回退到旧基线，不能代替这里的 last。
V3.14 没有新的官方成绩，不能预先保证达到 97。

## 为什么做这一版

V3.12–V3.13 一直冻结最后的 `Linear(512,2)`，主要靠上游特征迁就原来的决策边界。
V3.13 的两视图策略又使 Noisy 分类预算从 V3.12 的 50% 降到实际约 26%。
V3.14 保留 V3.12 已学到的特征和有界 adapter，让最终分类层一起适应，并恢复 Noisy 预算。
这不是把分类器适配当成全新算法；之前也在更早的表示上试过，这次使用的是训练后的 V3.12 实际表示。

## 两个阶段

| 项目 | Stage A：最终分类层适配 | Stage B：联合微调 |
|---|---|---|
| 冻结部分 | 编码器、MultiConv、有界 adapter、第一层 FC | 编码器前部 |
| 可训练部分 | 最后的 512→2 Linear，共 1026 个参数 | 编码器最后八层、MultiConv、已有 adapter、两层 FC |
| 初始化 | 完整保留 V3.12 last 的权重和偏置 | Stage A 通过保护条件则接续，否则从 V3.12 last 接续 |
| 目标 | 真假交叉熵 + 相对起点的参数 L2 | 真假交叉熵 + AdamW 参数正则 |
| 数据预算 | 四个 EN/ZH×真假组各 25%；普通/Noisy 各 50% | 同样的预算，每次抽一个普通视图和一个 Noisy 视图 |
| 训练长度 | 固定最多 80 次 PyTorch L-BFGS 迭代，梯度达标可提前结束 | 默认最多 2 轮，每半轮验证；连续 2 次没有足够进展就停 |

Stage A 重新提取 **V3.12 last** 的 512 维句子向量。旧 V3.7 向量仅用于读取和核对已有数据清单，不能拿来训练这个分类器。
求解使用 PyTorch，避免复用曾崩溃的 SciPy/OpenBLAS 求解路径。目标为加权 CE 加 `0.5 × 0.1 × (||W−W0||² + ||b−b0||²)`，只约束参数位移。
原始权重不重置，也不做阈值搜索。拟合结束后通过实际完整模型重放 Dev，核对缓存打分与部署打分。

Stage B 学习率：编码器顶层 `2e-7`，向前逐层乘 `0.8`；MultiConv/第一层 FC `2e-6`；adapter `1e-5`；最终 Linear `2e-5`。
使用 5% warmup、余弦衰减、梯度裁剪和 BF16 训练；Dev/导出为完整录音 FP32。
移除语言对抗、配对、困难样本排序、教师保留和特征/分数尺度匹配损失。
仍然保留有限值检查、已有 adapter 范数上限和异常停止。去掉语言对抗不等于证明语言差异无关，这版直接检验最终真假决策能否改善。

普通视图在官方 Offline/Online 之间轮换，Noisy 在已有 A/B 之间轮换。缺失官方 Online 的源使用 Offline，**不会降低该源的 50% Noisy 预算**。
源池跨轮继续轮换；“一轮”是固定数量的平衡抽样，不承诺每轮遍历全部多数类源。采样覆盖、重复次数和实际损失预算保存在 `sampling_epoch_*.json`。

## 评价与三个导出选择

只使用官方 Train 更新参数；Dev 用于验证和选模。Progress/Eval 不参与拟合、阈值搜索或参数选择。
本地 `Weighted = 0.3 × Online Clean Macro-F1 + 0.7 × mean(Seen,Heldout Macro-F1)`，固定阈值 0.5，分数为 P(fake)。
官方 Offline 仅在起点和整轮验证时作诊断，**不计入 Weighted**。缺少已验证的 Offline 元数据时明确记录不可用。

| 选择 | 含义 |
|---|---|
| `best_weighted` | 起点及所有已验证候选中本地 Weighted 最高的状态，不因保护条件被拒就丢弃 |
| `best_guarded` | 同时满足分类保护条件的最佳状态；默认导出选择 |
| `last` | 最后一次成功验证且已提交保存的联合训练状态；未发生联合更新则是实际进入联合阶段的模型 |

保护条件相对 V3.12 last 固定：Weighted 至少 +0.02 个百分点；Noisy 不下降；Clean 最多下降 0.3 个百分点；每个语言/处理组 fake recall 最多下降 0.5、real recall 最多下降 1.5、AUC 最多下降 0.2 个百分点；在 fake recall≥99% 时的 real recall 最多下降 2 个百分点。
后一个值是排序诊断，不会成为部署阈值。保护条件不会阻止保存 `best_weighted` 或 `last`。
连续两次验证没有比此前进展最好值提高至少 0.02 个百分点就停止；比起点 Weighted 下降超过 2 个百分点会提前停止。
每次验证打印完整指标、两个 best 的实际 tag、错误救回/新增错误统计位置和早停状态。

三种选择在一个事务检查点中保存，相同的状态共用张量，未选中的历史半轮权重不累积。
最终每次导出仍是**一个模型**。`best_weighted.json`、`best_guarded.json`、`last.json` 是可读的选择记录。
`submission_meta.json` 记录实际 tag、源权重哈希、当前权重哈希和协议哈希，避免再次混淆基线和新模型。

## 时间与磁盘

- 不重新生成 WAV，不生成帧级缓存，不增加在线 RawBoost，不进行额外教师前向。
- 以 153105 条 Train 和 22943 条 Dev 计，512 维 FP32 向量约 **344 MiB**；加分数与元数据通常是数百 MiB，总量以启动打印为准。
- 第一次运行多一次完整 Train 的冻结前向提取。后续恢复复用已提交的缓存。**首轮总时长包含这次提取，不能只按每步时间估计。**
- 联合训练每源两个视图，微批立即反向并释放，不再同时保留配对图；最多两轮。实际 A100 速度仍须以日志 `seconds/update` 为准。
- 每次验证保存最后八层及后端的部分权重、Adam 状态和至多两个不同的最佳候选，不复制冻结前端，不保留每轮完整模型。
- `storage_budget.json` 和启动日志给出按实际参数量估计的磁盘峰值。**344 MiB 只指向量，不包含训练检查点**；检查点仍需数 GiB，原子保存时会暂时保留旧、新两份。
- 空间不足时保留上一次已提交状态并报出所需空间，不自动删除数据、原始 best 或 V3.12 last。

## 部署和运行

在已有仓库及 `sdd` 环境执行；不重新安装 CUDA/PyTorch，不下载语言模型：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash setup_w2v_v314.sh
bash run_w2v_v314.sh \
  --source-run exp/w2v_v312_20261006_004830_7d2f \
  --upload-temp
```

终端保留每次验证结果。关闭查看器不会停止后台训练：

```bash
bash watch_w2v_v314.sh
bash show_w2v_v314.sh
```

恢复会读取原运行配置，不修改学习率、轮数或起点；只重放上次成功验证后的未提交步骤：

```bash
bash run_w2v_v314.sh --resume exp/你的V3.14运行目录 --upload-temp
```

停止当前 V3.14 及其已核实的子进程：

```bash
python -m w2v_v314.stop --apply
```

完成后导出，默认 `best_guarded`；环境变量可指定其他已完成的 V3.14 运行：

```bash
bash run_eval_w2v_v314.sh --checkpoint best_guarded --upload-temp
bash run_eval_w2v_v314.sh --checkpoint best_weighted --upload-temp
bash run_eval_w2v_v314.sh --checkpoint last --upload-temp
```

三条是不同选择的导出方式，按本地验证结果决定使用哪一个，不通过反复查看 Progress 反馈来选参数。上传失败时 ZIP 仍保存在日志所示本地目录。

## 完成后的清理

只清理**已完成的 V3.14** 自有内容，保留三种导出选择及依赖的原始权重。先预览，再执行：

```bash
python -m w2v_v314.cleanup --run "$(cat exp/.latest_v314_run)" --remove-features
python -m w2v_v314.cleanup --run "$(cat exp/.latest_v314_run)" --remove-features --apply
```

清理去掉 optimizer/RNG，并可删除本轮句子向量。权重压缩到 `inference.pt`，三种导出仍有效；该已完成运行不再支持训练恢复。
清理使用先保存新文件、切换完成记录、再删除旧文件的顺序，所以也需要临时空闲空间。
保留 V3.12 的 `last.pt/config.json/completed.json`、原始 base checkpoint、配置引用的旧数据/特征清单。
这版不顺带删除旧版本代码，因为它们仍参与来源校验和模型重建。

## 验证范围

本地测试使用真实的小型 w2v-BERT＋MultiConv 与生成的测试波形，检查分类层/后端更新、半轮断点恢复、缓存重放/损坏检测、50% Noisy 预算、三种 submission 导出以及清理后的预测一致性。
这些测试验证实现正确性，不代表已在 A100 或比赛完整数据上证明性能提升。
