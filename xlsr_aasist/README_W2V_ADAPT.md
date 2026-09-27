# 保留模型，修正训练控制并小比例补充噪声覆盖

上一轮本地 RobustF1 从 95.483 提升至 95.544 / 95.537，但严格的七项零退步条件拒绝了两轮候选。第二轮刚降低学习率就早停，降低后的学习率没有用于训练。本入口解决这些控制问题，并复用已经生成的多样噪声训练库。

## 固定设置

- 从原始 `w2v_rebuild_20260920_093548/stage3/best_model.pt` 开始新实验；对照旧 config 的初始化 SHA256，不覆盖原文件，也不读上一轮 last 作为初始化。
- w2v-BERT 2.0 + 当前 AASIST 不变；仅最后 4 层及分类头更新，前 20 层和特征投影冻结。
- encoder/head 学习率 `1e-7 / 2e-6`，real CE 成本 `1.25`，ordinary 仍是原采样及逆频率权重；一致性损失关闭。
- 总 batch、训练步数、原 noisy CE 权重 30%、配对损失及其 warmup 不变。
- 最多 3 轮。第一轮新库比例从 0 逐步增至 20%（整轮约 10%），以后每轮约 20%；这是保守起点，不保证最优。
- 每个 noisy 配对 batch 整体选择同一个库，内部仍为 2 fake + 2 real 源。因此两类进入新库的比例完全相同。每个源在每个库独立轮转四个噪声档，不增加每步前向次数。
- 只使用现成 `train_g0` 和 `train_g1`。Dev Seen/Heldout 固定，不进入训练。当前 full metadata 校验继续检查角色、源文件、标签、处理算法及噪声划分。
- `data.py`、特征提取及特征缓存身份保持原样；不重建或清理任何音频/特征缓存。

## 保存与选模

原 best 文件永久保留。新实验最多保留三个 checkpoint 文件：

| 文件 | 含义 |
|---|---|
| `candidate_best.pt` | `(RobustF1, -BalancedCE)` 最好的新候选；即使有 real/noisy 取舍也独立保存，避免被后续 last 覆盖。没有更好的候选则不创建。 |
| `best_model.pt` | 通过下面目标约束的最佳模型；没有合格改善时保留初始化权重。 |
| `last.pt` | 最新模型、优化器、调度、采样和进展状态，供本版本精确恢复。 |

目标约束：RobustF1 必须严格超过原始基线，且优于已有合格 best；Online real recall、八档平均 noisy real recall、Seen/Heldout 平均 noisy Macro-F1 均不低于原始基线。Offline、单独 Seen/Heldout F1 和逐档退步仍完整记录为 warnings，不再各自一票否决。

因此上一轮第 1 轮会保存为综合分数候选，但因平均 noisy F1 下降而不升级目标 best；第 2 轮可升级目标 best，Offline 下降会明确显示。这些都是 Dev 选择规则，不代表官方未知 noisy 数据必然改善。

保存候选的磁盘空间已加入启动预留，并采用原子写入。不会保存所有 epoch 权重。日志与 `completed.json` 明确区分 `improved`、`candidate_only` 和 `no_eligible_improvement`。

## 调度与早停

早停监测综合候选分数、平均 noisy F1 或 BalancedCE 是否取得新进展，不再把“保护条件拒绝”当成“训练无进展”。候选分数采用 F1 优先、CE 决胜，CE 单独进展要求相对改善至少 0.1%。

LR 减半当轮不触发提前停止，至少允许下一轮实际使用较低 LR 后再判断。总预算仍最多 3 轮；若最后一轮才触发降 LR，日志会明确说明预算已到，不能宣称尝试过该低 LR。

新参数都为显式启用。旧启动器保留原行为；不能把旧源码下的 last.pt 当作新源码的精确 resume。

## 服务器启动一次

无需安装新依赖或重新生成缓存。新入口从上一轮 config 读取路径，默认在 Dev Seen 缓存的同级目录查找已有的 `train_g1`；找不到会明确停止，可使用 `--diverse-cache` 指定已有训练库。

```bash
conda activate sdd &&
cd "$HOME/LXT/RTC-w2v-improved" &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash run_w2v_adapt.sh exp/w2v_refine_20260927_101702_caf4
```

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
tail -n 60 -f "$(cat exp/.latest_train_log)"
```

若先查看配置：`python -u start_w2v_adapt.py --from-run exp/w2v_refine_20260927_101702_caf4`。增加 `--run` 才会训练；正式脚本自动在后台执行并使用已有互斥锁。

回退点：`backup/w2v-refine-before-control-noise-20260927` 指向上一版 `ce79d3c`。原始最佳版本的备份分支也保留。上一轮 `last.pt` 不会被本入口修改。

## 验证范围

81 项 CPU 回归检查通过，覆盖三阶段训练、冻结层不更新、候选与目标 best 分离、截图指标重放、第三轮实际使用减半 LR、断点状态与噪声配额恢复、磁盘预留、原子保存，以及原生 FFmpeg 音频缓存和官方特征提取的衔接。实际音频集成测试中的 WebRTC 模块使用测试替身；本次服务器训练复用已生成的 WebRTC 音频，不会再次运行该处理模块。

CPU 回归检查验证代码行为，不证明真实 GPU 训练精度提高；最终根据这一轮固定 Dev 结果选择候选。
