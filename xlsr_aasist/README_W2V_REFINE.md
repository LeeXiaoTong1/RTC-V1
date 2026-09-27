# 从原 best 做一次保守微调

八轮组合实验的 Online real recall 从 90.77% 降至 85.97%，本地 RobustF1 从 95.483 降至 94.531。新入口回到原 best 的训练数据配方，缩小参数更新范围；不需要手动跑一系列对照，也不需要重新生成缓存。不能据此保证新一轮一定超过原 best。

## 审核与取舍

未发现标签颠倒、分支切片错位或特征缓存改变输入。撤回三项组合是基于整体实验失败，不能断言每一项单独有害。

- ordinary 恢复原来的逐样本打乱、逆频率 CE。此前 balanced 在固定步数内每轮只覆盖约 61.7% fake，real 每条平均抽取 2.64 次；原 CE 已经做过频率补偿。
- 训练只用原 `train_g0`。新库混入比例原先约占 noisy 分支一半，而 processed noisy 原有 CE 权重为 30%，因此这不是很小的分布变化。新缓存保留但不参加本次训练。
- 关闭新增分数一致性。它的教师是同一训练模型的 detach 分数，不是冻结的原 best，无法阻止整个模型偏移。
- 保留新的固定 Dev Seen/Heldout 缓存，继续用相同条件衡量退步或改善。这些是本地模拟验证，不能等同于官方 noisy 分数；回退新训练库后 `seen` 只是沿用已有验证文件名。
- 保留 FP32 输入特征缓存、噪声 LRU、微批、磁盘空间预留和原子保存。此次不修改 `data.py` 或特征缓存身份，不会因为本次源文件更新废弃整套已有特征缓存。

## 唯一默认配置

| 项目 | 设置 |
|---|---|
| 起点 | 旧实验 config 中记录的原 best；核对 SHA256 |
| 模型结构 / 推理 | 不变，输出仍为 fake 概率，判定阈值仍为 0.5 |
| 更新范围 | 最后 4 个 encoder 层和完整 AASIST head |
| 冻结范围 | feature projection、前 20 层及对应 dropout |
| 学习率 | encoder `1e-7`；head `2e-6` |
| ordinary | legacy 顺序打乱 + 原逆频率 CE |
| real 误判代价 | 所有 CE 分支乘 `1.25`，各分支按权重和归一化 |
| noisy CE / 特征对比 | 保持原 beta=0.3、原对比损失权重和 pair warmup |
| 新分数一致性 | 关闭 |
| 训练预算 | 最多 3 轮；连续 2 轮没有合格提升就停止 |
| LR 调度 | 用起点的 Dev CE 初始化；patience=1，保留原 cooldown/floor |
| LR warmup | 0.25 轮 |

real=1.25 是温和的错误成本调整，不是再次对所有分支套逆频率。配对分支本身已均衡，real 占该分支 CE 权重从 50% 变成 55.56%。它可能改善边界附近的 real，不能保证解决深层区分错误。

冻结前缀不再反向传播，也不保存它的 Adam 动量；仍需全部 24 层前向。实际加速由服务器测量，不能承诺某个倍数。冻结部分在每次 `model.train()` 后仍保持 eval，避免只冻结梯度却继续扰动 dropout。

## 自动保留 best 的条件

先在原样 Dev 上验证起点，并把原权重保存为新实验的 epoch=0 best。候选须同时满足：

1. RobustF1 严格超过起点，且 `(RobustF1, -RobustCE)` 优于当前 best；仅 CE 下降不算超过原 best。
2. Online / Offline real recall 不低于起点。
3. Online、Seen、Heldout 的 Macro-F1 均不低于起点。
4. Seen / Heldout 各自四个 SNR 档平均 real recall 不低于起点。

每档 recall 和相对起点的差异也写入日志；上述均值保护不代表每一档都一定不降。门槛是保守的 Dev 模型选择约束，不是对未知评测集的性能保证。

无合格候选时，`completed.json` 为 `status=no_eligible_improvement, best_epoch=0`，新实验的 `best_model.pt` 仍是起点权重。`last.pt` 只是最新训练状态，不应自动拿它替换原 best。原 checkpoint 路径从不写入或删除。

## 服务器运行

以下代码块均为 ASCII，并少于 2000 字符。旧训练结束后，在原有 `sdd` 环境运行。无需安装新依赖。

```bash
conda activate sdd
cd "$HOME/LXT/RTC-w2v-improved" &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
python -u start_w2v_refine.py --from-run exp/w2v_recovered_20260925_235901_5249
```

上面只显示配置并核对原 best，不训练。正式启动一项后台任务：

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
bash run_w2v_refine.sh exp/w2v_recovered_20260925_235901_5249
```

查看日志：

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
tail -n 60 -f "$(cat exp/.latest_train_log)"
```

启动一次即可，避免重复启动。启动器使用已有 recovery 文件锁，锁会一直保持到子训练进程结束。`Ctrl+C` 只退出上述日志查看，后台训练继续。初始化现在会输出“检查音频”和“加载模型”进度；完整 Dev 起点评估仍需时间。取消单独的重复 preflight，首个真正训练 batch 自动审计冻结状态、梯度和参数更新。

## 回退

- 原 best 对应代码：`backup/w2vbert2-score-91.6866-20260925` / `8b99d367468963121334d5d6bc99b75cb2cf4e7a`。
- 本次修改前代码：`backup/w2v-improved-before-refine-20260927` / `5b171a9a65d83e22e60d35536f8e81afeb98dbe4`。
- 旧的 `run_w2v_improved.sh` 仅保留复现旧实验；本轮使用 `start_w2v_refine.py`。
- 更新源码后不能将旧 `last.pt` 当作精确 resume。新入口只读取 config 的路径，从原 best 开始新实验。旧源码分支可继续用于旧实验的精确恢复。

## 本地验证范围

CPU 测试覆盖真实 HF 小随机模型的冻结行为、checkpoint 梯度、旧 checkpoint 结构、CE 成本及梯度、原训练引擎、选择门槛、提前停止、恢复和磁盘保护。合成测试不代表服务器上的真实精度或 GPU 加速；最终以此次 Dev 结果为准。
