# V4.1：执行效率优化与 V4 原进度续训

V4.1 使用 V4 的 w2v-BERT 2.0、24 层 LoRA/多层融合、MHASP、MLP。
八组 CE、Online/Noisy PCGrad、官方与模拟双 PCL、数据划分、增强配方、来源采样、物理 microbatch 和选模方式保持一致。
本版只改变执行方式。没有加入降噪预处理。

## 本版优化

- 官方 Mel/CMVN/帧堆叠在 CPU worker 内完成；主进程只重组原有物理批次并传入 GPU。
- 每个 worker 默认保留 64 MiB 有界 LRU，复用原始 PCL 参考和原始 Online 的 Mel。动态增强不使用静态缓存。
- 保留官方特征的奇数帧尾部和填充值，完整波形、PCL 时间网格不变。
- Noisy PCL 使用同一次增强已经生成的基础波形，不再重复做一次速度/频响/幅度增强。
- FP32 梯度缓冲复用，梯度合并不再逐参数复制到新存储。CE 与 PCL 仍分别求导，PCGrad 仍只作用于 Online/Noisy CE。
- 特征、logit、loss 的非有限检查和日志统计延迟至整次更新结束统一读取；错误在优化器提交前拒绝。
- 默认在服务器上探测激活重计算的显存需求，尝试 24/18/12/6/0 层。所有 24 层 LoRA 始终训练。

显存探测使用允许的 Train 长度上界、物理批大小和帧预算内的最坏形状，运行前向及 CE/PCL 两次反向，不使用 Dev/Test，也不提交优化器更新。
保留至少 2 GiB 或总显存 10%（取较大值）的余量，并计入尚未分配的 Adam 状态。
探测恢复 RNG；结果记录在 `execution_memory.json`。无足够余量时保留全层重计算。
若后续训练在优化器提交前 OOM，则恢复全层重计算、恢复同一批 RNG 并重试；不会缩短音频、跳样本或减少 PCL。
优化器提交阶段的 OOM 不自动重试，以防重复/部分更新。

## 在 V4 第 2 轮结束后升级

先看到第二轮评估结束，再在运行训练的终端按 **一次 Ctrl+C**，等待 `PAUSED_RUN=...`。
退出 `tail` 日志查看只会关闭查看器；这种情况下可在能执行命令的终端运行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
bash stop_w2v_v4.sh
```

等待训练输出 `PAUSED_RUN` 后，首次更新：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch --track origin/v4.1-training-efficiency

source /home/ubuntu/LXT/v318-runtime/envs/sdd-v4/bin/activate
cd xlsr_aasist
bash run_w2v_v41.sh \
  --continue-from "$(cat exp/.latest_v4_run)" \
  --upload-temp
```

继续使用现有 sdd-v4 环境（Transformers 4.38.2），无需重新下载模型或准备音素。
V4 的文件保持原样，旧代码/数据的哈希校验继续生效。
新建 V4.1 实验目录，原 V4 checkpoint 不被改写。

迁移从 `last.pt` 恢复模型、Adam 动量、GradScaler、RNG、PCL 校准权重、累计指标、来源采样位置及历史 best。
不从 best 重新开始，不重复头部预热或 PCL 校准，不重启学习率日程。
原来总预算为 6 轮，第 2 轮完成后就继续第 3～6 轮；**不要添加 `--epochs 6`**。
如果停下时第 3 轮已进行几步，也会从保存的下一步接续。
训练参数覆盖在迁移时会明确拒绝，不会看似接受却忽略。
原 V4 的自动 submission 请求会一并继承。

V4.1 再次中断后使用：

```bash
bash run_w2v_v41.sh --resume "$(cat exp/.latest_v41_run)" --upload-temp
```

查看日志或安全停止：

```bash
bash watch_w2v_v41.sh
bash stop_w2v_v41.sh
```

仅首次迁移可选择执行选项：`--workers 6`、`--eval-workers 2`、`--feature-cache-mib 64`。
默认沿用 V4 worker 数量；每个 worker 的 Mel 缓存会额外使用至多 64 MiB 内存。
保持全部重计算可用 `--checkpoint-policy fixed --checkpoint-layers 24`。
显存探测本身只在首次进入联合训练时运行；日志中的探测耗时不是每个 epoch 的固定成本。

## 验证与性能边界

53 项 CPU 测试通过，覆盖：完整 Mel（包括奇数尾部）的精确重组；两种 PCL 与 PCGrad 的梯度/Adam 更新一致性；
静态缓存与动态增强隔离；基础增强只计算一次；所有 LoRA 层的梯度；部分重计算的输出/梯度/RNG；
V4 第 2 轮迁移后仅训练剩余轮数；真实 spawn worker；模拟 OOM 后保留 RNG/Adam 的重试；中断恢复、best 选择、单模型导出和数据来源隔离。

数学目标不变，浮点归约顺序变化可能产生微小数值差异，不承诺跨版本训练轨迹逐位相同。
本地 CPU 测试使用 PyTorch 2.11 / Transformers 5.8 的小型原生模型；生产继续固定原 V4 的 Transformers 4.38.2。
本地没有服务器 GPU；完整生产模型的实际显存和提速幅度以服务器探测及稳定训练日志为准。
正常训练的 CE/PCL 仍需分开反向，这是保留原有梯度协调目标的成本。
