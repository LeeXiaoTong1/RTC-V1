# V3.18.3：从已审计的 C3 epoch 10 增量训练

本版落实已选定的两项改进：改善采样与 RTC 覆盖，并加入中间层融合。
它是待 GPU 实验验证的方案，不能把实现完成等同于泛化成绩提高。
旧代码标为 V3.18、对话中称 V3.18.1 的父模型是同一次训练：
`w2v_v318_20261010_135116_c6fa / epoch_10_step_2417`，完整 Dev raw Weighted 约 85.9684%。
父 checkpoint SHA256：`cb355866a76d9a1e959e3941747f600b2280c2f4244601c1d74fd198eea6149d`。

## 数据与目标

- 每次逻辑更新：独立抽取 16 条官方 Online，再抽 16 个原始来源，各生成 Noisy A/B。总计 48 条输入，约 32 个独立来源；两条流可能碰到同一原始来源。
- 原始 Offline 只用于生成 Noisy，不输入检测器、不计算 Offline CE、不参与对比损失。
- 每条流的 EN-fake / EN-real / ZH-fake / ZH-real 配额为 **4/2/7/3**。保留每条流 50%、每个组 25% 的监督预算：每个来源的损失系数为 `0.125 / 该组配额`。
- 保留 C3：Online 使用 CE，Noisy 使用 `0.75 * mean(CE_A, CE_B) + 0.25 * max(CE_A, CE_B)`。较难视图承担 62.5%、较易视图承担 37.5% 的该来源损失。label smoothing 仍为 0.02。
- 同一来源的 Noisy A/B 不跨物理批次；各流各组循环覆盖，循环边界也不会在同一逻辑批次内重复同一来源。
- 逻辑批次现在有 33 条 fake、15 条 real，但损失预算仍是 50% / 50%。继续冻结父模型 BN 运行统计，保留 BN affine 学习。

以原清单和 2,417 次更新/轮计算，EN-real Online 的平均曝光从每轮约 6.8 次降至约 3.4 次。
具体来源覆盖、最大重复和各组损失预算写入 `coverage_epoch_*.json`。采样变化不会增加原始来源的总量。

## 多层融合与效率

Omni-W2V-3B 仍读取最终输出；同时在**同一次编码**中读取第 15、30、45 层（从 1 开始计数）。
中间层做逐帧无参数 LayerNorm，并匹配最终特征的均值/标准差；参考统计不反传。
三个 softmax 权重混合中间层，再计算：

`H = H_final + 0.25 * tanh(g) * (H_mid_mix - H_final)`。

三个混合 logits 和门控 g 均初始化为 0，共新增 **4 个可训练参数**。
零门控时输出等于父模型；训练时保留门控梯度，随后逐渐学习中间层贡献。
残差系数允许正负，绝对值不超过 0.25；因此这不是强制非负的凸融合。
这是一种可兼容旧后端的轻量起点，尚未证明这些层或门控上限最优。

AASIST、窗口 128 / 步长 64、局部贡献模块以及最后 16 层 Q/K/V/O LoRA 保持不变。
第 15/30 层是冻结特征，第 45 层位于 LoRA 范围中，最终层也可适应；不会为了读取早期层而解冻整个前端。
没有 InfoNCE 的双前向重算。中间特征会增加显存及少量融合计算，实际速度需 GPU 测量。
FFmpeg/WebRTC 沿用原环境；新 RTC 变换在 CPU 工作进程运行。常规统计合并传回 CPU，每 100 步记录一次模块梯度范数。

## RTC 改进

Noisy A 保持旧配方；Noisy B 首轮以 17.5%、第二轮起以 35% 概率额外选一种机制：

- 0.5–2 秒变化的增益包络，范围约 ±2–6 dB；
- 60–180 Hz 高通与 3.4–5.6 kHz 带宽限制；
- 受控软限幅；
- Opus 码率覆盖扩至 12–48 kbps。

先混入环境噪声，再施加输入变换、NS/AGC 与编解码。原有 SNR 范围保留。
新增机制最多影响全部训练视图的约 11.7%；两份 Noisy 仍选择不同基础处理机制。
没有把简单波形处理称为完整 AEC、真实网络丢包或 PLC，这些机制本版尚未覆盖。
固定 Dev、Train probe、独立机制 panel 均继承父运行，探针仍使用原增强配方。
panel 中的 G.711 / anlmdn 不加入训练，也不参与 Weighted 选模。

## 增量训练设置

| 模块 | 本次最高学习率 |
|---|---:|
| LoRA | 2e-6 |
| AASIST | 6e-6 |
| 窗口贡献模块 | 1e-5 |
| 融合参数 | 1e-4 |

默认追加 **6 轮**，新目录内编号为 1–6。继承已训练模型的全部 LoRA、head 参数与 BN 缓冲区。
因为优化器增加了融合参数组且训练分布改变，本次新建 AdamW，不继承旧 Adam 动量。
前 200 次更新进行学习率预热（极短实验最多占总更新 10%），随后 cosine 衰减至峰值的 20%。
不重新进行分类头预热，不关闭已训练 LoRA，不重新估计 BN。C3 risk=0.25 从开始保留。
weight decay=0.01、全局梯度裁剪=1，梯度记录用于发现模块间更新不协调，不自动修改参数。
这些学习率是受控实验起点，尚无新的 GPU 对照成绩。

启动首先运行小型真实 fairseq2 接口检查，验证层钩子、重计算、梯度和保存。
随后加载父模型并完整重跑固定 Dev，比较父运行保存的逐条 logits：
`rtol=0.001, atol=0.003`，Weighted 差值还须 ≤0.1 个百分点。未通过时在任何训练更新前停止。
预检会多花一次完整 Dev 推理时间，只在新运行开始时执行完整父模型复现。
结果在 `parent_baseline.json`，应接近 85.9684%，而不是从随机后端分数开始。

当前运行的 best 只在本次训练的候选中选择；epoch 0 是独立参照，不会暗中替代退步的新模型。
报告每轮相对父模型的 selection Weighted 变化，最终 `completed.json` 也记录差值。
固定阈值的 recall/F1、AUC/EER 和 Real@95/99Fake 同时保留。后两项是 ROC 诊断，不作为部署阈值。
单一全局 affine 校准仅拟合预留 20% 来源，在 selection 上决定使用校准或 identity。
完整 Dev 包含校准来源，不能作为独立测试成绩；Progress/Eval 不参与后续选模或调参。

## 服务器运行

在原仓库与原 V3.18 环境中更新，不需要上传本地文件，也不需要重装依赖。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash run_w2v_v3183.sh \
  --parent-run exp/w2v_v318_20261010_135116_c6fa \
  --epochs 6
```

启动器与旧版一样后台运行，`Ctrl+C` 仅关闭进度查看器。
`bash watch_w2v_v3183.sh` 重新查看进度，`bash show_w2v_v3183.sh` 查看最近日志。
新运行使用 `.latest_v3183_run`，不会覆盖父运行或 `.latest_v318_run`。

只想先验收父模型再开始训练：

```bash
bash run_w2v_v3183.sh --baseline-only --no-watch
# 等到日志显示 BASELINE_VERIFIED_RUN 和正常退出，再执行：
bash run_w2v_v3183.sh --resume "$(cat exp/.latest_v3183_run)"
```

所有新运行都自动执行父模型复现，无需单独先跑上述预检。
中断后用相同 `--resume` 命令恢复：从最后完整提交的 epoch 回放，恢复优化器、融合参数和随机状态，清理未提交的步骤日志。
`--resume` 不接受学习率、采样、特征层等覆盖，以免同一运行身份混乱。

验证与导出：

```bash
bash run_validate_w2v_v3183.sh --checkpoint best --raw
bash run_eval_w2v_v3183.sh --checkpoint best
```

导出默认应用本次选定的全局校准；`--raw` 导出原始分数。
沿用已有评测路径参数与 `EVAL_PROTOCOL` / `EVAL_AUDIO_ROOT` 环境变量。
报告本地保存至 `/home/ubuntu/LXT/temp`；仅显式 `--upload-temp` 上传报告包。

## 消融开关与验证边界

默认运行同时使用两项改进，因此单次结果不能证明哪一项贡献了增益。
可从相同父模型另起新目录逐项控制：

```bash
# 只改采样（保留 C3、旧增强、最后层）
bash run_w2v_v3183.sh --no-fusion --rtc-stress-probability 0
# 改采样与 RTC，保留最后层
bash run_w2v_v3183.sh --no-fusion
# 完整方案；还可以用 --group-quotas 4 4 4 4 控制采样差异
bash run_w2v_v3183.sh
```

CPU 验证：`bash test_w2v_v3183.sh`，使用小编码器、真实 AASIST 和合成音频。
包含旧模型零门控精确复现、融合/LoRA 梯度、采样预算、微批次梯度等价、RTC 长度/确定性、训练/恢复/校准/导出。
发布前通过 11 项 CPU 检查和 8 个 Bash 脚本语法检查，并核对原始 75,785 条训练清单的采样预算。
本机不能执行真实 Omni-3B CUDA 训练；服务器启动检查和首轮结果仍是必要验收。
