# V5.4：同源接收端特征修正

V5.4 从**已完成的 V5.3 `final.pt`** 增量训练。保留 Omni、60 层动态融合、AASIST、分类配方、输入处理和模拟链，在融合后的 128 维时序表示与 AASIST 之间接入约 19.7 万参数的修正模块。默认只训练 **1 个 epoch**，新模块预热包含在其中。

这是 DKDSSD 的“处理分支、原始证据融合、配对指导”机制在现有隐藏表示上的迁移，**不是论文的原样复现，也没有尚未实测的提分承诺**。参考：[IEEE/ACM TASLP 2024，DOI](https://doi.org/10.1109/TASLP.2024.3389643)。Omni 隐藏通道不是频率轴；这里采用时序卷积，不做谱图掩码。

## 启动与恢复

等待原 V5.3 输出完成后的 `final.pt`，然后执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch --track origin/v5.4-paired-feature-repair

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash run_w2v_v54.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v53_20261021_200839_f923 \
  --epochs 1 --submit-best
```

上述路径也是默认父运行目录。它必须具备相互匹配的 `completed.json`、`final.pt`、`final_config.json`、锁定配方和来源清单；不会把 V5.3 的 `final/last.pt` 或暂停断点冒充完成模型。若已经建立本地分支，用 `git switch v5.4-paired-feature-repair`。不需要新建环境或安装新依赖。

`Ctrl+C` 会在下一次已提交训练步、或准备阶段的小批次边界保存。等待 `V5.4_PAUSED=...` 后恢复：

```bash
bash run_w2v_v54.sh --resume /完整路径/w2v_v54_日期_标识 --submit-best
```

恢复沿用保存的设置；不要同时传 `--parent-run`、学习配方或 epoch 参数。准备统计每 32 对保存一次，正常暂停保存当前边界；完成的尺度标定会复用。训练断点包含修正模块、尺度、优化器、混合精度状态、随机状态及采样游标。

默认导出本地 submission/report。需要临时下载链接时，在启动或恢复命令追加 `--upload-submission --upload-temp`；上传失败后可以单独重试，不重训：

```bash
bash upload_w2v_v54.sh --run /完整路径/w2v_v54_日期_标识
```

## 数据流与梯度

```text
官方 Online ------------------------\
Offline → 同源 Clean-RTC / Noisy-RTC --+→ 原输入策略 → Omni → V5.3 动态融合 H[T,128]
                                     |                              |
                                     |                    Hhat = H + R(H)
                                     |                    a = sigmoid(Q(H,Hhat,差值))
                                     |                    Hstar = H + a*(Hhat-H)
                                     |                              |
                                     +------------------------- AASIST → CE
```

- `R`：两层残差时序块，包含核大小 3/7/15 的深度卷积。`Q`：逐帧逐通道交互门控。
- 修正输出层零初始化，初始门控 0.1，初始预测与已加载的 V5.3 一致；门控不能同时初始化为零，否则会堵住分类梯度。
- 所有分类音频和推理音频走同一模块，不要求知道测试样本是 Clean 还是 Noisy。
- 原始特征旁路保留；官方平台 PCL 继续作用于修正前的融合特征。当前工程的这一路实际是 **官方 Online ↔ 同源模拟 Clean-RTC**，使用已有 Offline 音素映射。
- 新 `LF` 使用停止梯度的 **原始 Clean-RTC 融合特征**，监督 Noisy 的修正分支 `Hhat`，不强制最终融合表示全部变成 Clean。
- `LF` 的 Noisy 输入也停止梯度：它只更新 `R`，不更新 Omni、动态融合、交互门控和分类器。CE 仍可训练完整开放路径；KD 可训练学生路径。
- `KD` 的教师输出来自同一个模型处理后的 Clean-RTC，停止梯度。它不是第二个大模型。

## 固定目标与预算

| 项目 | 默认行为 |
|---|---|
| 分类监督 | 官方 Online 25%、模拟 Clean-RTC 25%、Noisy 50%；继承 label smoothing 与置信度正则 |
| 原模拟 PCL | 移除，包括它的方差项 |
| 官方平台 PCL | 保留原有效系数；移除另一分支后不放大其方差项 |
| `LF` | 固定通道尺度后的逐帧 Huber；全局初值 λF=0.05 |
| `KD` | 温度 3，λD=0.02；Clean 正确且真实标签概率 ≥0.90，并比 Noisy 高至少 0.02，配对有效时启用 |
| A/B 逻辑流 | 新配对辅助项只作用于 A；显式补偿 A 的 1/2 外层系数，不让 λ 意外减半 |
| 归一化 | 逐来源平均，再按计划的四组预算加权；无效配对贡献零，不把剩下一个样本放大成整组 |
| 新模块预热 | 前 `min(200, ceil(总步数×0.1))` 步，只训练 R/Q；LF 渐入，KD 为零 |
| 联合微调 | 开放原融合、后端及最后 16 层 LoRA，保留所有已有 LoRA 的贡献；KD 在之后 200 步渐入 |
| 学习率 | R 为 1e-4，Q 为 5e-5；原模块从父运行末尾学习率继续，新模块不继承已衰减的旧学习率 |
| 随机性 | LoRA dropout 全程为零，避免把编码器自身随机差异当成处理差异；CE 的后端 dropout 保留，KD 后端使用确定性模式 |

默认不追加第二轮训练。每轮步数从原全来源计划计算，当前规模预计约 2229，但以启动日志为准。预热、坡度以已提交 update 计数；AMP 重试不消耗新来源或额外预热步。

## 配对有效性与准备

仍使用现有同源、同标签、共同增益、相同 RTC 设置、分别重置处理器状态的两路音频。Noise 分支可能还包含场景回声，所以此处学习的是接收端扰动，不声称唯一差别只是加性噪声。

新配对不依赖 CTC：已经纳入父模型训练的 Dev Offline 来源同样会生成 Clean-RTC/Noisy-RTC 并参与 LF；原官方 PCL 的数据范围不扩大。正负类都使用各自同源参照，Fake 不与其他 Real 强行对齐。

先用四组均衡的 512 个允许训练来源、每个 2 个处理版本，统计固定帧级尺度、对齐覆盖、分类翻转与通道漂移。延迟估计和局部相关性复用原有保守规则；短语音人工补齐、重复内容、强衰减区间与不可靠边界不参与 LF。两条音频的 CE 始终保留。没有有效 KD 时允许零损失；任何组完全没有有效帧配对会留下诊断报告并停止，避免静默进行无效辅助训练。

标定使用逐来源等权帧统计，尺度下限为 `max(1e-4, 0.1×通道标准差中位数)`，不复用旧音素均值尺度。这些标定数据属于允许训练数据；它们不是未见验证集。

## 日志与计算量

训练日志的 `LF`、`KD`、`PCL-p` 都是对总损失的有效贡献。`stage=repair_warmup` 时 KD=0 正常；联合阶段 KD 仍可能因可靠性门槛为零。

`final/pair_preflight.json` 记录尺度、通道漂移及初始配对结果；`final/repair_summary.json` 分组汇总有效率、修正前后特征误差和 KD 启用率；`final/hard_examples_summary.json` 汇总真实 CE 前向中的困难来源。详单在各自 JSONL 中。困难样本捕获不会把 KD 的额外分类前向误认成训练 CE。`final/training_profile.json` 在联合训练稳定后记录 50 步的中位/P90 用时、编码量与显存峰值，供与同预算基线比较。

训练步复用已编码的 Clean/Noisy，不增加第二个 Omni，也不重新编码配对参照。LF 会多运行一次小型 R，KD 会增加 A 侧配对的确定性后端前向；准备阶段单独编码固定小样本。新增开销的实际比例需服务器 GPU 实测，不能据 CPU 测试宣称训练更快。

最终模型继续使用父运行事先锁定的单模型参数平均规则；新 R/Q 也参与平均，固定尺度保持不变。`export_repair_audit.json` 记录平均前后修正参数范数。不存在多模型预测融合，也不重新用已训练 Dev 挑最佳权重。

## 单变量对照

默认 `--repair-arm paired`。另提供：

- `baseline`：原 V5.3 分类/PCL，统一 LoRA dropout=0，绕过 R/Q。
- `ce_only`：新 R/Q、移除模拟 PCL，保留 CE/官方平台 PCL，不启用 LF/KD。

三者从相同 V5.3 final 出发，配方、来源顺序、总 epoch 和导出规则一致；`ce_only` 与 `paired` 的预热和新模块学习率一致。它们是三个独立运行，默认不会自动连跑。A 侧辅助覆盖和额外计算不同，因此不能把“同 epoch”说成“完全相同 FLOPs”。父模型已经训练过 Dev，这些运行不能再声称是未见来源的 Dev 对照。

## 验证范围

33 项 CPU 测试通过，覆盖真实项目 AASIST 与小型 Omni 形状编码器：零初始化等价、padding/批次同伴不变性、配对对齐、梯度隔离、A/B 系数、原基线更新一致、Dev 无 CTC 配对、准备恢复、跨预热的完整断点恢复、AMP 失败回滚、父权重完整性、单模型 submission 导出与重试。5 个 Shell 入口通过语法检查。未在本地运行真实 3B Omni/GPU 全量训练，性能提升仍待实际提交验证。

```bash
python -m unittest discover -s w2v_v54 -t . -v
```
