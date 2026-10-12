# V3.18.3-R2：全层融合与更积极的适应

原 V3.18.3 完成六轮后，用户回报 Dev Weighted 约 87%。R2 将“最后层加有限残差”改为完整的全层投影、SwiGLU 门控及层权重融合，并提高联合训练学习率。它是一次新实验，不能将代码完成解释为已经超过原模型。旧 `w2v_v3183` 代码、运行目录、checkpoint 和导出入口均保留。

## 模型与初始化

- 前端仍为 Omni-W2V-3B，后端仍为窗口 AASIST 和局部贡献汇聚，窗口 128 帧、步长 64 帧。
- 读取全部 60 个编码器层：第 1–59 层通过同一次前向的钩子读取，第 60 层使用编码器最终输出。不把前端卷积输出计作第 61 层。
- 各层特征经归一化、共享 `2048 -> 128` 投影、SwiGLU 门控后，用 60 个可学习 softmax 权重融合。初始权重均为 `1/60`，不设最后层旁路、残差门控或 25% 上限。
- SwiGLU 初始化为恒等变换，同时保留参数梯度；层权重从第一个训练步骤参与学习。融合输出直接进入 AASIST，旧 AASIST 输入投影移至共享融合投影位置。
- 默认 `--init-mode parent-transfer`：从**已完成的 V3.18.3 运行的 best** 继承 LoRA 和兼容的 AASIST、局部贡献参数；旧 `head.aasist.LL` 权重迁移到共享投影；丢弃旧四参数残差融合。不是从最后一轮盲目取权重，也不继承旧优化器。
- 默认父运行从 `exp/.latest_v3183_run` 读取；可以用 `--parent-run` 指定已完成的 V3.18.3 运行。新运行使用 `exp/.latest_v3183r2_run`，不会覆盖旧指针。
- 新特征分布改变，BN 运行均值、方差和计数在新运行初始化时重置一次；之后预热和联合阶段均以 momentum=0.01 更新。恢复 R2 训练时保留已保存的 BN 状态，不再次重置。

这借鉴了旧有效方案的全层投影、门控和层融合，但**不是整个 V3.17 的复现**：前端、AASIST、C3 和窗口汇聚仍与 V3.17 不同。转移兼容参数也不意味着初始 Dev 仍为 87%；需要重新适应全层输入。

`--init-mode pretrained` 是独立对照：只使用官方预训练前端，不转移父运行的已训练 LoRA/检测器权重，仍继承同一父运行固定的数据与验证划分。两种初始化均新建 R2 运行目录。不要把旧运行路径传给 `--resume`。

## 训练预算与参数协调

默认总计 **6 轮：1 轮 head/fusion 预热，随后 5 轮联合训练**。

| 模块 | 最高学习率 |
|---|---:|
| LoRA（联合阶段） | 2e-5 |
| AASIST | 1e-4 |
| 窗口贡献模块 | 1e-4 |
| 投影、SwiGLU 等融合参数 | 3e-4 |
| 全层混合 logits | 1e-3 |

预热阶段冻结 LoRA 参数更新，但父模型转移的 LoRA 增量仍参与前向；不能为了“冻结”而把它关掉。预热正常训练新融合和检测后端。联合阶段再开启 LoRA 更新，使用新 AdamW 状态、分阶段学习率预热与余弦衰减，最低为各组峰值的 10%。LoRA 的位置仍为最后 16 层 Q/K/V/O；较早层提供冻结特征，融合与后端可以学习如何使用它们。

报告记录各模块学习率与梯度统计；首步及每 100 步还检查 AdamW 实际参数更新量、相对更新量，预热中 LoRA 的更新量应为零。审计只在指定步骤复制小规模可训练参数，不增加编码器前向。每轮 `[Fusion]` 输出层权重熵、有效层数和最高权重层，完整权重保存在运行历史与 `diagnostics/fusion.csv`。有效层数为层权重熵的指数，均匀使用 60 层时等于 60，并不直接等于“有用层数”。每轮另外记录包含训练与验证的 CUDA 峰值显存。

## 保持的数据与目标

- 每次逻辑更新包含独立抽取的 16 条官方 Online，以及 16 个原始来源各生成的 Noisy A/B，共 48 条波形。两流可能抽到同一来源；原始 Offline 不直接送入检测器。
- 两流的 EN-fake / EN-real / ZH-fake / ZH-real 配额均为 `4/2/7/3`。每条流损失占 50%，每组监督预算仍占 25%。原有循环覆盖、同批去重和曝光审计保留。
- Online 使用 CE；Noisy 使用 `0.75 * mean(CE_A, CE_B) + 0.25 * max(CE_A, CE_B)`，label smoothing=0.02。保留 C3，不同时叠加 TFCL、InfoNCE 或 PCL。
- 沿用 V3.18.3 的受控 RTC 增强：Noisy A 为原配方，Noisy B 以受控概率额外采用变化增益、带宽限制、软限幅或扩展 Opus 码率。首轮额外概率 17.5%，之后 35%。它不包含完整 AEC 或网络丢包/PLC 模拟。
- 为配合新融合的预热，首轮沿用原增强器的预热 SNR 范围 10–25 dB，联合阶段为 5–25 dB。旧 V3.18.3 没有头部预热，从首轮即使用 5–25 dB；因此这里并非逐条完全相同的训练增强。固定 Dev 和探针不变。
- 固定 Dev、Train probe、独立机制 panel 及来源隔离的 selection/calibration 划分继承父运行。Progress/Eval 不参与训练、选模或校准。

## 启动检查、选模与效率

新运行检查父配置、checkpoint 身份、best 标签与固定验证预测。父分数是历史参照，R2 不再运行“新架构必须逐条复现旧分数”的检查。初始化报告会明确记录它不是精确复现。启动还执行小型真实 fairseq2 接口检查；失败则在训练更新前停止。

全层特征通过单次编码前向读取并逐层投影汇聚，不额外跑第二遍整条编码器。训练时投影重计算会增加少量算术，且需要保存更多中间层输入用于融合梯度；**不能承诺与旧版相同显存或耗时**。实际 CUDA 峰值和每步耗时以服务器日志为准。无需新增依赖、重新下载官方前端或重装已工作的环境。

best 只在本次 R2 已训练轮次中按原始 selection Weighted 选择，不会悄悄回退旧 V3.18.3。父模型只用于差值参照。完成后仍只在保留的 calibration 来源拟合全局校准，在 selection 上决定 identity/affine；完整 Dev 包含该校准子集，应和 raw 指标及分区指标一起看。

## 服务器命令

以下代码块全部为 ASCII 字符。先确认数据盘挂载；旧训练已完成后再更新并新建 R2 运行。

```bash
(
set -e
mountpoint -q /home/ubuntu/LXT
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
V3183_PARENT="$(cat exp/.latest_v3183_run)"
bash run_w2v_v3183r2.sh \
  --parent-run "$V3183_PARENT" \
  --init-mode parent-transfer \
  --epochs 6 \
  --submit-best --upload-submission \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
)
```

该命令训练后自动导出**本次 R2 best** 的 Progress submission，再上传 `submission.zip` 至 temp.sh。需要 Eval 时，启动前替换最后两项为对应的无标签 Eval 清单和音频目录。`--submit-best` 单独使用只导出；两项均省略则只训练。启动前核对 ID、音频存在与清单哈希，恢复时不能静默更换评测集合。

启动器在后台运行，`Ctrl+C` 仅退出查看器。重新查看：

```bash
bash watch_w2v_v3183r2.sh
bash show_w2v_v3183r2.sh
```

仅检查初始化并保存启动状态，可以使用 `--preflight-only --no-watch`。通过后用下面的 R2 恢复命令继续。无需额外执行一次完整 Dev 父模型推理。

```bash
bash run_w2v_v3183r2.sh --resume "$(cat exp/.latest_v3183r2_run)"
```

恢复总共完成原计划的 6 轮，不是重新追加 6 轮；从最后已提交的完整 epoch 回放，未提交的一轮会重跑。恢复保留模型、BN、优化器、随机状态和提交请求。不要在恢复同一次 R2 运行前修改代码或学习率；若需更改实验条件，应新建运行。

最终终端打印 `SUBMISSION_ZIP`、`TEMP_DOWNLOAD_URL` 和 `TEMP_LINK_FILE`。查看保存的链接：

```bash
V3183R2_RUN="$(cat exp/.latest_v3183r2_run)"
V3183R2_OUT="/home/ubuntu/LXT/temp/$(basename "$V3183R2_RUN")_submission_best"
cat "$V3183R2_OUT/temp_download_url.txt"
```

仅上传失败时 ZIP 会保留，退出码非零。无需重新推理，直接重试：

```bash
bash upload_w2v_v3183r2.sh --out "$V3183R2_OUT"
```

已有成功链接时默认复用；链接过期需要重新上传时加 `--force`。只有 `scores.txt` 被打入提交 ZIP，上传不会发送 checkpoint、音频或诊断报告。训练尚未完成前没有实际 submission 下载链接。

独立验证/导出入口仍可用：

```bash
bash run_validate_w2v_v3183r2.sh --checkpoint best --raw
bash run_eval_w2v_v3183r2.sh \
  --run "$(cat exp/.latest_v3183r2_run)" --checkpoint best \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

提交元数据记录新架构、父初始化来源、best 标签、校准模式、清单顺序与 ZIP 哈希。模型 checkpoint 中 best 可能不是最后一轮；导出显式读取 `best_model`，不会误用 `model` 最后状态。导出先写暂存目录，验证成功才成为最终目录；网络上传失败可复用已验证的 ZIP。

## 验证边界

`bash test_w2v_v3183r2.sh` 使用小编码器、真实 AASIST 与合成音频，检查全层融合梯度、权重迁移、采样和目标、BN、断点恢复、best/last 区分、校准与导出。上传网络在测试中模拟，不向 temp.sh 发送测试提交包。真实 3B CUDA 开销、六轮收敛和泛化提升需要服务器实验确认。

发布前完整 CPU 套件通过 32 项测试；新 Bash 脚本全部通过语法检查。检查点流式融合与独立完整计算的输出和梯度一致，断点恢复与连续训练权重一致，已覆盖上传失败后的重试。旧版 130 个受保护依赖与训练文件保持原 Git 内容。

R2 同时改变融合方式、BN 统计策略与学习率。一次整体实验可以判断新组合是否更好，但不能单独归因；应保留本次完整报告，并用冻结相同数据的消融决定后续改进，而不是根据 Progress 反复调参。
