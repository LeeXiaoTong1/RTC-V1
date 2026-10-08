# 归档：OmniASR 路线（当前 V3.16 主入口已用于 TFCL 重设计）

# V3.16：OmniASR W2V 7B + LoRA，配合改进后的 TFCL 执行

这一版落实两件事：优化 V3.15 的配对约束与执行开销，换用 OmniASR 的 7B 自监督音频编码器并加入 LoRA。数据策略仍沿用 V3.15，尚未改为只用 Online，也没有加入音素识别或语言对抗。

这是一个**新的检测器训练起点**：OmniASR 的预训练特征与 w2v-BERT 的特征坐标不同，不能将旧检测器的成绩当成新模型的初始成绩。V3.15 的 best、基础依赖和已提交分数全部保留；新模型能否超过它，要看真实训练和提交。

## 前端和微调位置

- 使用官方 `omniASR_W2V_7B` **SSL 版本**，约 64.9 亿参数、128 层、2048 维。没有加载 LLM 解码器、CTC 识别头或 tokenizer。
- 从原始 16 kHz 完整波形直接提取特征；不再将 w2v-BERT 的滤波器组输入送给新模型。每条完整音频独立归一化，CNN/位置前端按真实长度计算，之后才对编码器输入补零并提供长度掩码。
- 官方自监督模型的训练 `forward` 包含随机遮挡和量化目标。本实现直接调用 `encoder_frontend` 和 `encoder`，不会误用那个预训练目标。
- 冻结基础权重；默认只在最后 16 层的 Q、K、V、输出投影上加 LoRA：`rank=16, alpha=32`，共 4,194,304 个 LoRA 参数。其余 112 层只前向，不保存反向图。
- 从第 16、32、64、96、112、120、124、128 层取得表示（文档从 1 开始编号）。使用归一化、可学习层权重、2048→128 投影和门控融合，继续使用 MultiConv 的检测结构。
- 新投影和检测头重新初始化，默认先用 200 次更新适配检测头，再开启 LoRA 和 TFCL；这 200 次计入原有训练预算，不额外增加一轮。极小数据集自动缩短预热。
- LoRA 学习率 `5e-5`、检测头 `3e-4`、TFCL 分支 `1e-4`，有学习率预热和余弦衰减。默认最多 4 轮。新头尚未适配时不会因为“低于旧模型两个点”立刻停止；至少跑完两轮后，才按连续四次验证无进展早停。

这些超参数是本项目的工程起点，并非已有本竞赛实验确认的最优值。7B 的规模不保证真假检测一定更好。

## TFCL 和效率改动

普通视图、通信参照、含噪通信仍为每来源三视图，逻辑更新仍为 16 来源 / 48 视图。
EN-fake、EN-real、ZH-fake、ZH-real 的分类权重各 25%。普通视图继续按约 1:2 使用 Offline / 已验证同源 Online；主阶段普通/参照/含噪分类损失占比仍为 30/10/60，前四分之一轮从 40/10/50 过渡。

时间对齐仍使用共享的双向交叉注意力，两侧都参与反传；结构项仍按每个来源分别计算通道 CKA，只有结构支路池化到 201 格。

1. **辅助比较先做逐帧幅度归一化。**让约束更关注方向和关系，减轻更换前端后的尺度差异；分类支路使用自己的原始融合表示。
2. **多对样本一起计算双向注意力。**保留每对的掩码与独立 CKA，不把不同录音混成一个结构目标。使用不返回注意力矩阵的执行方式，减少额外分配。
3. **按新前端的实际时间几何排除缺失帧。**检查 320 采样点步长和 400 点感受野，屏蔽已知短时缺失及其边缘，不再套用旧滤波器组帧数。补零不参与对齐或分类池化。
4. **保留全部有效帧。**不通过裁成四秒、截掉长录音或提前缓存固定特征来省计算。
5. **同一训练 DataLoader 跨半轮验证复用。**工作进程、噪声读取缓存不重复启动；进程间仍传 NumPy，避免 Torch storage 文件描述符问题。
6. **默认不落盘保存增强音频或特征。**同源增强现场生成，避免缓存目录全量扫描、文件锁和逐条写盘。噪声 RAM 缓存上限每工作进程 64 MiB；默认四个进程。
7. **长度分组与可回滚测速。**物理批次与逻辑批次分开；参照/含噪端不拆开。用中位、长、最长 Train 来源测试微批次和是否重计算，预留 4 GiB 显存，恢复参数、Adam 状态及随机状态后正式训练。测速没有保留训练收益。
8. **验证只做一遍固定 Dev 主指标。**不重复跑 Offline/额外模拟面板；历史 Online、Seen、Heldout 及其保护条件保留。额外面板不再属于本版 `best_guarded` 条件。

损失仍是 `CE + ramp × (0.15 × 时间项 + 0.045 × 结构项)`，TFCL 在新头预热结束后逐步开启。日志同时显示 `TOTAL`、`CE`、原始/加权时间损失、原始/加权 CKA、单样本最大 CE、有效配对数、计算/等数据时间、实际分配/预留显存。因此 `CE=0.0000x` 不再容易被误读成所有约束都没有作用。

这些优化消除的是冗余开销。**7B 有 128 层，完整前向仍比当前前端贵；LoRA 节省的是可训练参数、优化器和保存空间，不会省掉基础模型前向。不能承诺比 V3.15 更快。**启动测速和前百步日志提供真实吞吐。

## 指标与导出

固定阈值 `P(fake) >= 0.5`。每次验证输出各语言/条件的 AP、AUC、EER、fake/real Recall、macro F1，以及 real@99%fake。AP 以 fake 为正类，EER 使用 ROC 插值；这些诊断不会改变提交阈值。JSON 同时保存分类型 F1、混淆矩阵和来源分组。

历史口径保持为：`Weighted = 0.3 × Online Clean + 0.7 × mean(Seen,Heldout)`。这是本地 Dev 指标，不是排行榜分数。

- `best_weighted`：本次已训练 Omni 检测器里 Weighted 最高者。
- `best_guarded`：比 V3.15 参照更好且通过 Clean/Noisy、真假 Recall、AUC、matched recall 保护的 Omni 候选；可能不存在。
- `last`：最后成功验证并保存的 Omni 状态。

默认导出 `best_guarded`。没有通过保护的新模型就明确报出原因，继续保留原 V3.15；不会静默导出旧模型并称作 Omni 的成绩。要提交新模型进行对比，可显式选择 `best_weighted` 或 `last`。

## 磁盘与环境

官方权重约 25 GiB，只下载一份；使用 BF16 加载，**不会额外生成一份转换后的完整权重**。每次只保存 LoRA、新检测头、TFCL、Adam 和最多两个不同的精选状态。单个 `last.pt` 原子更新，没有每轮堆积整模文件。

新训练运行的检查点/报告原子峰值硬上限 4 GiB，默认配置通常明显低于该上限；另外保留 10 GiB 空闲。`storage_budget.json` 按真实参数量计算预算。还应为独立 Python/PyTorch/CUDA 环境预留数 GB。你之前反馈的约 66 GB 空间应有实施余地，但脚本以服务器实时容量为准。

单独使用 `sdd-omni` 环境，不升级现有 `sdd`。固定 PyTorch 2.8.0/CUDA 12.6、fairseq2 0.6、Omnilingual ASR 0.2.0。fairseq2 原生扩展必须与 PyTorch ABI 匹配。安装脚本先进行真实小型 fairseq2 API/变长/梯度预检；完整 7B 预检另行执行。系统需有支持 CUDA 12.6 的驱动、libsndfile 和可编译 WebRTC 包的工具；已有 FFmpeg 二进制沿用 V3.15 记录的那个版本。

## 清理规则

`cleanup_w2v_v316_omni.sh` 只处理可确认的旧衍生产物：

- V3.15 之前、具有独立 best 且没有被配置或选择记录引用的 `last/epoch/step` 等中间 checkpoint。
- 所有权与完成哈希均吻合的旧 `x.npy/logits.npy`，保留 owner、来源清单和元数据。
- `data/rtc_v35/train` 中所有权明确且没有被来源清单引用的旧训练音频代次。

保留 V3.15 及以后所有文件、各版 best、预训练权重、原始音频、固定 Dev。**V3.12 last 仍可能是 V3.15 best 的基础依赖；这样的 last 必须保留。best 若嵌在 last 内也不能删除。**未知格式不删。旧特征数组删除后，依赖它们继续训练的旧版本需重新生成；当前 V3.15/3.16 的元数据读取和已有 best 权重不受影响。

清理先生成精确清单，应用前再核对元数据和文件状态；删除后复核受保护权重哈希。训练/评估/缓存写入尚在运行时拒绝清理，不终止正在训练的 V3.15。不要用 `rm exp/*/last.pt` 替代此脚本。

## 部署

在当前 V3.15 工作结束后执行。以下脚本不自动删除原始数据或 best。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist

# 显示并执行已经授权的旧文件清理，输出实际释放空间与清单。
bash cleanup_w2v_v316_omni.sh --apply

conda create -n sdd-omni python=3.11 -y
conda activate sdd-omni
bash setup_w2v_v316_omni.sh

# 默认走 Meta 官方下载地址；中断后仍使用同一个 .part 文件续传。
python -m w2v_v316.prepare
python -m w2v_v316.preflight --omni-assets pretrained/omniASR-W2V-7B/assets.json

# 自动读取最近 V3.15 的已提交分数与数据清单；最多四轮。
bash run_w2v_v316_omni.sh --upload-temp
```

如已有官方原始权重：`python -m w2v_v316.prepare --checkpoint /实际路径/omniASR-W2V-7B.pt`，会核对官方 SHA256，不复制大文件。Meta 网络不可用时可以用 `--provider hf`，但应先处理已有未完成的 `.part`，避免两条下载路线各占一份空间。下载不需要 tokenizer 或语言识别模型。

指定来源或物理批次、恢复、查看、导出：

```bash
bash run_w2v_v316_omni.sh --source-run exp/你的V3.15运行目录 --upload-temp
bash watch_w2v_v316_omni.sh
bash show_w2v_v316_omni.sh
bash run_w2v_v316_omni.sh --resume exp/你的V3.16运行目录 --upload-temp
bash run_eval_w2v_v316_omni.sh --checkpoint best_weighted --upload-temp
```

手动小批次新运行：`--microbatch 2 --frame-budget 1200 --no-autotune`。这不改变每次更新的 16 个来源及其分类比例；参照/含噪配对至少两条一起执行，极长录音仍可能超出显存，不会静默截断。

Ctrl+C 只关查看器。需要真正停止本版本时，使用 `python -m w2v_v316.stop --apply`。从 last 恢复会重放最后一次验证后尚未保存的步骤，保留此前数值执行设置，不重跑测速改变路径。

## 验证边界与资料

CPU 测试覆盖 LoRA 零初始化、冻结权重不变、两侧 TFCL 梯度、掩码、微批一致性、恢复、测速回滚、常驻工作进程、指标方向/并列值及安全清理。完整 fairseq2 原生运行、7B 文件加载、A100 显存/吞吐和官方成绩需由服务器预检及训练验证；本地测试不等于已经测得竞赛收益。

- [官方 SSL 7B 模型及容量](https://huggingface.co/facebook/omniASR-W2V-7B)
- [官方文件 SHA256](https://huggingface.co/facebook/omniASR-W2V-7B/blob/main/omniASR-W2V-7B.pt)
- [Omnilingual ASR 官方代码](https://github.com/facebookresearch/omnilingual-asr)
- [fairseq2 0.6 的安装与 ABI 配套要求](https://github.com/facebookresearch/fairseq2/tree/v0.6.0#installation)
- [TFCL 作者实现](https://github.com/JunXue-tech/TFCL)：本项目继续使用适配后的约束，未宣称逐字复现。
