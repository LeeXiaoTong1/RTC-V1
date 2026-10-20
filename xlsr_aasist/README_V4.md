# V4：w2v-BERT 2.0、MHASP、双路 PCL 与 Online/Noisy 梯度协调

本次修改完整采用已实现的 V3.20 检测结构。版本仍为 V4，checkpoint schema 更新为 `rtc_v4_bert_mhasp_dual_pcl_pcgrad_v2`。旧 Omni V4 实验不能在新架构下续训。

## 模型与初始化

16 kHz 完整录音 → 官方 SeamlessM4T 80 维 Mel 特征、相邻两帧堆叠为 160 维 → w2v-BERT 2.0 的 24 层 Conformer → 24 层融合（1024 维）→ 4-head MHASP（2048 维）→ MLP 2048→1536→1024→512→2。

- 全部 24 层 Q/K/V/O 使用 LoRA，rank 16、alpha 32，共 96 个投影。官方基座冻结。
- 各层无仿射 LayerNorm 后 softmax 加权；初始最后一层权重 0.5，其余层共享 0.5。
- MHASP 为四个 256 维子空间分别做加权均值/标准差后拼接，这是原 V3.20 的明确实现选择。
- 移除 Omni、AASIST、重叠特征窗口、MultiConvAdapter。V4 不再下载、准备或使用 XEUS，不再做近邻换样、真假配对间隔损失或教师校准。
- **从官方 w2v-BERT 预训练权重新建检测器**。`--data-run`（兼容别名 `--parent-run`）只复用来源登记、噪声资产与音素元数据，不读取旧 detector checkpoint。不能将 Omni 权重移植到 w2v-BERT。
- 推理只有一个模型、一个 P(fake) 输出，固定阈值 0.5，不需要音素识别器或参照音频。

这是采用 V3.20 中的 Teffic-Audio 风格结构，并非声称逐项复现原论文的所有未公开实现。

## 双路 PCL

共享同一个前端和融合表示；分类使用官方 Online 和当前模拟 Noisy。PCL 位于 1024 维融合输出、MHASP 之前，匹配音素区间的均值做 MSE，**两端都有梯度**。

1. **官方分支**：原始 Offline ↔ 同源原始官方 Online。每次逻辑更新每种语言×真假至多选一个已抽中的 Online 来源，最多四对。增强后的 Online 不冒充原始 Online；分类输入本来就是原始 Online 时复用编码。
2. **模拟分支**：原始 Offline ↔ 该来源本次已经生成的 Noisy＋RTC。每种语言×真假至多一对，最多四对。复用当前 Noisy 的波形和编码，仅增加原始 Offline 参照，不生成第二条噪声、不额外生成 clean RTC。

模拟对齐先使用已记录的变速时间映射，再估计常数传输延迟；检查局部信号关联和能量，并剔除活动门控抹掉的区间。w2v-BERT 使用 **560 samples 支持范围、320 步长、280 中心**，静音屏蔽边界按 280 samples 扩展。镜像补长和 batch padding 不参与音素损失。无法可靠对齐的区间只取消 PCL，分类样本仍完整保留。

没有官方 Online 的合法 Offline 仍能进入模拟分支。所有参照要求相同 canonical source、语言及真假标签，而且只能来自最终 Train 登记表。PCL 不使用检测分数筛选来源。

每个分支先按来源和实际有有效音素的语言×真假组平均，缺失组明确记录。每次更新最多新增 12 个参照编码，分类来源曝光预算不变。分类与 PCL 共用本次特征图；参照不进入分类器。

第 1 轮只预热融合/后端，冻结 LoRA，不读取 PCL 参照。第 2 轮开始开放全部 24 层 LoRA，使用六个固定 Train 任务分别标定官方/模拟 PCL：各自共享 LoRA 梯度范数目标为 CE 的 5%，λ 上限 1，随后 300 次更新逐渐增加至所标定权重。记录实际达到的比例；这不是对后续 Adam 更新比例的保证。校准不更新参数、恢复随机状态，并用 FP32 后端与显式梯度缩放避免此前 FP16 校准问题。

## 数据和优化

保持既有 V4 配方：按完整 PCM/官方配对来源，将 Dev EN-real 约 75% 迁入 Train、25% 留出，固定 split seed 4001，并在父 Select/Audit 内按配对可用性分别分层。迁入的 Offline 参加动态增强→单次 Noisy→RTC，Online 进入官方流。来源清单和保留验证固定，不使用错误分数挑选。

沿用现有噪声配方与两流曝光预算：通常每流 EN-fake/EN-real/ZH-fake/ZH-real 为 9668/8000/8000/8000，约每轮 1403 次逻辑更新；每次最多 24 Online＋24 Noisy 分类来源。8000 是来源访问次数，不是独立录音数。物理微批次默认 4、帧预算 9000，参考波形也计入预算；不扩大逻辑 batch。

两流八组均衡 CE 梯度累积完一个逻辑 batch 后做对称 PCGrad，再加入独立的两路 PCL 梯度，统一裁剪并提交一次 AdamW 更新。保留条件轮换，不加入 MLDG 临时更新。

默认训练 **10 轮（1 轮后端预热＋9 轮联合训练）**。LoRA 前/中/后各八层学习率 5e-6/1e-5/2e-5，新融合/MHASP/MLP 为 1e-4；LoRA 200 次 joint update 预热，联合阶段余弦衰减到 10%。基座 BF16，MHASP 统计与损失 FP32，MLP FP16，主参数和梯度 FP32。完整时序不裁剪，资源超限在启动前报告。

选模保持 V4 定义：面板内 EN/ZH 等权，再按 0.30 Online＋0.35 Regular＋0.35 Heldout 加权。只从**完成评估的训练轮次**选择 best，随机初始化与旧 Omni 不参与竞争。结束后仅评估所选 best 的锁定 Audit。历史 Dev 已参与开发，不是全新盲测；Progress 不用于选模或损失校准。

## 更新、环境和启动

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch v4-noisy-generalization
git pull --ff-only origin v4-noisy-generalization

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash setup_w2v_v4.sh
source /home/ubuntu/LXT/v318-runtime/envs/sdd-v4/bin/activate

bash run_w2v_v4.sh \
  --data-run "$(cat exp/.latest_v31813_run)" \
  --epochs 10 \
  --submit-best --upload-submission --upload-temp
```

`setup_w2v_v4.sh` 创建隔离 venv，复用现有 CUDA PyTorch/RTC 库，固定 Transformers 4.38.2，不升级旧 Omni 环境；随后在服务器运行真实小型 HF 模型的 CUDA/FP16 双 PCL＋PCGrad 验收。默认官方模型存于 `/home/ubuntu/LXT/v318-runtime/hf-w2v-bert-2.0`，可用 `--bert-model-dir` 指向已有完整模型。音素缓存仍复用旧版本已经核验的记录，仅为新增来源识别；音素识别环境独立于检测器。

网络需要镜像时可加 `--hf-endpoint https://hf-mirror.com`。`--local-files-only` 禁止下载；资产校验失败会报错，不能静默换模型。`--prepare-only` 只准备数据/官方资产/音素；`--preflight-only` 创建新 run 并检查官方完整模型加载，之后通过该 run 的 `--resume` 开始训练。

```bash
# 日志查看；退出查看不影响训练
bash watch_w2v_v4.sh

# 请求当前完整更新结束后保存并暂停
bash stop_w2v_v4.sh

# 等到 PAUSED_RUN 后恢复，配置不允许中途变更
bash run_w2v_v4.sh --resume "$(cat exp/.latest_v4_run)" --upload-temp
```

旧 Omni V4 run 的 `--resume` 会明确拒绝；新架构需要新 run。尚未完成任何轮次评估就暂停时，不允许把随机检测器导出为 best。

## 对照、观测与验收

- `--pcl none`：真正关闭两条 PCL 分支和音素准备；保留 CE/PCGrad。
- `--pcl official`：只保留原始 Offline↔Online。
- `--gradient-coordination none`：去掉 PCGrad，仍保留相同采样和所选 PCL。
- `--dev-real-train-fraction 0`：关闭数据迁移。改变该比例会改变验证集合，不能跨划分直接比较 SelectS。

终端显示 `PCL-o`、`PCL-n` 和各分支有效配对数。`steps.jsonl` 另外保存实际加权损失、权重/ramp、全可训练参数上的两路 PCL 梯度范数、两流 CE 梯度及协调情况、时间映射/延迟/静音屏蔽覆盖率。`auxiliary_calibration.json` 保存六个 Train 任务的梯度观测与实际比例。`coverage_epoch_*.json` 保存采样次数；`resource_audit` 保存完整长度及参照预算。

本地检查使用真实小型 HF Conformer，不下载完整基座、不启动服务器正式训练；测试涵盖 24 层 LoRA、MHASP、双端 PCL、变速/传输延迟/静音排除、原始与模拟编码复用、来源隔离、预算、精确续训和单模型导出。完整模型显存、吞吐及未知 Noisy 泛化收益仍需服务器实测；安装脚本的 CUDA 验收也不是完整基座峰值显存测试。

本次本地共 40 项 CPU 测试通过，Python 3.11 语法检查通过，新增安装脚本 Bash 语法检查通过。本地实际运行环境为 PyTorch 2.11.0+cpu、Transformers 5.8.0；服务器固定的 4.38.2/CUDA 组合由安装脚本做启动前验收，未声称已在本地完成该组合的 GPU 验证。
