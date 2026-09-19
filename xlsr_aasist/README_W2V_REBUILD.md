# w2v-BERT + V2：全编码器训练重构版（rebuild v1）

本版本是独立训练入口，不覆盖历史入口，不删除旧实验。源仓库审计基点：
`9d925407af846b2c544c8136433f49694bbcec00`。

## 改了什么，以及没有证明什么

之前的实现同时改变微调层数、类别权重、学习率、选模标准和优化器恢复行为，不能从单次掉点反推出唯一原因。此次重构不把某个参数称作“理论最优”。工程目标是：真实可测的全编码器学习、可控的训练/验证差异、统一的三阶段状态管理。

- 所有阶段训练特征投影和全部 24 层，不再使用 8/4/2 层限制。仅不参与前向的 SpecAugment placeholder 不放入优化器。
- 使用原始 `facebook/w2v-bert-2.0` 通用语音预训练权重。没有引入外部 Deepfake 检测器、教师权重或外部语音数据。
- 使用官方 `AutoFeatureExtractor`，在 CPU DataLoader worker 中进行波形到 160 维特征的转换，携带 attention mask 并去掉共同的尾部 padding。进入编码器后的图不 detach。
- 编码器保持原始结构；所有阶段统一关闭 LayerDrop/额外 SpecAugment，保留普通 dropout 的正常 train/eval 行为。没有在 forward 内偷偷调用 eval()。
- 使用 `use_reentrant=False` 的激活检查点。冻结输入不会因此丢掉模块参数的梯度，不需要把波形伪装成可学习参数。
- AASIST 保留频谱/时序图和双路图聚合拓扑，**不是原版逐参数兼容后端**：卷积 BatchNorm 换为 GroupNorm，图节点 BatchNorm 换为 LayerNorm；修正原残差块中预激活结果被 `conv1(x)` 丢弃的问题。没有新增第二个编码器、Adapter 或多层融合分支。
- 归一化改动针对批次统计依赖，不意味着已经证明历史所有退步都是 BN 造成的。
- 新架构参数名和状态格式不同，明确拒绝旧 XLS-R/w2v 的裸 state_dict。需要从官方预训练权重启动新的 Stage 1。

## 训练逻辑固定下来

| 阶段 | 实际每次 optimizer.step 的波形组成 | 编码器最高 LR | 后端最高 LR | 最多 epoch |
|---|---|---:|---:|---:|
| 1 | 40 普通样本 | 1e-6 | 1e-4 | 30 |
| 2 | 32 普通 + 4 真实 RTC 对×2 | 5e-7 | 1e-5 | 10 |
| 3 | 24 普通 + 4 真实 RTC 对×2 + 4 含噪 RTC 对×2 | 5e-7 | 1e-5 | 20 |

每阶段 LR 在第一个 epoch 内按 step 从 10% 平滑升到表中值，**不是冻结一部分 epoch 后突然解冻**。这些是可追踪的起始配置，不是经过 RTC 完整实验证明的最优参数。

AdamW 的参数和动量保留 FP32，计算使用 BF16（A100）；bias 和归一化缩放不做 weight decay，其余 weight decay=1e-4，梯度裁剪=1.0。验证的加权 balanced CE 连续 3 轮没有有效改善则 LR×0.5，最低为初始 LR 的 5%，冷却 1 轮。

**不再自动回滚模型并清空 Adam**。`best_model.pt` 用于选模/阶段迁移；`last.pt` 保存模型、AdamW、调度器、随机状态、V2 轮换状态，用于同阶段恢复。LR 降低不会抹除已学习的优化器状态。

普通分支采用从 Train 实际条目数计算的全逆频率 CE 权重 `N/(2*N_c)`，组内使用 normalized weighted mean；两个已做真假均衡采样的配对分支使用等权 CE，不重复纠偏。没有平方根权重这个额外经验参数。

Stage 2：分类 CE 按 ordinary/paired 波形数加权；真实配对 InfoNCE 系数前两轮从 0 升至 0.1。

Stage 3：原 V2 分组 CE 不变，含噪处理组 CE 系数 0.3，其他三组共同 0.7，内部按 24/8/4 计数分配。真实 RTC 系数保留 0.1；新引入的含噪配对系数前两轮升至 0.1。

RawBoost 参数、MUSAN 概率 0.5 / SNR 10–30 dB、已验证的真实配对、V2 四视图轮换与 21/6 通信组合划分均复用原数据模块。完整逻辑 batch 始终为 40。`MICROBATCH=4` 只把前向计算分块，所有输出仍合在一起计算一次损失、做一次参数更新，**不会把 4 对的对比负样本拆开**。关闭 dropout 的回归测试验证了分块前向的梯度等价性；有 dropout 时不承诺改变分块尺寸仍逐位相同。

## 固定验证、可观察的训练

三个阶段全部验证相同的 clean Online、noisy seen、noisy heldout Dev。

- 选模：`0.3 Online Macro-F1 + 0.35 Seen Macro-F1 + 0.35 Heldout Macro-F1`；同分看 balanced CE。
- 调 LR：对应条件的相同系数 balanced CE，不依赖单次离散 F1 波动。
- 分数阈值固定 0.5，不调 Progress/Eval 阈值。
- 每轮保存真假召回率、混淆矩阵、预测 fake 比例、按真值分类的平均 fake 分数、各 SNR 档指标。
- 每轮第一个 optimizer step 保存特征投影、每个编码器层及后端的真实梯度和实际 FP32 参数增量。发现梯度丢失/非有限/没有更新就报错，不继续烧 GPU。
- 保留完整数据与代码哈希。相同阶段按 epoch 边界恢复；不支持从中断 batch 中间接着跑。最后一个未完成 epoch 会重做。不同硬件/版本不承诺逐位复现。

Noisy Dev 是利用官方 Dev 构造的本地代理，并非官方提供的含噪 Dev。验证缓存只用于无梯度验证。训练/选择不读取 Progress/Eval。

## 文件与环境

新增 `w2v_rebuild/` 包，以及两个新 shell 入口。旧 `main_train*.py`、旧 `model/`、`dataset/`、`exp/` 不修改。

服务器沿用当前环境：Python >=3.10、已有 PyTorch 2.5.1、`transformers==4.38.2`、已有 V2 音频依赖。不要求安装 fairseq，不需要升级 CUDA。

新版本每阶段需要约一份模型最佳权重和一份完整训练状态的磁盘空间；600M 级 FP32 模型和 AdamW 状态合计约 9 GB/阶段量级，三阶段之外还要保留缓存及原有文件。保存时临时文件会额外占空间。显存以 preflight 实测为准。

## 第一次运行

更新只复制新目录和新入口到训练目录后，在项目根目录执行：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC/xlsr_aasist
python -m w2v_rebuild.tests
RUN_DIR="./exp/w2v_rebuild_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN_DIR"
# 使用已有 V2 缓存，先验证标签、配对、缓存角色、噪声隔离。
bash run_w2v_rebuild.sh 1 "$RUN_DIR" --check_data
# 真正加载官方大模型，对一组真实 Train 输入执行前向/反向/参数更新与重复评估。
bash run_w2v_rebuild.sh 1 "$RUN_DIR" --preflight
# preflight 不保存训练权重；正式 Stage 1 重新从官方预训练开始。
bash run_w2v_rebuild.sh all "$RUN_DIR"
```

训练器的每一轮本身也有梯度/实际更新检查。preflight 不能证明泛化性能，只验证真实执行链；不要把 PASS 当作“已经超过 88.7144”。

`all` 会依次训练 Stage1→Stage2→Stage3，后一阶段直接读取**同一 RUN_DIR** 内前一阶段的 `best_model.pt`。不再用 `ls -t` 猜测权重。

也可按顺序单独运行：

```bash
bash run_w2v_rebuild.sh 1 "$RUN_DIR"
bash run_w2v_rebuild.sh 2 "$RUN_DIR"
bash run_w2v_rebuild.sh 3 "$RUN_DIR"
```

同阶段恢复示例（不要同时传 --init）：

```bash
bash run_w2v_rebuild.sh 1 "$RUN_DIR" --resume "$RUN_DIR/stage1/last.pt"
```

修改代码、训练数据或主要优化器配置后不能冒充精确恢复；脚本会拒绝。OOM 时应先重新用更小 microbatch 做 preflight；不要在同一个恢复状态中悄悄换 microbatch。

## 提交

```bash
bash run_eval_w2v_rebuild.sh "$RUN_DIR/stage3/best_model.pt" "$RUN_DIR/submission" progress
unzip -l "$RUN_DIR/submission/submission.zip"
```

ZIP 根目录仅 `scores.txt`，每行原协议 ID + fake 概率。评估只推理，不产生伪标签、阈值选择或训练梯度。

## 本次实际验证范围

本地 PyTorch 2.10.0 CPU / Python 3.13：26 项测试中 **25 项通过，1 项跳过**。测试覆盖梯度检查点问题重现、非重入梯度、24 层 test-double 实际更新、AASIST 所有权重反传、分块梯度等价、归一化批次独立性、V2 损失公式、状态保留、完整优化器/RNG恢复、三阶段端到端调用衔接和 ZIP 格式。

**不包含官方 580M 权重、真实 RTC 数据、A100/BF16 或比赛效果验证。** 本地无法联网安装 transformers，因此真实 HF 小模型测试标记跳过，没有伪装通过；在你的现有服务器环境中同一测试会执行。真实大模型/真实 batch 的检查由上述 preflight 执行。

## 原始依据

- PyTorch 2.5 checkpoint 文档：明确区分 reentrant/non-reentrant；后者不要求输入 requires_grad=True。
  https://docs.pytorch.org/docs/2.5/checkpoint.html
- Meta/Hugging Face 原始模型配置及 API：
  https://huggingface.co/facebook/w2v-bert-2.0/blob/main/config.json
  https://huggingface.co/docs/transformers/v4.38.2/model_doc/wav2vec2-bert
- Group Normalization, Wu & He, ECCV 2018：归一化不依赖 batch size 的设计依据，不是 RTC 效果证明。
  https://arxiv.org/abs/1803.08494
- AASIST, Jung et al., ICASSP 2022；拓扑依据用户仓库 `model/model.py`。
- 官方竞赛规则：
  https://www.junxue.tech/rtc-sdd-challenge/
