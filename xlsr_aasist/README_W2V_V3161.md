# V3.16.1：从 V3.16 LAST 继续四轮

这是 V3.16 的续训更新。它读取 **V3.16 的 `last` 检测器权重和 Adam 状态**，不重新初始化检测器，不以 V3.15 的 `best_guarded` 开始训练。源实验目录保持原样；旧代码指纹与当前执行指纹分别记录。

## 训练

服务器当前环境沿用 `(sdd)`；无需下载新的大模型或生成音频缓存。

推荐沿用原 Git 分支更新，无需上传压缩包：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
```

若选择代码包方式，将 `w2v_v3161_code.zip` 上传到 `/home/ubuntu/LXT/temp/` 后执行：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
python -m zipfile -e /home/ubuntu/LXT/temp/w2v_v3161_code.zip .
```

此次修复只更新 V3.16.1 的入口、新模块和说明，不改写旧实验目录。完成其中一种更新方式后训练：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
bash run_w2v_v3161.sh \
  --source-run exp/w2v_v316_tfcl_20261009_020301_4d95 \
  --epochs 4 \
  --upload-temp
```

`--epochs 4` 表示**追加四个完整轮次**。若源 LAST 来自第 4 轮，新日志显示第 5–8 轮。此次不因分组守门或短期停滞提前结束；非有限值、磁盘不足等运行错误仍会停止。完整轮次结束后提交一次 LAST；中途断电会从上一次完整轮次恢复并重放未提交部分。

若源 LAST 在某轮中间停止，会从下一条未使用的来源批次继续，追加 `4 × 每轮步数` 次更新；每增加一轮更新预算做一次 Dev。不会跳过原来剩余的半轮，也不会重新播放已经提交的批次。日志中的原始 epoch/step 与“追加第几轮”分别记录。

```bash
bash watch_w2v_v3161.sh
bash show_w2v_v3161.sh
# 中断后恢复本次续训：
bash run_w2v_v3161.sh --resume "$(cat exp/.latest_v3161_run)" --upload-temp
```

原始 V3.16 不能先做 inference-only 压缩：续训需要 LAST 的 Adam 和随机状态。程序会验证版本、文件哈希、训练状态和数据来源；如果状态缺失，会明确报错而不会偷偷重新训练。

## Git 更新后的旧指纹兼容修复

旧提交 `22e4252` 训练时记录的 `w2v_v316_tfcl/config.py`，在 `672953d` 的启动参数修复中发生了变化。早期 V3.16.1 将这次参数解析重构误判为不可兼容的训练代码变动，导致在加载模型前退出。本次仅允许这组已核对的精确 SHA256 转换，LF/CRLF 两种已知编码分别列入；保留原始配置和 checkpoint 身份，不重写旧哈希、不关闭校验。

迁移记录保存在新配置及 `source_code_migrations.json`。模型、数据、预训练权重、优化器来源和环境版本仍按原校验执行；未知代码变化会列出路径和新旧指纹并停止。历史文件的实际差异经过 AST 对比：除移走 parser 和替换启动参数检查外，保留函数体一致。

## 本次模型和损失变化

1. **约束回到完整 SSL 输出。** w2v-BERT 2.0 最后一层的 1024 维逐帧输出直接进入 TFCL。保持原有 MultiConv 分类结构和已训练参数；训练最后八层编码器、MultiConv、适配器和分类器。没有在一致性分支前额外压成 128 维。
2. **恢复双向软对齐。** 共享八头交叉注意力分别执行 Offline 查询 Online/Noisy，以及处理版本查询 Offline，再计算双向余弦损失。两端参与反向传播，不把 Offline 当成固定老师，不按预测置信度筛掉困难样本。删除旧版硬匹配、逐帧梯度重要性和局部容差机制。
3. **恢复完整轨迹结构比较。** 两端有效 SSL 轨迹分别池化为 201 个时间格，经过共享可训练线性投影，以通道为统计单元计算 `1-CKA`。不再只比较通过硬匹配的小窗口。只有结构分支池化；分类和时间注意力使用完整有效内容。
4. **不改变监督数据预算。** 每次更新 16 个均衡来源，四个语言×真假组各占 25% CE 质量。每来源 Offline/官方 Online/模拟 Noisy 的 CE 质量为 10/50/40；缺少已验证 Online 时为 20/80。两个辅助关系各占每来源辅助预算的一半。所有视图仍独立接受真假标签监督。
5. **重新安排四轮学习率。** 编码器最高 `5e-6`，头部/适配器/分类器 `2.5e-5`，新辅助模块 `1e-4`；恢复检测器 Adam 动量，辅助模块使用新动量。新日程短预热后衰减，不沿用旧四轮末尾的低学习率。前四分之一轮渐入 TFCL；不再进行随机检测头预热。

总损失以 CE 总质量为 1：`CE + ramp × (0.15 × time + 0.15 × structure)`。结构权重从旧版的 0.045 提高到 0.15；时间项恢复全量双向软匹配，所以相同数字不代表相同有效梯度。逐轮记录原始损失、加权损失、CE/TFCL 在 SSL 特征上的梯度，以及各语言/类别/关系整轮实际参与数量。

参考 [TFCL 论文](https://arxiv.org/html/2607.17761v2) 和 [作者公开模型实现](https://github.com/JunXue-tech/TFCL/blob/main/code/model.py)。这是保留主要机制后的任务适配，不能称为逐行复现：

- 前端/后端保持本项目 w2v-BERT + MultiConv，而非原文 XLS-R + AASIST。
- 原作者代码要求固定 201 帧；本项目不裁掉长录音，只让结构分支适配到 201 格。
- 已知填充/局部缺失不参与比较。真实 Online 的未知局部变化由软注意力学习，不再根据人工硬匹配筛掉。
- 原作者代码将 batch 和时间展开后计算一次通道 CKA。本项目对每个来源计算完整轨迹 CKA 后按固定来源预算求和，防止自动调节微批大小改变损失定义。其单来源公式与公开代码一致，有数值和梯度测试。
- 论文使用相同权重的时间/结构项；公开代码另外有 `lambda_d=0.3` 的相对结构系数。本配置选择前者，以本项目 CE 归一化约定设定 0.15/0.15，不声称与仓库所有训练参数完全相同。

## 模型选择与 submission

`best`（别名 `best_weighted`）只比较：源 V3.16 保存的、真正训练过的最佳/最后模型，以及本次四轮候选。按完整固定 Dev 的 Weighted 最高者选择；相同 Weighted 时先比较 Noisy。不再要求众多分组守门同时通过，也不混入本地辅助面板 F1 排名。**绝不回退到 V3.15 `starting_parent`。**

`last` 始终是本次续训最后一个完成验证的模型。放宽选择不保证分数提高；如果续训均退步，`best` 可以是源 V3.16 中保存的训练模型，终端和导出元数据会明确写 `source:epoch_...`。

```bash
# 生成上述 V3.16 系列中 Dev Weighted 最好的 submission：
bash run_eval_w2v_v3161.sh --checkpoint best --upload-temp

# 如需强制本次最后一轮：
bash run_eval_w2v_v3161.sh --checkpoint last --upload-temp

# 单独复核同一套官方 Online / 本地模拟 Noisy Dev：
bash run_validate_w2v_v3161.sh --checkpoint best
```

默认读取 `exp/.latest_v3161_run`，也可用 `--run exp/w2v_v3161_...` 指定。不会读取旧 `.latest_v316_tfcl_run` 来导出旧选择器。输出在 `/home/ubuntu/LXT/temp/<本次run>_submission_best/`，包含 `submission.zip`、`submission_meta.json` 与临时下载地址。元数据明确记录选择标签、实际版本、源 LAST 和文件哈希；推理只保留单个检测器，不带 TFCL 辅助分支。

已存在非空输出目录时拒绝覆盖，用 `--out /home/ubuntu/LXT/temp/新的目录` 重跑。

## 速度、终端、磁盘

- 每轮完整 Dev 一次，六组 Recall(fake/real)、F1、AP、AUC、EER 常驻终端；训练进度只刷新一行。
- 每轮输出一次均值 CE、加权时间/结构损失、计算/等待耗时和 BEST。详细逐步记录写 `training_steps.jsonl`；验证计数和调速详情写 `details.log`。
- 去掉硬对齐 CPU 动态规划和额外重要性梯度计算；复用同一次 SSL 前向的特征。保留持久 CPU worker、有限 RAM 原音频/噪声缓存、按完整长度组批和 BF16 训练。
- 在 A100 上实测完整新目标的微批/重计算配置，测试后回滚权重、Adam 和 RNG。默认物理微批上限 24、帧预算 14400，不截断长音频；逻辑 batch 仍为 16 个来源。实际吞吐必须由服务器测量。
- 保存一个原子 LAST，内含当前可训练权重、Adam、辅助模块和至多一个最佳权重副本；冻结前缀引用现有文件。启动检查临时双份写入空间，并保留 10 GiB 余量。没有逐轮音频或特征磁盘缓存，也不删除源模型依赖。
- 额外机制面板只在最后做一次诊断，不参加新选择器。旧版把模拟 seen/heldout 当成官方对应关系的诊断已隔开：没有已验证 Dev Offline–Online 映射时明确标为 unavailable。

## 尚未解决的限制

本轮保留原 V3.16 的 Train 增强分布，避免在续训时同时改动过多数据因素。已有三条模拟链仍以 Opus 为主，不能等同于真实平台完整的 AEC/NS/AGC/VAD/丢包恢复行为；一次静态回声不是真正 AEC。官方 Online 已进入一致性学习，但增加轮次和恢复 TFCL 都不保证解决官方 Noisy 的分布差距。

四轮可能不足，尤其原来检测头随机初始化；是否欠拟合需看继续四轮后的训练与 Dev 曲线。由于本次同时修正目标，不能把后续提升全部归因为轮数。所有本地 Weighted 仍沿用固定 Dev 的定义，不是排行榜成绩预测。

## 验证

测试使用离线小型真实 w2v-BERT 和 CPU，不下载权重：

```bash
python -m unittest discover -s w2v_v3161 -t . -v
```

覆盖双向梯度、原作者单来源 CKA 数值、有效位置掩码、微批不变性、Adam 恢复、完整轮次续训/中断重放、真实最佳权重导出、无历史回退和诊断分组。新增发行兼容测试保存旧版本指纹，通过真实输入校验、WAV 读取、官方特征提取、小型 SSL 训练、Dev 重放和 submission 打包；只替代本机不可用的 Linux 原生 APM 边界。另跑当前流程依赖的历史模块测试。A100 显存/速度、Linux 原生通信处理和真实官方数据分数需要服务器运行后确认。
