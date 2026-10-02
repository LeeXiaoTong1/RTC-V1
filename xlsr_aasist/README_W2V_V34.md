# V3.4：从已提交模型出发，扩大编码器适配范围

本版保留 w2v-BERT 2.0 + MultiConv、完整原始语音及 V3.3 两种完整 noisy，重点改变编码器更新范围与分层学习率。只启动一次训练，默认两轮，半轮与整轮验证；不再先跑一轮 control。

用户提供的平台截图为 Clean 98.0803、Noisy 91.3858、Weighted 93.3941。相对早期 Weighted 91.68，提高约 1.7141 个百分点；这不能单独证明 V3.3 配对损失有效，因为默认导出可能选中了保留的基线权重。本地 Dev 的约 95.7 与平台 93.3941 属于不同评测，不能混用。若 Clean 保持 98.0803，按 0.3 Clean + 0.7 Noisy 达到 Weighted 97，需要 Noisy 约 96.5370。97 是后续目标，不是本版承诺或本地验收结果。

## 在服务器启动

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v34.sh &&
bash run_w2v_v34.sh --upload-temp
```

首次启动默认且明确使用：

```text
exp/w2v_v33_20261002_004018_198a
/home/ubuntu/LXT/temp/w2v_v33_20261002_004018_198a_submission/submission_meta.json
```

这对应用户已确认的 `env -u V33_CHECKPOINT V33_RUN=exp/w2v_v33_20261002_004018_198a bash run_eval_w2v_v33.sh --upload-temp` 导出。V3.4 读取完成的 `comparison.json` 和所选 arm，检查 `best_model.pt` 的 schema、tag、完整 SHA256、配置、Dev 记录和提交元数据。现存 submission.zip 也核对其导出 hash。若提交 ZIP 已移走，可以使用原始元数据；若元数据缺失，不猜测权重，需通过 `--submission-meta /原始文件路径/submission_meta.json` 指定。

启动会打印 `V34_WARM_CHECKPOINT`、`V34_WARM_SHA256`、`V34_WARM_TAG`、`V34_SOURCE_ARM` 及 `V34_SOURCE_BASELINE_FALLBACK`。`baseline` 是合法起点，表示该 V3.3 arm 没有替换原权重，不能宣传为新配对训练获胜。系统核对本地导出来源，不能从元数据独立认证平台截图与上传行为。

setup 复用已有环境，不安装依赖或替换 Torch。先运行测试，再启动后台任务并打开终端查看器。若还有旧训练/评估在运行，启动器会退出，避免同时占用 GPU。只需要停止仍运行的 V3.3 时：

```bash
python -m w2v_v33.stop --version v33 --apply
```

## 实际改变

| 项目 | V3.4 默认设置 |
|---|---|
| 编码器 | 24 个 encoder block 全部可训练 |
| 非编码块前端参数 | 输入投影等继续冻结；不把此设置称为前端所有参数全量更新 |
| 最高编码器学习率 | 第 24 层 `5e-7` |
| 逐层衰减 | 每向前一层乘 0.9，第 1 层约 `4.4315e-8` |
| MultiConv | 整个后端训练，学习率 `2e-6` |
| 日程 | 100 次更新预热，随后余弦衰减到峰值的 20% |
| 训练预算 | 默认 2 轮，每轮半程与结束时各验证一次 |
| 优化器 | 新建 AdamW；本轮回退时恢复权重及配套 Adam 状态 |

相对 V3.3，最高前端学习率提高了 10 倍，训练预算从默认一轮改为两轮，并扩大到所有编码器块。新解冻块按既有实现进入 train 模式，包含其 dropout。因此这是整体编码器适配策略，不能把收益都归因于层数，不是严格单变量层数消融。

较早层可以较小幅度调整对通信失真的表示，较后层承担更大的任务适配。当前 MultiConv 同时融合各层输出，因此中间层是否允许适配是有实际意义的限制；但全层参与不保证域外泛化提升。已有较高训练指标也不是无限延长训练的理由。

每层分别保存学习率与 Adam 状态；预热、衰减及恢复不会把各层学习率重新设成同一个值。`optimizer_groups.json` 记录完整分组，验证 JSON 记录逐层学习率与采样权重变化；终端只展示编码器最小/最大学习率和后端学习率。采样诊断用于确认参数确实调整，不代表完整层范数或泛化证据。

## 保持的训练输入与目标

- 原 Train、Dev、对应关系和已完成的 V3.3 noisy 缓存全部复用，开始、结束及恢复均核对输入指纹。不会新建或删除缓存，也不修改已有 best。
- 每轮按独立原始录音完整遍历；每个 source 的 Offline、可用 Online、两份 noisy，保留完整/短片段监督。
- 真假类别权重、普通/noisy 系数、整段/短片段系数、RawBoost/局部静音及逻辑 source batch 均继承所选提交组。
- 配对目标也继承该组：若选中 control，配对系数继续为 0；若选中 candidate，保留其原系数与预热。不会因版本升级偷偷启用配对项。
- 保留已有 MultiConv block CKA，不同时更换后端、损失或阈值。
- 只用官方 Train 语音及现有 Train 噪声；Dev 只验证，Progress 不进入梯度计算或模型选择。

保留按长度微批、BF16、激活重计算、GPU 驻留/CPU 卸载，以及修复过的三个共享存储的数据加载方式。前向输入次数没有增加；24 层反向传播与 Adam 状态会增加计算、显存和 checkpoint 占用。A100 40GB 的完整数据运行尚需服务器验证，CPU 小模型测试不能证明其峰值显存足够。磁盘检查会在优化器更新前确认保存事务所需的余量，空间不足不会自动删除旧权重。

## 选模与查看

训练开始先重新评估起点作为固定本地基线。Dev 条件、FP32、完整原始语音、既有短 noisy Dev 及统一 0.5 阈值保持不变。继续保护 Clean、每组英文 real、noisy fake recall 与 Noisy，只有合格的 Weighted 提升才替换 `best_model.pt`。最多一次从合格 best 恢复权重与 Adam、降低学习率，并留出两个验证间隔观察。最终仍可能回退到起点，这不影响旧提交。

```bash
# 动态进度；每次 Dev 结果保留在上方
bash watch_w2v_v34.sh

# 查看已完成的所有验证，包括起点及最终选择
bash show_w2v_v34.sh
```

Ctrl+C 仅关闭查看器，不会停止后台训练。实际停止：

```bash
python -m w2v_v34.stop --apply
```

恢复使用保存的 V3.4 配方和最后提交的验证边界，不能借恢复修改训练配置。验证边界后尚未保存的更新会重放：

```bash
bash run_w2v_v34.sh --resume "$(cat exp/.latest_v34_run)" --upload-temp
```

仅检查来源与完整缓存、不训练：

```bash
bash run_w2v_v34.sh --prepare-only --upload-temp
```

准备完成后的 `RUN` 可传给 `--resume` 正式训练；不会重复生成缓存。

## 提交与报告

完成后导出受保护的单模型 winner：

```bash
bash run_eval_w2v_v34.sh --upload-temp
```

导出使用原有 Progress 路径、整段 FP32 推理、官方 ID 顺序和 `P(fake)`，生成 `submission.zip`、`submission_meta.json` 并打印下载链接。元数据包括最终权重、起点权重 SHA256、来源 tag 和是否回退。它不会自动向比赛平台提交。

训练报告 ZIP 仅包含诊断、配置与逐样本分数，不含语音、权重或优化器。结束打印 `TEMP_DOWNLOAD_URL`；上传失败时本地文件仍保存。最终平台能否超过 97，必须由该模型实际提交结果确认。
