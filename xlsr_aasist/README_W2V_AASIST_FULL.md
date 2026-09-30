# 原 best + AASIST 整段音频微调

本次恢复原 91.68 checkpoint 的 **完整 w2v-BERT 和 AASIST 权重**，严格检查参数键和 SHA256。没有导入 MultiConv 第一轮权重，没有重新初始化 AASIST，也没有保留 MultiConv 的多层融合与 CKA。

唯一默认起点：

```text
/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt
SHA256=db3f8167742bf2fe41cfad028dec962d56f6c61295442870620421d7f3a9bbee
```

## 部署、保留权重、启动

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash setup_w2v_aasist.sh &&
python -m w2v_aasist.maintenance --apply &&
bash run_w2v_aasist.sh --upload-temp
```

`maintenance` 将原 best、最高 Dev 分数的 MultiConv 候选及至多两个不同权重的 AASIST 候选复制到 `checkpoints/retained/`，每份复制后校验 SHA256，并写入 `catalog.json`。候选排名使用各自架构内记录的 Dev 代理指标，不将其当作平台分数。原实验权重不删除；新实验也不会覆盖它们。

Git 更新会移除已跟踪的废弃脚本。维护程序对清单中仍残留的旧脚本先写入 `code_archives/` ZIP 并验证，再删除这些具体文件。清单不能包含 exp、数据、缓存、预训练模型或 checkpoint；也不会递归删除带有未知文件的目录。历史完整代码仍可从回退分支恢复。

查看进度：

```bash
tail -n 60 -f "$(cat exp/.latest_aasist_log)"
```

首先打印 checkpoint/缓存检查进度，再评估 `Epoch 0`：原 best 权重直接使用整段 Online/Offline 音频，没有训练更新。这样可以看到输入策略改变后的起点，再比较后续微调是否改善。

训练后台运行；退出日志查看不会停止训练。停止/崩溃后只承诺从最后完整 epoch 恢复：

```bash
bash run_w2v_aasist.sh --resume "$(cat exp/.latest_aasist_run)" --upload-temp
```

恢复要求配置、代码、依赖版本和输入元数据相同；不会静默套用新的 CLI 学习率。

## 整段音频与 noisy 组合

普通 Train、Online/Offline Dev 和新模型 Eval 都保留整段，不设置最大长度，不取随机 4 秒窗口。仅短于 0.4 秒的极短语音重复至 AASIST 所需最小帧数；已有 noisy 缓存维持原 64600 点。

每次 noisy 抽样仍只产生一个输入/一个分类损失项，使用以下固定预算，真假使用相同组合日程：

| 方式 | 预算 | 输入构成 |
|---|---:|---|
| 单个缓存 | 50% | 原有处理结果，不做二次增强 |
| 同源条件切换 | 25% | 同一录音两个不同 SNR/处理版本，在对应时间位置接续；边界 20ms 交叉淡化；总长仍约 4 秒 |
| 含噪首段接原始尾段 | 25% | 将该录音的缓存首段放回完整原录音的开头，接回同一时间位置之后的原始尾段；边界 20ms 淡化 |

第三种是“局部含噪整段”，尾段没有被缓存处理，不能称为整段 noisy；缓存没有尾段信息，也不会凭空生成。源音频不超过 64600 点时回退到单个缓存，并在组合计数中单独记录。接回尾段之前核对原录音与缓存的源 SHA256。

不混合不同录音、不把 real 和 fake 叠加、不把同一句话重复拼成虚假的长录音。多版本切换使用同源同时间位置，但不同处理链仍可能存在微小延迟；保留 50% 原始缓存并限制切换比例，避免所有训练输入都变成合成切换条件。组合只在 Train 的内存中生成；已有 WAV、清单、标签及 Dev 条件不改写。

默认只使用旧的 `train_noisy_cache`。若确有需要可用 `--include-extra-cache` 加入原配置中已有的第二个 Train bank；不会使用 heldout family，默认不同时增加噪声库变化。

这是本项目的待验证组合设计，不是论文已经证明能提分的配方。音频反欺骗中的增强/混合研究提供一般动机：[RawBoost](https://arxiv.org/abs/2111.04433)、[Mixup regularization strategies](https://www.isca-archive.org/interspeech_2022/kang22b_interspeech.html)。本文没有复现这些论文的全部方法，也不引用它们的提升幅度作为本方案预期。

## 训练预算与速度

| 项目 | 默认设置 |
|---|---|
| 参数起点 | 原 91.68 checkpoint 全部前后端权重 |
| 特征 | 仅 w2v-BERT 最后一层；原 1024→128 投影与 AASIST 不变 |
| 可训练范围 | 编码器最后 4 层 + 完整 AASIST；前 20 层、特征投影冻结 |
| 学习率 | encoder 1e-7，head 2e-6；100 step 升温，然后余弦降到初值的 10% |
| 轮数 | 最多 2 轮；连续 2 轮无 Weighted Dev 改善停止 |
| 逻辑批次 | 24 ordinary + 4 noisy；ordinary 完整遍历，每条每轮一次 |
| 真假 | ordinary 逆频率权重；noisy 每批 2 fake + 2 real，不再乘类别频率权重 |
| 分类目标 | 0.7 × ordinary weighted CE + 0.3 × noisy CE |
| 普通增强 | 沿用 RawBoost 算法 5；noisy 缓存不重复执行 RawBoost |
| 参数更新 | AdamW，weight decay=1e-4，梯度裁剪 1.0 |
| 精度 | Train BF16；Dev/Eval FP32，与原 best 的审计精度一致 |

本次不恢复旧 RTC/noisy 配对对比损失，也不加入蒸馏、CKA、额外 real 或英文权重。它是从旧权重出发的输入扩展微调，不是重跑原三阶段训练。

每个输入只有一次前向/反向；没有 MultiConv 的逻辑批次 CKA 和两遍整网前向。相同帧数的样本最多 4 条一起计算，默认总帧预算 1600；不同帧数分开，避免时间补零改变 Conformer 边界以及 AASIST 的 GroupNorm/图池化统计。末四层仍使用梯度检查点控制显存。

整段语音仍比 4 秒输入更耗时、耗显存；极长录音不会被偷偷裁短。可通过 `--microbatch 1` 降低多样本并行显存，但它无法消除单条超长语音的开销。日志每 100 步给出平均 seconds/step，实际 GPU 速度需由服务器运行确认。

## 保存、报告和提交

已停止训练后，可清理旧的中间 checkpoint 来释放空间：

```bash
python cleanup_w2v_checkpoints.py --apply
```

它只扫描当前项目和原 AASIST 项目的 `exp`，清理可识别的 `last`、`latest`、按 epoch 编号的权重；所有命名含 best/candidate 的文件、留存目录及其目录清单指定的源权重均受保护。没有命名 best 的实验、未知命名的权重会跳过。原 91.68 best 在删除前后检查 SHA256。检测到同一用户仍有 Python 训练或启动器运行时停止清理；执行期间不要启动新的训练。删除 last 后不能从该文件恢复优化器状态，但保留的 best 仍可用于微调。此命令不删除音频、noisy 缓存或报告，也不复制大文件。省略 `--apply` 只预览。

每个新实验只保留 `best_model.pt`（Weighted 最佳）、`best_noisy.pt`（Noisy 最佳）、`last.pt`（完整 epoch 的优化器状态）。最佳记录包含 Epoch 0，因此微调未改善时仍保留原权重＋整段策略。原 91.68 权重文件单独保留，绝不覆盖。

每轮保存 JSON、逐样本分数、英文/中文与真假指标、Offline/Online/Seen/Heldout、组合实际曝光计数。默认选模指标为 `0.3 Online F1 + 0.35 Seen F1 + 0.35 Heldout F1`；Seen/Heldout 使用原四档平均方式。它是固定 Dev 代理指标，不是平台 91.68。

结束或训练进程报错后，启动器导出诊断 ZIP 到 `/home/ubuntu/LXT/temp/`；启用 `--upload-temp` 时打印 `TEMP_DOWNLOAD_URL=https://temp.sh/...`。诊断 ZIP 不包含音频或模型。

训练后生成 Progress submission：

```bash
bash run_eval_w2v_aasist.sh --upload-temp
```

ZIP 内只有 `scores.txt`，每行是官方原始 ID 与 P(fake)，固定 0.5 判定规则。若选择 noisy 最佳：

```bash
AASIST_CHECKPOINT="$(cat exp/.latest_aasist_run)/best_noisy.pt" bash run_eval_w2v_aasist.sh --upload-temp
```

原版 checkpoint 可传给相同评估器，它会识别旧 schema 并使用旧首段裁剪/短音频重复规则。当前 MultiConv checkpoint 仅留存供回退；推理它时使用 `archive-before-aasist-full-20260929` 的代码。

## 验证范围

本地 CPU 检查覆盖：原 AASIST 参数和输出兼容、变长输入、单遍梯度与整体 CE 一致、冻结参数、同源组合与文件不变、真实小型 HF w2v-BERT 训练/恢复/导出、保留 checkpoint 和脚本清理。没有在本地加载服务器 600M 权重或运行生产 GPU；不会声称已经提高真实 Dev 或平台成绩。
