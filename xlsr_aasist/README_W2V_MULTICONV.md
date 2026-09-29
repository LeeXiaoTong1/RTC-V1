# w2v-BERT 2.0 + MultiConv 部署

本实现是独立候选方案：从原 best 导入 **w2v-BERT 编码器**，重新初始化并训练 MultiConv 后端。
原 AASIST、旧训练入口、旧缓存和原 best 均不修改。代码能完成训练、Dev 验证、断点恢复、逐样本报告及 submission.zip 导出。
只有官方 Train 的语音和已经生成的 Train noisy 缓存参与参数更新。Dev 只验证与选模；Eval 只推理。

## 1. 服务器启动

在已有 GPU 环境 `sdd` 和当前项目中执行。无需重新生成全量缓存，也不下载新的语音数据或反欺骗模型权重。

如果使用提供的本地 `w2v_multiconv_deploy.zip`，先将它上传到服务器的 `/home/ubuntu/LXT/temp/`，再执行：

```bash
conda activate sdd
python -m zipfile -e /home/ubuntu/LXT/temp/w2v_multiconv_deploy.zip /home/ubuntu/LXT/RTC-w2v-improved
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
bash setup_w2v_multiconv.sh
```

部署包只包含新增的 MultiConv 文件，不包含旧模型、旧训练文件、缓存或数据集。
若 GitHub 提交已获授权并成功推送，也可使用下面的更新方式；两种方式选一种即可：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash setup_w2v_multiconv.sh
```

安装入口保留现有 CUDA PyTorch，固定 `transformers==4.38.2`，然后运行 CPU 功能测试。
测试使用合成音频和缩小的 w2v-BERT，不加载原 best、不读取真实数据、不占用训练 GPU。
本地测试通过不代表服务器的完整 600M 模型已做 GPU 验证；启动训练时会检查实际 CUDA、BF16、数据与空间。

随后启动一项后台训练：

```bash
bash run_w2v_multiconv.sh --upload-temp
tail -n 60 -f "$(cat exp/.latest_multiconv_log)"
```

`--upload-temp` 仅上传报告 ZIP；不会上传音频或模型。结束后日志打印 `TEMP_DOWNLOAD_URL`。
`Ctrl+C` 只结束上述 `tail` 日志跟踪，后台训练继续。不要重复启动同一训练。
看到 PID 只表示启动命令已发出；以日志中的 epoch 进度、`TRAINING_COMPLETE=True` 或具体报错判断运行状态。

默认继承最新一份 `exp/w2v_*/stage3/config.json` 的数据路径，打印 `SOURCE_CONFIG` 供核对。
不会继承该配置的 real/英文额外权重。若自动选中的数据路径不是所需实验，可明确指定：

```bash
bash run_w2v_multiconv.sh --source-config /absolute/path/to/stage3/config.json --upload-temp
```

原 best 默认路径固定为：

```text
/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt
```

训练目录是独立的 `exp/w2v_multiconv_<时间>_<随机后缀>`。加载前后核对原 best SHA256。
预检查会打印缓存检查进度；哈希读取 2 GB 级权重可能需要一段时间。
空间预算按实际模型大小估算，保存使用临时文件完成后替换；每个实验仅保留 best 和 last 两份模型。

## 2. 模型与适配

```text
16 kHz 单声道音频
  → 官方 SeamlessM4T fbank / 每条语音 CMVN / 160 维输入
  → 原 best 的 w2v-BERT 2.0（24 层，输出包含输入层在内的 25 组表示）
  → 各层共享 1024→128 投影和 SwiGLU，按层求和
  → 4 个顺序 MultiConv block（每块含 3/7/11/15 多尺度卷积）
  → 按时间位置拼接四块输出
  → 4 头注意力均值与标准差池化
  → 512 维全连接、SELU、2 类 logits
```

参考 [MultiConv 官方代码](https://github.com/hoanmyTran/dissimilarity_deepfake_detection) 和
[ACM MM 2025 论文](https://arxiv.org/abs/2509.03409)。第三方许可见 `THIRD_PARTY_LICENSES/MultiConv.txt`。
官方方案前端是 XLS-R；这里适配为 w2v-BERT，因此不能直接引用论文成绩作为本系统成绩。

核对到的上游实现有两处影响训练的细节，本实现作了修正：

- 上游 CKA 经 `.item()` 再构造 Tensor 后失去梯度；这里保留整个计算图，确保 CKA 真正参与学习。
- 上游直接把 `B,block,T,D` 的堆叠结果 view 为 `B,T,block*D`；这里沿通道拼接，保持时间对齐。

同时使用带有效帧掩码的卷积和注意力池化；保留 LayerNorm，无 BatchNorm。
CKA 是四个 block 的池化均值在**整个逻辑批次**上的线性 CKA 均值，最小化它以减少表示冗余。
它不是 clean/noisy 配对对比损失，也不要求把所有噪声痕迹对齐成同一个表示。

## 3. 默认训练配置

| 项目 | 设置 |
|---|---|
| 阶段一 | 2 轮；冻结整个前端；只训练新 MultiConv 后端 |
| 阶段一后端学习率 | 1e-4，开始 200 step 逐渐升至该值 |
| 阶段二 | 最多 4 轮；仅解冻编码器最后 4 层，特征投影及前 20 层继续冻结 |
| 阶段二学习率 | 编码器 2e-7，后端 1e-5；阶段开始有 200 step 升温 |
| 优化器 | AdamW；非 bias/一维参数 weight decay=1e-4；梯度裁剪 1.0 |
| 逻辑批次 | 16 条 ordinary + 4 条缓存 noisy；末尾不足时并入上一批 |
| 实际前向批次 | 1 条变长音频；两遍重算累计精确逻辑批次梯度 |
| ordinary 监督 | 完整遍历所有 Train，每条每轮一次；CE 类别权重 N/(2*N_class) |
| noisy 监督 | 每批 2 fake + 2 real；源录音队列循环、四个 SNR 档轮转；不叠加逆频率权重 |
| 总损失 | 0.7 ordinary weighted CE + 0.3 noisy CE + λ CKA；λ 前 200 step 升至 0.05 |
| 输入增强 | ordinary 使用原 RawBoost 算法 5；已有处理缓存不再重复增强 |
| early stopping | 阶段二连续 3 轮未改善停止；连续 2 轮未改善时下一轮减半学习率 |

这些学习率、CKA 权重和冻结策略是本次迁移的初始配置，不是已经证明最优的参数。
新后端随机初始化，不能沿用旧 AASIST 微调时极低的后端学习率，也不应要求第 1 轮立即超过旧 best。
训练阶段切换保留后端 Adam 历史。旧 RTC 配对损失、模拟 noisy 对比损失、额外 real/EN 权重不进入本方案。

**关于显存与速度：**各条音频先前向收集输出，在完整批次上计算 CE/CKA 对输出的梯度，再恢复每条音频的 dropout 随机状态重算并反传。
这保留了 CKA 的跨样本意义，避免物理 batch=1 时 CKA 退化；代价是额外前向计算。
首个训练 step 自动检查重算输出是否一致，发现异常会在 optimizer 更新前停止。

**关于音频长度：**默认 ordinary、clean Dev、Eval 保留整段；已有 noisy 缓存保持原有 64600 点（约 4.04 秒）。
不会把现有短缓存描述成整段 noisy 训练。小于 0.25 秒的输入重复至最小分析长度。
单条非常长音频仍可能超显存。若实际 GPU 无法承载，需要重新启动一个显式限制长度的新实验，例如 `--max-seconds 8`；该策略写入 checkpoint，并同样用于 Dev/Eval。
程序不会悄悄截短音频、减小 CKA 批次或改用 CPU。

默认使用原 `train_noisy_cache`。`--include-extra-cache` 可显式启用历史配置中的额外 Train bank，首轮部署不默认同时改变噪声库。
缓存角色、官方协议摘要、全量覆盖、标签、SNR、Train/Dev 噪声分离和 heldout 处理族分离都会检查。

## 4. 验证、保存和结果判断

每轮输出并保存：

- Online、Offline、Seen、Heldout 的混淆矩阵、fake/real recall、Macro-F1、CE。
- 英文/中文分组指标及四个 SNR 档指标。
- `epoch_N_scores.jsonl`：逐样本来源、标签、logits、P(fake)；`epoch_N.json`：汇总。
- `report.md`、`inputs.json`、`config.json`；结束时 `completed.json`，出错时 `failed.json`。
- `best_model.pt`：新后端最佳候选；`last.pt`：恢复训练用，包含 Adam 状态。

候选按 `0.3 Online F1 + 0.7 × (Seen F1 + Heldout F1)/2` 选择；Seen/Heldout 按原实现取四档 Macro-F1 的均值。
同分时比较 noisy F1。该指标是固定 **Dev 代理指标**，不是最终评测的 clean/noisy/weighted 分数。
报告引用旧 best 的历史 Dev 值时，也注明旧模型使用首段输入；不能将该比较当成受控的后端消融。
原 best 始终保留，新的 best 仅指新候选内部最佳，不代表已经超越原系统。

## 5. 中断恢复

已有 `last.pt` 时，从上一轮完整保存处继续：

```bash
bash run_w2v_multiconv.sh --resume "$(cat exp/.latest_multiconv_run)" --upload-temp
```

恢复读取保存配置，并检查代码、依赖版本及协议/缓存元数据摘要；不接受悄悄修改训练参数。
支持 epoch 边界恢复，不承诺从中断的具体 step 接着跑。首轮尚未保存时直接启动新实验。
恢复后完成的运行可能仍保留此前 `failed.json` 作为错误记录，以 `completed.json` 和最新执行日志判断状态。

## 6. 评估并导出 submission.zip

训练完成后，默认对现有 `dataset/progress.txt` 和 `dataset/wav/progress` 推理：

```bash
bash run_eval_w2v_multiconv.sh --upload-temp
```

该脚本读取最新 MultiConv 运行的 `best_model.pt`，在 `/home/ubuntu/LXT/temp/<运行名>_submission/` 生成 `submission.zip`，并打印 temp.sh 下载链接。
ZIP 内仅有 `scores.txt`；逐行保持官方协议 ID 顺序，格式为 `id P(fake)`；fake=0、real=1。
不会根据 Eval 标签选模型或调阈值。

如正式 Eval 的文件位置不同，明确提供：

```bash
EVAL_PROTOCOL=/absolute/path/eval.txt \
EVAL_AUDIO_ROOT=/absolute/path/wav/eval \
bash run_eval_w2v_multiconv.sh --upload-temp
```

也可用 `MULTICONV_RUN=/path/to/run` 选择旧实验，或 `MULTICONV_CHECKPOINT=/path/to/model.pt` 指定本架构权重。
上传失败不会删除本地 ZIP。报告只打包当前运行的文本/JSON，不包含任何 `.pt`、wav 或缓存文件。

## 7. 验证范围

单元及集成检查覆盖：可微 CKA、时间/通道拼接、padding 不变性、整批图与逐条重算的梯度一致性、冻结层、真实 HF 小型编码器梯度检查点、官方特征提取器、RawBoost 随机复现、采样覆盖与真假均衡、缓存元数据、原编码器严格导入及文件保护、失败保存保护、模型训练/恢复/推理、submission 格式。
集成测试使用合成音频；它验证程序行为，不提供模型提分证据。
生产 GPU、真实全量 Train 的训练速度/显存及最终性能必须以服务器运行结果为准。
