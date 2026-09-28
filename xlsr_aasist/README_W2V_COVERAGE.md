# 一轮内容与噪声条件覆盖修复

从原 `w2v_rebuild_20260920_093548/stage3/best_model.pt` 开始，复用已有音频缓存。改动只通过新入口启用；原始 best、此前候选及现有实验目录均保留，不删除 checkpoint。

## 本轮改变什么

- **ordinary 完整遍历**：所有原有训练文件仍每轮一次。长于 64,600 点的录音，以 50% 概率保留首段、50% 概率提议随机窗口；每次仍只前向一个约 4.0375 秒窗口。短录音沿用原重复补齐。四组使用同一规则。
- **有限的有效性筛选**：在增强前的干净信号上，用 20 ms 帧能量检查随机窗口，至少 20% 帧超过相对/绝对能量门槛。最多尝试 8 次，失败回到首段。它是避免近乎静音的检查，不是 VAD，也不保证尾段能解决困难来源。只适用于官方整句真假标签，不应用于局部伪造数据。
- **增强顺序保留**：选择裁剪位置后，RawBoost 仍按原方式处理整条音频，再取所选窗口、执行原环境加噪。裁剪使用独立随机流，不消耗 RawBoost/环境噪声的随机数；仅保留首段时可逐值复现旧增强。随机位置由 seed、epoch 和文件索引决定，不随 worker 调度变化。
- **noisy 分层轮转**：仍每步 4 个来源、真假各 2 个。一个 batch 共用同一缓存库、算法家族和 SNR 档；按库内的算法×SNR 格子轮转，格子内各语言/真假来源打乱后依次使用，减少重复。缺失关键格子时明确报错，不悄悄用另一类替代。
- **避免重复语言补偿**：noisy 在每个条件内用累计抽样配额实现英文 fake 40%、real 35%，累计取整偏差小于一个样本。该分支 reference/processed 的语言 CE 系数均改为 1。ordinary 和真实 RTC pair 仍用原分支频率计算语言系数。因此没有再提高英文目标份额。分层改变了来源组合及对比项看见的样本，不能宣称与旧梯度完全相同。

不改变模型结构、Dev 输入、推理阈值、每步前向数量或原噪声生成器。真实 RTC pair、noisy reference 和已生成的 noisy 视图仍使用固定首段，不能把缓存误当作完整录音。

## 保留的配方

只训练最后 4 层和分类头，encoder/head LR 为 `1e-7 / 2e-6`，全局 real cost 仍为 `1.25`。固定 1 个 epoch，ordinary 仍逆频率 CE；noisy processed CE 仍占 30%；新库仍在首轮从 0 增至 20%（平均约 10%），pair warmup 沿用此前设置。一致性附加损失关闭。

这次没有继续搜索权重、扩大输入长度、生成新缓存或额外执行预训练模型。条件共享减少 batch 内条件种类，但提高短程训练各条件的可核查覆盖；是否提高精度需看固定 Dev，不能预先保证。

## 在服务器启动

使用已有 `sdd` 环境，不需重装依赖或重新生成 full 缓存。上一轮英文运行只提供路径、原 best 的 SHA256 和参考结果；不会加载它的候选权重。

```bash
conda activate sdd
cd "$HOME/LXT/RTC-w2v-improved"
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash run_w2v_coverage.sh exp/w2v_en_20260928_161650_59ab --upload-temp
```

查看进度：

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist"
tail -n 60 -f "$(cat exp/.latest_train_log)"
```

`Ctrl+C` 只退出上述日志查看，训练由后台进程继续。启动日志依次显示原 checkpoint 校验、缓存检查、原 best 固定 Dev 评估、`S3 epoch 1`。仅运行一次基线评估，不再独立执行重复 preflight。

只预览路径与配置、不训练：

```bash
python -u start_w2v_coverage.py --from-run exp/w2v_en_20260928_161650_59ab
```

如果上一轮目录名称不同，替换 `--from-run`/第一个位置参数即可。该目录必须有 `stage3/config.json` 且能验证相同原 best；不自动猜测最新 checkpoint。

## 结果与下载

新运行创建于 `exp/w2v_coverage_DATE_ID`。在 baseline SHA256 不匹配时拒绝开始，结束再核验。原 best 保存在原路径，不在新运行的保存范围内。

`stage3` 中保留：

- `coverage_plan_epoch_001.json`：计划的语言×真假×算法×SNR 曝光、各格可用源数。
- `coverage_actual_epoch_001.json`：已完成优化步骤的真实计数、不同源数、首段/随机/回退比例；每 256 步及首步保存，预取不算作训练。
- `baseline_scores.jsonl`、`epoch_001_scores.jsonl`：本轮实际前向得到的固定 Dev 分数。
- `metrics.jsonl`、`epoch_001_evaluation.json`：四组 Train/Dev 指标、覆盖、损失及耗时。
- `best_model.pt`、可能存在的 `candidate_best.pt`、`last.pt`：沿用既有目标保护与候选保存。候选不等同于全面超过原 best。

结束时自动生成报告 ZIP，复制到 `/home/ubuntu/LXT/temp`；加 `--upload-temp` 才上传报告到 temp.sh 并输出 `TEMP_DOWNLOAD_URL`。ZIP 包含计划、实际覆盖、配置、日志和分数，不含音频、模型、优化器或特征缓存。上传失败保留本地 ZIP；训练失败也尽量导出部分诊断并明确标记失败。

## 回退与验证范围

本次更新前的 Git 回退分支：`rollback-before-coverage-20260928`，指向 `9df30b44d43ce91b07f5eaf92bfc03fb9fcbde65`。原始训练最佳模型通过服务器文件路径与 SHA256 保护；Git 分支不包含几 GB 的模型文件。

新增 CPU 测试覆盖裁剪位置/静音/短文件、旧增强不变性、固定计算预算、真假条件对称、每格英文配额、完整 ordinary 遍历、无重复语言加权、实际计数、预取与恢复、真实训练引擎更新/冻结、checkpoint 保护和报告导出。CPU 验证不能替代服务器 GPU 实训，也不证明精度会提升。
