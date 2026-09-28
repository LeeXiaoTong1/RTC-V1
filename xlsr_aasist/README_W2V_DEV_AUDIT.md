# 原 best 与第一轮候选的逐样本 Dev 诊断

只运行推理，复用现有 Dev 音频和声学特征缓存。原 best、第一轮候选、训练脚本、阈值和缓存均不修改；缺失的声学特征在内存中重新计算，不生成新的磁盘缓存。下载包只包含表格、报告、评分、元数据和日志，不包含音频、模型或特征文件。

## 服务器启动

```bash
conda activate sdd &&
cd "$HOME/LXT/RTC-w2v-improved" &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash run_w2v_dev_audit.sh exp/w2v_adapt_20260927_225221_104a "$HOME/LXT/temp"
```

后台依次评估原 best、第一轮候选，每次只加载一个模型。各自依次运行 clean（包含 Online/Offline）、Seen、Heldout。需要两遍完整 Dev 前向，时间取决于 GPU 和磁盘；不是几秒钟的日志读取。不会重新训练或生成全量缓存。输出单独保存在 `exp/w2v_dev_audit_时间_PID`。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
tail -n 30 -f "$(cat exp/.latest_dev_audit_log)"
```

看到 `AUDIT_COMPLETE=True` 后，在网页文件管理器进入 `/home/ubuntu/LXT/temp`，下载对应 `w2v_dev_audit_时间_PID.zip`。`DOWNLOAD_ZIP` 会打印完整路径，旁边的 `.zip.sha256` 可用于核验完整性。退出日志跟踪按 Ctrl+C，后台诊断继续运行。

## 下载内容

- `report.md`：中文汇总，对比 F1、real recall、CE、AUC、救回/新增/持续错例，以及边界附近和高置信度错误。
- `summary.json`：机器可读汇总及逐噪声档指标。
- `predictions.csv`：逐条 ID、真实标签、两个模型的原始 logits、fake 概率、预测、条件、SNR、处理参数及来源路径。
- `changed_errors.csv`：被候选救回的样本和候选新增的误判；`persistent_errors.csv`：两模型都错的样本。
- `groups.csv`：按条件、噪声档、处理算法、路径目录统计；目录不代表已核实的说话人或设备。
- `metric_parity.json`：复算结果与本次训练保存的 baseline/epoch1 日志是否一致。有差异先查环境与输入，不把差异视为模型改善。
- 原始逐条件评分 JSONL、原训练指标、输入/模型指纹与报告文件 SHA256 清单。

CSV 使用 UTF-8 BOM，可用 Excel 打开。评分不截断到几位小数，防止 0.5 附近判断被显示精度改变。fake=0、real=1；p(fake)>=0.5 判为 fake。AUC 使用原始 logit 差排名，避免 FP32 概率饱和造成假并列。

## 与训练验证保持一致

- 沿用训练配置的 `eval_batch` 和 `eval_microbatch`，纯 FP32、eval 模式、尾批补齐；保持协议顺序和 noisy 数据的 source/band 排序。
- Seen/Heldout 分别取四档指标平均，RobustF1 = 0.3 Online + 0.35 Seen + 0.35 Heldout。分组 CSV 中的组内合并 F1 不代替这个正式口径。
- 原 best SHA256 必须等于本次 adaptation 的 `init_sha256`。两模型使用本次固定 Dev；原 checkpoint 过去用的旧 noisy Dev 不会混入比较。
- 核对 Dev 协议、特征提取器和缓存 manifest/config 的哈希。实际音频的大小及 mtime 在两遍评分期间及续跑时检查；这不是所有音频的内容哈希。
- 每个模型的混淆矩阵、BalancedCE 会与本次保存的基线/第一轮日志核对。包内 `completed.json` 的完成状态表示诊断流程完成，不意味着候选一定更好。
- 默认 400 次按真假分层的成对重采样，同一 noisy 源的八个视图共同抽样。Online 与 noisy 的源依赖关系未经确认，不计算 RobustF1 联合区间。Dev 已用于选模，这些区间不能作为独立测试集证据。

## 中断后继续

不要在同一个诊断仍运行时续跑。保存的每个模型 × 条件评分文件都有完整性标记；只有模型、输入、音频文件身份、代码、批大小及评分哈希相同才复用。未完成的一个条件从头计算。旧原始模型和缓存始终只读。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
OUT=$(cat exp/.latest_dev_audit_dir) &&
LOG="${OUT}_resume.log" &&
nohup python -u audit_w2v_dev.py \
  --run-dir exp/w2v_adapt_20260927_225221_104a \
  --out "$OUT" --resume --download-dir "$HOME/LXT/temp" \
  --log-file "$LOG" > "$LOG" 2>&1 < /dev/null &
```

完成后不要再次续跑：已有同名 ZIP 会被保留，不会静默覆盖。需要重新打包时可用 `--resume` 并指定另一个下载目录；已完成的评分无需再次前向。

## 本地验证范围

回归测试覆盖评分与原 `Metrics` 的口径、FP32 尾批补齐、误判转换、AUC 并列/饱和、按源抽样、缓存只读命中/缺失、模型与音频保留、续跑身份校验、ZIP 清单和哈希。端到端流程采用小型模型和合成输入，不能替代服务器实际模型的复算；真实评分一致性由报告中的 `metric_parity.json` 检查。
