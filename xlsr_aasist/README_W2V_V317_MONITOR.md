# V3.17 全量 Train 复测与曲线

这是 V3.17 的独立观测入口。模型、LoRA、采样、优化器、四轮预算、Dev 与 best 选择保持原样；原 V3.17 的 26 个发布文件没有修改。只新增模块和脚本，当前后台进程不会因旧文件指纹改变而失败。

## 当前训练正在运行：先查看已有结果

在原 `sdd` 环境中：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash monitor_w2v_v317.sh --watch
```

这条命令不加载模型、不使用 GPU，只读取已保存的逐步日志与各轮 logits。每 30 秒更新曲线；每轮完整指标只在变化时输出。Ctrl+C 只关闭观察窗口。指定其他实验可用 `--run exp/w2v_v317_...`。

当前旧进程不会自动加载新增的每轮全量推理。旧轮次只能显示已记录的 256 条固定 Train Online 探针，明确标为 sample，不会伪装成全量 Train，也不会编造缺失的 Train Noisy 指标。

## 本次实验从已保存 epoch 接续，启用每轮全量评估

仅在原训练进程已退出时执行：

```bash
bash run_w2v_v317_monitored.sh \
  --resume "$(cat exp/.latest_v317_run)" \
  --upload-temp
```

使用同一个实验、已有 LoRA/检测头、Adam、学习率进度与 RNG，不重新初始化。最近已保存 epoch 的全量 Train 指标缺失时，先补测该 checkpoint，再继续剩余轮次；总训练预算仍是原配置的最多四轮。未提交的半轮按原版恢复规则重放。

如果原实验已完成，恢复入口不会追加轮数；请使用下面的补测命令。不要为启用监控而重训已经完成的实验。

后续新实验需要全量监控时，使用：

```bash
bash run_w2v_v317_monitored.sh \
  --data-run exp/w2v_v316_tfcl_20261009_020301_4d95 \
  --epochs 4 \
  --upload-temp
```

日常查看仍使用 `bash watch_w2v_v317.sh`。验证和 submission 命令完全相同，仍只使用原 V3.17 的本次 best/last。

## 全量的范围

每次复测覆盖全部官方 Train 来源，独立输出 `offline/en`、`offline/zh`、`online/en`、`online/zh`、`noisy_train/en`、`noisy_train/zh`：

- 每条官方 Offline 评估一次。
- 每条已验证官方 Online 评估一次；缺少 Online 时不拿 Offline 冒充，不计入 Online 分母。
- 每个原始 Train 来源生成一份固定 Noisy+RTC 音频，使用 Train 噪声库与训练机制；配方、随机种子及强度跨 epoch 固定。全量指每个来源都覆盖，不代表穷举无限种随机增强。

`noisy_train` 是训练增强分布的固定复测，不等于 Dev `seen`，更不是 Dev `heldout`。Dev 仍打印原有 Online/Seen/Heldout 中英文六组指标。en/zh 是数据集标签，不保证整条录音只含一种语言。

每组打印 fake/real 数量、各自 Recall、宏 F1、AP、AUC、EER 和类别平衡 CE。AP 的正类为 fake，受真假比例影响，不能跨不同样本比例直接等量比较。CSV 同时保存 fake CE、real CE 与完整分组指标。

终端还会保留旧版的 `Fixed Online replay` 小探针行，以免更改原训练文件；新增 `[Train FULL]` 表才是全量结果。自动拟合提示在全量结果可用时，优先比较同一定义的全量 Online Train/Dev，而不混用不同大小的探针。

## 损失、曲线和拟合状态

实验的 `diagnostics/` 下包含：

- `curves.html`：离线可打开的交互曲线，点击图例隐藏/显示系列，悬停读数。
- `loss_steps.csv`：每次更新的 TOTAL、CE、原始及加权 TFCL-time/CKA、最大单样本 CE、梯度范数、各参数组学习率与耗时。
- `group_metrics.csv`：每轮各组完整指标，明确区分全量 Train、旧样本探针与 Dev。
- `observations.json`、`summary.txt`：可机器读取与终端查看的诊断结果。
- `panel_epoch_*.json` 和小体积 `panel_scores_epoch_*.npz`：全量复测的指标、两个 logits、样本身份、配方摘要和模型权重指纹。

图中训练损失按每 100 次更新均值汇总；最大单样本 CE 单独绘图，保留异常峰值。原始更新均保存在 CSV。中断后重放产生的重复 cursor 会覆盖旧尝试的未提交尾部，不把两次尝试拼成一条虚假曲线。尚未提交的更新会明确标注。

训练 step CE 来自随机增强及训练模式；全量复测 CE 来自固定输入及 eval 模式。拟合趋势只使用后者进行同口径比较。仅一轮时报告差距并等待趋势；Train CE 下降而 Dev CE 上升时提示过拟合或域偏移风险，不宣称已经确定原因。

每轮先保存 checkpoint，再做全量评估。额外推理恢复模型模式和随机状态，不更新梯度或 Adam，不参与 best 选择。如果诊断失败，终端明确报告并保留 `diagnostics/failure_*.log`，训练仍可继续；缺失结果不会填为零。

## 已训练完成：补测 best/last 与下载

```bash
bash run_train_audit_w2v_v317.sh --checkpoint best --upload-temp
bash run_train_audit_w2v_v317.sh --checkpoint last --upload-temp
```

只补评估，完全不训练。best 与 last 若实际指向同一组权重，会复用匹配的复测结果。没有保留权重的历史中间轮次，不能事后补出真实的全量 Train 指标。

只下载当前已有曲线和指标，不额外推理：

```bash
bash monitor_w2v_v317.sh --upload-temp
```

报告 ZIP 包含 HTML、CSV、JSON 与日志，不包含 checkpoint、音频或大型特征。`--watch` 和 `--upload-temp` 分开使用，避免循环上传。

## 时间与磁盘

全量评估额外覆盖大约 `2×Offline数 + Online数` 条完整音频，耗时可能显著超过小探针，也可能超过训练本身；程序打印实际评估耗时。观察器本身只读小型日志/分数，不占用训练 GPU。

生成音频只在内存中流式处理，不落地波形或帧缓存；只增加曲线、表格与小体积 logits。全量评估不需要第二份模型或新的 checkpoint 副本。现有版本没有为随机增强分配所谓 Train Heldout，不会为凑表格创建错误指标。

## 验证

```bash
python -m unittest w2v_v317_monitor.test_monitor -v
```

测试覆盖分组口径、重放日志去重、全量及缺失 Online 覆盖、固定噪声重现、多进程一致性、异常后的 RNG/模式恢复、从旧 V3.17 保存点恢复并补测，以及带全量观察与原版训练的参数/Adam/RNG 一致性。CPU 集成使用真实 WAV、特征提取和小型 SSL；原生 Linux APM 用测试边界替代，A100 行为仍需服务器执行。
