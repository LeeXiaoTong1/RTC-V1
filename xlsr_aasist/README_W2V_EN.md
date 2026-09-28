# 一轮英文加权微调

本入口从原始 `w2v_rebuild_20260920_093548/stage3/best_model.pt` 开始，只调整真假各自内部的英文分类损失权重。原 best、此前 E1 候选、旧实验和音频缓存均保留。

英文占训练 real 约 20%、fake 约 24%，但固定 Dev 中分别约 34%、40%。因此将英文预期损失系数份额设为 real 内部 35%、fake 内部 40%。这些是初始设定，并非已验证的最优比例，也不是强制的实际梯度份额。各分支依据自身样本构成归一化，保持真假总成本；不叠加英文重采样。

## 固定配方

- ordinary、RTC 配对、noisy 配对分类损失使用组内英文权重；配对对比损失不改。
- ordinary 保持原完整遍历和逆频率类别权重；real cost 仍为 1.25；一致性损失关闭。
- 只训练后 4 层与分类头，encoder/head 学习率仍为 `1e-7 / 2e-6`。前 20 层和特征投影冻结。
- 固定 1 轮；沿用上一轮种子、batch、pair warmup、增强、截取、推理阈值和 Dev 条件。
- 原 noisy 库及既有 diverse 库均复用；额外库比例仍在首轮从 0 增至 20%，一轮平均约 10%。本轮没有增大困难噪声配额。
- 训练过程中复用已有 logits 统计四组 Train 指标；Dev 导出现有前向得到的分数，不额外执行模型。Train 指标来自增强后的训练样本，与固定 Dev 口径不同。

## 启动

使用已有 `sdd` 环境，无需安装包或重建缓存。以下默认 source 是上次有 E1 候选的 adaptation 实验，只读取它的路径、原始初始化 SHA256 和参考成绩，不使用它的模型初始化。

```bash
conda activate sdd
cd "$HOME/LXT/RTC-w2v-improved"
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
bash run_w2v_en.sh exp/w2v_adapt_20260927_225221_104a --upload-temp
```

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist"
tail -n 60 -f "$(cat exp/.latest_train_log)"
```

预览（不训练）：

```bash
python -u start_w2v_en.py --from-run exp/w2v_adapt_20260927_225221_104a
```

正式启动只创建新的 `exp/w2v_en_DATE_ID`，用原 SHA256 校验 best，之后在训练进程中进行一次数据检查和基线验证，不另跑重复 preflight。若需要读取其他参考运行，可加 `--reference-run PATH`；默认取 source 的 `completed.json` 所记录的 candidate epoch，绝不将最后一轮冒充 E1。参考缺失或固定 Dev 配置不一致时会明确标记不可比较，而不会阻挡这次训练。

## 结果与下载

报告包括原 best、本轮和可用的历史 E1 总体指标；新增的英文/中文分组指标和 AUC 只对本次实际测量结果报告，不为历史 E1 补造分组数值。评价仍使用固定阈值，不根据本次 Dev 重新找阈值。

训练通过原有目标保护规则才更新本次运行的 `best_model.pt`；`candidate_best.pt` 独立保留综合分数候选，`last.pt` 记录最后一步。原始 best 文件不会覆盖。只有 real recall 上升而 noisy fake 或 noisy F1 退步时，报告会保留这种取舍，不能认定为全面改善。

运行末尾打印 `REPORT`、`DOWNLOAD_ZIP` 和 `EN_TRAINING_RESULT`。默认将诊断 ZIP 复制到 `/home/ubuntu/LXT/temp`。使用 `--upload-temp` 才上传到 temp.sh，并打印 `TEMP_DOWNLOAD_URL=https://temp.sh/...`；上传失败保留本地 ZIP，训练结果不受影响。ZIP 仅含日志、配置、指标和逐样本分数，不包含音频、模型或优化器权重。

即使训练或保存 checkpoint 失败，入口也尽量导出已有诊断，并明确记录 `failed`，不能将部分报告误认成完整训练。中止或机器断电时，已有原始日志仍在新运行目录。

## 验证边界

CPU 测试验证配方继承、原 checkpoint 保护、字面路径参数、候选参考选择、失败报告以及上传失败处理；分组损失和模型测试由相应训练测试覆盖。代码检查不能保证精度改善，最终看这一轮固定 Dev 的英文真假区分、noisy 表现和中文退化情况。
