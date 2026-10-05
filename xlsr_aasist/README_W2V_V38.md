# V3.8：保留原判别器，验证小幅修正是否真的改善判别

起点是已完成 V3.7 运行所保留的原 best。对当前运行 `w2v_v37_20261005_001824_8787`，它仍是此前提交过的 V3.3 **control/baseline** checkpoint，不是 V3.5，也不是未通过验收的 V3.7 language_debias。代码核对 checkpoint、V3.7 patch、缓存、预处理及特征生成代码的身份。

这是一版可运行的完整实验，不保证官方 Weighted 超过 97。V3.7 救回了一些 EN-real，同时增加了假音漏检；因此，本版要回答的是：能否纠正部分错误而保住已有的真假边界，以及语言信息是否带来了简单校准以外的收益。

## 启动

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
cd xlsr_aasist &&
bash setup_w2v_v38.sh &&
bash run_w2v_v38.sh \
  --source-run exp/w2v_v37_20261005_001824_8787 \
  --upload-temp
```

`--source-run` 必须指向已完成、最终选择 baseline 的 V3.7 运行。省略时读取 `exp/.latest_v37_run`。旧运行、原 best、V3.7 的 detector/teacher 特征目录需要保留。若缺少完整缓存，本版直接说明缺少什么，不会悄悄重做数十 GB 的音频。

默认使用可用 GPU 训练小模块；无 GPU 时使用 CPU。可以指定 `--device cpu`。服务器原有 CUDA PyTorch 保持不变，setup 不安装包、不下载教师。BLAS 和 CPU 数学线程固定为 1；优化使用 PyTorch AdamW，不调用曾崩溃的 SciPy L-BFGS/OpenBLAS 求解流程。

```bash
bash watch_w2v_v38.sh
bash show_w2v_v38.sh
```

结果会保留在终端，Ctrl+C 只关查看器。后台父进程另行记录退出码和信号，例如 `SIGSEGV`；不会把“没有新的输出”当作仍在训练。退出记录在对应日志旁的 `.exit.json`。

中断后用新 V3.8 路径恢复：

```bash
bash run_w2v_v38.sh --resume exp/w2v_v38_实际运行目录 --upload-temp
```

恢复复用已经原子提交的拟合阶段，重做尚未完成的小模块阶段；不承诺从中断的某个 minibatch 精确续接。代码、数据身份、配置或设备改变时拒绝混用旧阶段。需要停止时：

```bash
python -m w2v_v38.stop --apply
```

## 实际训练了什么

1. **大模型冻结。** w2v-BERT 2.0、MultiConv 和原二分类层均不更新。直接读取完整 Train 的既有 512 维向量，以及已缓存的 256 维教师向量。训练阶段不再执行编码器、不生成 WAV、不复制特征。
2. **按来源划分 Train。** 同一原始录音及 Offline、Online、noisy_a、noisy_b 都留在同一侧。80% 拟合，20% 选小模块的轮数与正则强度，再用选定配置在全部 Train 上重训。四个语言/真假组预算均衡；每个来源普通条件总预算 50%，两种 noisy 各 25%，避免有多个版本的来源获得额外总权重。
3. **先检查语言信息。** 教师目标先减去 Train 真音均值，再按总变化能量缩放。小分支只用 Train 真音拟合这部分变化；真音上的 EN/ZH 线性探针也只在 Train 拟合。报告中心化重建相对常数均值的 R²、教师/学生的语言探针准确率及分条件结果，不把原始 cosine 0.90 当作 90% 语言准确率。
4. **运行三个对照明确的候选。** `calibration` 只学习一个统一正比例缩放和偏移，保持排序不变；`residual_control` 不使用语言分支；`language_residual` 使用冻结的内部语言分支与原检测特征共同决定修正。两个非线性候选可训练参数数量近似相同，避免把更多容量直接说成语言收益。小分类器的最后一层从零初始化，因此开始时完整保留原输出。
5. **保住已有的正确判断。** 两个非线性分支只给 fake-minus-real margin 加一个绝对值不超过 2 的修正；原 logits 始终保留。分类监督之外，惩罚大修正，并额外惩罚削弱原来判对的 fake 的 margin；原来判对的 real 也有反向保护。它是软训练约束；最终仍要看实际假音召回，不保证所有样本都不退化。统一校准的比例项不受上述总修正幅度限制，但同样参加这些训练保护和 Dev 验收。
6. **最终才看固定 Dev。** 三个候选的轮数/强度只由 Train 决定；随后一次性比较原模型与候选，不按 Dev 反复找阈值或调强度。Clean 只统计 Online；Noisy 为 Seen/Heldout 的平均。输出始终是 `P(fake)`，阈值仍为 0.5。

语言学生默认需要比常数均值多解释至少 2% 的变化，Train 来源留出上的语言探针准确率至少 65%，教师探针至少 70%。不合格就跳过语言候选，但继续两个非语言对照。阈值是预先固定的工程验收条件，不是论文证明的最佳值。

需要注意：原编码器以前已经见过这些 Train。这个来源留出集可以用于小模块选参，不能冒充完全未见来源的泛化测试。学生向量也可能含有说话人、来源和处理条件；即使语言候选胜出，也不能据此证明已经分离出纯粹的语言因果因素。

## 如何选择提交模型

统一保护条件：相对原模型，本地 Weighted 至少提升 0.1 个百分点；Noisy 不下降；Clean 最多下降 0.1 个百分点。对 Online/Seen/Heldout 的 **EN 和 ZH** 分别检查 fake/real recall、AUC，以及固定 fake recall 下的 real recall；不会仅盯着 EN-real。

默认每个组 fake recall 最多下降 0.2 个百分点、real recall 最多下降 0.5 个百分点、AUC 最多下降 0.05 个百分点，99% fake recall 下的 real recall 最多下降 0.5 个百分点。

语言候选还必须：

- 学生信息检查通过，EN-real 三个条件平均至少提升 0.5 个百分点。
- Weighted 比统一校准和普通修正分别高至少 0.05 个百分点。
- 在固定 99% fake recall 的诊断下，EN-noisy real recall 平均至少提高 0.2 个百分点，证明收益不只是统一平移分数。

固定 fake recall 对应的阈值只用于 ROC 诊断，**绝不会拿去导出提交**。通过条件的候选按 Weighted 选优，否则保存精确的原模型回退。`calibration` 或 `residual_control` 获胜时，报告明确写出语言修正没有被选中。

报告包括各组原始指标、逐样本分数、救回多少 real、增加多少 fake 漏检，以及每个拒绝原因。真实服务器训练和官方 Progress/Eval 成绩才能验证性能，测试通过只证明实现链路和保护机制有效。

## 时间、磁盘和导出

本版避免了整段语音编码和反向传播，计算集中在少量线性层。缓存常驻 GPU 后按批拟合，语言上下文不在每个优化 step 重跑。常规拟合最多 24 轮、学生最多 30 轮，这些是**缓存小模块轮数**，不能与此前每轮数小时的大模型训练相比；实际时间以终端为准。

不产生新的音频或特征缓存，也不另存一份大 best。新增的是小模块、短拟合记录和 Dev 分数，启动要求 128 MiB 空闲余量。原 best 加 `best_patch.pt` 才是完整部署依赖；不能删除原 best，也不要删除恢复所引用的 V3.7 缓存。本版不会自动删除旧实验或正在使用的数据。

```bash
bash run_eval_w2v_v38.sh --upload-temp
```

也可显式设置 `V38_RUN=exp/w2v_v38_实际运行目录`。导出用一个冻结检测器加选中的内部小模块；不需要外部语言教师、不读取语言标签，不融合多个模型的分数。`submission_meta.json` 记录实际选中方案、原 best 和 patch 的 SHA256，避免把回退模型误称为新算法成果。
