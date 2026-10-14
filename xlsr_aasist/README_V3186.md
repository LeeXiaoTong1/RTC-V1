# V3.18.6：恢复重叠窗口与注意力统计汇聚，保留双分支音素一致性

此修复从**已完成的 V3.18.5 best 或 V3.18.4 AUG5 ASP best**完整继承模型，恢复父模型的重叠窗口、贡献权重、注意力统计汇聚和完整 320 维分类器；保留全层 Omni 融合、AASIST、V3.18.5 数据扩充规则、C3 分类及双分支音素约束。公开版本仍是 **V3.18.6**，内部 recipe 改为 `v3186_window_asp_dual_pcl_v2`，防止误接旧整句模型的 checkpoint。旧版本源码与 checkpoint 不改写。

父版本兼容修复：服务器上实际存在的是 `v3184_aug5_asp_v1`，此前只接受版本号 3.18.5，导致已封存的 AUG5 训练也被拒绝。两者的模型和 ASP 模块代码相同，现在分别通过各自原版本的加载器验证 checkpoint 哈希、配置身份、schema 与 best 记录，再执行严格的张量迁移；不修改版本号绕过校验。普通 V3.18.4、R2、未完成运行仍不可替代。即使继承 AUG5 权重，新训练也使用 **5/5/1/3**，不继承旧版四组统一 ×5。

## 训练实际做什么

每次更新仍是独立的 16 条官方 Online，以及 16 个 Offline 来源各生成的 Noisy A/B，共 48 条分类音频。两条流配额都为 EN-fake / EN-real / ZH-fake / ZH-real = **4/3/5/4**；四组监督各 25%，Online/Noisy 各 50%。Noisy 使用原 C3：`0.75 * mean(CE_A, CE_B) + 0.25 * max(CE_A, CE_B)`。原始 Offline、模拟干净 RTC 均不参与分类。

一个 Offline 来源先应用一次基础增强，再生成两份独立的 Noisy RTC。该来源本次选择 A 或 B（随同一来源访问次数轮换），额外生成同基础增强、同公共增益、同 RTC 配方的干净 RTC：

```text
official Online ----------------- phoneme PCL ----------------- clean RTC
                                                                  |
                                                              phoneme PCL
                                                                  |
augmented Offline -> environmental perturbation -> same RTC -> Noisy A or B
```

- 噪声分支：每个 Offline 来源的 `clean RTC <-> matched Noisy RTC`。
- 平台分支：每组至多两个已有官方配对的来源，`official Online <-> clean RTC`。不存在官方配对时保留原分类来源与噪声分支，不用其它来源替换。
- 参照只编码到全层融合后的 128 维帧特征，不通过 AASIST、不更新其 BN、不增加 CE。
- 按对应音素片段平均特征，L2 归一化后计算 `1 - cosine`，两端均有梯度；不是教师蒸馏，也不使用跨录音负例。
- 损失为 `L_C3 + ramp * (0.15 * L_platform + 0.15 * L_noise)`。先在来源内部平均音素，再按预定来源配额使四组各占 25%；缺失/无效配对保留零权重，不将其它有效来源重新放大。
- 两个辅助系数在前 300 次更新渐入，无整轮头部预热。默认 6 轮联合训练。

这是对音素一致性思路的**接收端适配**，不是原 benchmark 的逐项复现。音素由固定版本 `facebook/wav2vec2-xlsr-53-espeak-cv-ft` 产生，区间是 CTC 对齐支持附近推算的近似值，不是真实人工边界。中英文的有效覆盖分别记录。它能否改善未知平台/噪声，需要 GPU 对照验证，不能保证 Progress 分数。

## 增强与对齐复核

固定版本预算继承 V3.18.5：**EN-fake ×5、EN-real ×5、ZH-fake ×1、ZH-real ×3**，Online 和 Offline 均适用。×5 包括原始版本，不是原始再加五份；增强按需确定性生成，不把 WAV 数据集复制五份。ZH-fake 没有固定扩充，但仍有共同的轻量动态增强。原始/增强曝光保留 20%/80%。新视图不等于新说话人或独立录音。

Offline 的增强始终在环境噪声与 RTC 之前；分类端只收到最终 Noisy。噪声库和 RTC 策略继承 V3.18.5：沿用 V3.18.4 的五类录制场景加回声模拟、three-profile RTC，以及 Noisy B 的受控压力增强。

音素仅对原始 Train Offline/Online 准备一次。训练时用确切变速比例、实际裁剪偏移映射区间，再估计模拟 RTC 的波形延迟；检查局部删失/无信号。官方配对按音素序列单调对应，重复音素的歧义匹配屏蔽。不会把相同帧下标当作对齐，也不要求检测器先高置信分对才提供一致性监督。相同 RTC 配方只保证参数一致，不声称输入自适应的降噪/AGC 内部状态相同。

启动时生成 `augmentation_budget_audit.json`，按服务器实际 Train 清单计算所有轮次的来源及增强版本覆盖，区分 `all_expanded_views_covered`、`all_source_recordings_seen`、原始槽覆盖及重复次数。每轮另有 `coverage_epoch_N.json` 和实际 loss/采样统计。已有数量审计在 2417 步×6轮下覆盖两类 EN 的全部五版本和 ZH-real 的全部三版本；这不是服务器音频字节的远程核验。

## 模型迁移与参数

完整录音只经过一次 Omni 编码与多层融合，再对融合帧进行重叠分窗。窗口长度、步长、窗口物理批次完整继承父配置（当前目标父模型为 128 帧窗口、64 帧步长）。每窗经 AASIST 得到 160 维表示，结合窗口覆盖与贡献权重、可学习注意力计算整句的加权均值和加权标准差，拼成 320 维后分类。PCL 仍作用于窗口分类前的融合帧特征；干净 RTC 和官方配对参照不进入窗口分类头。

父版本全部可训练张量及 BN 缓冲严格加载，包含 LoRA、全层融合、AASIST、窗口贡献模块、注意力模块和完整 320 维分类器。**不截列、不重置后端、不丢弃标准差特征。**优化器新建；best 仍只从 V3.18.6 已完成 epoch 选择，不暗中回退到父模型。

开始优化之前必须对新模型执行一次完整固定 Dev 推理，按父模型的物理推理批次复现已保存的 best 原始 logits，核对行身份、顺序和固定验证划分；同时核对全 Dev、选择集、校准划分的 raw Weighted。logits 容差为 `atol=0.003, rtol=0.001`，Weighted 绝对差不超过 `0.001`（指标内部比例口径）。失败时保存审计并停止，不能继续更新。实际新推理与父缓存分别保存为 `parent_baseline_scores.npz`、`parent_reference_scores.npz`，均进入 report；不是只读父缓存就宣称复现。

| 模块 | 最高学习率 |
|---|---:|
| 最后 16 层 LoRA | 1e-5 |
| AASIST 主体 | 3e-5 |
| 全层投影与门控 | 1e-4 |
| 层混合权重 | 3e-4 |
| 完整 320 维分类器 | 1e-4 |
| 窗口贡献模块 | 3e-5 |
| 注意力统计汇聚模块 | 3e-5 |

LoRA 学习率前 200 更新渐入，各模块沿用余弦衰减、末端比例 0.1。BN 只在分类窗口上更新；AASIST 窗口按实际相同长度分组，避免填充帧污染 BN/图结构。

## 速度与存储

- 每条输入在一次逻辑更新中只编码一次；Noisy 特征同时用于 CE 和 PCL。参照在微批次内反传后释放；无效配对不额外编码。
- 最多 48 条分类 + 16 条干净参照 + 8 条官方参照。辅助监督仍有真实计算成本，不能宣称与旧版一样快。
- 不缓存 60 层全量特征，不保存增强 WAV；音素准备只存 JSON 元数据。模型下载、准备环境、pip/HF/TMP 均在数据盘。
- 音素片段用前缀和汇聚；有效性检查每对录音只各降采样一次，局部只计算所需相关延迟；完整训练预算审计和 checkpoint 快照不重复生成。
- 默认 `microbatch=8, frame-budget=12000`，包含辅助输入；AASIST 使用父模型的 `window_batch`，图规模按窗口限制，不再套用旧整句图的平方预算。超过默认 3000 帧（约一分钟）的录音明确报错，不偷偷裁剪；应根据编码器显存显式调整 `--whole-max-frames`。窗口不会重复调用 Omni，也不落盘缓存完整特征。
- 启动前扫描 Train/Dev 的长度信息，并在指定自动提交时读取目标音频文件头；考虑变速后的最长输入与整组辅助输入预算。超限会在训练前指出样本和需调整的资源参数，避免训完后才发现不能导出。预检不读取评测标签或用于调参。
- 每 100 更新抽样审计一次：辅助/分类梯度只测首个有效辅助微批次的共享投影；参数变化最多取每个 optimizer 参数组前 1024 个元素。不是完整模型梯度冲突估计，也没有每步额外三次大模型反传。
- `performance_epoch_*.json` 记录训练计算、数据等待、各评估阶段及保存耗时；峰值显存包括训练与评估。没有本地 A100 吞吐实测，不预设“显存用满就是更快”。

## 服务器部署

先确保数据盘已挂到 `/home/ubuntu/LXT`，父版本已训练完成。若旧 V3.18.6 正在运行，先按下 `Ctrl+C` 退出查看器（不会停止训练），然后在更新源码前停止它。以下脚本先 fetch，不改变工作区；从已发布分支取出独立停止工具，校验本项目的 supervisor、进程会话、日志和 run 身份后发送 TERM。它不使用宽泛的 pkill，不自动 KILL；正在计算的未完成轮不保存，已有完整 checkpoint 保留。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git fetch origin w2vbert2-balanced-robust-fast &&
mkdir -p /home/ubuntu/LXT/temp &&
git show origin/w2vbert2-balanced-robust-fast:xlsr_aasist/w2v_v3186/stop.py > /home/ubuntu/LXT/temp/stop_v3186.py &&
python /home/ubuntu/LXT/temp/stop_v3186.py \
  --root /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist \
  --run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v3186_20261014_144952_c2b6
```

确认工具成功返回停止状态，再更新源码并启动新运行；不要恢复失败的旧整句运行。

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git switch w2vbert2-balanced-robust-fast &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
source "$(conda info --base)/etc/profile.d/conda.sh" &&
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318 &&
cd xlsr_aasist &&
PARENT_RUN=exp/w2v_v3184_aug5_20261013_105027_64d9_sealed_e4_6bf9956295 &&
python -m w2v_v3186.parent_discovery --parent-run "$PARENT_RUN"
```

上述路径为服务器截图中已有的第 4 轮封存快照，使用其中记录的 best（不一定是第 4 轮），不需要再次封存或复制 checkpoint。不要求 `exp/.latest_v3185_run` 存在；也不要修改旧 config/completed 文件中的版本号。

其它部署可使用 `PARENT_RUN=$(python -m w2v_v3186.parent_discovery)`：若有有效的 latest 记录则复用；否则在当前项目以及同级 `RTC` / `RTC-w2v-improved` 的 `xlsr_aasist/exp` 下读取实际 recipe 与完成状态。只有一个兼容候选时自动选用；多个候选会要求明确 `--parent-run`，不会按日期或分数挑选。找不到时可运行 `python -m w2v_v3186.parent_discovery --list` 查看记录；目录移动过可加 `--search-root EXP_DIR`。

已完成的 V3.18.6 音素元数据与 V3.18.4 噪声库直接复用，无需重跑准备。以下命令仅适用于尚未准备音素的新环境：

```bash
bash prepare_phonemes_v3186.sh \
  --parent-run "$PARENT_RUN" \
  --hf-endpoint https://hf-mirror.com
```

脚本在数据盘建立独立音素环境，复用原环境 Torch/NumPy，仅在独立环境安装固定 HF 依赖，校验不改变训练环境。首次在线下载固定音素模型；中断后重跑会复用已完成记录。输出 `PHONEME_MANIFEST=...` 才代表缓存可用于训练。准备不读取 Progress/Eval。

若此前报 `We couldn't connect to https://huggingface.co`，失败发生在音素模型下载，六轮检测器训练尚未启动。上述命令优先使用 HF Mirror，失败后尝试官方地址；`--hf-endpoint auto` 则优先采用 `HF_ENDPOINT`（未设置时为官方地址）。镜像能否访问取决于服务器网络，不保证能绕过所有网络故障。每个文件每个地址默认最多尝试两次，可用 `--download-retries 3` 调整。

下载固定提交 `2c733782da5604684829819a5eb744c193fe9398` 的六个必要文件，不调用完整仓库列表 API，不发送登录 token；HF 缓存负责断点续传。权重约 1.26 GB，SHA256 固定为 `04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551`，依据[官方权重页面](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/blob/2c733782da5604684829819a5eb744c193fe9398/pytorch_model.bin)。JSON 文件验证格式和关键模型配置，并在数据盘 `hf-phonemes/v3186_model_download.json` 记录下载来源、字节数和哈希；JSON 哈希是本地来源记录，不声称是独立固定的上游校验值。

权重仍保存在 `/home/ubuntu/LXT/v318-runtime/hf-phonemes`，不额外复制一套模型，不重装已完成的独立音素环境，不清空下载/音素缓存。打印 `PHONEME_MODEL_READY=...` 后，特征提取器、分词器与模型只从已验证的本地快照加载，禁止再联网寻找可选文件。完整缓存可使用 `--local-files-only`。若两个地址都无法访问，程序保留部分下载并明确退出；修好 HTTPS/代理连接后重跑同一命令。

开始六轮，并自动导出本次 best、上传两个压缩包：

```bash
bash run_w2v_v3186.sh \
  --parent-run "$PARENT_RUN" \
  --epochs 6 \
  --submit-best \
  --upload-submission \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

训练启动前自动执行小型原生 fairseq2/CUDA 接口验收，检查全层融合、PCL 反传及重叠窗口 ASP；随后实际加载父 best 并进行上述完整 Dev 复现。完整 Dev 需要一次验证耗时，此时还没开始 epoch 1。终端保留固定屏幕进度和每轮评估，详细日志留在 `exp/v3186_*.log`。关闭查看器不会停止后台训练：

```bash
bash watch_w2v_v3186.sh
bash watch_w2v_v3186.sh --panel train --once
bash watch_w2v_v3186.sh --panel scenes --once
```

完成后获取两个独立链接；上传失败只重试交付，不重新训练或推理：

```bash
RUN=$(cat exp/.latest_v3186_run)
cat "$RUN/download_links.txt"
bash upload_w2v_v3186.sh --run "$RUN"
```

report ZIP 包含指标、对齐覆盖、采样、性能与配置；不包含录音、模型权重或大特征。只有在训练、导出、上传实际成功后才有 temp.sh 链接。

如果启动时没有指定自动提交，可在新运行训练完成后手动导出 **best** 并上传 submission 与 report：

```bash
RUN=$(cat exp/.latest_v3186_run)
bash export_w2v_v3186.sh \
  --run "$RUN" \
  --checkpoint best \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --upload-temp
```

若训练因关机/中断停止，从最后完整保存的 epoch 恢复，未完成轮会重放：

```bash
RUN=$(cat exp/.latest_v3186_run)
bash run_w2v_v3186.sh --resume "$RUN"
```

恢复仅适用于本次窗口修复之后的新 recipe，沿用已保存训练与交付配置，不接受改学习率/轮数/损失。旧 `v3186_whole_dual_pcl_v1` checkpoint 不能 `--resume` 到新结构。要验证 PCL 增益，另建相同起点的 CE-only 运行，在新启动命令加入 `--consistency none`；它完全省去参照准备和计算。两者分类抽样/波形一致，但因辅助输入影响微批次分组，BN 物理批次未保证逐步相同，因此不称为逐位严格控制实验。

## 验收边界

CPU 测试覆盖实际 AASIST、双 PCL 梯度、分类监督质量守恒、无参照 CE、每输入一次编码、版本覆盖、时间映射、缓存完整性、父模型迁移、epoch 断点恢复、best 导出及上传失败重试。没有运行完整 Omni-3B GPU 训练，也没有新的 Progress 结果。

窗口恢复测试验证：完整父状态继承后的多窗口输出一致、320 维分类器保留、恢复模块获得梯度、零更新 Dev 真实推理及不一致拒绝、两个兼容父版本的双 PCL/断点恢复/best 导出、进程停止身份校验、上传失败重试。端到端测试使用小型替代编码器和真实 AASIST，不将它称为完整 Omni GPU 验证。

本次在与远端 `e97672b` 的 1085 个文件逐一核对过的干净发布副本中验证：V3.18.6 模块的 106 项 CPU 测试、终端查看器的 6 项测试全部通过；停止、训练、导出 Bash 入口语法检查通过。未修改本次清单外的源码，未在目标 GPU 上训练。

下载兼容修复的验收覆盖地址切换、部分下载保留、缓存损坏检测与单文件修复、严格离线加载、CLI 参数转发及简短错误退出。网络测试使用受控替身，未在目标服务器完成真实模型下载或 GPU 验证。

下一步判断依据应是未见来源/噪声/RTC 下的 AUC/EER、EN/ZH 的 real/fake recall、固定 fake recall 的 real recall，以及两分支有效覆盖和梯度方向。不能仅凭辅助 loss 变小或本地 Weighted 上升认定泛化改善；这一配对拓扑与旧 V3.19 TFCL 有相似处，音素级目标是否更有效仍需实验回答。
