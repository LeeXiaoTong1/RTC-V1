# w2v-BERT 2.0：均衡采样、处理多样性与训练效率

本版本从已训练好的 Stage3 最佳权重开启独立微调实验。模型结构、24 层全量微调、输入裁剪、逻辑 batch=40、训练 BF16、验证 FP32/microbatch=4 和每轮完整验证保持原设置。三项训练改进可能改变效果，必须在 Dev 验证；这里不承诺分数或加速比例。

## 版本与回退

- 原代码：`8b99d367468963121334d5d6bc99b75cb2cf4e7a`。
- 远端回退分支：`backup/w2vbert2-score-91.6866-20260925`。
- 新分支：`w2vbert2-balanced-robust-fast`。
- 原 `v2-w2vbert2` 分支没有改动。
- Git 保存代码；模型权重仍在服务器原来的 `exp/.../stage3/best_model.pt`。新入口只读取它，拒绝在该目录及其子目录内写新实验。
- `--finetune_from` 只载入 Stage3 模型权重，新建 optimizer、warmup、调度和采样历史；这不是旧实验的精确 resume。原始权重 SHA256 和路径记录在新 config 中。
- 新实验先完整评估起点，保存 `baseline_dev.json` 和 epoch=0 的 `best_model.pt`。后续只有新的 Dev 综合指标改善才替换新实验的 best；原文件始终不修改。

## 服务器操作

### 优先清理已完成上一版的训练断点，保留特征缓存

上一版如果已经完成训练，回退、推理和继续微调只需保留它的 `stage3/best_model.pt` 及配置/日志。可清理该同一实验的 `stage1/{best_model.pt,last.pt}`、`stage2/{best_model.pt,last.pt}` 和 `stage3/last.pt`。这会失去旧实验各阶段的精确续训状态；正在训练的新实验仍保留自己的 best 和 last。不要对所有实验目录运行通配删除。

恢复脚本新增 `--prune-old-checkpoints --keep-feature-cache`。它从指定失败实验的配置定位原始 baseline，验证保留的 Stage3 最佳权重可读取，仅列出上述最多 5 个文件及总大小。执行前再检查清单是否变化，拒绝沿符号链接删除。其他实验、自定义权重、数据、特征缓存、原配置和日志保留。

```bash
conda activate sdd &&
cd "$HOME/LXT/RTC-w2v-improved" &&
git pull --ff-only &&
cd xlsr_aasist &&
python recover_w2v_storage.py \
  --failed-run exp/w2v_improved_20260925_134727 \
  --prune-old-checkpoints --keep-feature-cache
```

上面只预览，不删除、不启动训练。仅在尚未启动另一项恢复训练时执行下一段；它根据指定标志清理旧断点，复用全部缓存，检查空间，自动执行 preflight 和新实验训练。无需再使用 `--release-feature-cache`。若已启动前一种恢复任务，不要重复启动。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
LOG="$PWD/exp/recovery_keep_cache_$(date +%Y%m%d_%H%M%S).log" &&
{
  nohup python -u recover_w2v_storage.py \
    --failed-run exp/w2v_improved_20260925_134727 \
    --prune-old-checkpoints --keep-feature-cache --run > "$LOG" 2>&1 < /dev/null &
  printf 'LOG=%s\n' "$LOG"
}
```

```bash
tail -n 80 -f "$LOG"
```

只修改恢复工具及文档，不改变训练源文件，避免正在运行的实验因这次工具更新而改变源码指纹。定向检查：`python -m unittest test_w2v_recovery_cleanup w2v_rebuild.storage_tests -v`。

### 2026-09-25 存储不足修复

若第 1 轮在保存 `last.pt` 时出现 `PytorchStreamWriter failed writing file`，先检查磁盘及保存文件。仅有 `epoch=0, kind=weights` 的 `best_model.pt` 时，第 1 轮的更新没有可恢复的断点，不能宣称从第 2 轮精确续训。已生成的完整音频缓存可以复用。

修复增加模型实际大小对应的空间预检：计入完整 Adam 状态、保留中的 last 和其临时替换文件，以及新实验的最佳权重和余量。每轮开始及每次写检查点前再次检查。特征缓存遇到空间下限或 ENOSPC/EDQUOT 时直接计算相同的 FP32 输入，不继续写盘；已有有效缓存仍可读取。原 Linux 特征缓存键保持兼容。每轮 Dev 结果会先打印并写入 `epoch_NNN_evaluation.json`，再保存权重；文件中的 `checkpoint_saved` 表示本轮完整断点是否成功落盘。空间检查无法阻止其他进程同时占用磁盘，也不代替平台配额检查。

针对当前独立目录 `~/LXT/RTC-w2v-improved/xlsr_aasist`，提供 `recover_w2v_storage.py`。它读取指定失败实验的配置，复用原始 baseline、所有数据/缓存路径、学习率和训练参数，创建新实验。恢复实验关闭输入特征缓存以控制占用；不会改变模型结构、增强、训练步数或精度，但会失去这项缓存带来的加速。其他效率改进保留。

以下命令均为 ASCII，每段少于 2000 字符。在 `sdd` 环境执行：

```bash
conda activate sdd &&
cd "$HOME/LXT/RTC-w2v-improved" &&
git pull --ff-only &&
cd xlsr_aasist &&
python recover_w2v_storage.py --failed-run exp/w2v_improved_20260925_134727
```

上面仅检查并预览。确认打印的失败实验和原始 baseline 路径正确后，下面的 `--release-feature-cache` **明确授权删除此 checkout 的 `data/w2v_feature_cache` 中可重新计算的、符合哈希目录格式的 `.npy` 文件**。脚本不遍历符号链接，不删除音频、检查点或原始实验。如果已有完整 `last.pt` 或 best 的轮次大于 0，会拒绝这种重启方式。释放后仍不满足检查点预算时，会在训练前停止并要求继续释放空间或扩容。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
LOG="$PWD/exp/storage_recovery_$(date +%Y%m%d_%H%M%S).log" &&
{
  nohup python -u recover_w2v_storage.py \
    --failed-run exp/w2v_improved_20260925_134727 \
    --release-feature-cache --run > "$LOG" 2>&1 < /dev/null &
  printf 'LOG=%s\n' "$LOG"
}
```

```bash
tail -n 80 -f "$LOG"
```

启动段只执行一次。脚本也使用 Linux 文件锁阻止同一 checkout 内重复启动恢复任务。日志打印新的 `RUN=...`，随后自动执行 preflight 和训练，不再生成 54 GB 音频缓存。原始回退分支不变。新的源码保护逻辑不能用于对旧源码的 `last.pt` 声称精确 resume；有完整旧断点的其他实验应保留对应源码。

回归检查：`python -m unittest w2v_rebuild.tests w2v_rebuild.improvement_tests w2v_rebuild.storage_tests -v`。

### 新建实验的通用流程（Git 工作目录干净时）

先进入原来的 `sdd` 环境。以下路径取自此前训练记录；如果实际最佳权重不在此位置，只修改 `BASELINE`，不要自动选择“最新”目录。已使用独立工作目录的服务器，沿用上面的独立路径，不再切换原目录。

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC
git fetch origin
git switch w2vbert2-balanced-robust-fast
git pull --ff-only
cd xlsr_aasist

BASELINE=/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt
RUN="$PWD/exp/w2v_improved_$(date +%Y%m%d_%H%M%S)"
test -f "$BASELINE"
```

`git switch` 会自动跟踪同名远端分支。若服务器有阻止切换的本地改动，先单独保存这些改动，不要使用强制 reset。

只增加离线 WebRTC 处理依赖，不升级原环境的 torch、transformers、CUDA、librosa 等：

```bash
python -m pip install -r requirements_rtc_diverse.txt
```

`webrtc-audio-processing==0.1.3` 在 Linux 通常需要 C++ 编译器、SWIG 和与当前 Python 对应的开发头文件。缺少时先在平台安装这些构建依赖；例如 Ubuntu 系统 Python 可使用 `sudo apt-get install build-essential swig python3-dev`，Conda Python 则应使用该环境对应的头文件。安装失败时生成器会明确报错，不会偷偷跳过 WebRTC。

先生成极小 smoke 缓存，实际经过 FFmpeg/WebRTC；通过后生成完整的新缓存：

```bash
bash run_prepare_w2v_improved.sh smoke
bash run_prepare_w2v_improved.sh full
```

默认新缓存为 `dataset/rtc_noisy_improved_v1`，包含 `train_g1`、`dev_seen`、`dev_heldout`。原来的 `rtc_noisy_cache_v2` 不修改。可通过 `RTC_IMPROVED_CACHE_ROOT` 改位置。生成脚本也支持此前的 `DATASET_ROOT`、`NOISE_ROOT`、`RTC_B_NOISE_MANIFEST`、`RTC_DEV_NOISE_MANIFEST`、`CACHE_WORKERS` 和 `FFMPEG_BIN`。缓存配置不同会拒绝覆盖，使用新目录即可；相同配置可继续完成尚未生成的文件。

完整缓存就绪后，做一次真实训练 batch 检查，再启动训练：

```bash
bash run_w2v_improved.sh "$BASELINE" "$RUN" --preflight --profile_steps 1
bash run_w2v_improved.sh "$BASELINE" "$RUN"
```

`--preflight` 检查全 24 层和 feature projection 确实更新、验证无跨样本依赖，写 `preflight.json`，不保存训练后的模型。训练会重新从原始 BASELINE 初始化，不会继承这一步测试更新。

只检查数据、不加载模型时可单独使用 `--check_data`；preflight 已包含这些数据检查，无需重复运行。

默认微调 8 轮，encoder LR=2e-7、head LR=5e-6，沿用已有 warmup/plateau 机制；这是较保守的新实验起点。可用 `EPOCHS`、`ENCODER_LR`、`HEAD_LR`、`CONSISTENCY_WEIGHT` 设置。新入口依然使用约 3158 次更新/轮（若普通训练集为 75785 条），不通过减少训练步数伪造提速。

需要继续这个新实验时，保持原 `RUN`、训练参数和代码版本：

```bash
bash run_w2v_improved.sh "$BASELINE" "$RUN" --resume "$RUN/stage3/last.pt"
```

不要用新代码 `--resume` 原来的旧实验。精确恢复会校验代码、数据指纹、采样策略和训练配置，防止不知情地改变实验。

## 三项改进

1. ordinary 从随机取样+逆频率 CE 改为每批 12 fake + 12 real、等权 CE。RTC/noisy 各 4 pairs 不变。按每类连续循环打散，跨轮保留覆盖；重复抽取拥有不同 draw ID，增强可复现且不会每次重复同一波形。
2. noisy 同时使用原缓存和新缓存。新增真实本地 WebRTC NS/AGC + Opus、温和合成房间响应和间歇性环境噪声；原 FFmpeg 处理链仍保留。real/fake 使用相同生成规则，不添加外部语音或重新合成 fake。新的 mixed 轮换从第一轮就抽取新旧库，每个源在完整周期遍历 bank×SNR 的所有视图组合。
3. noisy pair 增加 `0.02 × KL(stopgrad(p_reference) || p_processed)`，沿用两轮 warmup。仅 reference 预测正确且置信度>=0.8 时启用，对两类规则相同。保留原分类 CE 和 InfoNCE，复用现有 logits，没有额外的大模型前向。

Dev-seen 使用 FFmpeg/WebRTC 混合；Dev-heldout 使用训练中未出现的 FFmpeg `anlmdn` 处理算法。两者使用同一 Dev 源、加性噪声混合和 SNR，算法划分、噪声录音隔离均有检查。新的模拟 Dev 与旧分数不是同一评估条件；用 `baseline_dev.json` 在新条件下公平比较。它仍是本地近似，不代表已经验证所有未知平台。

新增条件均在本地离线处理，不把官方数据上传到真实 RTC 平台。参考官方允许范围：<https://www.junxue.tech/rtc-sdd-challenge/>。WebRTC API 来源：<https://github.com/xiongyihui/python-webrtc-audio-processing>。

## 效率优化和边界

- 固定 pair/noisy/Dev 缓存官方 FP32 输入特征。缓存以实际裁剪波形内容、提取器配置、实现和依赖版本区分；不是 SSL 隐状态缓存。动态 ordinary 增强不缓存。首轮填充会更慢，后续才获益；PCM 仍会读取，用于内容校验。
- 默认 `FEATURE_CACHE=dataset/w2v_feature_cache`，建议放本地 SSD。每个独立固定视图约 0.13 MB，十万视图约 13 GB，具体取决于特征帧数。另有完整 PCM 新噪声库的空间需求，生成器会打印估算。用 `FEATURE_CACHE` 改到有足够空间的盘。
- 解码/重采样后的噪声采用有界缓存，默认每个普通数据 worker 128 MiB；`RTC_NOISE_CACHE_MB=0` 可关闭。只复用确定性结果，不改变混噪 RNG 或输出。语音活动 mask 不再重复计算。
- mask 和配对标签一次在 CPU 检查；损失标量集中复制、训练指标在设备上累计；保留非有限值及梯度检查。拼接已锁页的批次时保留 pinned memory。Adam、训练/验证精度、模型计算目标均未因这些工程优化改变。
- 默认不增大 microbatch、不降低验证精度、不减少验证样本、不关闭 checkpointing、不冻结层。`--eval_microbatch` 是显式实验开关，只有与参考结果核对后再使用。
- 每轮 `metrics.jsonl` 新增 `timing`：训练、主线程等数据、验证、best/last 保存耗时；`seconds` 现在包含 last 保存。`data_wait_host_seconds` 是主线程等待时间，不能直接等同于 GPU 空闲时间。
- `--profile_steps 20` 会用 CUDA events 对前 20 步分段计时，同时引入额外同步，只用于诊断。总耗时还应以正常训练为准。

训练结果仍需实际 Dev 判断。不要根据 Progress/Eval 样本分析、调参或选择模型。

## 测试与回退命令

```bash
python -m unittest w2v_rebuild.tests w2v_rebuild.improvement_tests -v
python check_rtc_noisy_v2.py --logic
```

`RTC_TEST_FFMPEG=ffmpeg` 可启用测试中的原生 FFmpeg 项；WebRTC 的原生处理由前面的 smoke 生成完整检查。CPU 单元测试中的 WebRTC adapter 测试使用替身，不能替代服务器原生检查。

另有 `RTC_TEST_FFMPEG=ffmpeg python -m unittest w2v_rebuild.integration_tests -v`，用合成波形贯通缓存生成、角色/噪声隔离检查、12:12 采样及官方特征缓存。该集成测试同样替代了 WebRTC 原生模块，避免把测试替身误报为真实 DSP 验证。

代码回退不删除任何权重或训练结果：

```bash
cd /home/ubuntu/LXT/RTC
git fetch origin
git switch --detach origin/backup/w2vbert2-score-91.6866-20260925
```

此时继续使用原 `exp/w2v_rebuild_20260920_093548/stage3/best_model.pt`。返回新版：

```bash
git switch w2vbert2-balanced-robust-fast
```
