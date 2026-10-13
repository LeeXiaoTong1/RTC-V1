# V3.18.5：定向基础增强与均衡监督

本版本沿用 V3.18.4 AUG5 的同一个已完成／已封存 R2 best，重新建立独立实验。保留 Omni-3B 全层融合、AASIST、注意力统计池化、C3 分类目标、分模块学习率，以及 1 轮预热＋5 轮联合训练。不会继承正在训练的 V3.18.4 权重或优化器，也不会停止它。

## 默认数据规则

| 分组 | Online 基础版本预算 | Offline 基础版本预算 |
|---|---|---|
| EN-fake | 原始＋4 个固定增强版本，共 5 倍 | 原始＋4 个固定增强版本，共 5 倍 |
| EN-real | 原始＋4 个固定增强版本，共 5 倍 | 原始＋4 个固定增强版本，共 5 倍 |
| ZH-real | 原始＋2 个固定增强版本，共 3 倍 | 原始＋2 个固定增强版本，共 3 倍 |
| ZH-fake | 不专门倍增；保留共同轻量随机增强 | 不专门倍增；保留共同轻量随机增强 |

EN 的四类基础变换为轻微降速、轻微加速、频率响应变化、时间裁剪／短音频增益包络。沿用 AUG5 的强度，避免同时改变变换幅度与组间分配。ZH-real 的两个增强版本在来源间均衡分配四类变换，不是简单截取“慢速、快速”两类。ZH-fake 的随机变化可由 seed 重建，审计明确将它记为动态增强，而非五倍固定样本库。

**扩充倍率与抽样概率分开。** 各组每个来源连续五次曝光中一次使用原始基础波形、四次使用增强基础波形；来源相位错开，单轮实际比例允许有限偏差。EN 的四个增强版本各出现一次；ZH-real 的两个版本会重复使用。不同组的固定库大小不会决定其原始／增强比例，从而避免“只有 Real 带有增强痕迹”。分语言、真假和流记录实际 family 分布，不能仅靠设计声明保证不存在捷径。

原始／增强 Online 输入分类器。Offline 先做基础变换，再分别生成两份独立的环境噪声＋RTC 视图；同一来源的基础变换只计算一次。原始 Offline 从不直接分类。沿用已经准备好的 `assets/noise_v3184_rnnoise/manifest.json`，无需重新下载。

每条流仍抽 16 个来源，配额按 EN-fake／EN-real／ZH-fake／ZH-real 为 `4/3/5/4`；每次更新共 16 Online＋32 Noisy，四组分类损失各占 25%，Online／Noisy 各占 50%。步数依原始来源清单计算，不随虚拟版本数膨胀。增强不增加前端重复前向；没有磁盘波形或特征缓存。

五／三倍是基础版本的数量，不是独立说话人数量，也不承诺一轮遍历全部版本。`input_audit.json`、每轮 `coverage_epoch_N.json` 和 `sampling_and_loss_epoch_*.json` 记录原始来源、基础版本、动态增强、原始比例、family 与实际监督。Dev、固定 Train probe、六场景诊断和来源隔离选模口径不变；Progress 不用于决定增强参数。

默认固定基础版本。可选 `--refresh-base-variants` 在完整增强版本轮转后更新参数种子，属于额外对照；审计标明 generation，不能将它描述成始终只有五／三个固定波形。默认训练命令不启用此选项。

## 服务器部署

V3.18.4 仍训练时，可以拉取本次新增文件。等它完成并退出查看器后，再执行缓存清理和 V3.18.5 训练。下列命令全部为 ASCII；每块成功后再执行下一块。

```bash
mountpoint -q /home/ubuntu/LXT
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
export TMPDIR="$PWD/.tmp_v3185"
mkdir -p "$TMPDIR"
```

先预览可回收空间。预览会核验准备完成的噪声音频，不删除素材；首次核验可能需要时间。

```bash
bash cleanup_w2v_v3185.sh
```

等所有训练／评估／噪声准备任务结束后再清理。工具只删除经过校验且已经生成规范噪声音频的 RNNoise 下载包、选中 RAW、ESC 原始下载、准备阶段 VAD 权重。保留所有 checkpoint、prepared noise、清单、数据集、环境、模型缓存和未知文件。`last.pt` 同时保存 best、last 和优化器，不能当作非 best 文件删除。若此前已清理，回收量为零；不保证固定回收空间。

```bash
bash cleanup_w2v_v3185.sh --apply
```

工具在检测到活动训练时拒绝清理，并持有训练启动锁和噪声准备锁。输出 `RECLAIMABLE_GIB`、`REMOVED_PAYLOAD_BYTES`、`FREED_BYTES`；后者是文件系统实际空闲量变化，可能受其他进程和清理回执写入影响。清理回执保存在噪声库 `raw_cleanup_receipt.json`。修改噪声准备配方后重建素材可能需要重新下载原始包。

使用当前 V3.18.4 真正使用的 R2 parent，避免最新指针误指向另一次实验。以下命令只读取配置，不加载权重：

```bash
OLD_RUN=$(cat exp/.latest_v3184_aug5_run)
PARENT_RUN=$(python -c 'import json,sys; from pathlib import Path; print(json.loads((Path(sys.argv[1])/"config.json").read_text())["initialization"]["run"])' "$OLD_RUN")
printf 'PARENT_RUN=%s\n' "$PARENT_RUN"
test -f "$PARENT_RUN/completed.json"
```

开始独立六轮训练。默认固定一屏显示进度和每轮指标，详细信息保存在日志中。结束后用本次 best 生成 submission，并将 submission.zip 与 report.zip 分别上传 temp.sh：

```bash
bash run_w2v_v3185.sh \
  --parent-run "$PARENT_RUN" \
  --epochs 6 \
  --warm-epochs 1 \
  --submit-best \
  --upload-submission \
  --upload-temp \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

两个链接会出现在终端及运行目录 `download_links.txt`。上传失败可以单独重试，不重新训练、不重复推理。没有完成实际训练前不会生成虚假的下载链接。

```bash
RUN=$(cat exp/.latest_v3185_run)
cat "$RUN/download_links.txt"
bash upload_w2v_v3185.sh --run "$RUN"
```

关闭查看器的 Ctrl+C 不停止后台训练。重新查看、断点恢复分别使用：

```bash
bash watch_w2v_v3185.sh
bash run_w2v_v3185.sh --resume "$(cat exp/.latest_v3185_run)"
```

恢复仅接受 V3.18.5 自己的配置与 checkpoint，按最后完整提交的 epoch 续训。不能把 V3.18.4 目录传给 `--resume`，也不能在恢复时更改增强配置。所有新文件独立发布，旧版本源码保持原样。

## 验证边界

本地验证使用合成录音、轻量前端和真实 AASIST，检查定向版本预算、波形确定性、来源覆盖、Offline 处理顺序、均衡损失、断点恢复、选模与导出、终端显示、两类上传重试及清理拒绝条件。网络上传在测试中模拟。实际 Omni-3B GPU 吞吐和 Progress 分数需要服务器训练验证；不能由这些测试推断 V3.18.5 已提高泛化成绩。
