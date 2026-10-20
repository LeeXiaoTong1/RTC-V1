# V5：从 V3.18.9 best 开始的竞赛优化

起点固定为已提交 Weighted 93.82 的 V3.18.9 best。完整保留 Omni、60 层已训练 LoRA、多层融合、AASIST 与窗口池化权重；只更新最后 16 层 LoRA、融合和后端。初始化不丢弃前 44 层适配器，不加载与 best 不对应的 last 优化器。

没有加入 XEUS、新前端、元学习或新的辅助损失。沿用已有的配对分类：Noisy 50%、官方 Online 25%、同源 Clean-RTC 25%，每条流内语言/真假组均衡；继承 Train PCL。

## 五项改动

| 项目 | 实现 |
|---|---|
| 权重平均 | 最后 25% 更新中保存四个位置的运行平均；比较最后权重、尾部平均、尾部平均与父 best 的 50:50 参数平均。合并后实际运行模型评测，最终仍是一个模型。LoRA A/B 参数平均不等于有效矩阵 B×A 的平均，也不等于预测分数集成。 |
| 完整 Dev 收尾训练 | 四类 Dev 的真实 Offline/Online 均加入；不把旧模拟 Dev 缓存当新来源。Offline 生成 Clean-RTC/Noisy-RTC，Online 直接分类。完整 PCM 来源登记表贯通采样，允许没有 Online 的 Offline。 |
| 连续状态 RTC | 同一录音的循环尾段作为 2 秒上下文，将上下文与目标一次性通过 RTC 后裁掉上下文。两视图独立初始化处理器，参数与原始语音相同。保留继承的单次编解码路径；这是状态历史的近似，不宣称精确复现真实平台。 |
| 短句处理 | 候选为不额外重复、重复至 3 秒。保留全部原始录音，不裁长音频，不删除内部静音；重复部分不进入 PCL。 |
| 输入标准化 | 候选包括 identity、6800 Hz 温和低通及短句组合。Omni 原本就做整段波形归一化，单纯改音量通常被抵消，因此不浪费 GPU 筛选响度候选。只用 Dev Select 选择，最终训练和提交使用同一冻结配置。identity 是合法回退，不能承诺每项处理都会启用。 |

选模直接使用固定 0.5 阈值下的 `0.3 × Clean Macro-F1 + 0.7 × Noisy Macro-F1`，Noisy 合并两个面板的样本后计算。四组召回仍报告，但不再使用“每一组都不能退步”的逐组否决条件。内部 Noisy 是既有模拟 Dev，不能当作比赛 Noisy 的成绩。

## 默认会跑多少

默认 `--phase all --epochs 1`，实际是 **两段、每段一轮**：

1. **Screen**：原 Train 训练一轮，Dev 不进入训练。先按完整来源抽取最多每类 64 个来源，比较四种输入策略；胜者再与 identity 在完整 Select 上确认。随后进行一轮训练，并对三种真实合并权重评测。方案锁定后，历史 Audit/Calibration 仅报告一次，不再反过来改方案。这些 Dev 在往期研究中已被查看过，不是全新的盲测。
2. **Final**：重新从同一个 V3.18.9 best 出发，按锁定的输入、轮数与平均方式，使用完整 Train＋Dev 训练一轮。不会从筛选阶段的最后权重继续，也不会再在已训练的 Dev 上选 best 或报告泛化提升。

Screen 沿用原 1,403 步预算。Final 扩大抽样预算，保证各流每个来源至少被访问一次；基于既有清单规模预计约 2,229 步，**以实际启动打印的预算为准**。不是三个训练实验各跑一轮。预处理筛选、权重评测和最终提交推理另需时间，不能把 epoch ETA 当完整流程 ETA。

若所有训练候选都低于父模型，报告会明确记录负增益；锁定的是三个训练候选中分数最高者，仍可进行全 Dev 收尾。这不代表已经证明训练有效。最终全 Dev 模型没有独立 Dev 成绩，比赛收益需提交后验证。

PCL 范围明确限定于已经准备过真实音素记录的原 Train。新增 Dev 使用完整分类监督，不虚构其音素标注；不为这次收尾重新下载或运行新的教师模型。

## 更新与启动

先结束旧训练，使用 V3.18 环境：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch v5-competition
git pull --ff-only origin v5-competition

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist

bash run_w2v_v5.sh \
  --parent-run /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist/exp/w2v_v3189_20261016_183410_a437 \
  --epochs 1 \
  --submit-best \
  --upload-submission \
  --upload-temp
```

需要父 run 的 `last.pt`（内含经过校验的 best）、配置、训练/Dev 清单和 partition，及原模型、噪声、PCL 资产。父目录只读，必须保持停止状态。无需把 last 模型当作初始化，也无需另找 V3.18.11 checkpoint。

默认提交位置沿用旧目录；可以通过 `--submission-protocol`、`--submission-audio-root` 指定。Progress/Eval 音频仅在最终模型锁定后读取，用于正常推理，不参与选择、训练或预处理研究。

只想先看 Screen 的结果：

```bash
bash run_w2v_v5.sh --phase screen --parent-run /path/to/v3189_run
```

完成后执行固定方案的 Final：

```bash
bash run_w2v_v5.sh --resume /path/to/w2v_v5_run --phase final \
  --submit-best --upload-submission --upload-temp
```

## 停止、恢复与下载

脚本前台记录日志，建议在已有 tmux/screen 会话中运行。另开终端查看：

```bash
bash watch_w2v_v5.sh
bash stop_w2v_v5.sh
```

停止请求在当前更新/评测结束的边界生效；不要强杀正在写 checkpoint 的进程。`Ctrl+C` 同样请求保存并停止。恢复保存的模型、优化器、随机状态和尾部平均计数：

```bash
bash run_w2v_v5.sh --resume /path/to/w2v_v5_run \
  --submit-best --upload-submission --upload-temp
```

恢复时不附带 `--epochs`、`--workers` 或其他训练配置覆盖。启动会重新校验源文件身份。已完成的候选评测按模型和输入配置校验后复用。

最终模型：`RUN/final.pt`，绑定 `final_config.json` 与 `recipe_lock.json`。训练恢复点分别保存在 `RUN/screen/last.pt` 和 `RUN/final/last.pt`。最终没有额外拟合阈值或分数偏置。

输出日志打印 `SUBMISSION_ZIP`、`TEMP_DOWNLOAD_URL` 和报告链接。若上传失败，已有结果保留，可不占 GPU 重试：

```bash
bash upload_w2v_v5.sh --run /path/to/w2v_v5_run
```

报告包括数据审计、输入候选、锁定方案、训练概况和四组指标；不上传音频或 checkpoint。

## 实验边界

本地提供 CPU 测试，覆盖真实 PCM 归并、数据分区、来源覆盖、PCL 时间范围、参数平均、恢复、固定方案和提交包一致性。完整 3B GPU 训练、处理器环境与最终比赛收益需要在服务器实际运行验证；此实现没有声称已经获得新分数。
