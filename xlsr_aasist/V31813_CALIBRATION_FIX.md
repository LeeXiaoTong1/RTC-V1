# V3.18.13：第二轮辅助损失校准修复

适用错误：`No usable prior calibration gradient; inspect Train pair coverage`，发生在第一轮完成后、第二轮尚未更新时。

旧校准直接通过 FP16 AASIST 做未缩放的反向传播，存在小梯度下溢风险；同时将所有 `<=1e-12` 的辅助梯度当成异常停止。完整训练中的 GradScaler 并没有保护这次独立的校准。真实语音和伪造语音已经很好分开的配对，也可能只有很小的 softplus margin 梯度；零梯度不能直接证明没有配对。

修复只在短暂的 Train 校准中使用 FP32 后端，结束后恢复原 FP16 设置，不更新参数。校准记录每个任务的配对数量、损失、梯度范数、计算图连通性；存在计算路径但梯度很小时，沿用原有有界系数上限（XEUS 0.1、PCL 0.01），保留后续困难配对的学习机会。缺少有效配对、计算图断开、非有限损失或梯度仍会停止。

此次报告的第二轮采样计划有 5,193 组 XEUS 配对，四个 PCL 组均有覆盖。旧报告未保存六次校准的具体梯度，不能仅凭该报告区分数值下溢和原本极小的信号。

## 更新和恢复

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved &&
git fetch origin &&
git switch v3.18.13-xeus-pcl &&
git pull --ff-only origin v3.18.13-xeus-pcl &&
source "$(conda info --base)/etc/profile.d/conda.sh" &&
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318 &&
cd xlsr_aasist &&
FIXED_RUN=$(python -m w2v_v31813.recover_calibration \
  --run "$(cat exp/.latest_v31813_run)") &&
bash run_w2v_v31813.sh --resume "$FIXED_RUN" --upload-temp
```

恢复工具只接受已知旧代码、此错误和“完成第一轮、尚未校准或开始第二轮”的完整 checkpoint。它将模型、best、Adam 状态、GradScaler、随机数状态、历史和游标复制到新的 `_calfix_` 目录，更新代码身份记录，保留原目录作为备份；不会重跑 XEUS/音素准备或第一轮。总轮数沿用原配置，此次报告为总共 3 轮，恢复后还训练第 2、3 轮。已经记录的 submission 请求保留。

不要直接编辑旧 `config.json` 的哈希，也不要直接对旧目录使用普通 `--resume` 绕过代码一致性检查。恢复后打印 `Auxiliary Train calibration ... | FP32 | pairs=... | ce=... prior=... pcl=...`，详细结果写入新目录的 `auxiliary_calibration.json`。

验证覆盖：FP16 下溢的最小复现、FP32 测量恢复、真实零梯度有界处理、缺配对/断图/非有限仍拒绝、精度/RNG/模型状态恢复，以及旧 checkpoint 的模型/优化器/随机数状态逐项保留。未在本地复跑完整服务器 Omni 模型。
