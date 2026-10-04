# 结束 V3.5 后释放空间

此入口用于明确放弃当前 V3.5 的断点续训。它不修改训练策略，不运行训练。

先停止正在运行的 V3.5；停止命令会核对进程归属，不强制杀死其它任务：

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
python -m w2v_v35.stop --apply
```

拉取新增入口后，预览精确清理清单：

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
cd xlsr_aasist
python retire_w2v_v35.py --run exp/w2v_v35_20261003_161443_4fb5
```

执行清理：

```bash
python retire_w2v_v35.py --run exp/w2v_v35_20261003_161443_4fb5 --apply
```

- 仅删除指定实验的 `last.pt`（原始权重、Adam、EMA、随机状态），以及配置所指向、完整验证归属的 V3.5 Train `epoch_*` 派生音频缓存。多个 V3.5 实验可能共用这些缓存；未来若重新训练，需要按相同种子重新生成。
- 保留所有 `best*.pt`，包括 `best_model.pt`、`best_candidate.pt`、救回的 best；保留最初 91.68 best、提交对应的旧 best、其它实验全部 checkpoint、报告、逐样本分数和元数据。
- 保留官方原始音频、固定 Dev、旧 V3.3 完整 noisy 缓存、预训练资源，以及 V3.5 缓存根目录的 owner/recipe/sources 元数据。
- `best_model.pt` 可以仍是旧模型的兜底；`best_candidate.pt` 是这次 V3.5 自己训练出的最佳 EMA。两者均验证为包含完整架构和模型参数、无需 `last.pt` 即可加载的独立权重。
- 清理后**不能 `--resume` 这个实验**。可从保留的 best 权重新建训练或显式指定权重导出；默认 V3.5 导出流程仍要求正常结束记录，清理工具不会伪造 `completed.json`。工具写入 `retired.json` 和完整删除清单；训练器本身没有新增退役策略。

生产入口仅允许 Linux，持有训练启动锁和缓存锁，并再次检查活跃任务；有训练、推理、缓存任务时拒绝执行。有 `.previous` 未提交的选模事务、未知文件、元数据变化、符号链接/路径越界时也拒绝，不递归盲删。

输出 `LOGICAL_GiB` 是待删除文件的逻辑大小，`MEASURED_FREE_CHANGE_GiB` 是文件系统可用空间的实测变化；硬链接、仍被其它进程打开的文件和同期磁盘写入会使两者不同。最后仍显示实际 `FREE_GiB`。

新增清理入口位于包外，不改变 V3.5 的训练源码指纹。无需重新安装依赖；未执行 `--apply` 不会删除 checkpoint 或音频。
