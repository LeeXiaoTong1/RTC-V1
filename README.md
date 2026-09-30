# RTC V3：w2v-BERT 2.0 + MultiConv

当前推荐入口是 `xlsr_aasist/w2v_v3`：原平台 91.68 checkpoint 的 w2v-BERT 编码器＋新 MultiConv 后端，普通 Train 与每条 Offline Train 的两个整段 noisy 版本完整遍历；半轮验证、阶段最佳恢复、低学习率联合适配及召回保护。原 AASIST 入口保留供回退。

启动：在 `xlsr_aasist` 下执行 `bash setup_w2v_v3.sh`，然后 `bash run_w2v_v3.sh --upload-temp`。流程自动准备/复用整段 noisy 缓存，实际读取验证成功后清理所属旧 Train 缓存；原始语音、固定 Dev、所有 checkpoint 保留。请在旧任务结束后更新和执行。

实时进度：`bash watch_w2v_v3.sh`，Ctrl+C 只退出查看。训练后生成提交：`bash run_eval_w2v_v3.sh --upload-temp`。

V3 部署、训练配方、资源预算、恢复和提交导出见 [V3 使用说明](xlsr_aasist/README_W2V_V3.md)；历史 AASIST 见 [原使用说明](xlsr_aasist/README_W2V_AASIST_FULL.md)。实际性能需服务器训练与评估确认，不承诺必然超过历史 best。

仓库仅保留当前训练/推理包、原 AASIST 模型定义、必要的音频/缓存工具。历史分析脚本、失败实验入口及重复嵌套副本已移除。

- 清理前完整代码：分支 `archive-before-aasist-full-20260929`，提交 `ab937d63293670935905bd960186976777c9272e`。
- V3 前的 AASIST＋整段 noisy＋动态进度版本：分支 `archive-before-v3-20260930`，提交 `4395fd93a45d7babcb5588093d02920a8ebfb7e5`。
- 原最佳权重、实验目录、数据集及缓存均不由 Git 管理；清理清单禁止删除这些内容。
- `cleanup_manifest.json` 列出废弃源文件。服务器维护命令会先复制并校验最佳 checkpoint，再把残留旧脚本归档后移除。
- 旧 91.68 checkpoint 可由当前评估器按其原来的 64600 点裁剪方式推理。新 checkpoint 自带整段输入策略。

训练语音只来自官方 Train，Dev 仅用于验证和选模。
