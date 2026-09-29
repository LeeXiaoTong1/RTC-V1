# RTC：w2v-BERT 2.0 + AASIST

当前维护入口是 `xlsr_aasist/w2v_aasist`：从原平台 Weighted 91.68 的完整 AASIST checkpoint 微调，普通音频使用整段，已有 noisy 缓存只读并在内存组合。MultiConv 已撤出当前分支。

部署、训练、保留 checkpoint、日志、恢复及 submission 导出见 [使用说明](xlsr_aasist/README_W2V_AASIST_FULL.md)。

仓库仅保留当前训练/推理包、原 AASIST 模型定义、必要的音频/缓存工具。历史分析脚本、失败实验入口及重复嵌套副本已移除。

- 清理前完整代码：分支 `archive-before-aasist-full-20260929`，提交 `ab937d63293670935905bd960186976777c9272e`。
- 原最佳权重、实验目录、数据集及缓存均不由 Git 管理；清理清单禁止删除这些内容。
- `cleanup_manifest.json` 列出废弃源文件。服务器维护命令会先复制并校验最佳 checkpoint，再把残留旧脚本归档后移除。
- 旧 91.68 checkpoint 可由当前评估器按其原来的 64600 点裁剪方式推理。新 checkpoint 自带整段输入策略。

训练语音只来自官方 Train，Dev 仅用于验证和选模。
