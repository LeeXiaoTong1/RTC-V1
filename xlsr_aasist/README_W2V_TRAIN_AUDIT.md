# Train 四组数据核查

从已有实验 `stage3/config.json` 读取实际 Train 协议、音频根目录、RTC 配对清单和训练 noisy 缓存。不会扫描整个 LXT，也不会载入模型、调用 GPU、训练或重建缓存。脚本在训练源码指纹范围之外，新增脚本不会改变原训练代码的 resume 指纹。

报告包含：

- en-real、en-fake、zh-real、zh-fake 的文件数、字节 SHA256 去重数、明确 Offline/Online 配对合并后的已知源组件数。
- 各组及来源目录的时长、短于 4.0375 秒数量、首段外时长；采样率与通道。
- 原音频首段、中段的音量与能量活跃窗比例，低能量/近满幅等人工核查线索。
- 重复 ID、相同文件异标签、音频读取失败、训练清单指纹变化。读取失败可能来自缺失、损坏或当前解码器不支持，需按具体错误判断。
- 现有训练缓存的源覆盖、四档覆盖、语言组×真假×处理算法×SNR 分布；核对当前源 hash、标签、角色和文件存在。

“来源目录”不是已核实的说话人/设备/录音库身份。“源组件”只合并文件字节相同或配对清单明确建立的关系；不同编码的同一录音仍可能重复，不能称为完全独立的录音数。音量指标不是 VAD，不证明文件无语音或标错。本次不做跨 Train/Dev 的内容去重。

**本步骤不测量四组 Train recall。** 旧日志的总体训练 recall 不能推断每组表现。需要时另做固定条件下的只读分组推理；不会悄悄在本核查中启动。

## 使用

在现有 `(sdd)` 环境中运行。以下每个命令块都低于网页终端的 2000 字符限制，并且只含 ASCII。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
git pull --ff-only origin w2vbert2-balanced-robust-fast &&
RUN="$PWD/exp/w2v_adapt_20260927_225221_104a" &&
bash run_w2v_train_audit.sh "$RUN" --upload-temp
```

`--upload-temp` 在报告完成后上传报告 ZIP 到 temp.sh，打印下载链接；ZIP 只含统计、源路径、hash 和元数据，不含音频/权重。省略该参数则仅保存本地 ZIP。文件托管服务的有效期由服务端决定。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
tail -n 30 -f "$(cat exp/.latest_train_audit_log)"
```

会打印 `Audio scan: 已完成/总数`，随后打印缓存清单扫描进度。全文件 hash 需要读取 Train 音频全部字节，但只解码每个文件的首段和中段；用时主要取决于存储速度。默认 4 个工作线程、单线程数值库。使用 `--workers 2` 可减少并发读盘。

`AUDIT_COMPLETE=True` 表示报告 ZIP 已完成并通过 CRC 检查；输入有异常时仍保存完整核查结果，状态是 `complete_with_input_issues`。`UPLOAD_COMPLETE=True` 后的 `TEMP_DOWNLOAD_URL=...` 是下载链接。如果上传失败，本地报告仍保留；不需要再次扫描音频。

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist" &&
OUT=$(cat exp/.latest_train_audit_dir) &&
python -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "$OUT/completed.json" > "$OUT/zip_path.txt" &&
ZIP=$(cat "$OUT/zip_path.txt") &&
curl --fail --show-error -F "file=@$ZIP" https://temp.sh/upload
```

`Ctrl+C` 只退出上面的日志查看，后台核查继续。核查结果目录见 `exp/.latest_train_audit_dir`，ZIP 在 `/home/ubuntu/LXT/temp`。每次运行使用新目录，不覆盖既有报告。

直接运行或自定义输出：

```bash
python -u audit_w2v_train.py --run-dir /path/to/experiment --out /path/to/new_report --download-dir /path/to/downloads --workers 4
```

只依赖当前训练环境已有的 NumPy、soundfile 和标准库。测试：`python -m unittest test_w2v_train_audit -v`。
