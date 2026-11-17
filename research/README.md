# 外部源码研究材料

- `source-manifest.json`：固定提交、下载文件及 Git/SHA256 校验值。
- `source-index.json`：本地固定快照的函数和类行号。
- `sources/`：只供阅读的源码快照，已排除版本控制。
- `probe_upstream.py`：只提取已审阅的纯算法 AST 定义，使用内存合成图和消息记录；不导入原项目 GUI、捕获模块或 OCR 引擎。
- `probe-results.json`：实际离线实验结果。

完整分析与改造方案见 [OCR 改造方案](../docs/ocr-memory-bubbles-redesign.md)。没有向 WeLog 生产模块复制第三方实现，没有操作真实微信窗口，没有新增截图文件。

重现：

```powershell
.\.venv\Scripts\python.exe research/fetch_snapshots.py
.\.venv\Scripts\python.exe research/probe_upstream.py
```
