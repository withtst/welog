# WeLog：内存 OCR 聊天采集

项目代码、问题与改动通过 GitHub 维护。每完成一项完整功能、修复或改动并通过验证后，集中提交；开发和调试中的中间状态不频繁提交。具体约定见 [AGENTS.md](AGENTS.md)。

Windows 本地工具：用户指向微信窗口并校准聊天区，程序按选定方向采样，在后台用 RapidOCR 识别，检测气泡并组装跨屏消息。

**默认采集不保存任何截图，包括原图、裁剪、拼接图和异常诊断图。** 只有显式勾选“调试采集（保存截图）”才留存聊天区原图。离线回放只引用已有图片，不再复制图片。

## 安装与启动

推荐 Python 3.12 x64，在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,windows,gui,ocr]"
.\.venv\Scripts\welog-gui.exe
```

也可运行 `.venv\Scripts\python.exe -m welog gui --no-auto-pick`。复现实测版本：先安装 `requirements.lock`，再 `pip install -e . --no-deps`。OCR 使用 CPU ONNX Runtime、PP-OCRv6 small，已验证模型直接接受 numpy BGR 数组，不使用临时图片。

交互框选由 PySide6 完成，不调用 OpenCV 的窗口 API。Windows 的 WGC 依赖 `windows-capture==2.0.1`，其依赖声明使用 `opencv-python`，RapidOCR 则使用 `opencv-python-headless`；Windows extra 和锁定文件将两个发行版固定为同一版本。升级已有环境时按 `requirements.lock` 安装，不单独卸载其中一个共享 `cv2` 的发行版；需要修复环境时重新安装锁定依赖。非 Windows 环境仅使用 headless 发行版。

## 用户交互

1. 打开程序，光标移到可见的微信主窗口内，停留约 1.5 秒锁定。默认使用“后台窗口采集”，锁定后可以用其他窗口遮住微信。
2. 首次在内存截图上框选可滚动消息区，排除标题、输入框和滚动条；可再框选固定会话名称。
3. GUI 默认“完整采集”，不受帧数上限限制。选择“底部 → 顶部”前先定位到最底部；选择“顶部 → 底部”前，用户需自行切换到最早一条聊天记录。程序从当前位置开始，不替用户寻找起点。
4. 倒计时后开始；后台采集保留本程序的停止按钮，不响应其他软件中的全局 Esc。在本程序内按 Esc 仍可停止；前台兼容模式也保留全局 Esc。不会切换联系人或发送聊天消息。
5. 结束后自动打开“记录与导出”，也可点击“查看本次记录”。窗口关闭/最小化、布局或会话变化、画面过期、跨页连续性不能确认时暂停。后台模式不因切换到其他软件或被遮挡而停止。

完整模式在有实际滚动后，连续三次确认无位移时停止；候选边界处会等待再次尝试，允许客户端继续加载。若滚动恢复则继续采集。只想试采几屏时可切换“限量采集”并设置帧数。相同尺寸可复用校准；字体、主题或布局变化后需重新框选。

**两种采集方向都按从早到晚显示和导出。** 消息按全局纵坐标升序排列，向上采集的负坐标不会导致时间线反转；同一长消息内部的文字行也保持从上到下。这里遍历的是用户选定起点到客户端当前可滚动边界的内容；未检测到任何滚动、连续性失败、用户停止和达到限量帧数都不会标记为完整遍历。

## 后台窗口采集

默认采用 Windows Graphics Capture 按精确 HWND 取图，裁剪到已校准的客户区 ROI；允许窗口未最小化且被其他软件遮挡。取图不包含鼠标，不依赖当前屏幕可见内容。校准和采集都不会主动激活微信；本程序的结果窗口也以不激活方式显示。

后台滚动仅向锁定窗口树中选定的聊天子控件发送 `WM_MOUSEWHEEL`，不调用 `SetCursorPos`、`SendInput` 或 `SetForegroundWindow`，不会随鼠标位置选择其他软件。它不自动切换到前台兼容模式。

当前实现使用定向窗口消息，尚未实现 UIA ScrollPattern 路径。不同微信版本对后台滚动的响应需要用户主动验证；消息投递成功不能证明已滚动，程序仍以实际图像位移为准。未产生位移会暂停提示，用户可主动选择“前台兼容采集”。校准会话标题区域有助于发现误切换会话，采集中应保持目标窗口的位置、尺寸、DPI 及会话不变。

最小化、隐藏、关闭、窗口身份或控件变化、WGC 不可用/返回黑屏、标题校验变化、失去新画面时停止；不会恢复、置顶或激活目标窗口。后台只缓存一对 ROI/标题图像，覆盖旧帧，不积压历史截图。超过 0.5 秒的缓存不作为稳定帧，定向请求目标重绘；超过 2 秒没有新帧则暂停，避免把冻结画面当作滚动边界。

详细设计和验证见 [后台采集](docs/background-capture.md)。

## 调试采集

在“采集工作台”勾选 **调试采集（保存截图）**，再按原流程选择窗口、校准并开始。该开关每次启动默认关闭，开启时保存本次聊天区送入 OCR 前的无损 PNG（未遮罩文件/图片区域），包括匹配失败、原位重试及超时诊断帧；不逐张留存稳定检测的中间采样，也不保存整窗校准图。

采集页及倒计时会明确显示“普通采集：不保存截图”或“调试采集：保存聊天区截图”；开启后按钮显示“选择窗口并开始调试”。开始前可据此确认模式。

图片位于 `runs/pointer-.../capture/frames/000000.png`，`frames.jsonl` 记录相对路径、SHA-256、动作和稳定性信息；`alignments.jsonl` 保留是否确认及累计位移。清单标为 `kind=live_debug`、`storage_mode=debug_images`，报告记录图片数量和字节数。未确认帧仅留作诊断，不计入累计高度。若写入失败，采集停止；诊断图无法写入时单独记录 `diagnostics.jsonl`，已有帧仍可回放。

默认截图额度 **1 GiB**，到限即停止，保留已存图片。CLI 可显式开启并调整额度（MiB）：

```powershell
.\.venv\Scripts\python.exe -m welog capture --config configs/wechat.json --output runs/debug-chat --direction up --debug-save-images --debug-max-mb 1024
.\.venv\Scripts\python.exe -m welog replay runs/debug-chat --output runs/debug-chat-replay
```

GUI 采集的回放输入为对应的 `pointer-.../capture` 目录。调试图片随该记录一起保留，删除记录时一并删除；普通采集的存储行为保持不变。

## 记录查看与导出

左侧“最近的记录”自动发现 `runs/` 下的消息结果；点击“刷新记录”可更新列表。只想查看和导出时，使用下面的命令打开工作台，不触发窗口选择：

```powershell
.\.venv\Scripts\welog-gui.exe --no-auto-pick
```

- “打开记录”可载入已有 `messages.sqlite3`、JSONL，或包含消息数组的 JSON。SQLite 以只读方式打开，源文件不被修改。
- “重命名”可将左侧记录名改为对方姓名等名称，不移动采集目录或改变消息 ID。“信息”显示采集起止时间、最近导出时间、总消息数、各类型/方向数量、待核对和人工修改数量及最近 20 次导出历史。
- 中间按正文搜索，每页最多预览 200 条，可按时间段筛选；选择一条可查看完整多行正文、语音时长与完整性状态。读取、附件复制和导出在后台线程执行。
- “编辑消息”可修改正文、方向、类型、状态和语音时长；“恢复原始”撤销该条消息的人工修改。编辑以独立覆盖层保存，原始 OCR 数据、位置及证据保持不变，修改后的消息以 `*` 标识。
- “替换附件”选择用户自己的真实图片或文件，程序复制一份并保存原文件名、MIME、字节数和 SHA-256；图片可直接预览，文件可点击“打开附件”。语音消息可关联本地音频并保留时长。用户原附件移动或删除后，不影响记录中的副本。
- 右侧分别设置对方（incoming）和我方（outgoing）昵称，即时更新预览。未知方向保留原昵称或显示“未知”。昵称只用于显示与导出，不修改原消息方向。
- 可多选 **TXT、Markdown、HTML、CSV、JSON、JSONL**，设置标题和目标目录，再点击“导出全部消息”。导出包含当前记录的全部消息及人工修改，搜索、时间筛选与分页只影响预览。
- 每次导出创建带时间和随机标识的新文件；文件名会清理 Windows 非法字符。导出成功后可直接打开目标目录。下次打开会记住最近一次导出的昵称、格式、标题及目录。
- 点击消息列表上方“删除记录”，确认后永久删除。应用采集目录下的记录会删除该记录目录（包括编辑、附件、元数据和报告），对 `pointer-.../capture` 会删除这次采集的整个 `pointer-...` 目录；确认框显示完整目标路径。导入的外部文件只删除选中的文件、对应编辑数据、附件目录及 SQLite 日志伴随文件，不删除其父目录。采集、读取、编辑或导出进行中不能删除。

HTML 可直接用浏览器离线查看、显示图片并下载附件；有附件时须与同名 `.assets` 目录一起移动。JSON/JSONL 保留消息 ID、方向、类型、位置、区间、人工修改和证据元数据，并添加 `sender`。每次导出同时生成 `.metadata.json`，包含导出时间、记录名、统计和采集信息，附件使用相对路径。CSV 使用 UTF-8 BOM，保留多行正文和时间区间，对可能被表格软件执行的公式文本加单引号。JSON 查看上限 64 MiB，大记录使用流式 JSONL 或 SQLite。导出保留待补全、识别失败等状态。

采集截图仍然不落盘。只有用户通过“替换附件”主动选择的真实图片或文件才会复制到记录附件目录，导出时按需复制附件副本。查看、编辑和导出不访问微信窗口。

## 时间区间与文件卡片

截图中居中的日期/时间分隔条会被记录为时间锚点，与消息使用相同的累计坐标。两个锚点之间的消息共享 `time_interval.start/end`，统一使用固定 **UTC+08:00**；区间不等于每条消息的精确发送时间。首尾缺少一个时间戳时保留开放端，没有证据时显示“未定位”，冲突顺序会标记异常。

支持完整年月日、“昨天/前天/星期几”、月日和时分。相对日期按该帧采集时点解析，跨午夜采集使用每帧的日期；保存原文、解析精度、帧号和坐标证据。只有确认位移的帧参与分段。已有记录首次打开时，可从 `alignments.jsonl`、`observations.jsonl` 和帧元信息回填区间，无需截图；原有日志未保留的时间信息无法恢复。

文件卡片结合布局、文件名后缀、大小和客户端来源文字识别，支持 PDF、Office、压缩包、媒体、文本及代码等后缀，保存 `file.filename/extension/size_display/download_status`。大小推算值标为 `size_bytes_estimate`；手动关联真实文件后才记录实际字节数。自动采集只提取卡片可见信息，原始图片、语音及文件内容由用户通过“替换附件”补入。

白底媒体会先进行附件证据检查，再依据密集小字、界面结构及空间分布识别网页/论文/目录截图；图片内部文字不作为聊天正文。单纯提到文件名或图片里的文件列表不会因此成为附件。缺乏足够证据的空白卡片仍保持未知。截断的未知片段与完整图片只有在横向位置、累计坐标高度及重叠范围符合条件时关联，避免跨屏重复。

可使用与实时采集相同的 OCR、分类和跨屏组装流程重新处理调试帧，结果写入独立目录，原记录保持不变：

```powershell
.\.venv\Scripts\python.exe tools/reprocess_debug_capture.py --source runs/某次调试采集/capture --output runs/离线重识别结果
```

旧记录中保存为文字的文件卡片，在查看和导出时会依据保留的“文件名＋大小＋客户端来源”重新分类，统计随之更新；原始数据库、正文证据和消息 ID 保持不变。普通文件名提及不会升级为文件，人工修改的类型优先。仅有图片/未知占位且未留下卡片文字的旧消息不会被猜测分类。

离线验证导出可运行：

```powershell
.\.venv\Scripts\python.exe tools/validate_exports.py --source runs/text-rules-validation/latest-session runs/text-rules-validation/long-session --output runs/frontend-validation
```

`validation.json` 记录条数、语音条数、格式、源文件哈希未变及 JSON/JSONL 回读一致性。`tools/workbench_preview.py` 仅用编造的消息离屏绘制本程序界面，输出样式预览，不查询或控制桌面窗口。

当前先完成一页 OCR 与连续性确认再执行下一批滚动，给识别提供背压。模型在独立单工作线程串行复用，UI 不执行识别。队列最多两项，图像缓存有字节额度。

**匹配失败只原位重试，不能确认就暂停，不再自动反向滚动。** 位移依次使用文字锚点、局部图像模板和经过几何/重叠内容校验的静态特征；正文 OCR 媒体遮罩与运动估计动态图遮罩独立。

## 内存与输出

应用图像额度默认 64 MiB，包含受管原图、灰度参照、边界条带和显式工作区额度。额度不足会暂停；不丢弃未处理帧。OCR 模型工作区和采集后端原生缓冲另计，这不是整个进程 RSS 的上限。

原图在处理完后释放，只保留一份受管灰度参照和滚动边界条带；条带独立复制，不钉住整图。跨缝 OCR 完成后替换旧条带，结束时全部归还额度。长消息的文字行与观察记录存入 SQLite，不累积历史整图。

```text
runs/pointer-日期时间-编号/
├─ calibration.json
└─ capture/
   ├─ manifest.json       # memory_only、配置、结束原因和内存指标
   ├─ frames.jsonl        # 帧编号、尺寸、滚动和稳定性元数据
   ├─ alignments.jsonl    # 位移、参照、方法、是否确认
   ├─ observations.jsonl  # 气泡/OCR 观察；未确认几何的内容也保留
   ├─ checkpoint.json     # 最后确认的坐标，不含图像路径
   ├─ messages.sqlite3    # 流式消息、文字行、来源框
   ├─ messages.jsonl      # 按自然顺序导出的消息
   ├─ record-state.sqlite3 # 首次编辑/重命名/导出或回填时创建；名称、修改历史、导出历史
   ├─ record-state.assets/ # 仅含用户主动替换的真实附件
   ├─ report.json
   └─ report.html         # 文字、完整性和指标，不嵌入图片
```

`complete=false`、`partial`、`ocr_failed` 和 unknown 媒体需要核对，不会静默删除。两个独立的“好”或“开会/开会取消”不会按文字相似度合并。

气泡顶部、底部分别记录 visible/clipped/uncertain。跨屏关联以全局区间、方向和横向位置识别同一气泡，再合并文字行；仅在需要时用滚动条带重新 OCR 截断行。支持向上前补、向下后补及多屏长气泡，不能用上下文补写没有像素证据的字。

OCR 后处理会删除带数字的“x条新消息”提示：混入正文时只移除提示文字，独立提示气泡则整条排除；普通 OCR、边界补识别和存储修复使用同一规则。单条内容完整符合 `19''`、`19‘’`、`19"` 等时长标记时，消息类型为 `audio`，保留原识别文本，并输出 `duration_seconds`。包含时长的普通句子不按语音分类，也不转写语音内容。

已有结果可以离线清理到新目录，原始结果保持不变：

```powershell
.\.venv\Scripts\python.exe tools/repair_structured_results.py --source runs/旧任务/capture --output runs/清理后结果
```

当前针对浅色私聊布局；文件卡片已用所提供的 PPTX/PDF 截图验证。复杂卡片、群昵称和其他主题需要更多样本，OCR 识别结果和完整性状态不等于人工确认的准确率。

## CLI

```powershell
.\.venv\Scripts\python.exe -m welog windows
.\.venv\Scripts\python.exe -m welog configure --hwnd 123456 --output configs/wechat.json --with-guard
.\.venv\Scripts\python.exe -m welog capture --config configs/wechat.json --output runs/chat-new --direction up --backend background
# 用户先手动定位到最早一条记录，再从顶部向下采集：
.\.venv\Scripts\python.exe -m welog capture --config configs/wechat.json --output runs/chat-down --direction down
```

句柄 123456 需替换为实际值。GUI 与 capture CLI 默认后台采集，可显式选择 `--backend foreground` 兼容旧方式；后台模式不允许 `--activate`。`--max-frames` 默认 0，表示采集到边界；指定正数启用限量采集。ROI 为相对窗口客户区的原生物理像素 x,y,width,height。`--wheel-units` 是单事件输入，默认 120；程序分隔发送多个事件，按实测位移规划批次。旧前台后端仍要求目标位于主显示器；后台后端按窗口句柄取图。

累计坐标约定：`global_y = scroll_offset_px + viewport_y - origin_y_px`。向上为负，不是错误。`--origin-y` 默认 0；可手动指定首帧目标消息顶部。消息使用独立 UUID，不使用累计高度作永久身份。

```powershell
.\.venv\Scripts\python.exe -m welog coordinate runs/chat-new --frame 000003 --y 150
Get-Content logs/welog.log -Tail 80
```

`boundary_reached` 表示完整模式到达当前方向可滚动边界，`boundary` 为 top/bottom，`timeline_order` 为 oldest_first。`limit_reached` 是限量采样上限；`visible_end_candidate` 是限量模式疑似边界；`continuity_unconfirmed`、StableTimeout 或 OCR 错误表示原位停止。启动后一直没有移动为 `no_scroll_motion`。后面这些状态都不代表全部聊天历史已采完。

## 离线验证，不操作微信

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/validate_memory_pipeline.py --source runs/pointer-20261116-204033-bf2111/capture --output runs/my-memory-validation --frames 6
```

第二条读取已有归档，在内存重识别，不复制或保存图片。输出目录需为空。它是一次新的离线计算，不追认源任务中原来未确认的帧。

本次示例结果在 `runs/memory-ocr-final/`。回归覆盖零图像写入的正常/超时/OCR 异常/预算耗尽/取消路径、长消息双向组装、跨缝重 OCR、重复内容保留、千次条带更新、队列背压和真实动图失败帧的静态特征备用配准。

单元测试默认使用模拟 OCR，实际 CPU 模型另由上述离线工具验证。不要把模拟识别测试当作真实字符准确率。

公开仓库不包含聊天数据、截图、附件、日志、校准配置、模型权重或第三方源码快照。涉及私有样本的回归会在没有样本时跳过；提供了 [Windows 离屏 CI 模板](docs/ci/README.md)，启用后仅运行合成、离线及 Qt 离屏测试，不控制桌面或下载聊天样本。`docs/` 中的历史验证记录仅记录方法和汇总指标，提及的本地归档不随仓库发布。

## 旧归档兼容

```powershell
.\.venv\Scripts\python.exe -m welog replay runs/旧任务/capture --output runs/旧任务-replay
```

只适用于已经保存图片的历史归档。默认普通帧沿用采集方向，恢复帧双向搜索，未确认帧排除；回放只保存原图引用和新计算数据，不复制 PNG。源图片必须继续存在。新 memory_only 会话没有像素回放，直接查看其文字报告。

`demo` 是显式生成合成 PNG 的开发工具，不是真实采集入口。原有图片不会自动删除。`native_smoke.py` 会打开人工测试窗口并发送真实输入，只有用户主动运行时使用；本次改造验证没有运行它。

## 设计与验证

- [源码研究和改造设计](docs/ocr-memory-bubbles-redesign.md)
- [内存 OCR 实现验证](docs/memory-ocr-implementation.md)
- [历史几何修复验证](docs/capture-fixes-validation.md)
