# 后台采集：Issue #1

## 数据与控制路径

`CaptureWorker → BackgroundSource → WgcStream(HWND)` 获取窗口原始 BGRA，在回调返回前复制 ROI/标题区域到单槽缓冲，后续仍走现有稳定检测、OCR、配准、消息组装与导出。窗口客户区相对 DWM 可见边框的偏移用于裁剪；支持 WGC 提供整窗或客户区尺寸，尺寸不匹配时拒绝，避免错误坐标。

定向滚动通过 `ChildWindowFromPointEx` 在锁定窗口自己的子树中寻找控件，绑定后每次核对 PID、父子关系、有效性和可见性。以该 ROI 的屏幕坐标构造轮消息，不读取当前鼠标位置。原生消息投递失败则暂停，不升级权限或改用全局输入。

后台校验保留窗口身份、标题、客户区尺寸、DPI、最小化状态和可选会话图像守卫；允许尺寸及DPI不变的窗口平移，前台/遮挡检查只用于用户主动选择的前台兼容后端。窗口被遮挡时仍由 WGC 获取其合成表面。

缓存过期时向目标窗口及其子控件异步请求重绘；没有新帧则拒绝继续使用旧画面。全局 Esc 只属于原前台模式，后台使用停止按钮、本程序 Esc 或 CLI Ctrl+C。

## 验证边界

本次没有操作用户微信窗口。GUI 和 Win32 调用使用模拟适配器测试；原生集成只创建两个人工测试窗口，均不接受焦点，叠放在已有窗口下方。粉色模拟遮挡窗覆盖内容窗，WGC 始终绑定内容窗。

测试窗口用程序生成的场景，每个定向滚轮事件移动 80 个原生像素。测试把激活目标、全局 SendInput、SetCursorPos 设为禁止调用，检查：

- 上、下方向分别采集 8 帧，均确认，实际移动分别为 −1040 / +1040 px；累计坐标误差 0 px。
- 采集图片与合成场景逐像素一致；测试窗口关闭圆角，避免把 Windows 的圆角透明遮罩当作图像内容差异。
- 两次离线几何回放均接受 8 帧，与采集检查点差异为 0 px。
- 完整采集模式双向各确认 15 帧，自动到达边界，分别移动 −1600 / +1600 px；回放各接受 15 帧，检查点差异为 0 px。
- 原生测试输出只保存在忽略的 `runs/background-validation/` 下。

此外用现有 68 张离线调试图运行同一 OCR/组装流水线，确认位移、消息顺序和分类不受采集适配器变更影响。未使用这些截图替代原生遮挡验证，也不以合成窗口测试宣称所有微信版本都支持后台轮消息。

本次全量 pytest 为 **261 passed**；不含私有样本的公开文件副本为 **258 passed, 3 skipped**。依赖检查通过。68 帧离线重识别均确认、无拒绝，269 条消息，累计位移 −18427 px，每帧坐标与原采集一致。

## 主动复现原生模拟测试

以下命令只操作工具自身的模拟窗口，不选择或控制用户微信。它会短暂创建两个不激活的本程序窗口，输出目录必须为空：

```powershell
.\.venv\Scripts\python.exe tools/background_smoke.py --output runs/background-validation/down --direction down
.\.venv\Scripts\python.exe tools/background_smoke.py --output runs/background-validation/up --direction up
```

原生测试不放入常规 pytest 或 CI。需要安装 Windows extra 和真实交互式图形环境；普通单元测试不需要窗口采集依赖。

## 限制

- 当前为 WGC + 定向轮消息；UIA ScrollPattern 尚未实现。
- 微信是否接受此类后台消息，必须由用户在自己的版本主动验证；失败可选前台兼容模式。
- 窗口最小化或隐藏期间暂停读取和滚动，等待用户恢复同一窗口。锁屏、改变目标窗口客户区尺寸或DPI、同时操作同一会话仍不受支持。同尺寸同DPI平移可继续采集。
- WGC/模型原生工作区与采集缓冲仍不计入受管图像额度；图像单槽和 OCR 背压保持有界，默认截图不落盘，调试模式及其配额仍需显式开启。

参考：[微软 HWND 窗口捕获接口](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.capture.interop/nf-windows-graphics-capture-interop-igraphicscaptureiteminterop-createforwindow)、[窗口消息投递](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-postmessagew)、[windows-capture Python 接口](https://github.com/NiiightmareXD/windows-capture/blob/main/windows-capture-python/README-Python.md)。项目使用其公开 API，没有复制第三方实现代码。

## 标题未读徽标与目标变化诊断

后台采集仍绑定同一 HWND、PID、客户区几何和滚动子控件。仅对已锁定进程为 weixin.exe/wechat.exe、基础标题为“微信”/“WeChat”/“Weixin”的情况，将应用名首尾的数字未读徽标视为可变标题，例如“微信”→“微信 (25)”→“微信”。数字、括号或新消息正文出现在其他标题中时，不进行这种归一化。可选会话图像守卫继续独立检查，不因未读数变化被关闭。

其他标题、进程、客户区尺寸、DPI和控件变化仍暂停采集。异常通过 source_error.code 区分 window_pid_changed、window_title_changed、client_geometry_changed、scroll_control_changed、scroll_control_pid_changed、conversation_guard_changed；诊断日志、事件索引和报告摘要保存该原因，界面显示对应中文说明。诊断仅记录标题长度和数值几何/标识，不记录变化后的标题文本或第三方异常正文。

窗口状态检查失败时，并未获取该时刻的目标截图；此类事件标明 image_evidence=last_confirmed_frame（或 last_attempt_only；无可用图时为 not_available），不能把保留的上一张成功图当成触发异常时的画面。没有新图时不会伪造失败帧或推进累计高度。旧日志只保存 BackgroundTargetChanged 类名时，不能事后从旧截图断言是哪项身份检查变化。

模拟 WGC/Win32 回归在每次读取时修改未读数，验证上下方向、定向滚动、消息顺序和累计位置；另验证真实身份变化仍停止、原因留存、无截图默认策略和离屏界面提示。这些测试不访问真实窗口。


## 允许窗口平移

客户区屏幕原点不是聊天内容坐标。后台校验现在区分原点平移与尺寸/DPI变化：同一目标进程、同尺寸和同DPI的移动继续采集。窗口目标描述保存DPI，新区域校准缓存也保存DPI；旧缓存缺少该字段时仍可读取。

WGC回调使用初始化时一致的客户区/窗口边框映射，裁剪只依赖两者的局部偏移和固定ROI，不逐帧拼接不同时间的屏幕原点。初始化映射采样前后若窗口移动，则最多重新采样三次；仍不一致时返回window_mapping_unstable，采集流尚未启动。

每次定向滚轮投递在检查绑定子控件后，重新查询当前客户区原点，计算WM_MOUSEWHEEL屏幕坐标；初次子控件查找也使用当前位置。始终向原绑定窗口子树投递，不读取或移动鼠标、不激活窗口。后台工作线程启动和校准复用同一校验，避免倒计时期间移动也被旧对象相等判断拦截。

client_geometry_changed现在表示客户区尺寸变化；client_dpi_changed表示DPI变化或无法核实。它们和其他身份/会话保护都保留，不通过移动旧ROI或猜测像素缩放继续采集。移动跨越不同DPI的显示器仍需要重新校准。

模拟回归覆盖正/负屏幕坐标、子控件检查期间移动、初始化映射重试、持续移动时拒绝启动、同像素尺寸下DPI改变、客户区/整窗两种表面，以及上下方向完整采集。窗口平移本身不贡献聊天累计高度，图像预算和默认不落盘策略不变。

## 短暂不可用时恢复采集

后台窗口隐藏或最小化后，采集等待最多 30 秒。界面显示等待状态，停止按钮在等待期间可用。同一窗口恢复可见后，重新核验已绑定目标，重建窗口采集流并取得新画面。可选会话守卫与上一次确认位置继续沿用。读取阶段重新采样稳定性，滚轮阶段只继续发送尚未投递的事件。

窗口关闭或身份变化时结束采集。等待超时、窗口关闭和身份变化的诊断区分具体状态、操作阶段以及当前批次已经投递的滚轮事件数。异常事件只保留已取得的调试证据，累计位置仍来自配准通过的帧。测试使用模拟窗口和图像流，覆盖滚轮批次中途隐藏、采样前最小化、恢复后会话守卫变化、主动停止及超时。
