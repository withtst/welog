# 后台采集：Issue #1

## 数据与控制路径

`CaptureWorker → BackgroundSource → WgcStream(HWND)` 获取窗口原始 BGRA，在回调返回前复制 ROI/标题区域到单槽缓冲，后续仍走现有稳定检测、OCR、配准、消息组装与导出。窗口客户区相对 DWM 可见边框的偏移用于裁剪；支持 WGC 提供整窗或客户区尺寸，尺寸不匹配时拒绝，避免错误坐标。

定向滚动通过 `ChildWindowFromPointEx` 在锁定窗口自己的子树中寻找控件，绑定后每次核对 PID、父子关系、有效性和可见性。以该 ROI 的屏幕坐标构造轮消息，不读取当前鼠标位置。原生消息投递失败则暂停，不升级权限或改用全局输入。

后台校验保留窗口身份、标题、几何、最小化状态和可选会话图像守卫；前台/遮挡检查只用于用户主动选择的前台兼容后端。窗口被遮挡时仍由 WGC 获取其合成表面。

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
- 不支持最小化、隐藏、锁屏、移动/缩放目标窗口或同时操作同一会话。
- WGC/模型原生工作区与采集缓冲仍不计入受管图像额度；图像单槽和 OCR 背压保持有界，默认截图不落盘，调试模式及其配额仍需显式开启。

参考：[微软 HWND 窗口捕获接口](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.capture.interop/nf-windows-graphics-capture-interop-igraphicscaptureiteminterop-createforwindow)、[窗口消息投递](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-postmessagew)、[windows-capture Python 接口](https://github.com/NiiightmareXD/windows-capture/blob/main/windows-capture-python/README-Python.md)。项目使用其公开 API，没有复制第三方实现代码。
