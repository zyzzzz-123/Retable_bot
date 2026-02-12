# Problem Log

## 2026-02-12

### 已解决：相机后台线程未启动导致帧为空

**问题**: 在 `wait_for_command` 阶段（READY 状态），UI 请求摄像头帧返回 404。

**原因**: 
- `OpenCVCamera` 的后台读取线程 (`_read_loop`) 是**懒加载**的，仅在首次调用 `async_read()` 时启动
- 首版实现直接访问 `cam.latest_frame` 属性，但在 wait-for-start 模式下从未调用过 `async_read()`，线程未启动，`latest_frame = None`

**解决**: 
- 修改 `save_camera_frames()` 函数，在 `robot` 路径下使用 `cam.async_read()` 代替直接访问内部属性
- `async_read()` 会自动启动后台线程（如果未运行），确保帧可用
- 推理循环路径保持不变（从 `obs` dict 提取，零开销）

**状态**: ✅ 已解决
