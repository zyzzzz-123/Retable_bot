# Progress Log

## 2026-02-12

### 完成：实时摄像头画面显示功能

**需求**: 在 UI 中实时显示机器人摄像头画面，且不能阻塞推理或影响相机读取。

**分析发现**:
- 官方 `lerobot-record` 使用 Rerun 显示画面，**不会**重新打开相机设备
- `OpenCVCamera` 使用后台线程 (`_read_loop`) 持续读取帧并缓存到 `latest_frame`
- `robot.get_observation()` 调用 `cam.async_read()` 返回缓存帧，无额外硬件 I/O
- 因此可以安全地将**同一帧数据**转发给 UI，零额外相机开销

**实现方案**:

1. **`eval_act_safe.py`**: 
   - 新增 `save_camera_frames()` 函数，将帧编码为 JPEG 写入 `/tmp/lerobot_frames/`
   - 推理循环: 从 `obs` dict 提取帧 (零额外 I/O)
   - 等待/暂停循环: 通过 `cam.async_read()` 获取帧 (~5fps)
   - 新增 `--frame-dir` 参数

2. **`config.py`**: 新增 `FRAME_DIR` 常量，命令中添加 `--frame-dir`

3. **`main_robot.py`**: 
   - `GET /api/cameras` — 返回摄像头名称列表
   - `GET /api/frame/{cam_name}` — 返回最新 JPEG 帧

4. **`App.tsx`**:
   - `CameraFeed` 组件: 预加载图片避免闪烁，~5fps 刷新
   - `CameraFeeds` 组件: 自动获取摄像头列表，响应式 grid 布局
   - Live 指示灯 + 无帧占位符

**技术细节**:
- JPEG quality 75，原子写入（write tmp → os.replace）避免半写读取
- RGB→BGR 转换（OpenCV 的 `imencode` 要求 BGR）
- 前端用 `new Image()` 预加载，避免 React 不必要的 re-render
- Warmup 阶段不显示摄像头 (`active={!isWarmup}`)
