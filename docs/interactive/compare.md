# 交互式模型对比 · 浏览器内跑 YOLO ONNX

> 一句话：上传你的图，浏览器里对比 YOLOv8n / YOLO11n / YOLO26n 的检测框、耗时、内存。**不需要 GPU、不需要 API Key**。

- ⏱ 首次加载：模型约 6-14MB，需 10-20 秒；后续走 IndexedDB 缓存
- 🔧 依赖：onnxruntime-web (WASM/SIMD 后端)，无 Python
- 🔗 相关：[基准数据](/benchmarks) · [选型决策树](/choose-a-model) · [ONNX 部署](/recipes/export-onnx)

## Demo

<interactive-compare />

## 典型性能数据（COCO val · 参考）

| 模型 | 参数 | ONNX 大小 | WASM 推理 (4 线程) | 备注 |
|------|------|----------|-------------------|------|
| yolo26n | 2.8M | 6MB | ~850ms/张 | 最新小模型 |
| yolo11n | 2.6M | 6.2MB | ~920ms/张 | 主流小模型 |
| yolov8n | 3.2M | 12.5MB | ~1200ms/张 | 上一代基线 |

（数字来自社区基准 + 本地实测，仅供参考。生产环境请用 GPU 推理。）

## 浏览器推理的 5 个限制

1. **首次加载慢**：模型需从 CDN 下载 6-14MB；本地缓存后可复用。
2. **WASM 比 GPU 慢 10-20 倍**：适合验证算法/结构，不适合实时。
3. **内存占用高**：640×640 一张图约 100-200MB 内存（含 wasm 缓冲区）。
4. **不能跑 ReID / Segment 完整流程**：Segment 需额外处理 mask；ReID 依赖后端特征库。
5. **浏览器兼容**：需要支持 SharedArrayBuffer（跨域 CORS header）的浏览器；无头环境（Selenium headless / Playwright 默认）不支持。

## 什么时候用这个 demo

- 想快速验证模型差异而不想搭环境 → 用这个。
- 想给业务同事展示 → 直接开链接，5 分钟讲完差异。
- 想验证「我的图片能不能被识别」→ 拖一张进去看结果。
- 教学场景 → 现场演示 NMS、letterbox、IoU 的实际效果。

## 什么时候不要用

- 生产环境推理 → 用 TensorRT / ONNX Runtime C++，速度快 20-50 倍。
- 视频流实时（>10 FPS）→ 用 GPU 版，WASM 顶多 2-3 FPS。
- 大量用户并发 → 后端推理 + WebSocket 推流，别把流量打到用户浏览器。
- 数据合规敏感 → 图片会走跨域 CDN 加载模型，注意本地缓存策略。

## 反哺信号

- **PainPoint 新增**：`pain-wasm-sharedarraybuffer-cors`（浏览器需要 COOP/COEP header 才能开 SharedArrayBuffer）
- **Trick 新增**：`trick-browser-onnx-warmup`（首次推理会因 JIT 慢，预热一次即可稳定）
- **Trick 新增**：`trick-browser-ort-wasm-threads`（`numThreads=4` + `proxy=true` 是稳定组合，2 线程以上需 SAB）
- **Metric 新增**：`metric-browser-inference-ms`（WASM 单张 640×640 端到端耗时）

---
**最后更新：2026-10-25**
