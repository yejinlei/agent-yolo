# models/yolov5s.yaml

YOLOv5s 网络配置：**24 层 yaml 驱动结构**（v4 是 162 层 cfg），通过 `depth_multiple`/`width_multiple` 一键缩放为 s/m/l/x。

> 快照路径：`source/yolov5/snapshot/models/yolov5s.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v5:models/yolov5s.yaml" />

**阅读路线**：

- 超参：[L1-4 nc + depth_multiple + width_multiple](#L1-L4)
- Anchors：[L6-10 3 组锚点（P3/P4/P5）](#L6-L10)
- 骨干：[L12-25 Focus + Conv + C3 ×3 + SPP + C3](#L12-L25)
- Neck：[L28-45 双向 PANet（上采样 + 下采样）](#L28-L45)
- Detect：[L47-48 Detect 头 ×3](#L47-L48)

## 与 v4 cfg 的对比

| 维度 | v4 cfg | v5 yaml |
|------|--------|---------|
| 层数 | 162 层（逐层手写） | 24 层（模块抽象） |
| 格式 | 自定义文本 | 标准 yaml |
| 缩放 | 需复制粘贴改通道数 | `depth_multiple`/`width_multiple` |
| 解析 | darknet parse_cfg | `yaml.load` + `parse_model` |
| 锚点 | 18 个（6 组） | 9 个（3 组） |
| 头数 | 3 个 yolo 层 | 1 个 Detect 模块（内部 3 分支） |
