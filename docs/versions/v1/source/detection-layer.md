# detection_layer.c

YOLOv1 检测头：**前向同时完成训练 loss/delta 计算**，另有 NMS 前的输出拷贝。本站所有 `detection_layer.c:行号` 引用均指向本页。

> 快照路径：`source/yolov1/snapshot/src/detection_layer.c`。点击左侧行号可复制/分享行锚点；如 `#L159-L161` 可高亮多行区间。

<SourceViewer file="src/detection_layer.c" />

**阅读路线**：[L42 前向入口](/versions/v1/source/detection-layer#L42) → [L75-80 noobj](/versions/v1/source/detection-layer#L75-L80) → [L102-127 选 best box](/versions/v1/source/detection-layer#L102-L127) → [L136-138 随机 warmup](/versions/v1/source/detection-layer#L136-L138) → [L157-161 conf/rescore](/versions/v1/source/detection-layer#L157-L161) → [L163-172 coord 与 IoU cost](/versions/v1/source/detection-layer#L163-L172)。
