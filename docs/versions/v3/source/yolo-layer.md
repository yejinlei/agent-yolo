# yolo_layer.c

YOLOv3 检测头（替代 v2 的 region_layer）：**前向同时完成训练 loss/delta 计算**，激活在 forward 开头原地做，推理时提前返回，解码与 NMS 在别处。本站所有 `v3 yolo_layer.c:行号` 引用均指向本页。

> 快照路径：`source/yolov3/snapshot/src/yolo_layer.c`。点击左侧行号可复制/分享行锚点；如 `#L159-L236` 可高亮多行区间。

<SourceViewer file="v3:src/yolo_layer.c" />

**阅读路线**：[L13 构造 make_yolo_layer](/versions/v3/source/yolo-layer#L13) → [L83-91 解码 get_yolo_box](/versions/v3/source/yolo-layer#L83-L91) → [L93-108 坐标 delta（动态 2−w·h）](/versions/v3/source/yolo-layer#L93-L108) → [L111-123 类别 delta（objectness-gated）](/versions/v3/source/yolo-layer#L111-L123) → [L137-146 前置激活](/versions/v3/source/yolo-layer#L137-L146) → [L159-192 第一遍循环（noobj+ignore 豁免）](/versions/v3/source/yolo-layer#L159-L192) → [L195-236 第二遍循环（GT 定主力）](/versions/v3/source/yolo-layer#L195-L236) → [L238 cost=Σdelta²](/versions/v3/source/yolo-layer#L238) → [L242-245 backward axpy](/versions/v3/source/yolo-layer#L242-L245) → [L247 letterbox 还原 correct_yolo_boxes](/versions/v3/source/yolo-layer#L247) → [L316 推理解码 get_yolo_detections](/versions/v3/source/yolo-layer#L316)。
