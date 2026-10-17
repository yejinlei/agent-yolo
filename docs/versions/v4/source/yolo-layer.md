# yolo_layer.c

YOLOv4 检测头：**前向同时完成 CIoU loss/delta 计算**，激活在 forward 开头原地做（含 scale_x_y），推理时提前返回，解码与 NMS 在别处。本站所有 `v4 yolo_layer.c:行号` 引用均指向本页。

> 快照路径：`source/yolov4/snapshot/src/yolo_layer.c`。点击左侧行号可复制/分享行锚点；如 `#L369-L545` 可高亮多行区间。

<SourceViewer file="v4:src/yolo_layer.c" />

**阅读路线**：[L13 构造 make_yolo_layer](/versions/v4/source/yolo-layer#L13) → [L119-133 解码 get_yolo_box](/versions/v4/source/yolo-layer#L119-L133) → [L153-240 坐标 delta（CIoU 分支）](/versions/v4/source/yolo-layer#L153-L240) → [L242-256 averages_yolo_deltas（multi-positive 平均）](/versions/v4/source/yolo-layer#L242-L256) → [L327-340 前置激活 + scale_x_y](/versions/v4/source/yolo-layer#L327-L340) → [L369-410 第一遍循环（noobj+ignore 豁免）](/versions/v4/source/yolo-layer#L369-L410) → [L447-490 第二遍循环（best_n 定主力）](/versions/v4/source/yolo-layer#L447-L490) → [L501-545 第三遍循环（multi-positive）](/versions/v4/source/yolo-layer#L501-L545) → [L556 delta 平均](/versions/v4/source/yolo-layer#L556) → [L604 cost=avg_iou_loss+class_loss](/versions/v4/source/yolo-layer#L604) → [L616-619 backward axpy](/versions/v4/source/yolo-layer#L616-L619) → [L625 letterbox 还原 correct_yolo_boxes](/versions/v4/source/yolo-layer#L625) → [L773 推理解码 get_yolo_detections](/versions/v4/source/yolo-layer#L773)。
