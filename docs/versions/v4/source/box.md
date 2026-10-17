# box.c

IoU 家族、NMS、CIoU 梯度手推：`box_iou`/`do_nms_sort` 供推理用；**v4 新增 `dx_box_iou`**（L258-476）手推 CIoU 对坐标的解析梯度——这是 v4 坐标损失的核心。

> 快照路径：`source/yolov4/snapshot/src/box.c`。点击左侧行号可复制/分享行锚点；如 `#L258-L476` 可高亮多行区间。

<SourceViewer file="v4:src/box.c" />

**阅读路线**：[L163 box_iou](/versions/v4/source/box#L163) → [L258-476 dx_box_iou（CIoU 梯度手推）](/versions/v4/source/box#L258-L476) → [L430-450 CIoU 长宽比项](/versions/v4/source/box#L430-L450) → [diounms_sort（DIoU-NMS，代码就绪）](/versions/v4/source/box) → [do_nms_sort（greedy NMS）](/versions/v4/source/box)。
