# utils/loss.py

YOLOv5 损失函数：**ComputeLoss（box CIoU + obj BCE + cls BCE）+ build_targets（anchor 匹配 + 邻居偏移）**。

> 快照路径：`source/yolov5/snapshot/utils/loss.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v5:utils/loss.py" />

**阅读路线**：

- 损失定义：[L15-29 BCEBlurWithLogitsLoss](#L15-L29) → [L32-59 FocalLoss](#L32-L59) → [L62-85 QFocalLoss](#L62-L85)
- ComputeLoss：[L88-112 __init__（BCE 定义 + balance）](#L88-L112) → [L114-161 __call__（前向）](#L114-L161)
- 关键损失：[L129-133 CIoU box loss](#L129-L133) → [L136 IoU ratio obj](#L136) → [L139-142 cls BCE](#L139-L142)
- 目标分配：[L163-214 build_targets](#L163-L214) → [L185-188 wh 比例匹配](#L185-L188) → [L190-197 邻居偏移](#L190-L197)

## 与 v4 的关键差异

- v4 `yolo_layer.c` 手写 delta 数组，v5 用 PyTorch autograd
- v4 的 objectness target 是固定 1.0，v5 是 `iou.detach()`（IoU ratio）
- v4 的 anchor 分配是 best_n 独占 + multi-positive，v5 是 wh 比例 + 邻居偏移
