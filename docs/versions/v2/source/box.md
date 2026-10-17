# box.c

YOLOv2 的框工具库：IoU / DIoU 实验代码 / NMS。本站所有 `v2 box.c:行号` 引用均指向本页。

- <SrcLink file="v2:src/box.c" :line="87" :end="96" />：`box_union` / `box_iou`（NMS 与 anchor 匹配的基础）
- <SrcLink file="v2:src/box.c" :line="249" />：`do_nms_sort()`——按类 qsort，IoU 超阈值把该类概率置零（**不删除框**）
- <SrcLink file="v2:src/box.c" :line="279" />：`do_nms()`——无序版 NMS

> 快照路径：`source/yolov2/snapshot/src/box.c`。此文件还包含一批 dxdydh 相关的 DIoU 实验函数（L140+），在本快照的检测路径中**未被调用**，属于 pjreddie 的草稿代码。

<SourceViewer file="v2:src/box.c" />
