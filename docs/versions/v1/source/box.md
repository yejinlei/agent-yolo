# box.c

box 几何与 NMS：

- <SrcLink file="src/box.c" :line="6" /> `float_to_box()`：把 4 个连续 float 解释为中心点格式的 box
- <SrcLink file="src/box.c" :line="93" :end="96" /> `box_iou()`：交集 / 并集
- <SrcLink file="src/box.c" :line="98" :end="104" /> `box_rmse()`：IoU 都为 0 时选 best box 的兜底度量
- <SrcLink file="src/box.c" :line="249" /> `do_nms_sort()`：按类做非极大值抑制

> 快照路径：`source/yolov1/snapshot/src/box.c`。

<SourceViewer file="src/box.c" />
