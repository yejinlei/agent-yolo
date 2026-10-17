# region_layer.c

YOLOv2 检测头（替代 v1 的 detection_layer）：**前向同时完成训练 loss/delta 计算**，推理时只做 softmax/sigmoid 激活，解码与 NMS 在别处。本站所有 `v2 region_layer.c:行号` 引用均指向本页。

> 快照路径：`source/yolov2/snapshot/src/region_layer.c`。点击左侧行号可复制/分享行锚点；如 `#L168-L174` 可高亮多行区间。

<SourceViewer file="v2:src/region_layer.c" />

**阅读路线**：[L14 构造 make_region_layer](/versions/v2/source/region-layer#L14) → [L73-85 解码 get_region_box](/versions/v2/source/region-layer#L73-L85) → [L87-106 坐标 delta（含 sigmoid/log 链式）](/versions/v2/source/region-layer#L87-L106) → [L108-131 类别 delta（softmax/层级）](/versions/v2/source/region-layer#L108-L131) → [L168-174 前向 softmax/sigmoid](/versions/v2/source/region-layer#L168-L174) → [L230-241 noobject delta](/versions/v2/source/region-layer#L230-L241) → [L243-254 12800 anchor warmup](/versions/v2/source/region-layer#L243-L254) → [L272-292 bias_match 选 anchor](/versions/v2/source/region-layer#L272-L292) → [L299-304 rescore objectness](/versions/v2/source/region-layer#L299-L304) → [L318 cost=Σdelta²](/versions/v2/source/region-layer#L318) → [L327 推理解码 get_region_boxes](/versions/v2/source/region-layer#L327)。
