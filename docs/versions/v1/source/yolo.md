# yolo.c

YOLOv1 的训练 / 测试 / 验证入口与**输出解码**：

- <SrcLink file="src/yolo.c" :line="86" :end="112" /> `convert_yolo_detections()`：把网络归一化输出解码成像素框（加 cell 偏移、宽高平方还原）
- <SrcLink file="src/yolo.c" :line="311" :end="361" /> `test_yolo()`：单图推理入口（predict → convert → `do_nms_sort` → draw）
- <SrcLink file="src/yolo.c" :line="15" :end="84" /> `train_yolo()`：训练循环

> 快照路径：`source/yolov1/snapshot/src/yolo.c`。

<SourceViewer file="src/yolo.c" />
