# yolo.c

YOLOv2 的训练 / 测试 / 验证入口：

- <SrcLink file="v2:src/yolo.c" :line="15" /> `train_yolo()`：训练循环（加载数据、network_train、定期存权重）
- <SrcLink file="v2:src/yolo.c" :line="111" /> `validate_yolo()`：VOC 验证入口
- <SrcLink file="v2:src/yolo.c" :line="285" /> `test_yolo()`：单图推理入口（predict → get_detection_boxes → `do_nms_sort` → draw）

> ⚠️ **快照已知 bug**：<SrcLink file="v2:src/yolo.c" :line="292" /> 写的是 `detection_layer l = net.layers[net.n-1];`，但 v2 的最后一层是 **region_layer**。pjreddie 在此 commit 未改这行，靠结构体前部字段重合才"碰巧能跑"。研究 v2 推理时请注意 `get_detection_boxes` / `l.side` 这些 v1 残留调用。

> 快照路径：`source/yolov2/snapshot/src/yolo.c`。

<SourceViewer file="v2:src/yolo.c" />
