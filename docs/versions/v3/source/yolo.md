# yolo.c

YOLOv3 训练/验证/测试命令入口。**注意：v3 快照目录结构调整，此文件从 v2 的 `src/` 移到了 `examples/`**——本站引用统一写作 `v3:examples/yolo.c`。

> 快照路径：`source/yolov3/snapshot/examples/yolo.c`。点击左侧行号可复制/分享行锚点；如 `#L5-L76` 可高亮多行区间。

<SourceViewer file="v3:examples/yolo.c" />

**阅读路线**：[L5 train_yolo 训练入口](/versions/v3/source/yolo#L5) → [L77 print_yolo_detections 结果写出](/versions/v3/source/yolo#L77) → [L98 validate_yolo 验证入口](/versions/v3/source/yolo#L98) → [L171 do_nms_sort 调用](/versions/v3/source/yolo#L171) → [L182 validate_yolo_recall 召回验证](/versions/v3/source/yolo#L182) → [L265 test_yolo（⚠️ 含未初始化 side 的祖传 bug）](/versions/v3/source/yolo#L265) → [L311 run_yolo 命令分发](/versions/v3/source/yolo#L311)。

> ⚠️ `test_yolo()`（L265）用 `l.side*l.side*l.n` 传框数，而 yolo 层从未赋值 `side`（=0）——NMS 空转、画框为空。v2/v3 已半废弃此入口，真正推理走 `examples/detector.c` 的 `test_detector`（详见[痛点 §5](/versions/v3/painpoints#_5-快照-bug-test_yolo-用未初始化的-l-side-pain-test-yolo-side-bug)）。
