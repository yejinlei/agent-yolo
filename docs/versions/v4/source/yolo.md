# yolo.c

YOLOv4 训练/验证/测试命令入口。**注意：v4 快照目录结构与 v1/v2 相同，此文件在 `src/`**（v3 曾移到 `examples/`）——本站引用统一写作 `v4:src/yolo.c`。

> 快照路径：`source/yolov4/snapshot/src/yolo.c`。点击左侧行号可复制/分享行锚点；如 `#L285-L310` 可高亮多行区间。

<SourceViewer file="v4:src/yolo.c" />

**阅读路线**：[L5 train_yolo 训练入口](/versions/v4/source/yolo#L5) → [L285 test_yolo（v2/v3 的 side bug 已修复）](/versions/v4/source/yolo#L285) → [run_yolo 命令分发](/versions/v4/source/yolo)。

> ⚠️ v2/v3 的 `test_yolo()` 用未初始化的 `l.side`（祖传 bug），v4 已修复——改用 `get_network_boxes` 返回的 `nboxes`。
