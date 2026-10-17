# yolo.cfg

YOLOv1 完整网络与训练配置（darknet 快照版，`cfg/yolo.cfg`）：

- <SrcLink file="cfg/yolo.cfg" :line="1" :end="14" />：`[net]` 训练超参（lr、steps、增强）
- <SrcLink file="cfg/yolo.cfg" :line="210" :end="219" />：两层 FC 与 dropout
- <SrcLink file="cfg/yolo.cfg" :line="221" :end="234" />：`[detection]` 头（side=7、num=2、sqrt=1、四项 scale）

> 注意：当前上游 darknet 仓库的 `cfg/yolov1.cfg` 已被 v2+ 实验值覆盖（`num=3`、`dropout=0.5` 等差异），研究 v1 一律以本快照为准。

<SourceViewer file="cfg/yolo.cfg" />
