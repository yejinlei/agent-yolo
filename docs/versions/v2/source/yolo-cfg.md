# yolo.cfg

YOLOv2 完整网络与训练配置（darknet 快照版，`cfg/yolo.cfg`，COCO 80 类）：

- <SrcLink file="v2:cfg/yolo.cfg" :line="1" :end="18" />：`[net]` 训练超参（batch=64、416×416、lr、decay、hue=.1）
- <SrcLink file="v2:cfg/yolo.cfg" :line="20" :end="183" />：backbone（22 个 conv，全部 `batch_normalize=1` + leaky，5 个 maxpool）
- <SrcLink file="v2:cfg/yolo.cfg" :line="203" :end="210" />：passthrough（route -9 → reorg stride=2 → route -1,-3 拼接）
- <SrcLink file="v2:cfg/yolo.cfg" :line="212" :end="225" />：检测头（3072→1024→1×1→425）
- <SrcLink file="v2:cfg/yolo.cfg" :line="227" :end="244" />：`[region]` 头（anchors、bias_match=1、softmax=1、四项 scale、random=0）

> 注意：本快照共 **31 层**（22 conv + 5 maxpool + 2 route + 1 reorg + 1 region），与论文/流传的"24 conv"口径不同，一律以本 cfg 为准。

<SourceViewer file="v2:cfg/yolo.cfg" />
