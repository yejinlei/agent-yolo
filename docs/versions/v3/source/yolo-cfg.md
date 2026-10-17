# yolov3.cfg

YOLOv3 网络配置（commit `e84933b`）：**107 层** = 75 conv + 23 shortcut + 4 route + 2 upsample + 3 yolo。三尺度头 + 9 anchors 三组 mask + 多尺度训练内置。

> 快照路径：`source/yolov3/snapshot/cfg/yolov3.cfg`。点击左侧行号可复制/分享行锚点；如 `#L607-L615` 可高亮多行区间。

<SourceViewer file="v3:cfg/yolov3.cfg" />

**阅读路线**：

- 骨干：[L8-23 net 段（burn_in/steps/增强）](/versions/v3/source/yolo-cfg#L8-L23) → [L25 第一个 conv](/versions/v3/source/yolo-cfg#L25) → [L59 第一个 shortcut](/versions/v3/source/yolo-cfg#L59) → [L115 段3 s2 入口](/versions/v3/source/yolo-cfg#L115) → [L286 段4 s2](/versions/v3/source/yolo-cfg#L286) → [L461 段5 s2](/versions/v3/source/yolo-cfg#L461)
- 检测头：[L551 头A 卷积栈](/versions/v3/source/yolo-cfg#L551) → [L599 1×1→255](/versions/v3/source/yolo-cfg#L599) → [L607 yolo@13×13](/versions/v3/source/yolo-cfg#L607-L615) → [L618-633 回连1（route→conv→upsample→concat 61）](/versions/v3/source/yolo-cfg#L618-L633) → [L693 yolo@26×26](/versions/v3/source/yolo-cfg#L693-L701) → [L705-720 回连2（concat 36）](/versions/v3/source/yolo-cfg#L705-L720) → [L780 yolo@52×52](/versions/v3/source/yolo-cfg#L780-L788)

> cfg 代码块在站内一律用 ```ini 展示（shiki 不识别 cfg 语言标签）。
