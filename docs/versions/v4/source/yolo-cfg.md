# yolov4.cfg

YOLOv4 网络配置（commit `4131ae1`）：**162 层** = 110 conv + 23 shortcut + 21 route + 3 maxpool + 2 upsample + 3 yolo。CSPDarknet53 骨干 + SPP + PANet + CIoU + Mosaic + 18 anchors。

> 快照路径：`source/yolov4/snapshot/cfg/yolov4.cfg`。点击左侧行号可复制/分享行锚点；如 `#L967-L982` 可高亮多行区间。

<SourceViewer file="v4:cfg/yolov4.cfg" />

**阅读路线**：

- 骨干：[L8-26 net 段（burn_in/mosaic/增强）](/versions/v4/source/yolo-cfg#L8-L26) → [L30 第一个 conv](/versions/v4/source/yolo-cfg#L30) → [L56 CSP 分流 route](/versions/v4/source/yolo-cfg#L56) → [L95 CSP concat](/versions/v4/source/yolo-cfg#L95) → [L186 CSP段3 s2 入口](/versions/v4/source/yolo-cfg#L186) → [L395 CSP段4 s2](/versions/v4/source/yolo-cfg#L395) → [L609 CSP段5 s2](/versions/v4/source/yolo-cfg#L609)
- SPP：[L763 maxpool 5×5](/versions/v4/source/yolo-cfg#L763) → [L772 maxpool 9×9](/versions/v4/source/yolo-cfg#L772) → [L779 maxpool 13×13](/versions/v4/source/yolo-cfg#L779) → [L790 concat 4 份](/versions/v4/source/yolo-cfg#L790)
- FPN + 头A：[L818 upsample](/versions/v4/source/yolo-cfg#L818) → [L826 route 85 回连](/versions/v4/source/yolo-cfg#L826) → [L950 1×1→255](/versions/v4/source/yolo-cfg#L950) → [L958 yolo@76×76](/versions/v4/source/yolo-cfg#L958-L982)
- PANet + 头B：[L978 conv s2 下采样](/versions/v4/source/yolo-cfg#L978) → [L986 concat](/versions/v4/source/yolo-cfg#L986) → [L1037 1×1→255](/versions/v4/source/yolo-cfg#L1037) → [L1045 yolo@38×38](/versions/v4/source/yolo-cfg#L1045-L1069)
- PANet + 头C：[L1065 conv s2 下采样](/versions/v4/source/yolo-cfg#L1065) → [L1073 concat](/versions/v4/source/yolo-cfg#L1073) → [L1124 1×1→255](/versions/v4/source/yolo-cfg#L1124) → [L1132 yolo@19×19](/versions/v4/source/yolo-cfg#L1132-L1155)

> cfg 代码块在站内一律用 ```ini 展示（shiki 不识别 cfg 语言标签）。
