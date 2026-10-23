# ultralytics/cfg/models/master/v0/det/yolo-master-l.yaml

YOLO-Master l 档结构配置：**44 行**，与 n 档**逐行结构完全一致**（13 层 backbone + 13 层 head + 4×ES_MOE + 2×A2C2f + Detect），唯一差别在 [L5-7 scales 复合缩放](#L5-L7)：`[1.00, 1.00, 512]`。官方摘要：560 层 / 30,163,180 参数 / 119.7 GFLOPs——是 n 档的 **~34× 参数、~13.6× FLOPs**。

> 快照路径：`source/yolomaster/snapshot/ultralytics/cfg/models/master/v0/det/yolo-master-l.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/cfg/models/master/v0/det/yolo-master-l.yaml" />

**阅读路线**：

- [L5-7 scales 复合缩放](#L5-L7)：`l: [1.00, 1.00, 512]`——**depth=1.00（无 depth 缩减）、width=1.00（通道不缩放）、max_channels=512**
- **max_channels 的隐性陷阱**：n 档 max_channels=1024，l 档反降到 512——`parse_model` 里的 `c2 = min(c2, max_channels) * width` 会让 l 档所有超过 512 的通道**都被截断**。这也是为什么 l 档 P5 层的 yaml 写 `[1024, ...]` 但实际建出的层通道是 `min(1024, 512) * 1.00 = 512`（[痛点 2 §max_channels 隐性降级](../painpoints#2-l-档-max_channels-隐性降级-pain-scales-mismatch)）
- 骨干 [L10-24](#L10-L24)：与 n 档完全一致——[L15 ES_MOE(256)](#L15)、[L18 ES_MOE(512)](#L18)、[L21 ES_MOE(512)](#L21)、[L24 ES_MOE(1024)](#L24)
- Head [L27-44](#L27-L44)：与 n 档完全一致，Detect 三尺度收尾在 [L44](#L44)

## 与 n 档 yolo-master-n.yaml 的对比

| 维度 | Master n 档 | Master l 档 |
|------|-------------|-------------|
| 行数 | 45 | 44 |
| scales depth | 0.50 | **1.00**（层数翻倍） |
| scales width | 0.25 | **1.00**（通道翻倍） |
| scales max_channels | **1024** | **512**（反而更小） |
| yaml 结构 | 4×ES_MOE + 2×A2C2f | 完全一致 |
| 官方参数 | 2,694,364 | 30,163,180（×11.2） |
| 官方 GFLOPs | 8.8 | 119.7（×13.6） |
| 实际 P5 通道 | min(1024, 1024)×0.25 = 256 | **min(1024, 512)×1.00 = 512** |

**max_channels 与 width 的耦合**：n 档是 `1024 × 0.25 = 256` 的最大通道，l 档是 `512 × 1.00 = 512`——**l 档的最大通道只比 n 档大 2×，但 depth 和 width 都翻倍了**，参数增长远大于通道增长。这是 Master 团队为保持「ES_MOE 稀疏计算在 l 档依然可控」做的取舍。

**结构一致性**：由于两个 yaml 的模块序列一致，`parse_model` 的走查路径完全相同——这也是为什么 [痛点 3 §只公开 n/l 两档](../painpoints#3-只公开-nl--1-个-esmoe-n-三个-yaml-pain-yaml-incomplete) 是 Master 的一个遗憾：中间档（s/m/x）的 scales 需要用户自己插值。

架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)，痛点见 [painpoints 页](../painpoints)。

## 常见问题

**Q：为什么 l 档 max_channels=512 而不是 1024？**
A：Master 团队的选择——l 档的 width=1.00 意味着通道不再缩放，但 max_channels=512 会强制所有通道上限为 512。这与「width=1.00 保持通道不缩放」的直觉相悖，但也限制了 l 档不会像 x 档那样膨胀。用户若想要更大的模型，需自己修改 yaml 或 scales。

**Q：l 档的训练时间？**
A：官方摘要 30,163,180 参数、119.7 GFLOPs——**是 n 档的 13.6× FLOPs**，训练时间也大致相同倍数（3-4× 显存 + 3-4× 时间）。

**Q：l 档在 VisDrone 上的表现？**
A：Master 论文报告 l 档在 COCO 上 mAP 55.5，比 n 档的 44.6 提升 ~11 mAP——但官方未公开 l 档的 VisDrone 权重，需自行训练。

**Q：如何自己写 s/m/x 档 yaml？**
A：参考 n/l 档的 scales，s 档可试 `[0.67, 0.33, 1024]`、m 档 `[0.80, 0.50, 768]`、x 档 `[1.00, 1.00, 768]`——**但官方未验证过这些 scales**，需要自己测试。

**Q：l 档的 ES_MOE 稀疏度会比 n 档更低吗？**
A：ES_MOE 的 top_k=3（默认）在两档完全一致——**sparse 度不变**，但 expert 的通道数翻倍（因为 width=1.00）。这导致 l 档的稀疏节省在 FLOPs 上被抹平，实际加速比不如 n 档。

**Q：l 档训练需要多少显存？**
A：按 8.8 GFLOPs × 119.7/8.8 ≈ 13.6× 估算，n 档 batch=64 单卡可跑，l 档需要 4-8 卡 × 32 batch，或使用 DDP + gradient checkpointing。

**Q：为什么 l 档 yaml 里 P5 写 1024 而不是 512？**
A：为了与其他 yaml（v12/x）保持一致的 yaml 语义——**yaml 里写「理论通道数」，实际通道由 max_channels 截断**。这是 Ultralytics 的通用做法。

**Q：l 档 checkpoint 兼容 n 档吗？**
A：不兼容——**通道数不同**，权重形状不匹配。用户若从 n 档迁移到 l 档，需从头训练或用 partial load。

**其他相关 yaml**：
- [yolo-master-n.yaml](./yolo-master-n-yaml)：n 档结构
- [yolo-master-esmoe-n.yaml](./yolo-master-esmoe-n-yaml)：COCO profile
