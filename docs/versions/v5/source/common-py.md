# models/common.py

YOLOv5 基础模块库：**Conv/Bottleneck/C3/SPP/Focus/Concat/NMS**。所有模块都是 `nn.Module` 子类，可被 yaml 引用。

> 快照路径：`source/yolov5/snapshot/models/common.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v5:models/common.py" />

**阅读路线**：

- 基础 Conv：[L33-45 Conv（BN+SiLU）](#L33-L45)
- 残差模块：[L94-104 Bottleneck](#L94-L104) → [L126-138 C3](#L126-L138) → [L107-123 BottleneckCSP（v4 遗留）](#L107-L123)
- 下采样与感受野：[L163-172 Focus](#L163-L172) → [L149-160 SPP](#L149-L160)
- 辅助模块：[L203-210 Concat](#L203-L210) → [L213-223 NMS](#L213-L223) → [L226-293 autoShape（推理包装器）](#L226-L293)

## 关键模块速查

| 模块 | 行号 | 功能 | 备注 |
|------|------|------|------|
| Conv | L33 | Conv + BN + SiLU | 全网络基础单元 |
| Bottleneck | L94 | 1×1 + 3×3 + shortcut | 残差块 |
| C3 | L126 | CSP Bottleneck 3 conv | 比 v4 少一次 BN/Act |
| SPP | L149 | MaxPool 5/9/13 | 感受野扩展 |
| Focus | L163 | 4 路切片 + conv | v5 stem |
| Concat | L203 | torch.cat | PANet 拼接 |
| NMS | L213 | torchvision.ops.nms | 推理后处理 |
