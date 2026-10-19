# yolov6/models/losses/loss.py

YOLOv6 的损失中枢：**ATSS 预热 + TAL 正式分配、VarifocalLoss、GIoU/SIoU、DFL** 全部在此接线。

> 快照路径：`source/yolov6/snapshot/yolov6/models/losses/loss.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v6:yolov6/models/losses/loss.py" />

**阅读路线**：

- 初始化：[L14-50 ComputeLoss.__init__](#L14-L50)——双分配器接线（[L41 warmup_assigner ATSSAssigner(9)](#L41)、[L42 formal_assigner TAL topk=13](#L42)）、[L46 proj 向量](#L46)（DFL 解码用 linspace(0,16)）
- 总调度：[L52-183 __call__](#L52-L183)——[L85-103 warmup/formal 分派](#L85-L103)（分配结果 `.detach()` 当常量）、[L163 VFL 分类](#L163)、[L175-177 总损失加权](#L175-L177)
- [L194-198 bbox_decode](#L194-L198)——softmax 后与 proj 相乘得期望距离
- [L201-211 VarifocalLoss](#L201-L211)——非对称加权，不是 BCE
- [L214-265 BboxLoss](#L214-L265)——GIoU/SIoU 主损失 + [L267-278 _df_loss](#L267-L278)（左右 bin 加权交叉熵）

## 与 v5 utils/loss.py 的对比

| 维度 | v5 loss.py | v6 loss.py |
|------|------------|------------|
| 标签分配 | wh 比例 + 邻居偏移（内联在 loss 里） | ATSS warmup + TAL 独立模块 |
| 分类损失 | BCE（含 objectness 分支） | VarifocalLoss（无 objectness） |
| 回归损失 | CIoU | GIoU/SIoU + DFL 可选 |
| 权重 | box 0.05 / obj 1.0 / cls 0.5 | class 1.0 / iou 2.5 / dfl 0.5 |
| objectness | tobj = IoU ratio | 无（导出补恒 1 占位） |

更多细节见[损失函数](/versions/v6/losses)与[反向传播推导](/versions/v6/loss-gradient)。
