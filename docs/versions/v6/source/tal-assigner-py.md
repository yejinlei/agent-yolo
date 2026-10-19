# yolov6/assigners/tal_assigner.py

**TaskAlignedAssigner**：TAL（Task Alignment Learning）的 PyTorch 实现，源自 TOOD、快照注明参考 PP-YOLOE。用"分类分数 × IoU"的对齐度量选正样本，让分类与定位目标显式对齐。

> 快照路径：`source/yolov6/snapshot/yolov6/assigners/tal_assigner.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v6:yolov6/assigners/tal_assigner.py" />

**阅读路线**：

- [L22-95 forward 主流程](#L22-L95)——整体 `@torch.no_grad()`，分配结果是常量
- 对齐度量：[L117-133 get_box_metrics](#L117-L133)——`align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)`（L131，α=1.0、β=6.0）
- top-k 选择：[L135-150 select_topk_candidates](#L135-L150)——top-13 且度量 >0
- 软标签生成：[L152-173 get_targets](#L152-L173)——one-hot × 归一化对齐度量，喂给 VarifocalLoss
- 冲突消解在 [assigner_utils.py 的 select_highest_overlaps](https://github.com/yejinlei/agent-yolo/blob/main/source/yolov6/snapshot/yolov6/assigners/assigner_utils.py#L46)（一个 anchor 被多个 GT 选中时保留 IoU 最高者）

## 与 v5 分配逻辑的对比

| 维度 | v5 | v6 TAL |
|------|----|--------|
| 正样本判据 | anchor/GT wh 比例 < 4.0 + 网格邻居偏移 | 对齐度量 top-13 且 anchor 中心在 GT 内 |
| 每头独立性 | 按 anchor 形状独立分配 | 全局统一分配到三尺度 |
| 分类目标 | 硬标签 + obj=IoU ratio | **软标签 = 归一化对齐度量** |
| 动机 | 提高 GT 覆盖 | 分类-定位一致性对齐 |
