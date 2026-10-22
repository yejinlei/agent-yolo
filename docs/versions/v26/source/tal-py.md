# utils/tal.py

**这个文件在 v26 里的角色**：v26 的 TAL（Task-Aligned Assigner）分配器文件。核心是 <SrcLink file="v26:utils/tal.py" :line="14" :end="14"/> `TaskAlignedAssigner`，v13 引入的按"分类分^α × IoU^β"综合分数做 anchor 分配的算法；v26 新增 <SrcLink file="v26:utils/tal.py" :line="421" :end="421"/> `RotatedTaskAlignedAssigner` 用于 OBB；并导出 `make_anchors / dist2bbox / bbox2dist / dist2rbox / rbox2dist` 一整套 anchor 几何工具函数。

> **注意**：`_grouped_topk` **不在此文件**，它作为 `Detect` 的 `@staticmethod` 定义在 <SrcLink file="v26:nn/modules/head.py" :line="92" :end="92"/>，服务于 head 端的 top-k 后处理；与 TAL 分配的 top-k 是两套独立的加速技巧。

## 关键 class 与函数速查

| 类/函数 | 行段 | 用途 |
| --- | --- | --- |
| <SrcLink file="v26:utils/tal.py" :line="14" :end="14"/> `TaskAlignedAssigner` | v26 主线分配器 | 核心 |
| <SrcLink file="v26:utils/tal.py" :line="396" :end="396"/> `PoseTaskAlignedAssigner` | Pose 专用分配器 | pose |
| <SrcLink file="v26:utils/tal.py" :line="421" :end="421"/> `RotatedTaskAlignedAssigner` | **v26 新增**——OBB 旋转框分配器 | obb |
| <SrcLink file="v26:utils/tal.py" :line="464" :end="464"/> `make_anchors` | 生成 anchor points | 通用 |
| <SrcLink file="v26:utils/tal.py" :line="493" :end="493"/> `dist2bbox` | dist → bbox 解码 | 通用 |
| <SrcLink file="v26:utils/tal.py" :line="515" :end="515"/> `bbox2dist` | bbox → dist 编码（`reg_max` 可选） | 通用 |
| <SrcLink file="v26:utils/tal.py" :line="533" :end="533"/> `dist2rbox` | dist + angle → 旋转框 | obb |
| <SrcLink file="v26:utils/tal.py" :line="554" :end="554"/> `rbox2dist` | 旋转框 → dist + angle | obb |

## TaskAlignedAssigner 关键行段

- <SrcLink file="v26:utils/tal.py" :line="31" :end="40"/> `__init__` 默认参数：`topk=13, alpha=1.0, beta=6.0, stride=[8, 16, 32]`——沿用 v13 默认。
- <SrcLink file="v26:utils/tal.py" :line="53" :end="61"/> 属性赋值：`topk`, `topk2 = topk2 or topk`, `num_classes`, `alpha`, `beta`, `stride_val = stride[1]`（多尺度时取中间层作为最小边阈值）。
- <SrcLink file="v26:utils/tal.py" :line="63" :end="64"/> `@torch.no_grad()` 装饰 forward——分配器不参与反传。
- <SrcLink file="v26:utils/tal.py" :line="89" :end="139"/> forward 主流程：
  - <SrcLink file="v26:utils/tal.py" :line="91" :end="99"/> `n_max_boxes == 0` 早退分支。
  - <SrcLink file="v26:utils/tal.py" :line="106" :end="128"/> OOM 兜底——**CUDA OOM 时逐图像重试**：`for i, self.n_max_boxes in enumerate(last_gt_idx): result.append(self._forward(...))`。
- <SrcLink file="v26:utils/tal.py" :line="142" :end="183"/> `_forward` 主体：`get_pos_mask → select_highest_overlaps → get_targets`。
- <SrcLink file="v26:utils/tal.py" :line="183" :end="211"/> `get_pos_mask`：候选池筛选。
- <SrcLink file="v26:utils/tal.py" :line="212" :end="242"/> `get_box_metrics`：核心公式 <SrcLink file="v26:utils/tal.py" :line="235" :end="235"/> `align_values = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)`——**分类分^α × IoU^β**。
- <SrcLink file="v26:utils/tal.py" :line="259" :end="277"/> `select_topk_candidates`：`torch.topk(metrics, self.topk, dim=-1)` 选 topk。
- <SrcLink file="v26:utils/tal.py" :line="288" :end="330"/> `get_targets`：生成 target labels/boxes/scores。
- <SrcLink file="v26:utils/tal.py" :line="328" :end="360"/> `select_candidates_in_gts`：把 anchor 按 GT 框位置分桶，用 `wh_mask` 把小 GT 边扩到 `stride_val`。
- <SrcLink file="v26:utils/tal.py" :line="361" :end="394"/> `select_highest_overlaps`：对每个 GT 保留最高 overlap 的 anchor，`topk2 != topk` 时先取 topk2 再取 top-1（配合 E2ELoss 的 `tal_topk=7, tal_topk2=1`）。

## v26 新增：RotatedTaskAlignedAssigner

<SrcLink file="v26:utils/tal.py" :line="421" :end="421"/> `class RotatedTaskAlignedAssigner(TaskAlignedAssigner)`——继承 TAL，用于 `OBB26` 与 `v8OBBLoss`：
- <SrcLink file="v26:utils/tal.py" :line="409" :end="409"/> `pair_overlaps`：用 <SrcLink file="v26:utils/loss.py" :line="229" :end="229"/> `RotatedBboxLoss` 提供的 IoU 计算（含旋转）。
- <SrcLink file="v26:utils/tal.py" :line="424" :end="428"/> `iou_calculation` 覆盖：走 `RotatedBboxLoss` 的 `_compute_ciou`。

## PoseTaskAlignedAssigner

<SrcLink file="v26:utils/tal.py" :line="396" :end="396"/> `class PoseTaskAlignedAssigner(TaskAlignedAssigner)`——`sigmas` 参数驱动，`pair_overlaps` 额外融合关键点可见度。

## Anchor 几何工具

- <SrcLink file="v26:utils/tal.py" :line="464" :end="464"/> `make_anchors(feats, strides, grid_cell_offset=0.5)`：从特征图 shape 生成 anchor points。
- <SrcLink file="v26:utils/tal.py" :line="493" :end="493"/> `dist2bbox(distance, anchor_points, xywh=True, dim=-1)`：解码。
- <SrcLink file="v26:utils/tal.py" :line="515" :end="515"/> `bbox2dist(anchor_points, bbox, reg_max=None)`：`reg_max` 传 `None` 时不做 DFL 分箱量化——v26 主线 `reg_max=1` 也走这条路径。
- <SrcLink file="v26:utils/tal.py" :line="533" :end="533"/> `dist2rbox`：`pred_dist + pred_angle + anchor_points → 旋转框`。
- <SrcLink file="v26:utils/tal.py" :line="554" :end="554"/> `rbox2dist`：反向编码。

## 与 E2ELoss 的对接

<SrcLink file="v26:utils/loss.py" :line="1288" :end="1289"/> `E2ELoss.__init__` 里 one2many 用 `tal_topk=10`，one2one 用 `tal_topk=7, tal_topk2=1`——`topk2` 参数正是 <SrcLink file="v26:utils/tal.py" :line="384" :end="387"/> `if self.topk2 != self.topk: max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices` 触发的双阶段 top-k。

## SourceViewer

<SourceViewer file="v26:utils/tal.py"/>

## 参考

- [End2End 概念](../concepts.md)
- [Losses 页](../losses.md)
- [头文件源码](./head-py.md)
- [Loss 源码](./loss-py.md)
- [v26 主页](../index.md)
