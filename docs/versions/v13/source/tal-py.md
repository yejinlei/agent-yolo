# ultralytics/utils/tal.py

[TAL](/glossary#tal-task-alignment-learning)（Task Alignment Learning）的标签分配器与锚点工具箱：**385 行文件里，`TaskAlignedAssigner` 在 L14 主体定义**——把 GT 按对齐度量分给 anchor，同时提供 `make_anchors / dist2bbox / bbox2dist` 三个坐标工具函数。v13 对 tal.py 的净改动**为零**——分配逻辑与默认参数完全沿用 v11/v12（仅 OOM 回退机制与 v12 一致）。

> 快照路径：`source/yolov13/snapshot/ultralytics/utils/tal.py`（385 行，commit `73289949`）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：v13 论文「超图消息传递 + 门控连接」是纯结构改动，标签分配侧完全沿用 v11/v12。所以 v13 的 tal.py 与 v11/v12 行为等价（仅行号整体后移）——**`TaskAlignedAssigner.__init__` 的默认参数仍是 topk=13 / α=1.0 / β=6.0（L29）**。但由于 [loss.py L176](/versions/v13/source/loss-py) 显式传参覆盖，检测训练实际用的仍是 **topk=10、α=0.5、β=6.0**（v8 参数），与 v11/v12 完全一致。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="11" /> | TORCH_1_10 | 版本兼容标记 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="14" :end="258" /> | **TaskAlignedAssigner** | TAL 分配器主体 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="29" :end="37" /> | __init__ | **v13 默认 topk=13, α=1.0, β=6.0**（与 v11/v12 相同） |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="39" :end="88" /> | forward | 五步流水：mask → metric → 去重 → targets → 归一化 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="90" :end="100" /> | get_pos_mask | 三重 mask 相乘得正样本位 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="102" :end="121" /> | get_box_metrics | L120 `align_metric = s^α × u^β` |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="123" :end="125" /> | iou_calculation | [CIoU](/glossary#ciou-loss) 且 clamp 到 0 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="127" :end="161" /> | select_topk_candidates | topk 决定每 GT 多少正样本 anchor |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="163" :end="210" /> | get_targets | one-hot scatter 造分类目标 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="212" :end="229" /> | select_candidates_in_gts | 锚点中心落在 GT 框内 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="231" :end="258" /> | select_highest_overlaps | 一 anchor 被 GT 抢占时按 IoU 归属 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="268" :end="298" /> | RotatedTaskAlignedAssigner | OBB 换 probiou，结构复用 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="301" :end="313" /> | make_anchors | 逐层网格中心 +0.5 偏移 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="316" :end="326" /> | dist2bbox | ltrb 距离 → xywh/xyxy 框 |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="329" :end="332" /> | bbox2dist | xyxy 框 → ltrb 距离（DFL 目标） |
| <SrcLink file="v13:ultralytics/utils/tal.py" :line="335" :end="355" /> | dist2rbox | 旋转框解码 |

<SourceViewer file="v13:ultralytics/utils/tal.py" />

## 类默认与训练实际值（三代之争）

| 项 | tal.py 类默认 | loss.py 显式覆盖 | 训练实际值 |
|-----|--------------|------------------|-----------|
| `topk` | **13**（L29） | loss.py L160 `tal_topk=10` | **10** |
| `α` | 1.0（L29） | loss.py L176 硬编码 `0.5` | **0.5** |
| `β` | 6.0（L29） | loss.py L176 硬编码 `6.0` | **6.0** |

**这个"隐性坑"从 v11 起就有**——loss.py L176 的显式覆盖让「训练实际值」与「tal.py 类默认」不一致，读代码极易误判。

## TAL 五步流水（forward L39-L88）

```python
def forward(self, pred_scores, pred_bboxes, anchor_point, gt_bboxes, gt_labels, mask_gt):
    # 1. get_pos_mask: 三重 mask（in_gt + center_pos_mask + is_visible）
    pos_mask, fg_mask, mask_gt = self.get_pos_mask(...)

    # 2. get_box_metrics: align_metric = s^α × u^β
    align_metric = pred_scores.pow(alpha) * pred_bboxes.pow(beta)

    # 3. 去重：一个 anchor 被多个 GT 抢占时按 IoU 归属
    fg_mask, mask_gt = self.select_highest_overlaps(...)

    # 4. 选 topk：每个 GT 至多 topk 个正样本
    target_scores, target_gt_idx = self.get_targets(...)

    # 5. 归一化：Σ_q = max(sum(target_scores), 1)
    target_scores = target_scores / target_scores.sum()
    return target_scores, target_bboxes, target_scores.detach(), target_gt_idx.detach()
```

- **正样本判定**：三重 mask 相乘——`in_gt`（锚点中心在 GT 内）× `center_pos_mask`（topk 选中的）× `is_visible`（IoU 大于阈值）；
- **对齐度量**：`align_metric = s^α × u^β`，其中 `s = pred_scores.sigmoid()`（分类置信度）、`u = pred_bboxes`（边界框质量，即 CIoU clamp 到 0）；
- **topk 分配**：每个 GT 选出 topk 个 anchor 作为正样本，剩余负样本用 BCE 惩罚。

## 与 v12/v11 的差异

| 项 | v11 | v12 | **v13** |
|-----|-----|-----|---------|
| 文件行数 | 355 | 385 | **385** |
| `topk` 类默认 | 13 | 13 | **13** |
| `α` 类默认 | 1.0 | 1.0 | **1.0** |
| `β` 类默认 | 6.0 | 6.0 | **6.0** |
| OOM 回退 try/except | 无 | 有（L73-80） | **有** |
| `_forward` 拆分 | 无 | 有 | **有** |

**结论**：**v13 tal.py 与 v12 完全一致**，相对 v11 只多了 OOM 回退机制（与 v12 一致）。

## 想快速对上位读者的话

- **v8 读者**：v13 的 tal.py 与 v8 相比，只有类默认参数被改了（topk 10→13、α 0.5→1.0），但 loss.py L176 显式覆盖，**训练实际值不变**；
- **v11/v12 读者**：v13 tal.py 与 v11/v12 完全一致（v12 只加了 OOM 回退，v13 沿用）；
- **v13 结构改动如何影响 TAL？**——**不影响**：v13 论文的超图消息传递与 FullPAD 门控都是「结构骨架层的替换」，标签分配完全沿用 v8 逻辑。

## 参考

- [YOLOv13 损失函数](/versions/v13/losses)
- [loss.py 源码页](/versions/v13/source/loss-py)
- [v12 tal.py](/versions/v12/source/tal-py)
- Yang W. *YOLOv13.* arXiv:2506.17733v2, 2025-11. https://arxiv.org/abs/2506.17733
