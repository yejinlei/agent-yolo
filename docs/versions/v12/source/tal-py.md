# ultralytics/utils/tal.py

[TAL](/glossary#tal-task-alignment-learning)（Task Alignment Learning）的标签分配器与锚点工具箱：**385 行文件里，`TaskAlignedAssigner` 在 L14 主体定义**——把 GT 按对齐度量分给 anchor，同时提供 `make_anchors / dist2bbox / bbox2dist` 三个坐标工具函数。v12 对 tal.py 的净改动只有一处：**新增 OOM 时回落 CPU 的 `try/except` 与 `_forward` 拆分**（L73-80，v11 没有），分配逻辑与默认参数完全沿用 v11。

> 快照路径：`source/yolov12/snapshot/ultralytics/utils/tal.py`（385 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v12 里的角色**：v12 论文「Attention 骨干内嵌」是纯结构改动，标签分配侧完全沿用 v11。所以 v12 的 tal.py 与 v11 行为等价（仅新增 OOM 回退、行号整体后移）——**`TaskAlignedAssigner.__init__` 的默认参数仍是 topk=13 / α=1.0 / β=6.0（L29）**。但由于 v12 [loss.py L176](/versions/v12/source/loss-py) 显式传参覆盖，检测训练实际用的仍是 **topk=10、α=0.5、β=6.0**（v8 参数），与 v11 完全一致。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="11" /> | TORCH_1_10 | 版本兼容标记 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="14" :end="258" /> | **TaskAlignedAssigner** | TAL 分配器主体 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="29" :end="37" /> | __init__ | **v12 默认 topk=13, α=1.0, β=6.0**（与 v11 相同） |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="39" :end="88" /> | forward | 五步流水：mask → metric → 去重 → targets → 归一化 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="90" :end="100" /> | get_pos_mask | 三重 mask 相乘得正样本位 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="102" :end="121" /> | get_box_metrics | L120 `align_metric = s^α × u^β` |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="123" :end="125" /> | iou_calculation | [CIoU](/glossary#ciou-loss) 且 clamp 到 0 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="127" :end="161" /> | select_topk_candidates | topk 决定每 GT 多少正样本 anchor |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="163" :end="210" /> | get_targets | one-hot scatter 造分类目标 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="212" :end="229" /> | select_candidates_in_gts | 锚点中心落在 GT 框内 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="231" :end="258" /> | select_highest_overlaps | 一 anchor 被 GT 抢占时按 IoU 归属 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="268" :end="298" /> | RotatedTaskAlignedAssigner | OBB 换 probiou，结构复用 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="301" :end="313" /> | make_anchors | 逐层网格中心 +0.5 偏移 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="316" :end="326" /> | dist2bbox | ltrb 距离 → xywh/xyxy 框 |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="329" :end="332" /> | bbox2dist | xyxy 框 → ltrb 距离（DFL 目标） |
| <SrcLink file="v12:ultralytics/utils/tal.py" :line="335" :end="355" /> | dist2rbox | 旋转框解码 |

<SourceViewer file="v12:ultralytics/utils/tal.py" />

## TaskAlignedAssigner（L14）：默认参数不变

<SrcLink file="v12:ultralytics/utils/tal.py" :line="29" :end="37" /> `__init__` 的默认参数与 v11 完全一致：

```python
def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
    super().__init__()
    self.topk = topk
    self.num_classes = num_classes
    self.bg_idx = num_classes
    self.alpha = alpha
    self.beta = beta
    self.eps = eps
```

- **v8 / v9 / v10**：`topk=10, alpha=0.5, beta=6.0`——一个 GT 最多监督 10 个正样本 anchor，对齐度量里定位项权重 β 主导、分类项 α 用「开方」压权；
- **v11 与 v12**：`topk=13, alpha=1.0, beta=6.0`——`topk` 从 10 提到 13（更接近「每个 GT 更多正样本」的 [multi-positive](/glossary#multi-positive-多正样本) 学习策略）、`alpha` 从 0.5 提到 1.0（分类项从「开方压权」变成「原权」，与定位项等权竞争）。

**关键提醒**：以上默认只对「外部直接实例化 `TaskAlignedAssigner()`」生效。[v8DetectionLoss.__init__（loss.py L176）](/versions/v12/source/loss-py) 显式传 `topk=tal_topk（默认 10）, alpha=0.5, beta=6.0`，因此 `yolov12.yaml` 检测训练实际用的仍是 v8 参数。

> **v1~v11 读者路标**：v6 读者注意 v6 论文的 TAL 分配器默认就是 topk=13 / α=1.0 / β=6.0——v11 把 v8/v9 时代的「topk=10 / α=0.5」回调到 v6 论文原值、v12 沿用，但**只在类默认层面**；检测损失走 loss.py L176 显式传参，仍用 v8 值。v8/v11 读者按 v8 tal.py 走一遍就能读懂 v12 的分配机制。v10 读者请把 v10 里 tal.py 的 topk=10 / α=0.5 视为历史遗留——v10 用两次实例化取 topk=10 与 topk=1 实现「一致性双分配」，v12 完全放弃了这套设计。

## forward / _forward 流水（L39-118）

整个 `forward` 挂在 <SrcLink file="v12:ultralytics/utils/tal.py" :line="39" /> 的 `@torch.no_grad()` 下——分配是纯训练期路由，永不回传梯度（loss.py 调用前还额外 detach 了一次预测，双保险）。

1. **空 GT 早退**<SrcLink file="v12:ultralytics/utils/tal.py" :line="64" :end="71" />：`n_max_boxes == 0` 时直接返回全背景（`bg_idx = num_classes`）；
2. **OOM 回退**<SrcLink file="v12:ultralytics/utils/tal.py" :line="73" :end="80" />：`_forward` 抛 `torch.OutOfMemoryError` 时把张量搬到 CPU 重算再搬回——**v12 新增、v11 没有**；
3. **求正样本位**<SrcLink file="v12:ultralytics/utils/tal.py" :line="102" :end="104" />：`get_pos_mask`——「中心在 GT 内」×「对齐度量 topk」×「GT 有效」三重 mask 相乘；
4. **抢占去重**<SrcLink file="v12:ultralytics/utils/tal.py" :line="106" />：`select_highest_overlaps`——一个 anchor 被多个 GT 选中时，归 IoU 最高者；
5. **造目标**<SrcLink file="v12:ultralytics/utils/tal.py" :line="109" />：`get_targets` 按 `target_gt_idx` 抄 GT 标签与框；
6. **归一化软标签**<SrcLink file="v12:ultralytics/utils/tal.py" :line="111" :end="116" />：`target_scores × pos_overlaps / pos_align_metrics`——对齐度量归一后乘回 one-hot，得到软分类目标。

## get_box_metrics（L132-155）：对齐度量一行算

- L143 `bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]`——取的是 **anchor 对该 GT 类别的预测分**，不是 max；
- L148 `overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)`——`iou_calculation`<SrcLink file="v12:ultralytics/utils/tal.py" :line="153" :end="155" /> 用 [CIoU](/glossary#ciou-loss) 且 `clamp_(0)`；
- <SrcLink file="v12:ultralytics/utils/tal.py" :line="150" /> `align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)`——**α 与 β 的幂运算决定分类项与定位项的相对权重**：
  - v8/v9/v10 与 v12 检测实际用：`s^0.5 × u^6.0`（loss.py L176 显式传 α=0.5）；
  - v12 类默认：`s^1.0 × u^6.0`——分类项原权、定位项六次方；
  - **检测训练实际用仍是 0.5**（loss.py L176 覆盖）。

## select_topk_candidates（L157-190）：topk 决定正样本数

- <SrcLink file="v12:ultralytics/utils/tal.py" :line="174" /> `torch.topk(metrics, self.topk, dim=-1, largest=largest)`——对每个 GT 取对齐度量前 topk 个 anchor 的索引与值。**topk=1 时每个 GT 只留下 1 个正样本（v10 one2one 分支就是这样）**；
- L178 `topk_idxs.masked_fill_(~topk_mask, 0)`——无效 GT 的索引统一填 0；
- <SrcLink file="v12:ultralytics/utils/tal.py" :line="181" :end="185" /> `count_tensor` 逐 k `scatter_add_` 计数——topk=1 时循环体只执行一次；
- <SrcLink file="v12:ultralytics/utils/tal.py" :line="188" /> `count_tensor.masked_fill_(count_tensor > 1, 0)`——计数超过 1 的位置清零；
- L190 返回 float mask。

## select_highest_overlaps（L265-295）：GT 抢占

<SrcLink file="v12:ultralytics/utils/tal.py" :line="265" :end="295" />：`fg_mask = mask_pos.sum(-2)` 统计每个 anchor 被几个 GT 选中；大于 1 时构造 `mask_multi_gts`，用 `overlaps.argmax(1)` 找出 IoU 最高的 GT，`torch.where` 把冲突 anchor 划给它；最后 `target_gt_idx = mask_pos.argmax(-2)` 给出每个正 anchor 的归属 GT。

**topk 越激进，冲突越频繁**——类默认 topk=13 时 GT 之间更容易抢同一个 anchor（v12 检测实际用 10），`select_highest_overlaps` 就承担更多工作量。这也是「topk 越大 ≠ 无脑好」的机制根源。

## get_targets（L192-238）：软标签的底料

<SrcLink file="v12:ultralytics/utils/tal.py" :line="192" :end="238" />：L216-219 算出展平索引；L222 抄 GT 框；L227-233 用 `scatter_` 造 one-hot（注释自称比 `F.one_hot()` 快 10 倍）；L235-236 用 `fg_scores_mask` 把非正样本位置清零。

## RotatedTaskAlignedAssigner（L298）：OBB 专用

<SrcLink file="v12:ultralytics/utils/tal.py" :line="298" :end="340" /> 只覆写两个方法：

- `iou_calculation`——把 `bbox_iou(CIoU=True)` 换成 `bbox_iou(RIoU=True)`（probiou）；
- `select_candidates_in_gts`——把「锚点中心在 GT 框内」的判据改成「四角点在旋转框内」。

主干 `forward / get_box_metrics / select_topk_candidates / get_targets` 全部靠继承，与标准 TaskAlignedAssigner 同源。v12 的 yolov12-obb.yaml 通过 [v8OBBLoss（loss.py L612）](/versions/v12/source/loss-py) 实例化这个类，显式传 `topk=10, alpha=0.5, beta=6.0`——**与检测主路径同值**，也不读 tal.py 的类默认。

## 工具函数：锚点与框坐标

- **make_anchors**<SrcLink file="v12:ultralytics/utils/tal.py" :line="333" :end="346" />：逐层生成网格中心（+0.5 偏移落在格子中心）与 `stride_tensor`——head 的 `_inference` 与 loss 的 `__call__` 都靠它；
- **dist2bbox**<SrcLink file="v12:ultralytics/utils/tal.py" :line="348" :end="357" />：ltrb 距离 + 锚点 → 框（xywh 或 xyxy），无 assert；
- **bbox2dist**<SrcLink file="v12:ultralytics/utils/tal.py" :line="360" :end="363" />：反向——xyxy 框转 ltrb 并 `clamp_(0, reg_max - 0.01)`，是 [DFL](/glossary#dfl-distribution-focal-loss) 的目标构造，与 dist2bbox 近似互逆；
- **dist2rbox**<SrcLink file="v12:ultralytics/utils/tal.py" :line="366" :end="385" />：OBB 的旋转框解码，检测主路径不涉及。

## 与 v6 / v8 / v10 / v11 TAL 的对比

| 维度 | v6 论文原 | v8 | v10 | v11 | **v12** |
|------|-----------|-----|-----|-----|---------|
| assigner 结构 | 独立实现 | 8.0.100 同结构 | 8.1.34 基线复用 | 8.3.0 类默认上调 | **与 v11 行为一致（多 OOM 回退）** |
| __init__ 默认 topk | 13 | 10 | 10 | 13 | **13** |
| __init__ 默认 α / β | 1.0 / 6.0 | 0.5 / 6.0 | 0.5 / 6.0 | 1.0 / 6.0 | **1.0 / 6.0** |
| 检测损失实际用 | — | topk=10, α=0.5 | topk=10 与 1 各一份 | topk=10, α=0.5 | **topk=10, α=0.5**（loss.py L176 覆盖） |
| OBB 用 | topk=13, α=1.0 | topk=10, α=0.5 | 同 v8 | 同 v8 | **同 v8（topk=10, α=0.5）** |
| 一致性约束 | 无 | 无 | 同分配器两次实例化 | 无 | **无** |

## 与其他文件的关系

- 分配器在损失里的实例化：<SrcLink file="v12:ultralytics/utils/loss.py" :line="176" /> `v8DetectionLoss.__init__`（显式传 topk=tal_topk=10, α=0.5）；<SrcLink file="v12:ultralytics/utils/loss.py" :line="618" /> `v8OBBLoss.__init__`（显式传 topk=10, α=0.5，与检测同值）
- 分配器在推理侧不出现——`forward` 挂在 `@torch.no_grad()` 下，仅训练期调用
- 建模入口：<SrcLink file="v12:ultralytics/nn/tasks.py" :line="299" /> `DetectionModel`
- 结构描述：<SrcLink file="v12:ultralytics/cfg/models/v12/yolov12.yaml" :line="6" :end="12" /> scales 五档

分配机制的公式解释见 [loss-gradient](/versions/v12/loss-gradient)；头侧的 Detect 见 [head.py 源码页](/versions/v12/source/head-py)；损失总装见 [loss.py 源码页](/versions/v12/source/loss-py)；结构配置见 [yolov12.yaml 源码页](/versions/v12/source/yolov12-yaml)；端到端训练链见 [code-walkthrough](/versions/v12/code-walkthrough)。
