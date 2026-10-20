# ultralytics/utils/tal.py

TAL（[Task Alignment Learning](/glossary#tal-task-alignment-learning)）的标签分配器与锚点工具箱：**TaskAlignedAssigner 把 GT 按对齐度量分给 anchor，make_anchors / dist2bbox / bbox2dist 负责锚点生成与框坐标互转**。v10 对这个文件是「零改动的最大受益者」——一致性双分配没有新建 assigner 类，[loss.py](/versions/v10/source/loss-py) L719-720 两次实例化只差 topk=10/1。

> 快照路径：`source/yolov10/snapshot/ultralytics/utils/tal.py`（345 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v10 里的角色**：v10 首发虽把 tal.py 列入修改文件清单，但其类结构与 ultralytics 8.1.34 基线基本一致（复用未改布局）；「一对一分配」的全部秘密在调用侧的 topk 参数，不在分配器本体。也就是说：**v8 读者对这份代码的所有理解原封不动适用于 v10**——读懂 topk 如何进 `select_topk_candidates`，就同时读懂了 v8 的多正样本分配和 v10 的一对一分配。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="13" :end="258" /> | TaskAlignedAssigner | TAL 分配器主体 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="28" :end="36" /> | __init__ | 默认 topk=13；v10 经 loss.py 传 10/1 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="38" :end="88" /> | forward | 五步流水：mask → metric → 去重 → targets → 归一化 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="90" :end="100" /> | get_pos_mask | 三重 mask 相乘得正样本位 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="102" :end="121" /> | get_box_metrics | L120 `align_metric = s^α × u^β` |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="123" :end="125" /> | iou_calculation | [CIoU](/glossary#ciou-loss) 且 clamp 到 0 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="127" :end="161" /> | select_topk_candidates | **topk=1 时每 GT 仅 1 个正样本** |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="163" :end="210" /> | get_targets | one-hot scatter 造分类目标 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="212" :end="229" /> | select_candidates_in_gts | 锚点中心落在 GT 框内 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="231" :end="258" /> | select_highest_overlaps | 一 anchor 被 GT 抢占时按 IoU 归属 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="261" :end="291" /> | RotatedTaskAlignedAssigner | OBB 换 probiou，结构复用 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="294" :end="306" /> | make_anchors | 逐层网格中心 +0.5 偏移 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="309" :end="319" /> | dist2bbox | ltrb 距离 → xywh/xyxy 框 |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="322" :end="325" /> | bbox2dist | xyxy 框 → ltrb 距离（DFL 目标） |
| <SrcLink file="v10:ultralytics/utils/tal.py" :line="328" :end="345" /> | dist2rbox | 旋转框解码 |

<SourceViewer file="v10:ultralytics/utils/tal.py" />

## forward 五步流水（L38-88）

整个 `forward` 挂在 <SrcLink file="v10:ultralytics/utils/tal.py" :line="38" /> 的 `@torch.no_grad()` 下——分配是纯训练期路由，永不回传梯度（loss.py 调用前还额外 detach 了一次预测，双保险）。

1. **空 GT 早退**<SrcLink file="v10:ultralytics/utils/tal.py" :line="62" :end="70" />：`n_max_boxes == 0` 时直接返回全背景（`bg_idx = num_classes`）；
2. **求正样本位**<SrcLink file="v10:ultralytics/utils/tal.py" :line="72" :end="74" />：`get_pos_mask`——「中心在 GT 内」（L92）×「对齐度量 topk」（L96）×「GT 有效」（L98）三重 mask 相乘；
3. **抢占去重**<SrcLink file="v10:ultralytics/utils/tal.py" :line="76" />：`select_highest_overlaps`——一个 anchor 被多个 GT 选中时，归 IoU 最高者；
4. **造目标**<SrcLink file="v10:ultralytics/utils/tal.py" :line="79" />：`get_targets` 按 `target_gt_idx` 抄 GT 标签与框；
5. **归一化软标签**<SrcLink file="v10:ultralytics/utils/tal.py" :line="82" :end="86" />：`target_scores × pos_overlaps / pos_align_metrics`——对齐度量归一后乘回 one-hot，得到软分类目标。

## get_box_metrics（L102-121）：对齐度量一行算

- L113 `bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]`——取的是 **anchor 对该 GT 类别的预测分**，不是 max；
- L118 `overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)`——`iou_calculation`<SrcLink file="v10:ultralytics/utils/tal.py" :line="123" :end="125" /> 用 [CIoU](/glossary#ciou-loss) 且 `clamp_(0)`；
- <SrcLink file="v10:ultralytics/utils/tal.py" :line="120" /> `align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)`——v10 双分支同用 α=0.5、β=6.0（loss.py L166 传入），定位项平方放大，分类项开方缓压。

## select_topk_candidates（L127-161）：topk=1 就是一对一

这段是 v10 一致性双分配的物理载体，逐行拆：

- <SrcLink file="v10:ultralytics/utils/tal.py" :line="145" /> `torch.topk(metrics, self.topk, dim=-1, largest=largest)`——对每个 GT 取对齐度量前 topk 个 anchor 的索引与值。**topk=1 时，每个 GT 只留下全场对齐度量最高的 1 个 anchor**；
- L149 `topk_idxs.masked_fill_(~topk_mask, 0)`——无效 GT（`topk_mask=False`）的索引统一填 0；
- <SrcLink file="v10:ultralytics/utils/tal.py" :line="152" :end="156" /> `count_tensor` 逐 k `scatter_add_` 计数——topk=1 时循环体只执行一次，每个有效 GT 恰有 1 个 anchor 被计 1；
- <SrcLink file="v10:ultralytics/utils/tal.py" :line="159" /> `count_tensor.masked_fill_(count_tensor > 1, 0)`——计数超过 1 的位置清零，主要清掉「多个无效 GT 都填索引 0」造成的假阳性；
- L161 返回 float mask。对比 topk=10：同一 GT 最多 10 个 anchor 拿正样本（[multi-positive](/glossary#multi-positive-多正样本)），这正是 v8 训练稳、推理却冒冗余框的根源。

**注意**：topk=1 不等于「全局不冲突」——两个 GT 的 top-1 可能是同一个 anchor，这正是下一步 `select_highest_overlaps` 存在的理由。

## select_highest_overlaps（L231-258）：GT 抢占

<SrcLink file="v10:ultralytics/utils/tal.py" :line="231" :end="258" />：`fg_mask = mask_pos.sum(-2)` 统计每个 anchor 被几个 GT 选中（L246）；大于 1 时（L247）构造 `mask_multi_gts`，用 `overlaps.argmax(1)`（L249）找出 IoU 最高的 GT，`torch.where` 把冲突 anchor 划给它（L254）；最后 `target_gt_idx = mask_pos.argmax(-2)`（L257）给出每个正 anchor 的归属 GT。

## get_targets（L163-210）：软标签的底料

<SrcLink file="v10:ultralytics/utils/tal.py" :line="163" :end="210" />：L189-190 算出展平索引；L194 抄 GT 框；L200-205 用 `scatter_` 造 one-hot（注释自称比 `F.one_hot()` 快 10 倍）；L207-208 用 `fg_scores_mask` 把非正样本位置清零。出来的是 `(b, h×w, 80)` 的 0/1 目标，再由 forward 第 5 步乘上归一化对齐度量变成软标签。

## 工具函数：锚点与框坐标

- **make_anchors**<SrcLink file="v10:ultralytics/utils/tal.py" :line="294" :end="306" />：逐层生成网格中心（+0.5 偏移落在格子中心）与 `stride_tensor`——head 的 `inference` 与 loss 的 `__call__` 都靠它；
- **dist2bbox**<SrcLink file="v10:ultralytics/utils/tal.py" :line="309" :end="319" />：ltrb 距离 + 锚点 → 框（xywh 或 xyxy），L311 的 `assert` 校验 4 通道；
- **bbox2dist**<SrcLink file="v10:ultralytics/utils/tal.py" :line="322" :end="325" />：反向——xyxy 框转 ltrb 并 `clamp_(0, reg_max - 0.01)`，是 [DFL](/glossary#dfl-distribution-focal-loss) 的目标构造，与 dist2bbox 近似互逆；
- **dist2rbox**<SrcLink file="v10:ultralytics/utils/tal.py" :line="328" :end="345" />：OBB 的旋转框解码，检测主路径不涉及。

`RotatedTaskAlignedAssigner`<SrcLink file="v10:ultralytics/utils/tal.py" :line="261" :end="291" /> 只覆写 `iou_calculation`（换 probiou）与旋转版 `select_candidates_in_gts`（角点判据），主干全靠继承。

## 与 v8/v10 的关系

| 维度 | v6 TAL | v8 TAL | v10 TAL |
|------|--------|--------|---------|
| assigner | 独立实现（见 [v6 tal_assigner 源码页](/versions/v6/source/tal-assigner-py)） | 8.0.100 同结构 | **8.1.34 基线复用，类未改** |
| topk | 13（默认值来源） | 检测用 10 | **10 与 1 各实例化一份** |
| α / β | — | 0.5 / 6.0 | 0.5 / 6.0（两分支相同） |
| 一致性约束 | 无 | 无 | 无需新增——同一度量天然一致 |

分配机制逐条讲解见 [concepts](/versions/v10/concepts)，损失侧的两次实例化见 [loss.py 源码页](/versions/v10/source/loss-py)，头侧 detach 见 [head.py 源码页](/versions/v10/source/head-py)，论文推导见 [paper](/versions/v10/paper)。
