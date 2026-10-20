# ultralytics/utils/loss.py

YOLOv10 的损失仓库：**v8.1.34 基线的全套损失类原样在列（v8DetectionLoss 总装 + seg/pose/obb 变体），文件末尾追加 11 行的 `v10DetectLoss`**——「一致性双分配」不引入任何新损失，只是把同一个 v8DetectionLoss 按 tal_topk=10 与 tal_topk=1 实例化两次，两路损失直接相加。

> 快照路径：`source/yolov10/snapshot/ultralytics/utils/loss.py`（727 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v10 里的角色**：与 [head.py 源码页](/versions/v10/source/head-py) 同一个故事——v10 的增量只在文件末尾一小段。损失本体（[CIoU](/glossary#ciou-loss) + [DFL](/glossary#dfl-distribution-focal-loss) + BCE）与 [TAL](/glossary#tal-task-alignment-learning) 调用逻辑完全复用基线；论文的「consistent dual assignments」在代码里就是 <SrcLink file="v10:ultralytics/utils/loss.py" :line="719" :end="720" /> 两行：同一 `TaskAlignedAssigner` 类、同 α=0.5/β=6.0，只差 topk（10 vs 1）。检测训练入口在 `nn/tasks.py` L640-642——`YOLOv10DetectionModel.init_criterion` 返回 `v10DetectLoss(self)`。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="14" :end="35" /> | VarifocalLoss | VFL 原式实现，存而未用 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="38" :end="60" /> | FocalLoss | 存而未用 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="63" :end="86" /> | BboxLoss | CIoU + DFL，target_scores 兼当回归权重 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="88" :end="103" /> | _df_loss | GFL 双 bin 插值 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="147" :end="247" /> | v8DetectionLoss | 检测损失总装，v10 双分支各实例化一份 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="150" /> | __init__ 签名 | `tal_topk=10` 形参——v10 双值化的入口 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="166" /> | assigner 实例化 | `TaskAlignedAssigner(topk=tal_topk, alpha=0.5, beta=6.0)` |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="196" :end="247" /> | __call__ | 前处理 → TAL 分配 → box/cls/dfl 三分量 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="221" :end="228" /> | assigner 调用 | pred 与 pred_bbox 都 detach 后进分配器 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="234" /> | 分类 BCE | L233 注释留档的 VFL 写法——实锤「存而未用」 |
| <SrcLink file="v10:ultralytics/utils/loss.py" :line="717" :end="727" /> | **v10DetectLoss** | v10 全部增量：双实例化 + 相加 |

<SourceViewer file="v10:ultralytics/utils/loss.py" />

## BboxLoss（L63-103）：CIoU + DFL

- **forward**<SrcLink file="v10:ultralytics/utils/loss.py" :line="72" :end="86" />：
  - L74 `weight = target_scores.sum(-1)[fg_mask]`——[TAL](/glossary#tal-task-alignment-learning) 软标签一物两用：既是分类目标，又当回归损失权重；
  - L75-76 `bbox_iou(..., xywh=False, CIoU=True)`，`(1−IoU) × weight` 求和后除 `target_scores_sum`；
  - L79-82 DFL 分支：`bbox2dist` 把目标框转 ltrb 距离 bin，`_df_loss × weight` 后同样归一。
- **_df_loss**<SrcLink file="v10:ultralytics/utils/loss.py" :line="88" :end="103" />：GFL 论文的双 bin 插值——L96-99 算出左 bin `tl/wl` 与右 bin `tr/wr`，L100-103 两次 `F.cross_entropy` 按左右权重求和。几何本体（`bbox_iou`）在同目录 metrics.py，`bbox2dist` 来自 tal.py——本文件是「损失胶水层」。

## v8DetectionLoss（L147-247）：v10 双分支的母体

- **__init__**<SrcLink file="v10:ultralytics/utils/loss.py" :line="150" :end="168" />：形参 `tal_topk=10` 是理解 v10 的钥匙——基线本就允许自定义 topk，v10 只是在外面调了两次。L156 `BCEWithLogitsLoss`；L166 `self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)`；L167 `BboxLoss(m.reg_max - 1, use_dfl=...)`；L168 存 DFL 投影向量 `proj`。
- **前处理**：`preprocess`<SrcLink file="v10:ultralytics/utils/loss.py" :line="170" :end="185" /> 把 batch 的 `batch_idx/cls/bboxes` 摊平到 `(b, max_boxes, 5)` 并转 xyxy；`bbox_decode`<SrcLink file="v10:ultralytics/utils/loss.py" :line="187" :end="194" /> 把分布回归 softmax 期望成框（L191），再 `dist2bbox` 转坐标。
- **__call__**<SrcLink file="v10:ultralytics/utils/loss.py" :line="196" :end="247" /> 流水线：
  1. L200-202 拼平三尺度并 split 出 `pred_distri`/`pred_scores`；
  2. L210 `make_anchors` 生成锚点与 stride 张量；
  3. L213-216 GT 前处理，`mask_gt` 标有效框；
  4. <SrcLink file="v10:ultralytics/utils/loss.py" :line="221" :end="228" /> 调分配器——`pred_scores.detach().sigmoid()`、`pred_bboxes.detach()` 都先切断梯度再进 TAL（分配不回传）；
  5. L230 `target_scores_sum = max(..., 1)` 防除零；
  6. <SrcLink file="v10:ultralytics/utils/loss.py" :line="233" :end="234" /> 分类用 BCE，上一行注释 `# loss[1] = self.varifocal_loss(...)` 把 VFL 写法留档；
  7. L237-241 有正样本才算 box/dfl；
  8. L243-245 乘增益 `hyp.box=7.5 / hyp.cls=0.5 / hyp.dfl=1.5`，L247 返回 `loss.sum() * batch_size` 与 detach 的三分量。

## v10DetectLoss（L717-727）：一致性双分配的总装

<SrcLink file="v10:ultralytics/utils/loss.py" :line="717" :end="727" /> 全部代码一目了然：

- **__init__（L718-720）**：`self.one2many = v8DetectionLoss(model, tal_topk=10)`、`self.one2one = v8DetectionLoss(model, tal_topk=1)`——两个完整独立的损失实例，各持一份 assigner、一份 BboxLoss。**同一 TaskAlignedAssigner 类、同 α=0.5/β=6.0，只差 topk：one2many 沿用 v8 的「一个 GT 监督 10 个 anchor」，one2one 收紧成「一个 GT 只给 1 个正样本」**。
- **__call__（L722-727）**：`preds` 是 head 返回的字典 `{"one2many", "one2one"}`，两路各算各的，L727 返回 `loss_one2many[0] + loss_one2one[0]`——**两分支权重均 1.0，直接相加**，loss items 拼成 6 维（box/cls/dfl × 2，训练日志里的 `box_om, cls_om, dfl_om, box_oo, cls_oo, dfl_oo`）。

没有一致性正则、没有蒸馏项、没有额外权重——论文机制的代码实现就是「同分配器、两种 topk、损失相加」。

## 基线里那些「没动」的类

| 类 | 行段 | 状态 |
|----|------|------|
| VarifocalLoss | L14-35 | 存而未用（__call__ L233 注释留档） |
| FocalLoss | L38-60 | 存而未用 |
| RotatedBboxLoss | L106-127 | OBB 专用（probiou），检测主路径不碰 |
| KeypointLoss | L130-144 | pose 专用 |
| v8SegmentationLoss | L250-430 | seg 总装，未动 |
| v8PoseLoss | L433-586 | pose 总装，未动 |
| v8ClassificationLoss | L589-596 | 分类交叉熵，未动 |
| v8OBBLoss | L599-715 | L607 换 `RotatedTaskAlignedAssigner`，未动 |

## 与 v8 站内快照的对比

| 维度 | v8.0.100（[v8 loss.py](/versions/v8/source/loss-py)） | v10（8.1.34 fork + 增量） |
|------|-------------------------------------------------------|----------------------------|
| 体量 | 75 行，无总装类 | 727 行（基线扩写）+ 11 行 v10DetectLoss |
| 总装位置 | yolo/v8/detect/train.py 的 Loss 类 | 本文件 v8DetectionLoss / v10DetectLoss |
| 标签分配 | tal.py 的 TaskAlignedAssigner（topk 固定） | 同一 assigner，topk 10 与 1 各一份 |
| 损失分量 | box/cls/dfl 三项 | 两分支共 6 项，直接相加 |
| 分类损失 | BCE（VFL 注释留档） | 同左，一字未改 |

逐条公式讲解见 [losses](/versions/v10/losses)，梯度推导见 [loss-gradient](/versions/v10/loss-gradient)，分配器本体见 [tal.py 源码页](/versions/v10/source/tal-py)，头侧的 detach 与双分支见 [head.py 源码页](/versions/v10/source/head-py)。`nn/tasks.py` L640-642（init_criterion）未设站内页，SrcLink 会回退 GitHub 快照外链。
