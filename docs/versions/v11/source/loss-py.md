# ultralytics/utils/loss.py

YOLOv11 的损失仓库：**745 行文件里，v11 检测主路径只用 L157 的 `v8DetectionLoss`**——v8 以来一字未改的三分量总装（[CIoU](/glossary#ciou-loss) + [DFL](/glossary#dfl-distribution-focal-loss) + BCE）；文件末尾只剩一个 v10 遗产 `E2EDetectLoss`（L730），**`yolo11.yaml` 不引用它**。旧类名 `v10DetectLoss` 在 tasks.py L825 被映射到 `E2EDetectLoss`，源码里已经没有这个类。

> 快照路径：`source/yolov11/snapshot/ultralytics/utils/loss.py`（745 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v11 里的角色**：v11 对 loss.py 的净增量为零。`DetectionModel.init_criterion`（<SrcLink file="v11:ultralytics/nn/tasks.py" :line="386" :end="388" />）是三元表达式——`E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)`；`yolo11.yaml` 的 Detect 没有 `end2end`，走 `v8DetectionLoss`。**v11 与 v8 的损失侧唯一差异不在 loss.py 本体，而在上游 `TaskAlignedAssigner` 的默认参数**（tal.py L29 把类默认从 topk=10 / α=0.5 提到 topk=13 / α=1.0），但 loss.py L176 显式传参覆盖了它，检测训练实际仍吃 v8 值。详见 [tal.py 源码页](/versions/v11/source/tal-py)。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="16" :end="37" /> | VarifocalLoss | VFL 原式实现，存而未用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="40" :end="62" /> | FocalLoss | 存而未用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="65" :end="88" /> | DFLoss | GFL 双 bin 插值 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="91" :end="113" /> | BboxLoss | CIoU + DFL，target_scores 兼当回归权重 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="116" :end="137" /> | RotatedBboxLoss | OBB 专用（probiou） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="140" :end="154" /> | KeypointLoss | pose 专用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="157" :end="261" /> | **v8DetectionLoss** | **v11 检测主用**：三分量总装 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="264" :end="445" /> | v8SegmentationLoss | 实例分割总装（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="447" :end="601" /> | v8PoseLoss | 关键点总装（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="603" :end="611" /> | v8ClassificationLoss | 分类交叉熵（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="613" :end="728" /> | v8OBBLoss | OBB 总装（v11 未动，用 RotatedTaskAlignedAssigner） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="730" :end="745" /> | E2EDetectLoss | **v10 端到端专用，v11 不使用**（旧名 v10DetectLoss） |

<SourceViewer file="v11:ultralytics/utils/loss.py" />

## BboxLoss（L91-113）：CIoU + DFL 双分量

<SrcLink file="v11:ultralytics/utils/loss.py" :line="91" :end="113" /> v11 检测的 box 损失本体：

- **__init__**（L94-97）：只接一个参数 `reg_max=16`，内部 `self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None`——增益不在这里，而在 `v8DetectionLoss.__call__` 末尾统一乘；
- **forward**（L99-113）：
  - L101 `weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)`——[TAL](/glossary#tal-task-alignment-learning) 软标签一物两用：既是分类目标，又当回归损失权重；
  - L102 `bbox_iou(..., xywh=False, CIoU=True)`——(1−IoU) × weight 求和后除 `target_scores_sum`；
  - L106-109 DFL 分支：`bbox2dist`（tal.py 里）把目标框转 ltrb 距离 bin，`self.dfl_loss(...) × weight` 后同样归一。
- **DFLoss**（L65-88）：GFL 论文的双 bin 插值——L80 clamp、L81-84 算出左 bin `tl/wl` 与右 bin `tr/wr`，L85-88 两次 `F.cross_entropy` 按左右权重求和。几何本体（`bbox_iou`）在同目录 metrics.py，`bbox2dist` 来自 tal.py——本文件是「损失胶水层」。

**v11 与 v8 的 BboxLoss 结构一致**（v8 是 `reg_max` 单参数版；v10 才短暂改成过 `reg_max - 1 + use_dfl` 双参数）。

## v8DetectionLoss（L157）：v11 检测主用的总装

<SrcLink file="v11:ultralytics/utils/loss.py" :line="157" :end="261" /> 是 v11 检测训练唯一会实例化的损失类：

- **__init__**（L160-178）：
  - L166 `BCEWithLogitsLoss(reduction="none")`——分类用交叉熵；
  - L176 `self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)`——**topk 来自 tal_topk 参数（默认 10）、α/β 仍硬编码 0.5/6.0**，v11 的检测损失默认仍走 v8 参数；
  - L177 `self.bbox_loss = BboxLoss(m.reg_max).to(device)`
  - L178 存 DFL 投影向量 `proj = torch.arange(m.reg_max, dtype=torch.float, device=device)`
- **前处理**：
  - `preprocess`（L180-196）：把 batch 的 `batch_idx / cls / bboxes` 摊平到 `(b, max_boxes, 5)` 并转 xyxy；
  - `bbox_decode`（L198-205）：分布回归 softmax 期望成框，再 `dist2bbox` 转坐标；
- **__call__**（L207-261）：
  1. L211-213 拼平三尺度并 split 出 `pred_distri / pred_scores`；
  2. L221 `make_anchors` 生成锚点与 stride 张量；
  3. L224-227 GT 前处理，`mask_gt` 标有效框；
  4. **L234-242 调分配器**——L236 `pred_scores.detach().sigmoid()`、L237 `pred_bboxes.detach() * stride_tensor` 都先切断梯度再进 TAL（分配不回传）；
  5. L244 `target_scores_sum = max(target_scores.sum(), 1)` 防除零；
  6. L248 分类用 BCE（`loss[1]`），上一行 L247 注释 `# loss[1] = self.varifocal_loss(...)` 把 VFL 写法留档；
  7. L251-255 有正样本才算 box/dfl；
  8. L257-259 乘增益 `hyp.box=7.5 / hyp.cls=0.5 / hyp.dfl=1.5`，L261 返回 `loss.sum() * batch_size` 与 detach 的三分量。

### 关于 v11 的 topk / α 争议

- <SrcLink file="v11:ultralytics/utils/loss.py" :line="176" /> 里 assigner 的构造参数写死为 `topk=tal_topk（默认 10）, alpha=0.5, beta=6.0`——这是 v8 时代的默认；
- <SrcLink file="v11:ultralytics/utils/tal.py" :line="28" :end="36" /> `TaskAlignedAssigner.__init__` 的类默认参数在 v11 里是 `topk=13, alpha=1.0, beta=6.0`；
- **实际生效的是 loss.py L176 的显式传参**——所以 v11 检测训练实际用的仍是 topk=10、α=0.5；
- 那么「v11 默认 topk=13、α=1.0」的说法来自哪里？来自「直接实例化 `TaskAlignedAssigner()` 而不带参数时」——即外部代码如果直接用默认值，会拿到 v11 的新默认。

**结论**：v11 里对 `yolo11.yaml` 训练而言，损失侧的默认参数**仍是 v8 的 topk=10 / α=0.5**；tal.py 类默认值上调到 topk=13 / α=1.0 是「面向未来 / 面向外部直接实例化」的一次架构级回调。读源码时以 loss.py L176 为准。

> **v1~v10 读者路标**：v8 读者看 v8DetectionLoss 就是读 L157-261，v11 逻辑一字未改。v10 读者请忽略 L730 的 E2EDetectLoss——它是 v10「端到端检测器」专用，v11 的 yaml 完全没有引用点；旧名 `v10DetectLoss` 只在 tasks.py L825 作为加载旧权重的字符串映射存在。

## v8SegmentationLoss（L264）：实例分割

<SrcLink file="v11:ultralytics/utils/loss.py" :line="264" :end="445" /> 是 Segment 头（[head.py L104](#)）配套的损失总装——三分量基础 + 掩码 BCE + 掩码 Focal 加权。v11 的 yolo11-seg.yaml 引用 Segment 头时，`SegmentModel.init_criterion` 会返回这个类。v11 对 L264 一字未改。

## v8PoseLoss（L447）：关键点检测

<SrcLink file="v11:ultralytics/utils/loss.py" :line="447" :end="601" /> 是 Pose 头配套的损失总装——三分量基础 + KLD 关键点损失（把每个关键点看作 2D 高斯）。v11 的 yolo11-pose.yaml 引用 Pose 头时用它。v11 对 L447 一字未改。

## v8ClassificationLoss（L603）：图像分类

<SrcLink file="v11:ultralytics/utils/loss.py" :line="603" :end="611" /> 极简 8 行——直接 `F.cross_entropy(pred, target)`。v11 分类模型（`yolo11-cls.yaml`）用它。

## v8OBBLoss（L613）：旋转框检测

<SrcLink file="v11:ultralytics/utils/loss.py" :line="613" :end="728" /> 与 v8DetectionLoss 结构对称——三分量基础（box 用 probiou）+ BCE + DFL，`__init__` 里的 `self.assigner = RotatedTaskAlignedAssigner(...)`（L619）用的是 [tal.py L268](/versions/v11/source/tal-py) 的旋转版分配器，参数同样是 **topk=10 / α=0.5 / β=6.0**。v11 的 yolo11-obb.yaml 引用 OBB 头时用它。

## E2EDetectLoss（L730）：v10 遗产

<SrcLink file="v11:ultralytics/utils/loss.py" :line="730" :end="745" /> `E2EDetectLoss`：v10 的「一致性双分配」总装——L733-734 两个 `v8DetectionLoss` 实例（one2many 用 topk=10，one2one 用 topk=1），`__call__` 里两分支损失直接相加。**v11 的 yaml 没有引用它**；旧类名 `v10DetectLoss` 已在 tasks.py L825 映射到本类，源码中不再单独存在。

## 计算分配器实例化流程

v11 里两个实际的 assigner 实例化点（`E2EDetectLoss` 内部的 second-level 复用不计）：

| 位置 | 类型 | topk | α / β | 使用者 |
|------|------|------|-------|--------|
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="176" /> | TaskAlignedAssigner | 10 | 0.5 / 6.0 | v8DetectionLoss → **yolo11.yaml 检测主用** |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="619" /> | RotatedTaskAlignedAssigner | 10 | 0.5 / 6.0 | v8OBBLoss → yolo11-obb.yaml |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="735" :end="736" /> | TaskAlignedAssigner × 2 | 10 与 1 | 0.5 / 6.0 | E2EDetectLoss（v11 不使用） |

注意 L176 与 L619 都是显式传参，不读 tal.py 的类默认；OBB 与检测**同值**（topk=10、α=0.5），并不是更激进的 topk=13。

> **v1~v10 读者路标**：v8 读者按 v8 loss.py 走一遍就能读懂 v11 的全部损失侧；v10 读者请把 L730 的 `E2EDetectLoss` 当作「历史遗留代码」看待，yolo11 训练时不会碰到它；顶多会在导出旧 v10 权重时命中 tasks.py L825 的类名映射。

## 与 v8 / v10 loss.py 的对比

| 维度 | v8.0.100 | v10（8.1.34 fork） | **v11（8.3.0）** |
|------|----------|--------------------|------------------|
| 文件行数 | 75 | 727 | **745** |
| 检测总装 | DetectionLoss（早期名） | v8DetectionLoss + v10DetectLoss | **v8DetectionLoss（E2EDetectLoss 保留不用）** |
| 标签分配 | TaskAlignedAssigner topk=10 | topk=10 与 topk=1 各一份 | **topk=10（loss.py L176 显式传参）** |
| 损失分量 | box / cls / dfl | 两分支共 6 项 | **三分量（回归 v8）** |
| 分类损失 | BCE（VFL 注释留档） | 同 v8 | **同 v8** |
| TAL 类默认 | topk=10, α=0.5 | topk=10, α=0.5 | **topk=13, α=1.0（tal.py 类默认上调）** |
| OBB 分配器 | topk=10, α=0.5 | topk=10, α=0.5 | **topk=10, α=0.5（同检测）** |

逐条公式讲解见 [losses](/versions/v11/losses)；分配器本体见 [tal.py 源码页](/versions/v11/source/tal-py)；头侧的 Detect 与 v10Detect 见 [head.py 源码页](/versions/v11/source/head-py)；端到端训练链路见 [code-walkthrough](/versions/v11/code-walkthrough)。
