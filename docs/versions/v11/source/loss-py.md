# ultralytics/utils/loss.py

YOLOv11 的损失仓库：**745 行文件里，v11 检测主路径只用 L157 的 `v8DetectionLoss`**——v8 以来一字未改的三分量总装（[CIoU](/glossary#ciou-loss) + [DFL](/glossary#dfl-distribution-focal-loss) + BCE）；v10 的 `v10DetectLoss`（L717）与「端到端」专用的 `E2EDetectLoss`（L730）都保留在文件末尾，但 **`yolo11.yaml` 不引用它们**，属于 v10 遗产。

> 快照路径：`source/yolov11/snapshot/ultralytics/utils/loss.py`（745 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v11 里的角色**：v11 对 loss.py 的净增量为零。`DetectionModel.init_criterion`（<SrcLink file="v11:ultralytics/nn/tasks.py" :line="298" /> 附近）返回 `v8DetectionLoss(self)`——与 v8/v9 完全一致。**v11 与 v8 的损失侧唯一差异不在 loss.py 本体，而在上游 `TaskAlignedAssigner` 的默认参数**（topk 从 10 提到 13、α 从 0.5 提到 1.0），详见 [tal.py 源码页](/versions/v11/source/tal-py)。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="14" :end="35" /> | VarifocalLoss | VFL 原式实现，存而未用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="38" :end="60" /> | FocalLoss | 存而未用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="63" :end="103" /> | BboxLoss | CIoU + DFL，target_scores 兼当回归权重 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="88" :end="103" /> | _df_loss | GFL 双 bin 插值 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="106" :end="127" /> | RotatedBboxLoss | OBB 专用（probiou） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="130" :end="144" /> | KeypointLoss | pose 专用 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="157" :end="258" /> | **v8DetectionLoss** | **v11 检测主用**：三分量总装 |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="264" :end="445" /> | v8SegmentationLoss | 实例分割总装（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="447" :end="599" /> | v8PoseLoss | 关键点总装（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="603" :end="611" /> | v8ClassificationLoss | 分类交叉熵（v11 未动） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="613" :end="715" /> | v8OBBLoss | OBB 总装（v11 未动，用 RotatedTaskAlignedAssigner） |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="717" :end="728" /> | v10DetectLoss | v10 双分支总装，**v11 不使用** |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="730" :end="745" /> | E2EDetectLoss | **v10 端到端专用，v11 不使用** |

<SourceViewer file="v11:ultralytics/utils/loss.py" />

## BboxLoss（L63-103）：CIoU + DFL 双分量

<SrcLink file="v11:ultralytics/utils/loss.py" :line="63" :end="103" /> v11 检测的 box 损失本体：

- **__init__**（L64-70）：`box_gain`（默认 7.5）、`cls_gain`（默认 0.5）、`dfl_gain`（默认 1.5）——增益来自 `hyps`，v11 与 v8 数值一致；
- **forward**（L72-86）：
  - L74 `weight = target_scores.sum(-1)[fg_mask]`——[TAL](/glossary#tal-task-alignment-learning) 软标签一物两用：既是分类目标，又当回归损失权重；
  - L75-76 `bbox_iou(..., xywh=False, CIoU=True)`——(1−IoU) × weight 求和后除 `target_scores_sum`；
  - L79-82 DFL 分支：`bbox2dist`（tal.py 里）把目标框转 ltrb 距离 bin，`_df_loss × weight` 后同样归一。
- **_df_loss**（L88-103）：GFL 论文的双 bin 插值——L96-99 算出左 bin `tl/wl` 与右 bin `tr/wr`，L100-103 两次 `F.cross_entropy` 按左右权重求和。几何本体（`bbox_iou`）在同目录 metrics.py，`bbox2dist` 来自 tal.py——本文件是「损失胶水层」。

**v11 与 v8 的 BboxLoss 完全一致**，一个字都没改。

## v8DetectionLoss（L157）：v11 检测主用的总装

<SrcLink file="v11:ultralytics/utils/loss.py" :line="157" :end="258" /> 是 v11 检测训练唯一会实例化的损失类：

- **__init__**（L160-179）：
  - L163 `BCEWithLogitsLoss(reduce="none")`——分类用交叉熵；
  - L170 `self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)`——**注意这里 loss.py 里仍然是硬编码 topk=10 / α=0.5**，也就是说 v11 的检测损失默认还是走 v8 参数！
  - L171 `self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=...)`
  - L172 存 DFL 投影向量 `proj = torch.arange(reg_max).float().view(1, -1)`
- **前处理**：
  - `preprocess`（L181-196）：把 batch 的 `batch_idx / cls / bboxes` 摊平到 `(b, max_boxes, 5)` 并转 xyxy；
  - `bbox_decode`（L198-205）：分布回归 softmax 期望成框，再 `dist2bbox` 转坐标；
- **__call__**（L207-258）：
  1. L211-213 拼平三尺度并 split 出 `pred_distri / pred_scores`；
  2. L221 `make_anchors` 生成锚点与 stride 张量；
  3. L224-227 GT 前处理，`mask_gt` 标有效框；
  4. **L232-239 调分配器**——`pred_scores.detach().sigmoid()`、`pred_bboxes.detach()` 都先切断梯度再进 TAL（分配不回传）；
  5. L241 `target_scores_sum = max(..., 1)` 防除零；
  6. L244-245 分类用 BCE，上一行注释 `# loss[1] = self.varifocal_loss(...)` 把 VFL 写法留档；
  7. L248-252 有正样本才算 box/dfl；
  8. L254-256 乘增益 `hyp.box=7.5 / hyp.cls=0.5 / hyp.dfl=1.5`，L258 返回 `loss.sum() * batch_size` 与 detach 的三分量。

### 关于 v11 的 topk / α 争议

- loss.py L170 里 assigner 的构造参数写死为 `topk=10, alpha=0.5, beta=6.0`——这是 v8 时代的默认；
- <SrcLink file="v11:ultralytics/utils/tal.py" :line="13" /> `TaskAlignedAssigner` 类的 `__init__` 默认参数在 v11 里改成了 `topk=13, alpha=1.0, beta=6.0`；
- **实际生效的是 loss.py L170 的显式传参**——所以 v11 检测训练实际用的仍是 topk=10、α=0.5；
- 那么「v11 默认 topk=13、α=1.0」的说法来自哪里？来自「直接实例化 `TaskAlignedAssigner()` 而不带参数时」——即外部代码（研究代码 / OBB 之外的场景）如果直接用默认值，会拿到 v11 的新默认。

**结论**：v11 里对 `yolo11.yaml` 训练而言，损失侧的默认参数**仍是 v8 的 topk=10 / α=0.5**；tal.py 类默认值上调到 topk=13 / α=1.0 是「面向未来 / 面向外部直接实例化」的一次架构级回调。读源码时以 loss.py L170 为准。

> **v1~v10 读者路标**：v8 读者看 v8DetectionLoss 就是读 L157-258，v11 一字未改。v10 读者请忽略 L717 的 v10DetectLoss——那是 v10Detect 双分支配套的；L730 的 E2EDetectLoss 更是 v10 论文的「端到端检测器」专用，v11 的 yaml 完全没有引用点。

## v8SegmentationLoss（L264）：实例分割

<SrcLink file="v11:ultralytics/utils/loss.py" :line="264" :end="445" /> 是 Segment 头（[head.py L104](#)）配套的损失总装——三分量基础 + 掩码 BCE + 掩码 Focal 加权。v11 的 yolo11-seg.yaml 引用 Segment 头时，`SegmentModel.init_criterion` 会返回这个类。v11 对 L264 一字未改。

## v8PoseLoss（L447）：关键点检测

<SrcLink file="v11:ultralytics/utils/loss.py" :line="447" :end="599" /> 是 Pose 头配套的损失总装——三分量基础 + KLD 关键点损失（把每个关键点看作 2D 高斯）。v11 的 yolo11-pose.yaml 引用 Pose 头时用它。v11 对 L447 一字未改。

## v8ClassificationLoss（L603）：图像分类

<SrcLink file="v11:ultralytics/utils/loss.py" :line="603" :end="611" /> 极简 8 行——直接 `F.cross_entropy(pred, target)`。v11 分类模型（`yolo11-cls.yaml`）用它。

## v8OBBLoss（L613）：旋转框检测

<SrcLink file="v11:ultralytics/utils/loss.py" :line="613" :end="715" /> 与 v8DetectionLoss 结构对称——三分量基础（box 用 probiou）+ BCE + DFL，`__init__` 里的 `self.assigner = RotatedTaskAlignedAssigner(...)` 用的是 [tal.py L268](/versions/v11/source/tal-py) 的旋转版分配器。v11 的 yolo11-obb.yaml 引用 OBB 头时用它。

## v10DetectLoss（L717）与 E2EDetectLoss（L730）：v10 遗产

<SrcLink file="v11:ultralytics/utils/loss.py" :line="717" :end="728" /> `v10DetectLoss`：v10 的「一致性双分配」总装——两个 v8DetectionLoss 实例（topk=10 与 topk=1），`__call__` 里两分支损失直接相加。**v11 的 yaml 没有引用它**，只在加载 v10 权重并调用 `init_criterion` 时可能命中。

<SrcLink file="v11:ultralytics/utils/loss.py" :line="730" :end="745" /> `E2EDetectLoss`：v10 论文的「端到端检测器」（RTDETR 系）专用，用 `MultiLabelClassifier` 而不是 TAL。v11 的 yolo11.yaml 完全不涉及它。

## 计算分配器实例化流程

v11 里三个可能的 assigner 实例化点：

| 位置 | 类型 | topk | α / β | 使用者 |
|------|------|------|-------|--------|
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="170" /> | TaskAlignedAssigner | 10 | 0.5 / 6.0 | v8DetectionLoss → **yolo11.yaml 检测主用** |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="643" /> | RotatedTaskAlignedAssigner | 13 | 1.0 / 6.0 | v8OBBLoss → yolo11-obb.yaml |
| <SrcLink file="v11:ultralytics/utils/loss.py" :line="718-719" /> | TaskAlignedAssigner × 2 | 10 与 1 | 0.5 / 6.0 | v10DetectLoss（v11 不使用） |

注意 L170 是显式传参（v8 时代遗留），L643 是显式传参（OBB 一直用 topk=13、α=1.0——OBB 从 v6 起就比检测更激进）；两者在 v11 里都是硬编码，不读 tal.py 的默认。

> **v1~v10 读者路标**：v8 读者按 v8 loss.py 走一遍就能读懂 v11 的全部损失侧；v10 读者请把 L717 / L730 两个类当作「历史遗留代码」看待，yolo11 训练时不会碰到它们；顶多会在导出旧 v10 权重时命中 `v10DetectLoss`。

## 与 v8 / v10 loss.py 的对比

| 维度 | v8.0.100 | v10（8.1.34 fork） | **v11（8.3.0）** |
|------|----------|--------------------|------------------|
| 文件行数 | 75 | 727 | **745** |
| 检测总装 | DetectionLoss（早期名） | v8DetectionLoss + v10DetectLoss | **v8DetectionLoss（v10DetectLoss 保留不用）** |
| 标签分配 | TaskAlignedAssigner topk=10 | topk=10 与 topk=1 各一份 | **topk=10（loss.py L170 硬编码）** |
| 损失分量 | box / cls / dfl | 两分支共 6 项 | **三分量（回归 v8）** |
| 分类损失 | BCE（VFL 注释留档） | 同 v8 | **同 v8** |
| TAL 类默认 | topk=10, α=0.5 | topk=10, α=0.5 | **topk=13, α=1.0（tal.py 类默认上调）** |

逐条公式讲解见 [losses](/versions/v11/losses)（若已发布）；分配器本体见 [tal.py 源码页](/versions/v11/source/tal-py)；头侧的 Detect 与 v10Detect 见 [head.py 源码页](/versions/v11/source/head-py)；端到端训练链路见 [code-walkthrough](/versions/v11/code-walkthrough)。
