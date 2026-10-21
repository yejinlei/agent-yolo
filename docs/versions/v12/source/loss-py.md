# ultralytics/utils/loss.py

YOLOv12 的损失仓库：**743 行文件里，v12 检测主路径只用 L157 的 `v8DetectionLoss`**——v8 以来的三分量总装（[CIoU](/glossary#ciou-loss) + [DFL](/glossary#dfl-distribution-focal-loss) + BCE）；文件末尾只剩 `E2EDetectLoss`（L728，端到端双分支）这一个遗产类，**`v10DetectLoss` 在 v11/v12 里都不存在**（旧类名由 tasks.py L825 映射到 `E2EDetectLoss`）。**`yolov12.yaml` 不引用它们**。

> 快照路径：`source/yolov12/snapshot/ultralytics/utils/loss.py`（743 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v12 里的角色**：v12 对 loss.py 的**训练行为净增量为零**——「Attention 骨干内嵌」是纯结构改动，损失侧完全沿用 v8 三分量总装。v12 与 v11 的 loss.py 差异仅限**代码风格**（license 头、`if n := matches.sum()` 写法、docstring 列表格式），`v8DetectionLoss` 的签名、L176 assigner 显式参数、L177 `BboxLoss(m.reg_max)` 全部逐行相同。**`tal_topk=10` 参数位在 v11 就已存在**，不是 v12 新增。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="16" :end="37" /> | VarifocalLoss | VFL 原式实现，存而未用 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="40" :end="62" /> | FocalLoss | 存而未用 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="65" :end="88" /> | DFLoss | GFL 双 bin 插值 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="91" :end="113" /> | BboxLoss | CIoU + DFL，target_scores 兼当回归权重 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="116" :end="137" /> | RotatedBboxLoss | OBB 专用（probiou） |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="140" :end="154" /> | KeypointLoss | pose 专用 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="157" :end="259" /> | **v8DetectionLoss** | **v12 检测主用**：三分量总装 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="160" /> | __init__ 签名 | **`tal_topk=10`（v11 起就有，v12 沿用）** |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="176" /> | assigner 实例化 | **显式传 topk=tal_topk, α=0.5, β=6.0** |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="205" :end="259" /> | __call__ | 三分量总装 |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="263" :end="444" /> | v8SegmentationLoss | 实例分割总装（v12 未动） |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="446" :end="599" /> | v8PoseLoss | 关键点总装（v12 未动） |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="601" :end="610" /> | v8ClassificationLoss | 分类交叉熵（v12 未动） |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="612" :end="725" /> | v8OBBLoss | OBB 总装（v12 未动，用 RotatedTaskAlignedAssigner） |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="728" :end="743" /> | E2EDetectLoss | 端到端双分支总装，**v12 不使用** |

<SourceViewer file="v12:ultralytics/utils/loss.py" />

## BboxLoss（L91-113）：CIoU + DFL 双分量

<SrcLink file="v12:ultralytics/utils/loss.py" :line="91" :end="113" /> v12 检测的 box 损失本体：

- **__init__**（L94-97）：只接 `reg_max=16`，内部按 `reg_max > 1` 建 `DFLoss`——**没有 `box_gain / cls_gain / dfl_gain` 参数**（增益在 v8DetectionLoss `__call__` 里乘）；
- **forward**（L99-113）：
  - L101 `weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)`——[TAL](/glossary#tal-task-alignment-learning) 软标签一物两用：既是分类目标，又当回归损失权重；
  - L102-103 `bbox_iou(..., xywh=False, CIoU=True)`——(1−IoU) × weight 求和后除 `target_scores_sum`；
  - L107-109 DFL 分支：`bbox2dist`（tal.py 里）把目标框转 ltrb 距离 bin，`DFLoss × weight` 后同样归一。

- **DFLoss**（L65-88）：GFL 论文的双 bin 插值——L80-84 算出左 bin `tl/wl` 与右 bin `tr/wr`，L85-88 两次 `F.cross_entropy` 按左右权重求和。几何本体（`bbox_iou`）在同目录 metrics.py，`bbox2dist` 来自 tal.py——本文件是「损失胶水层」。

**v12 与 v8 / v11 的 BboxLoss 完全一致**，`BboxLoss(m.reg_max)` 的构造也相同。

## v8DetectionLoss（L157-259）：v12 检测主用的总装

<SrcLink file="v12:ultralytics/utils/loss.py" :line="157" :end="259" /> 是 v12 检测训练唯一会实例化的损失类：

- **__init__**（L160-178）：
  - L160 `def __init__(self, model, tal_topk=10)`——**v11 已引入的参数位，v12 逐行沿用**；
  - L166 `self.bce = nn.BCEWithLogitsLoss(reduction="none")`——分类用交叉熵；
  - L174 `self.use_dfl = m.reg_max > 1`；
  - **L176** `self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)`——**显式传 α=0.5, β=6.0**；`topk` 默认 10，来自 tal_topk 参数；
  - L177 `self.bbox_loss = BboxLoss(m.reg_max).to(device)`——**v11 与 v12 都是这一行**；
  - L178 `self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)`。
- **preprocess**（L180-195）：把 batch 的 `batch_idx / cls / bboxes` 摊平到 `(b, max_boxes, 5)` 并转 xyxy；
- **bbox_decode**（L197-204）：分布回归 softmax 期望成框，再 `dist2bbox` 转坐标；
- **__call__**（L206-260）：
  1. L208-209 初始化 `loss = torch.zeros(3)`，并 `feats = preds[1] if isinstance(preds, tuple) else preds`——兼容 [Detect.forward](/versions/v12/source/head-py) 的 `(y, x)` 返回值；
  2. L210-212 拼平三尺度并 split 出 `pred_distri / pred_scores`；
  3. L220 `make_anchors` 生成锚点与 stride 张量；
  4. L223-226 GT 前处理，`mask_gt` 标有效框；
  5. L229 `pred_bboxes = self.bbox_decode(anchor_points, pred_distri)`——先解码出预测框；
  6. **L233-241 调分配器**——L235 `pred_scores.detach().sigmoid()`、L236 `pred_bboxes.detach() * stride_tensor` 都先切断梯度再进 TAL（分配不回传）；
  7. L243 `target_scores_sum = max(target_scores.sum(), 1)` 防除零；
  8. L246-247 分类用 BCE（L247），L246 注释 `# VFL way` 把 [VFL](/glossary#vfl-varifocal-loss) 写法留档（未启用）；
  9. L250-254 有正样本才算 box / dfl；
  10. L256-258 乘增益 `hyp.box=7.5 / hyp.cls=0.5 / hyp.dfl=1.5`，L260 返回 `loss.sum() * batch_size` 与 detach 的三分量。

### 关于 v12 的 topk / α 说明

- L176 里 assigner 的构造参数**显式传** `topk=tal_topk, alpha=0.5, beta=6.0`——即使 tal.py 类默认是 topk=13 α=1.0，**这里也是显式覆盖**；
- `tal_topk=10` 是 `__init__` 的默认参数——v12 检测训练实际用的仍是 **topk=10、α=0.5、β=6.0**；
- 与 v11 的关系：**v11 的 loss.py 也是 `tal_topk=10` + 同一行 L176 显式传参**——v12 没有把 topk「新提到参数位」，两代逐行相同；外部仍可 `v8DetectionLoss(model, tal_topk=13)` 覆盖；
- **α=0.5 在 v12 与 v11 都硬编码在 loss.py L176**，不受 tal.py 类默认上调影响——即使 tal.py L29 默认 `alpha=1.0`，实际走 0.5。

**结论**：v12 与 v11 在损失侧默认参数与代码行为**完全一致**（topk=10、α=0.5、β=6.0）。**「v12 把 tal_topk 参数化」的说法是错的**——那是 v11 就有的；**「v12 默认 topk=13、α=1.0」也是错的**——那是 tal.py 类默认，被 loss.py L176 显式覆盖了。

> **v1~v11 读者路标**：v8/v11 读者读 v8DetectionLoss 就是读 L157-259，v12 行为未改（仅代码风格差异）；v10 读者请忽略文件末尾的 `E2EDetectLoss`——那是端到端双分支配套的，v12 的 yaml 完全没有引用点。

## v8SegmentationLoss（L263）：实例分割

<SrcLink file="v12:ultralytics/utils/loss.py" :line="263" :end="444" /> 是 Segment 头配套的损失总装——三分量基础 + 掩码 BCE + 掩码 Focal 加权。v12 的 yolov12-seg.yaml 引用 Segment 头时，`SegmentModel.init_criterion` 会返回这个类。v12 对 L263 一字未改。

## v8PoseLoss（L446）：关键点检测

<SrcLink file="v12:ultralytics/utils/loss.py" :line="446" :end="599" /> 是 Pose 头配套的损失总装——三分量基础 + KLD 关键点损失（把每个关键点看作 2D 高斯）。v12 的 yolov12-pose.yaml 引用 Pose 头时用它。v12 对 L446 一字未改。

## v8ClassificationLoss（L601）：图像分类

<SrcLink file="v12:ultralytics/utils/loss.py" :line="601" :end="610" /> 极简 10 行——直接 `F.cross_entropy(pred, target)`。v12 分类模型（`yolov12-cls.yaml`）用它。

## v8OBBLoss（L612）：旋转框检测

<SrcLink file="v12:ultralytics/utils/loss.py" :line="612" :end="725" /> 与 v8DetectionLoss 结构对称——三分量基础（box 用 probiou）+ BCE + DFL，`__init__` L618 `self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)`，用的是 [tal.py L298](/versions/v12/source/tal-py) 的旋转版分配器。v12 的 yolov12-obb.yaml 引用 OBB 头时用它。

## E2EDetectLoss（L728）：端到端双分支遗产

<SrcLink file="v12:ultralytics/utils/loss.py" :line="728" :end="743" /> 只有两个 `v8DetectionLoss` 实例：L733 `self.one2many = v8DetectionLoss(model, tal_topk=10)`、L734 `self.one2one = v8DetectionLoss(model, tal_topk=1)`，`__call__` 把两分支损失直接相加。它依赖 `Detect.end2end=True` 与 `forward_end2end` 返回的字典预测，**v12 的 yolov12.yaml 完全不涉及**。

**关于旧类名**：v10 时代的 `v10DetectLoss` 在 8.3.x 主线里被重命名为 `E2EDetectLoss`；v12 的 <SrcLink file="v12:ultralytics/nn/tasks.py" :line="825" /> 保留了模块路径映射，加载旧权重时兜底。

## 计算分配器实例化流程

v12 里三个可能的 assigner 实例化点：

| 位置 | 类型 | topk | α / β | 使用者 |
|------|------|------|-------|--------|
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="176" /> | TaskAlignedAssigner | **10**（tal_topk 默认） | **0.5 / 6.0** | v8DetectionLoss → **yolov12.yaml 检测主用** |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="618" /> | RotatedTaskAlignedAssigner | **10** | **0.5 / 6.0** | v8OBBLoss → yolov12-obb.yaml |
| <SrcLink file="v12:ultralytics/utils/loss.py" :line="733" :end="734" /> | TaskAlignedAssigner × 2 | 10 与 1 | 0.5 / 6.0 | E2EDetectLoss（v12 不使用） |

注意三处**全部显式传参**，都不读 tal.py 的类默认（topk=13 / α=1.0）；OBB 与检测同值（topk=10 / α=0.5）。

> **v1~v11 读者路标**：v8/v11 读者按 v8 loss.py 走一遍就能读懂 v12 的全部损失侧；v10 读者请把 L728 的 `E2EDetectLoss` 当作「历史遗留代码」看待，yolov12 训练时不会碰到它；顶多会在加载旧 v10 权重时命中 tasks.py L825 的映射。

## 与 v8 / v11 / v12 的对比

| 维度 | v8.0.100 | v10（8.1.34 fork） | **v11（8.3.0）** | **v12（8.3.63）** |
|------|----------|--------------------|------------------|---------|
| 文件行数 | 75 | 727 | 745 | **743（仅代码风格差异）** |
| 检测总装 | DetectionLoss（早期名） | v8DetectionLoss + 双分支类 | v8DetectionLoss | **v8DetectionLoss** |
| tal_topk 参数化 | — | — | **✅ L160 参数位** | **✅ L160（同 v11）** |
| 标签分配 topk | 10 | 10 与 1 各一份 | 10（L176 显式传） | **10（L176 显式传）** |
| 标签分配 α | 0.5 | 0.5 | 0.5（L176 显式传） | **0.5** |
| 标签分配 β | 6.0 | 6.0 | 6.0 | **6.0** |
| 损失分量 | box / cls / dfl | 两分支共 6 项 | 三分量 | **三分量** |
| 分类损失 | BCE（VFL 注释留档） | 同 v8 | 同 v8 | **同 v8** |
| TAL 类默认 | topk=10, α=0.5 | topk=10, α=0.5 | topk=13, α=1.0 | **topk=13, α=1.0** |
| BboxLoss 构造 | BboxLoss(reg_max) | 同 v8 | BboxLoss(reg_max) | **BboxLoss(reg_max)** |
| 端到端损失类 | 无 | v10DetectLoss | E2EDetectLoss（L730） | **E2EDetectLoss（L728）** |

逐条公式讲解见 [losses](/versions/v12/losses)；分配器本体见 [tal.py 源码页](/versions/v12/source/tal-py)；头侧的 Detect 与 v10Detect 见 [head.py 源码页](/versions/v12/source/head-py)；结构配置见 [yolov12.yaml 源码页](/versions/v12/source/yolov12-yaml)；端到端训练链路见 [code-walkthrough](/versions/v12/code-walkthrough)。
