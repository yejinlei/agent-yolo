# ultralytics/yolo/utils/loss.py

YOLOv8 的回归损失仓库：**BboxLoss（CIoU + DFL，按 target_scores 加权）+ VarifocalLoss 存档 + KeypointLoss**。注意：检测总装的 Loss 类不在此文件，在 yolo/v8/detect/train.py（GitHub 快照外链）——v8 把「损失模块」与「损失总装」拆开了。

> 快照路径：`source/yolov8/snapshot/ultralytics/yolo/utils/loss.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v8:ultralytics/yolo/utils/loss.py" />

**阅读路线**：

- [L11-24 VarifocalLoss](#L11-L24)：VFL 原式实现（[L20 非对称加权](#L20)：负样本 α·p^γ、正样本直接用 gt_score）——但检测主路径用 BCEWithLogits，train.py L217-218 把 VFL 写法注释留档，「存而未用」的实锤
- [L27-49 BboxLoss](#L27-L49)：
  - [L37 权重](#L37) `weight = target_scores.sum(-1)[fg_mask]`——TAL 软标签兼当分类目标与回归权重
  - [L38-39 CIoU](#L38-L39)：`bbox_iou(..., xywh=False, CIoU=True)`，(1−IoU)·weight 求和后除 target_scores_sum
  - [L42-45 DFL 分支](#L42-L45)：`bbox2dist` 把目标框转 ltrb 距离 bin，`_df_loss × weight` 后同样归一
- [L51-60 _df_loss](#L51-L60)：GFL 论文的双 bin 插值——[L55-58 tl/tr 与 wl/wr](#L55-L58)、[L59-60 两次 CE 的加权和](#L59-L60)（左 bin 权重 wl + 右 bin 权重 wr）
- [L63-75 KeypointLoss](#L63-L75)：pose 任务的 OKS 式关键点损失
- 依赖外置：[L7 bbox_iou 来自同目录 metrics.py](#L7)（CIoU 几何实现不在本文件）、[L8 bbox2dist 来自 tal.py](#L8)——本文件是「损失胶水层」

## 与 v7 utils/loss.py 的对比

| 维度 | v7 loss.py | v8 loss.py |
|------|------------|------------|
| 体量 | 约 1650 行，四套 ComputeLoss | 75 行，无总装类 |
| 损失本体 | CIoU + BCEobj + BCEcls（obj 目标 = IoU） | BboxLoss（CIoU + DFL）+ BCE，无 objectness |
| 标签分配 | build_targets 系列写在 loss 文件内 | 独立成 tal.py 的 TaskAlignedAssigner |
| 总装位置 | 同文件 ComputeLoss / AuxOTA | yolo/v8/detect/train.py 的 Loss 类 |
| 动态权重 | dynamic_k + cost 匹配 | TAL align_metric + target_scores 软标签 |
| 备用损失 | Focal / aLRPLoss / APLoss / RankSort | VarifocalLoss 存而未用 |

三分量 box/cls/dfl 的增益加权与 TAL 调用在 [检测总装 Loss（train.py L138-230）](https://github.com/yejinlei/agent-yolo/blob/main/source/yolov8/snapshot/ultralytics/yolo/v8/detect/train.py#L138-L230)；逐条讲解见 [losses](/versions/v8/losses)，梯度推导见 [loss-gradient](/versions/v8/loss-gradient)。
