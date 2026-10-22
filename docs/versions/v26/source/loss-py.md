# utils/loss.py

**这个文件在 v26 里的角色**：v26 的损失文件——重头戏。核心新增 <SrcLink file="v26:utils/loss.py" :line="1263" :end="1263"/> `E2EDetectLoss`（旧版）与 <SrcLink file="v26:utils/loss.py" :line="1283" :end="1283"/> `E2ELoss`（带 decay 退火的新一代包装），把 one2many 与 one2one 两条分支的 `v8DetectionLoss` 加权合并；v26 还新增 <SrcLink file="v26:utils/loss.py" :line="1169" :end="1169"/> `DepthLoss26`（SILog + L1 梯度）与 <SrcLink file="v26:utils/loss.py" :line="1389" :end="1389"/> `SemanticSegmentationLoss`（CE + Dice）。

## 关键 class 行段速查

| 类 | 行段 | 用途 |
| --- | --- | --- |
| <SrcLink file="v26:utils/loss.py" :line="29" :end="29"/> `VarifocalLoss` | 分类辅助损失 | 通用 |
| <SrcLink file="v26:utils/loss.py" :line="61" :end="61"/> `FocalLoss` | Focal Loss | 通用 |
| <SrcLink file="v26:utils/loss.py" :line="96" :end="96"/> `DFLoss` | DFL 损失（`reg_max=1` 时不使用） | 通用 |
| <SrcLink file="v26:utils/loss.py" :line="118" :end="118"/> `BboxLoss` | CIoU / GIoU 基础框损失 | 通用 |
| <SrcLink file="v26:utils/loss.py" :line="166" :end="166"/> `RLELoss` | RLE 长度损失，`Pose26` 使用 | pose |
| <SrcLink file="v26:utils/loss.py" :line="229" :end="229"/> `RotatedBboxLoss` | 旋转框损失 | obb |
| <SrcLink file="v26:utils/loss.py" :line="278" :end="278"/> `MultiChannelDiceLoss` | 语义分割 Dice | sem |
| <SrcLink file="v26:utils/loss.py" :line="311" :end="311"/> `BCEDiceLoss` | BCE + Dice | 通用 |
| <SrcLink file="v26:utils/loss.py" :line="335" :end="335"/> `KeypointLoss` | 关键点 OKS 损失 | pose |
| <SrcLink file="v26:utils/loss.py" :line="354" :end="354"/> `v8DetectionLoss` | v8 系列通用检测损失，被 E2E 复用 | 核心 |
| <SrcLink file="v26:utils/loss.py" :line="514" :end="514"/> `v8SegmentationLoss` | 实例分割损失 | seg |
| <SrcLink file="v26:utils/loss.py" :line="679" :end="679"/> `v8PoseLoss` | 关键点损失 | pose |
| <SrcLink file="v26:utils/loss.py" :line="810" :end="810"/> `PoseLoss26` | v26 关键点损失，含 `RLELoss` | pose |
| <SrcLink file="v26:utils/loss.py" :line="985" :end="985"/> `v8ClassificationLoss` | 分类损失 | cls |
| <SrcLink file="v26:utils/loss.py" :line="995" :end="995"/> `v8OBBLoss` | OBB 损失 | obb |
| <SrcLink file="v26:utils/loss.py" :line="1169" :end="1169"/> `DepthLoss26` | **v26 新增**——SILog + 多尺度 L1 梯度 | depth |
| <SrcLink file="v26:utils/loss.py" :line="1263" :end="1263"/> `E2EDetectLoss` | 旧版 E2E 包装（无 decay） | detect |
| <SrcLink file="v26:utils/loss.py" :line="1283" :end="1283"/> `E2ELoss` | **v26 新增**——带 `decay` 退火的 E2E 包装 | detect |
| <SrcLink file="v26:utils/loss.py" :line="1318" :end="1318"/> `TVPDetectLoss` | 文本视觉 prompt 损失 | yoloe |
| <SrcLink file="v26:utils/loss.py" :line="1362" :end="1362"/> `TVPSegmentLoss` | 开集分割损失 | yoloe-seg |
| <SrcLink file="v26:utils/loss.py" :line="1389" :end="1389"/> `SemanticSegmentationLoss` | **v26 新增**——CE + Dice，`ignore_index=255` | sem |

## E2ELoss（v26 主线损失包装）

<SrcLink file="v26:utils/loss.py" :line="1283" :end="1297"/> `E2ELoss.__init__`：
- <SrcLink file="v26:utils/loss.py" :line="1288" :end="1288"/> `self.one2many = loss_fn(model, tal_topk=10)`——沿用 `v8DetectionLoss`，topk=10。
- <SrcLink file="v26:utils/loss.py" :line="1289" :end="1289"/> `self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)`——注意 one2one 使用 `tal_topk=7, tal_topk2=1`，TAL 分配器走"先取 top-7、再对每个 GT 取 top-1" 的双阶段筛选。
- <SrcLink file="v26:utils/loss.py" :line="1291" :end="1297"/> 权重初值：`self.total = 1.0; self.o2m = 0.8; self.o2o = 0.2; self.final_o2m = 0.1`——**one2many 初始权重 0.8，one2one 0.2**。

<SrcLink file="v26:utils/loss.py" :line="1299" :end="1305"/> `__call__`：合并两个分支，输出 `(loss_one2many * o2m + loss_one2one * o2o, loss_one2one 的 loss items)`——**日志只报 one2one 分支的细分项**，因为它是主要评价指标。

<SrcLink file="v26:utils/loss.py" :line="1307" :end="1315"/> `update` 与 `decay`：
- <SrcLink file="v26:utils/loss.py" :line="1310" :end="1311"/> `self.o2m = self.decay(self.updates); self.o2o = max(self.total - self.o2m, 0)`——每步退火。
- <SrcLink file="v26:utils/loss.py" :line="1313" :end="1315"/> `decay` 公式：`max(1 - x / (hyp.epochs - 1), 0) * (0.8 - 0.1) + 0.1`——**one2many 权重从 0.8 线性退火到 0.1**。

## E2EDetectLoss（旧版）

<SrcLink file="v26:utils/loss.py" :line="1263" :end="1269"/> `E2EDetectLoss.__init__`：`tal_topk=10` 与 `tal_topk=1`，不做退火；<SrcLink file="v26:utils/loss.py" :line="1271" :end="1280"/> `__call__` 直接相加两条分支。兼容保留，主训练用 `E2ELoss`（见 <SrcLink file="v26:nn/tasks.py" :line="650" :end="652"/>）。

## DepthLoss26

<SrcLink file="v26:utils/loss.py" :line="1169" :end="1169"/> `class DepthLoss26`；<SrcLink file="v26:utils/loss.py" :line="1172" :end="1173"/> docstring 明确写的是 **"SILog + gradient-matching loss, following the Depth Anything approach"**。
- <SrcLink file="v26:utils/loss.py" :line="1181" :end="1185"/> 权重：`silog_weight = h.dlog`, `grad_weight = h.dgrad`, `silog_lambda = h.dlam`（1.0 = scale-invariant, 0.0 = log-RMSE）。
- <SrcLink file="v26:utils/loss.py" :line="1188" :end="1198"/> `_grad_l1`：L1 梯度损失，`(pred_x[:,1:] - pred_x[:,:-1]) * valid_f`——**只在两个都是有效像素的位置计算**。
- <SrcLink file="v26:utils/loss.py" :line="1233" :end="1236"/> SILog 主项：`sqrt(((log_diff - m)^2).mean() + (1 - lambda) * m^2 + 1e-6)`。
- <SrcLink file="v26:utils/loss.py" :line="1244" :end="1257"/> 多尺度梯度：`grad_scales = 4`，每轮 `avg_pool2d(valid_f, 2)` 缩分辨率，`keep = (vp.sum > 0.7 * occupied.sum)` 过滤稀疏 LiDAR 图。

## SemanticSegmentationLoss

<SrcLink file="v26:utils/loss.py" :line="1389" :end="1389"/> `class SemanticSegmentationLoss(nn.Module)`；<SrcLink file="v26:utils/loss.py" :line="1394" :end="1395"/> docstring：`"CrossEntropyLoss with ignore_index=255, or BCE with logits when nc == 1"`。
- <SrcLink file="v26:utils/loss.py" :line="1418" :end="1418"/> 主损失：`nn.CrossEntropyLoss(ignore_index=255, reduction="sum")`——**`ignore_index=255` 是 Cityscapes 语义**。
- <SrcLink file="v26:utils/loss.py" :line="1415" :end="1416"/> 二分类退化：`nn.BCEWithLogitsLoss(reduction="sum")`。
- <SrcLink file="v26:utils/loss.py" :line="1410" :end="1413"/> Cityscapes 权重表：`use_cityscapes_weight` 时载入 `CITYSCAPES_WEIGHT`，并 `register_buffer(..., persistent=False)`——**不参与 ckpt 序列化**。

## 关于 `BNContrastiveLoss`

v26 中 **未** 定义独立的 `BNContrastiveLoss` 类。开集检测（YOLOE）里 `BNContrastiveHead` 输出的对比分数直接经 <SrcLink file="v26:nn/modules/head.py" :line="1158" :end="1158"/> 进入 <SrcLink file="v26:utils/loss.py" :line="354" :end="354"/> `v8DetectionLoss` 的 BCE 项；<SrcLink file="v26:utils/loss.py" :line="1318" :end="1318"/> `TVPDetectLoss` 则负责文本/视觉 prompt 场景下的分类损失裁剪。

## 说明

- <SrcLink file="v26:utils/loss.py" :line="354" :end="354"/> `v8DetectionLoss` 是 v26 检测损失的底座，`E2ELoss` 通过两次调用它完成 End2End 训练。
- <SrcLink file="v26:utils/loss.py" :line="96" :end="96"/> `DFLoss` 在 `reg_max=1` 时不会被触发（v26 主线 DFL 失效），但对兼容 v10/v11 权重仍然保留。

## SourceViewer

<SourceViewer file="v26:utils/loss.py"/>

## 参考

- [Losses 概念页](../losses.md)
- [End2End 概念](../concepts.md)
- [头文件源码](./head-py.md)
- [TAL 源码](./tal-py.md)
- [v26 主页](../index.md)
