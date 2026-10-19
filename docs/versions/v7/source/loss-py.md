# utils/loss.py

YOLOv7 的损失全家桶：**损失本体仍是 v3/v5 老三样（CIoU + BCEobj + BCEcls），变化全在标签分配**——静态匹配、SimOTA 式 OTA、coarse-to-fine 双头 AuxOTA 四套并存。

> 快照路径：`source/yolov7/snapshot/utils/loss.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v7:utils/loss.py" />

**阅读路线**：

- 备用损失：[L121-149 FocalLoss](#L121-L149)、[L151 QFocalLoss](#L151)、[L176 RankSort](#L176)、[L275 aLRPLoss](#L275)、[L345 APLoss](#L345)（默认训练不启用，留作实验）
- [L422-553 ComputeLoss](#L422-L553)（v3/v5 式静态 anchor 匹配）：[L430-439 BCE + label smoothing + 可选 Focal](#L430-L439)、[L442 balance](#L442) `{3:[4.0,1.0,0.4]}`、[L465-469 CIoU 回归](#L465-L469)、[L472 obj 目标](#L472) `(1-gr)+gr·iou`、[L500-553 build_targets](#L500-L553)（[L522-523 anchor 宽高比匹配](#L522-L523) + [L509-512 五偏移邻域](#L509-L512)）
- [L556-846 ComputeLossOTA](#L556-L846)（train.py 默认，SimOTA 式动态匹配）：[L582 __call__(p, targets, imgs)](#L582)、[L638-793 build_targets](#L638-L793) 逐图跨层收集 p_obj/p_cls/pbox、[L713 -log IoU](#L713)、[L715-716 dynamic_k](#L715-L716)（top-10 IoU 之和截断）、[L737-740 cost](#L737-L740) `cls + 3.0·(-log IoU)`、[L752-755 冲突消解](#L752-L755)（cost 最小者胜）、[L795-846 find_3_positive](#L795-L846)（[L803 g=0.5](#L803)）
- [L849 ComputeLossBinOTA](#L849)（回归量离散成 bin 的备用变体）
- [L1176-1650 ComputeLossAuxOTA](#L1176-L1650)（train_aux.py 用，论文 §4.2 的代码落点）：
  - [L1205-1206](#L1205-L1206) 同一批预测+GT 生成两套目标：`build_targets2`（aux）+ `build_targets`（lead），同 cost 匹配——lead-head guided
  - lead 走 [L1287-1289 build_targets → find_3_positive](#L1287-L1289)（g=0.5，fine），aux 走 [L1440-1442 build_targets2 → find_5_positive](#L1440-L1442)（[L1593 定义](#L1593)、[L1601 g=1.0](#L1601)，候选格更宽 = coarse）
  - aux 损失统一 ×0.25：[L1259 lbox](#L1259)、[L1269 lcls](#L1269)、[L1273 lobj](#L1273)

## 与 v5 utils/loss.py 的对比

| 维度 | v5 loss.py | v7 loss.py |
|------|------------|------------|
| 损失本体 | CIoU + BCEobj + BCEcls | 完全同 v5（obj 目标 = IoU ratio 不变） |
| 标签分配 | wh 比例 + 邻居偏移（单一） | 四套并存：静态 / OTA / BinOTA / AuxOTA |
| 动态匹配 | 无 | ComputeLossOTA：cost=cls+3·(-log IoU)、dynamic_k |
| 双头训练 | 无 | ComputeLossAuxOTA：fine 给 lead、coarse 给 aux |
| aux 权重 | — | aux 损失三处统一 ×0.25 |
| 备用损失 | Focal/QFocal | 另备 aLRPLoss/APLoss/RankSort |

辅助双头的结构见 [models/yolo.py](/versions/v7/source/yolo-py) 的 IAuxDetect，训练入口是 `train_aux.py`。
