# cfg/models/yolo26-*.yaml

**这个文件在 v26 里的角色**：`cfg_models_26/` 目录下共 11 个 yaml：主模板 `yolo26.yaml` + 10 个任务变体。所有变体共享 `Conv → C3k2 × 4 → SPPF → C2PSA` 骨干，仅在头/输出层上做增减，覆盖检测、姿态、OBB、实例分割、语义分割、深度估计、分类、开集检测 (YOLOE) 全部任务。

## 变体差异一览

| 变体 | 头 | 输出层 | 关键差异 |
| --- | --- | --- | --- |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="1" :end="52"/> `yolo26.yaml` | `Detect([nc])` | P3/P4/P5 | 主线基线 |
| <SrcLink file="v26:cfg_models_26/yolo26-p2.yaml" :line="1" :end="60"/> `yolo26-p2.yaml` | `Detect([nc])` | P2/P3/P4/P5 | 加 P2 极小目标层（第 44-46 行插入第二次 Upsample + C3k2[128]） |
| <SrcLink file="v26:cfg_models_26/yolo26-p6.yaml" :line="1" :end="62"/> `yolo26-p6.yaml` | `Detect([nc])` | P3/P4/P5/P6 | 骨干加一层 P6/64；`SPPF(k=5, shortcut=False)` |
| <SrcLink file="v26:cfg_models_26/yolo26-obb.yaml" :line="1" :end="52"/> `yolo26-obb.yaml` | `OBB26([nc, 1])` | P3/P4/P5 | 头换成 OBB26，输出 (x,y,w,h,angle) |
| <SrcLink file="v26:cfg_models_26/yolo26-pose.yaml" :line="1" :end="53"/> `yolo26-pose.yaml` | `Pose26([nc, kpt_shape])` | P3/P4/P5 | `kpt_shape: [17, 3]`，用 `PoseLoss26` + `RLE` |
| <SrcLink file="v26:cfg_models_26/yolo26-seg.yaml" :line="1" :end="52"/> `yolo26-seg.yaml` | `Segment26([nc, 32, 256])` | P3/P4/P5 | 实例分割，配 `Proto26`（SAVPE 增强） |
| <SrcLink file="v26:cfg_models_26/yolo26-sem.yaml" :line="1" :end="42"/> `yolo26-sem.yaml` | `SemanticSegment([nc])` | P3/P4 | 只有 P3/P4 输入，`SemanticSegmentationLoss` (CE + Dice, `ignore_index=255`) |
| <SrcLink file="v26:cfg_models_26/yolo26-depth.yaml" :line="1" :end="50"/> `yolo26-depth.yaml` | `Depth([256])` | P3/P4/P5 | `c_mid=256`，SILog + L1 梯度损失 |
| <SrcLink file="v26:cfg_models_26/yolo26-cls.yaml" :line="1" :end="33"/> `yolo26-cls.yaml` | `Classify([nc])` | 单分支 | 无 SPPF，`nc=1000`（ImageNet 默认） |
| <SrcLink file="v26:cfg_models_26/yoloe-26.yaml" :line="1" :end="53"/> `yoloe-26.yaml` | `YOLOEDetect([nc, 512, True])` | P3/P4/P5 | `text_model: mobileclip2:b` + `BNContrastiveHead` + `SAVPE` |
| <SrcLink file="v26:cfg_models_26/yoloe-26-seg.yaml" :line="1" :end="53"/> `yoloe-26-seg.yaml` | `YOLOESegment26([nc, 32, 256, 512, True])` | P3/P4/P5 | 开集实例分割 |

## 关键变体行段速查

### yolo26-p2：小目标层

<SrcLink file="v26:cfg_models_26/yolo26-p2.yaml" :line="44" :end="46"/> 在 backbone P2 位置插入 `-1, 2, C3k2, [128, True]`；<SrcLink file="v26:cfg_models_26/yolo26-p2.yaml" :line="60" :end="60"/> 头改为四通道 `Detect([19, 22, 25, 28], [nc])`。

### yolo26-p6：P6 高层

<SrcLink file="v26:cfg_models_26/yolo26-p6.yaml" :line="29" :end="34"/> 骨干比主线多一层 `Conv(1024, 3, 2)` + `C3k2(1024)` 到达 P6/64；<SrcLink file="v26:cfg_models_26/yolo26-p6.yaml" :line="33" :end="33"/> 唯一一处 `SPPF(1024, 5, 3, False)`——`shortcut=False`。

### yolo26-pose：关键点

<SrcLink file="v26:cfg_models_26/yolo26-pose.yaml" :line="11" :end="11"/> `kpt_shape: [17, 3]`；<SrcLink file="v26:cfg_models_26/yolo26-pose.yaml" :line="53" :end="53"/> 头 `- [[16, 19, 22], 1, Pose26, [nc, kpt_shape]]`。

### yolo26-sem：语义分割

<SrcLink file="v26:cfg_models_26/yolo26-sem.yaml" :line="8" :end="8"/> `nc: 19`（Cityscapes）；<SrcLink file="v26:cfg_models_26/yolo26-sem.yaml" :line="42" :end="42"/> 头只用 P3/P4 两个尺度，`- [[16, 13], 1, SemanticSegment, [nc]]`。

### yoloe-26：开集检测

<SrcLink file="v26:cfg_models_26/yoloe-26.yaml" :line="11" :end="11"/> `text_model: mobileclip2:b`；<SrcLink file="v26:cfg_models_26/yoloe-26.yaml" :line="53" :end="53"/> `YOLOEDetect([nc, 512, True])`——`embed=512, with_bn=True` 触发 `BNContrastiveHead`。

### yoloe-26-seg：开集分割

<SrcLink file="v26:cfg_models_26/yoloe-26-seg.yaml" :line="53" :end="53"/> `YOLOESegment26([nc, 32, 256, 512, True])`——比 YOLOEDetect 多了 `nm=32`（原型数）与 `npr=256`（原型维度）。

### yolo26-cls：分类

<SrcLink file="v26:cfg_models_26/yolo26-cls.yaml" :line="29" :end="29"/> 唯一与主线不同的骨架——**没有 `SPPF`**，`C2PSA` 直接跟在 P5 的 `C3k2` 后面；<SrcLink file="v26:cfg_models_26/yolo26-cls.yaml" :line="33" :end="33"/> 头 `- [-1, 1, Classify, [nc]]`。

## SourceViewer

<SourceViewer file="v26:cfg_models_26/yolo26-p2.yaml"/>

<SourceViewer file="v26:cfg_models_26/yolo26-sem.yaml"/>

<SourceViewer file="v26:cfg_models_26/yoloe-26.yaml"/>

## 参考

- [主线 YAML](./yolo26-yaml.md)
- [头文件源码](./head-py.md)
- [任务工厂 tasks.py](./tasks-py.md)
- [v26 概念页](../concepts.md)
- [v26 主页](../index.md)
