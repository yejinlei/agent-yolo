# nn/tasks.py

**这个文件在 v26 里的角色**：v26 的任务分发中心与模型工厂。`DetectionModel` 是所有检测/分割/姿态/OBB/深度模型的基类，`SegmentationModel / OBBModel / PoseModel / DepthModel / SemanticSegmentationModel / ClassificationModel / YOLOEModel / YOLOESegModel` 分别对应 v26 的 8 种任务；`parse_model` 是 yaml 到 `nn.Sequential` 的装配器；`end2end` 属性决定了 `init_criterion` 走 `E2ELoss` 还是 `v8DetectionLoss`。

## 任务类速查

| 类 | 行段 | 任务 |
| --- | --- | --- |
| <SrcLink file="v26:nn/tasks.py" :line="121" :end="121"/> `BaseModel` | 基类 | 所有模型的公共接口 |
| <SrcLink file="v26:nn/tasks.py" :line="494" :end="494"/> `DetectionModel` | v26 主线检测，`init_criterion` 判断 `E2ELoss` vs `v8DetectionLoss` | detect |
| <SrcLink file="v26:nn/tasks.py" :line="655" :end="655"/> `OBBModel` | 旋转框 | obb |
| <SrcLink file="v26:nn/tasks.py" :line="687" :end="687"/> `SegmentationModel` | 实例分割 | seg |
| <SrcLink file="v26:nn/tasks.py" :line="723" :end="723"/> `SemanticSegmentationModel` | 语义分割 | sem |
| <SrcLink file="v26:nn/tasks.py" :line="795" :end="795"/> `PoseModel` | 关键点 | pose |
| <SrcLink file="v26:nn/tasks.py" :line="838" :end="838"/> `DepthModel` | 单目深度 | depth |
| <SrcLink file="v26:nn/tasks.py" :line="865" :end="865"/> `ClassificationModel` | 图像分类 | cls |
| <SrcLink file="v26:nn/tasks.py" :line="957" :end="957"/> `RTDETRDetectionModel` | 有锚框检测器 | rtdetr |
| <SrcLink file="v26:nn/tasks.py" :line="1127" :end="1127"/> `WorldModel` | 世界模型 | world |
| <SrcLink file="v26:nn/tasks.py" :line="1258" :end="1258"/> `YOLOEModel` | 开集检测 | yoloe |
| <SrcLink file="v26:nn/tasks.py" :line="1534" :end="1534"/> `YOLOESegModel` | 开集分割 | yoloe-seg |

## End2End 分支

- <SrcLink file="v26:nn/tasks.py" :line="563" :end="571"/> `DetectionModel.end2end` property：从最末层的 `Detect.end2end` 冒泡，`setter` 反向写入 head。
- <SrcLink file="v26:nn/tasks.py" :line="650" :end="652"/> `init_criterion`：`E2ELoss(self) if getattr(self.model[-1], "one2one_cv2", None) is not None else v8DetectionLoss(self)`——**判断 `one2one_cv2` 而不是 `end2end` 属性**，因为融合后 one2many 分支可能被删除，训练仍需要 E2ELoss。
- <SrcLink file="v26:nn/tasks.py" :line="595" :end="597"/> `_predict_augment`：End2End 模型不支持 SA 增强，直接回退到 `_predict_once`。

## parse_model 装配

- <SrcLink file="v26:nn/tasks.py" :line="2071" :end="2071"/> `parse_model(d, ch, verbose=True)`——yaml 字典转 `nn.Sequential`。
- <SrcLink file="v26:nn/tasks.py" :line="2088" :end="2088"/> 关键解包：`nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))`。
- <SrcLink file="v26:nn/tasks.py" :line="2230" :end="2242"/> `Detect / YOLOEDetect / Segment / Segment26 / YOLOESegment / YOLOESegment26` 等头会把 `reg_max` 与 `end2end` 参数一并传入 `args.extend([...])`，是 yaml 中 `end2end: True` 与 `reg_max: 1` 真正生效的位置。
- <SrcLink file="v26:nn/tasks.py" :line="2241" :end="2242"/> `args.extend([reg_max, end2end, [ch[x] for x in f]])`——Detect 分支把通道数列表也压入 args。
- <SrcLink file="v26:nn/tasks.py" :line="2270" :end="2270"/> 兼容行：`SPPF` 遇到旧 yaml 少于 4 个参数时自动补默认值，说明 v26 的 SPPF 新增了一个 `shortcut` 开关。

## 权重兼容

- <SrcLink file="v26:nn/tasks.py" :line="1861" :end="1861"/> 兼容映射：`E2EDetectLoss → v10DetectLoss`（v10 加载 v26）。
- <SrcLink file="v26:nn/tasks.py" :line="1915" :end="1915"/> 反向：`v10DetectLoss → E2EDetectLoss`。

## SourceViewer

<SourceViewer file="v26:nn/tasks.py"/>

## 参考

- [End2End 概念](../concepts.md)
- [任务类与 losses 关系](../losses.md)
- [头文件源码](./head-py.md)
- [损失源码](./loss-py.md)
- [v26 主页](../index.md)
