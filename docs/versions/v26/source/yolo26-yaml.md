# cfg/models/yolo26.yaml

**这个文件在 v26 里的角色**：v26 主线检测配置，是 yolo26n/s/m/l/x 五档规模的公共模板。骨干完全回归 v8/v10/v11 家族——`Conv → C3k2 × 4 → SPPF → C2PSA`，头是 `Detect(end2end=True, reg_max=1)`；由于 `reg_max=1`，`DFL` 模块退化为 `nn.Identity`，坐标回归走一维 L1；由于 `end2end: True`，模型内部会 `copy.deepcopy` 出 `one2one_cv2/one2one_cv3` 双分支，训练时 one2many + one2one 并行，推理时只走 one2one。

## 关键行段速查

| 行段 | 内容 |
| --- | --- |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="7" :end="8"/> | `nc: 80`，COCO 默认 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="9" :end="9"/> | `end2end: True` —— 触发 End2End 双分支 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="10" :end="10"/> | `reg_max: 1` —— `DFL` 失效，退化为 `nn.Identity` |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="11" :end="17"/> | `scales`：n/s/m/l/x 五档复合缩放 (depth, width, max_channels) |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="20" :end="32"/> | backbone：P1/2 → P5/32，末尾 `SPPF(1024, 5, 3, True)` + `C2PSA(1024)` |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="35" :end="52"/> | head：PAN-FPN，三层输出 P3/P4/P5 → `Detect([nc])` |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="52" :end="52"/> | `- [[16, 19, 22], 1, Detect, [nc]]` |

## 结构要点

- **backbone**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="22" :end="32"/> 由 5 次 `Conv(k3, s2)` 下采样叠加 4 次 `C3k2` 组成，P5 位置串入 `SPPF` 与 `C2PSA`。`SPPF` 的 `shortcut=True` 是 v26 的默认。
- **head**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="36" :end="50"/> 是经典 PAN 结构，`C3k2(1024, True, 0.5, True)` 表示 `e=0.5, shortcut=True`；P3/P4/P5 三通道送入 `Detect`。
- **`end2end` 参数**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="9" :end="9"/> 会在 <SrcLink file="v26:nn/tasks.py" :line="2088" :end="2088"/> 被解析后透传给 `Detect.__init__`，触发 `copy.deepcopy(cv2), copy.deepcopy(cv3)` 生成 one2one 分支。
- **`reg_max: 1`**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="10" :end="10"/> 使 <SrcLink file="v26:nn/modules/head.py" :line="139" :end="139"/> 的 `DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()` 分支走 `nn.Identity`。

## SourceViewer

<SourceViewer file="v26:cfg_models_26/yolo26.yaml"/>

## 参考

- [v26 架构总览](../architecture.md)
- [v26 概念速览](../concepts.md)
- [v26 损失函数](../losses.md)
- [主线 YAML 变体一览](./yolo26-variants.md)
- [Detect 头源码](./head-py.md)
- [v26 主页](../index.md)
