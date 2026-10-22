# nn/modules/block.py

**这个文件在 v26 里的角色**：v26 的骨干积木库。所有 v26 yaml 中的骨干层——`C3k2`、`SPPF`、`C2PSA`——都在这里；此外还集中存放 `DFL`、`Proto` / `Proto26`、`SAVPE`、`BNContrastiveHead`、`RealNVP` 等 v26 特有模块，供 head.py 和 loss.py 引用。

## 关键类行段速查

| 类 | 行段 | 用途 |
| --- | --- | --- |
| <SrcLink file="v26:nn/modules/block.py" :line="60" :end="60"/> `DFL` | DFL 解码器，`reg_max=1` 时退化为 `nn.Identity` | 坐标解码 |
| <SrcLink file="v26:nn/modules/block.py" :line="85" :end="85"/> `Proto` | 实例分割原型基础类 | seg 头 |
| <SrcLink file="v26:nn/modules/block.py" :line="211" :end="211"/> `SPPF` | 空间金字塔池化，v26 新增 `shortcut` 参数 | 骨干 |
| <SrcLink file="v26:nn/modules/block.py" :line="780" :end="780"/> `BNContrastiveHead` | 开集检测的对比学习头，BN 层承载类别 embedding | YOLOE |
| <SrcLink file="v26:nn/modules/block.py" :line="1069" :end="1069"/> `C3k2` | v26 骨干主力 CSP 块，继承 `C2f` | 骨干 |
| <SrcLink file="v26:nn/modules/block.py" :line="1337" :end="1337"/> `PSABlock` | `PSA` 的注意力子块 | 骨干 |
| <SrcLink file="v26:nn/modules/block.py" :line="1387" :end="1387"/> `PSA` | Position-Sensitive Attention 独立模块 | 骨干 |
| <SrcLink file="v26:nn/modules/block.py" :line="1442" :end="1442"/> `C2PSA` | `PSA` 的 CSP 封装，v26 骨干顶部收尾块 | 骨干 |
| <SrcLink file="v26:nn/modules/block.py" :line="1497" :end="1497"/> `C2fPSA` | 无 v26 用，兼容旧模型 | 兼容 |
| <SrcLink file="v26:nn/modules/block.py" :line="1930" :end="1930"/> `SAVPE` | Spatial-Aware Visual Prompt Embedding，开集任务专用 | YOLOE |
| <SrcLink file="v26:nn/modules/block.py" :line="1998" :end="1998"/> `Proto26` | v26 分割原型，继承 `Proto`，输出通道翻倍 | Segment26 |
| <SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2045"/> `RealNVP` | 可逆耦合流模型，用于 Pose26 的 kpt 变换 | Pose26 |

## 骨干积木

### C3k2

<SrcLink file="v26:nn/modules/block.py" :line="1069" :end="1069"/> `class C3k2(C2f)`——继承 `C2f`，把内部 `Bottleneck` 换成 `BottleneckCSP`，用两个不同 `kernel_size=(3, 3)` 的分支。v26 全部 4 段骨干都由它组成；<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="24" :end="24"/> 参数形如 `[256, False, 0.25]`——`c2, shortcut=False, e=0.25`。

### SPPF

<SrcLink file="v26:nn/modules/block.py" :line="211" :end="211"/> `class SPPF(nn.Module)`；<SrcLink file="v26:nn/modules/block.py" :line="214" :end="214"/> 签名 `__init__(c1, c2, k=5, n=3, shortcut=False)`——**`shortcut` 参数是 v26 新增的**：`shortcut=True` 时最后 concat 前会跳过 5×5 分支，只保留 3 路 pool。主线 <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="31" :end="31"/> `SPPF, [1024, 5, 3, True]` 打开该开关。

### C2PSA / PSA / PSABlock

<SrcLink file="v26:nn/modules/block.py" :line="1442" :end="1442"/> `class C2PSA(nn.Module)`——v26 骨干的最后一层，把 `PSA`（Position-Sensitive Attention）包在 CSP 结构里；<SrcLink file="v26:nn/modules/block.py" :line="1337" :end="1337"/> `PSABlock` 是 `PSA` 的一个注意力子块，负责 `attn_ratio=0.5` 的 head 压缩；<SrcLink file="v26:nn/modules/block.py" :line="1387" :end="1387"/> `PSA` 本体提供位置敏感注意力。

## v26 特有模块

### Proto26

<SrcLink file="v26:nn/modules/block.py" :line="1998" :end="1998"/> `class Proto26(Proto)`——继承 `Proto`，用于 `Segment26`；相比 `Proto` 输出多一组通道，配合 SAVPE 做形状先验。

### SAVPE

<SrcLink file="v26:nn/modules/block.py" :line="1930" :end="1930"/> `class SAVPE(nn.Module)`——Spatial-Aware Visual Prompt Embedding，把外部视觉 prompt 注入到 `YOLOEDetect` 的 `cv3` 后，`forward(x, vp)` 在 <SrcLink file="v26:nn/modules/block.py" :line="1960" :end="1960"/>。

### BNContrastiveHead

<SrcLink file="v26:nn/modules/block.py" :line="780" :end="780"/> `class BNContrastiveHead(nn.Module)`——用 BN 的 `weight` 参数承载类别 embedding，`forward(x, w)` 在 <SrcLink file="v26:nn/modules/block.py" :line="812" :end="812"/>。开集检测（YOLOE）用它把文本/视觉 prompt 投影到特征空间。

### RealNVP

<SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2045"/> `class RealNVP(nn.Module)`——可逆耦合流，被 `Pose26` 用来自适应地变换 kpt 输出。

## 说明

- v26 骨干**没有** `A2C2f` / `DeformableA2C2f` 等 v13 的 deformable 系列，全部回归到 `C3k2 + C2PSA` 家族。
- <SrcLink file="v26:nn/modules/block.py" :line="1804" :end="1804"/> `A2C2f` 与 <SrcLink file="v26:nn/modules/block.py" :line="1886" :end="1886"/> `SwiGLUFFN` 等仍存在，但仅供其他 v26 yaml 外的旧模型使用，主线 v26 不引用。

## SourceViewer

<SourceViewer file="v26:nn/modules/block.py"/>

## 参考

- [Layers 层列表](../layers.md)
- [架构总览](../architecture.md)
- [头文件源码](./head-py.md)
- [主线 YAML](./yolo26-yaml.md)
- [v26 主页](../index.md)
