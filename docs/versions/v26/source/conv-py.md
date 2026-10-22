# nn/modules/conv.py

**这个文件在 v26 里的角色**：v26 的所有卷积积木。`Conv` 是所有层的基础（含 BN + 激活），`DWConv` 是深度可分离卷积，被 `Detect` 头的 `cv3` 分支使用；`Focus / GhostConv / RepConv / CBAM / Concat / Index` 等保留兼容，主线 v26 主要用 `Conv` 与 `DWConv`。

## 关键类行段速查

| 类 | 行段 | 说明 |
| --- | --- | --- |
| <SrcLink file="v26:nn/modules/conv.py" :line="30" :end="30"/> `autopad` | 自动 padding 工具函数 | 全局 |
| <SrcLink file="v26:nn/modules/conv.py" :line="48" :end="48"/> `Conv` | 主力卷积层：Conv2d + BN + 激活（默认 SiLU） | 骨干 / 头 |
| <SrcLink file="v26:nn/modules/conv.py" :line="101" :end="101"/> `Conv2` | 双分支卷积，兼容旧模型 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="152" :end="152"/> `LightConv` | 轻量化深度可分离卷积 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="188" :end="188"/> `DWConv` | `groups=out_channels` 的深度可分离卷积 | Detect cv3 |
| <SrcLink file="v26:nn/modules/conv.py" :line="205" :end="205"/> `DWConvTranspose2d` | 深度可分离反卷积 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="222" :end="222"/> `ConvTranspose` | 反卷积封装 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="274" :end="274"/> `Focus` | 空间降采样（v5 遗留） | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="314" :end="314"/> `GhostConv` | GhostNet 卷积 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="356" :end="356"/> `RepConv` | RepVGG 卷积 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="514" :end="514"/> `ChannelAttention` | CBAM 通道注意力 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="551" :end="551"/> `SpatialAttention` | CBAM 空间注意力 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="585" :end="585"/> `CBAM` | 组合注意力 | 兼容 |
| <SrcLink file="v26:nn/modules/conv.py" :line="618" :end="618"/> `Concat` | yaml 中的 concat 层 | 头 |
| <SrcLink file="v26:nn/modules/conv.py" :line="646" :end="646"/> `Index` | 索引重排工具 | 兼容 |

## 关键行段速查

- <SrcLink file="v26:nn/modules/conv.py" :line="48" :end="48"/> `class Conv(nn.Module)`——签名 `__init__(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)`；`act=True` 时默认接 `SiLU()`。v26 所有 `Conv(x, c2, 3, 2)` 都用它。
- <SrcLink file="v26:nn/modules/conv.py" :line="188" :end="188"/> `class DWConv(Conv)`——继承 `Conv`，把 `groups` 强制设为 `out_channels`。
- **在 v26 头中的使用**：<SrcLink file="v26:nn/modules/head.py" :line="132" :end="134"/> `Detect.cv3` 中的非 legacy 分支用 `DWConv(x, x, 3)` + `Conv(x, c3, 1)`——即深度可分离卷积先做空间混洗，1×1 卷积再做通道融合。

## 与 v13 的差异

- v26 的 `Conv` 与 v13 基本一致，默认激活仍是 `SiLU`。
- 主要变化在 **头侧**：v13 的 `Detect.cv3` 走普通 3×3 Conv；v26 <SrcLink file="v26:nn/modules/head.py" :line="130" :end="136"/> 走 `DWConv + Conv` 组合作为非 legacy 默认。

## SourceViewer

<SourceViewer file="v26:nn/modules/conv.py"/>

## 参考

- [Layers 层列表](../layers.md)
- [头文件源码](./head-py.md)
- [主线 YAML](./yolo26-yaml.md)
- [v26 主页](../index.md)
