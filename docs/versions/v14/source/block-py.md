# nn/modules/block.py

这个文件是 v14 的**骨干积木块仓库**：v14 的所有新模块（`CircularConv` 在配置层被解析为该文件的 `Conv` 变体入口 / `ViewEmbedding` / `DomainAdaptiveLayer` / `DeformableA2C2f` / `A2C2f` / `SphereAAttn` / `GradReverse`）以及 YOLO 各代沉淀下的 C2f/C3k2/PSA/AAttn 等通用积木，都集中在这里。它同时提供了 `is_fast_mode()` / `grl_lambda_schedule()` 两个训练-推理切换工具，控制 Fast Mode 下跳过场景分类器、DomainAdaptiveLayer、ViewEmbedding 注入以降低推理延迟。理论推导请回 [v14 论文页](../index.md)，本文件只做行号索引。

## 关键行段速查

### v14 新增 / 核心类

| Class / Func | 行号 | SrcLink |
| --- | --- | --- |
| `is_fast_mode()` / `reset_fast_mode_cache()` | L13–L37 | <SrcLink file="v14:nn/modules/block.py" :line="13" :end="37"/> |
| `GradReverse` (GRL) + `grl_lambda_schedule()` | L39–L70 | <SrcLink file="v14:nn/modules/block.py" :line="39" :end="70"/> |
| `AAttn` | L1251–L1336 | <SrcLink file="v14:nn/modules/block.py" :line="1251" :end="1336"/> |
| `ABlock` | L1339–L1387 | <SrcLink file="v14:nn/modules/block.py" :line="1339" :end="1387"/> |
| `A2C2f` | L1390–L1445 | <SrcLink file="v14:nn/modules/block.py" :line="1390" :end="1445"/> |
| `DeformableAAttn` | L1447–L1658 | <SrcLink file="v14:nn/modules/block.py" :line="1447" :end="1658"/> |
| `DeformableABlock` | L1661–L1675 | <SrcLink file="v14:nn/modules/block.py" :line="1661" :end="1675"/> |
| `DeformableA2C2f` | L1678–L1709 | <SrcLink file="v14:nn/modules/block.py" :line="1678" :end="1709"/> |
| `ViewEmbedding` | L1712–L1733 | <SrcLink file="v14:nn/modules/block.py" :line="1712" :end="1733"/> |
| `DynamicScaleRouter` | L1736–L1775 | <SrcLink file="v14:nn/modules/block.py" :line="1736" :end="1775"/> |
| `SphereAAttn` | L1778–L1868 | <SrcLink file="v14:nn/modules/block.py" :line="1778" :end="1868"/> |
| `DomainAdaptiveLayer` | L1871–L1973 | <SrcLink file="v14:nn/modules/block.py" :line="1871" :end="1973"/> |
| `ARMRouter` | L1976–L2033 | <SrcLink file="v14:nn/modules/block.py" :line="1976" :end="2033"/> |
| `OffsetSTN` | L2036–L2064 | <SrcLink file="v14:nn/modules/block.py" :line="2036" :end="2064"/> |

### YOLO 通用积木（v14 沿用，未修改）

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `DFL` | L131–L150 | <SrcLink file="v14:nn/modules/block.py" :line="131" :end="150"/> |
| `Proto` | L153–L170 | <SrcLink file="v14:nn/modules/block.py" :line="153" :end="170"/> |
| `HGStem` | L173–L201 | <SrcLink file="v14:nn/modules/block.py" :line="173" :end="201"/> |
| `HGBlock` | L204–L225 | <SrcLink file="v14:nn/modules/block.py" :line="204" :end="225"/> |
| `SPP` | L228–L242 | <SrcLink file="v14:nn/modules/block.py" :line="228" :end="242"/> |
| `SPPF` | L245–L264 | <SrcLink file="v14:nn/modules/block.py" :line="245" :end="264"/> |
| `C1` | L267–L279 | <SrcLink file="v14:nn/modules/block.py" :line="267" :end="279"/> |
| `C2` | L282–L297 | <SrcLink file="v14:nn/modules/block.py" :line="282" :end="297"/> |
| `C2f` | L300–L322 | <SrcLink file="v14:nn/modules/block.py" :line="300" :end="322"/> |
| `C3` | L325–L339 | <SrcLink file="v14:nn/modules/block.py" :line="325" :end="339"/> |
| `C3x` | L342–L349 | <SrcLink file="v14:nn/modules/block.py" :line="342" :end="349"/> |
| `RepC3` | L352–L366 | <SrcLink file="v14:nn/modules/block.py" :line="352" :end="366"/> |
| `C3TR` | L369–L376 | <SrcLink file="v14:nn/modules/block.py" :line="369" :end="376"/> |
| `C3Ghost` | L379–L386 | <SrcLink file="v14:nn/modules/block.py" :line="379" :end="386"/> |
| `GhostBottleneck` | L389–L407 | <SrcLink file="v14:nn/modules/block.py" :line="389" :end="407"/> |
| `Bottleneck` | L410–L423 | <SrcLink file="v14:nn/modules/block.py" :line="410" :end="423"/> |
| `BottleneckCSP` | L426–L445 | <SrcLink file="v14:nn/modules/block.py" :line="426" :end="445"/> |
| `ResNetLayer` | L465–L484 | <SrcLink file="v14:nn/modules/block.py" :line="465" :end="484"/> |
| `C2fAttn` | L522–L546 | <SrcLink file="v14:nn/modules/block.py" :line="522" :end="546"/> |
| `ContrastiveHead` | L595–L610 | <SrcLink file="v14:nn/modules/block.py" :line="595" :end="610"/> |
| `BNContrastiveHead` | L613–L635 | <SrcLink file="v14:nn/modules/block.py" :line="613" :end="635"/> |
| `RepNCSPELAN4` | L658–L680 | <SrcLink file="v14:nn/modules/block.py" :line="658" :end="680"/> |
| `ELAN1` | L683–L693 | <SrcLink file="v14:nn/modules/block.py" :line="683" :end="693"/> |
| `AConv` | L696–L707 | <SrcLink file="v14:nn/modules/block.py" :line="696" :end="707"/> |
| `ADown` | L710–L727 | <SrcLink file="v14:nn/modules/block.py" :line="710" :end="727"/> |
| `SPPELAN` | L730–L747 | <SrcLink file="v14:nn/modules/block.py" :line="730" :end="747"/> |
| `CBFuse` | L764–L776 | <SrcLink file="v14:nn/modules/block.py" :line="764" :end="776"/> |
| `CBLinear` | L750–L761 | <SrcLink file="v14:nn/modules/block.py" :line="750" :end="761"/> |
| `C3f` | L779–L797 | <SrcLink file="v14:nn/modules/block.py" :line="779" :end="797"/> |
| `C3k2` | L800–L808 | <SrcLink file="v14:nn/modules/block.py" :line="800" :end="808"/> |
| `C3k` | L811–L819 | <SrcLink file="v14:nn/modules/block.py" :line="811" :end="819"/> |
| `RepVGGDW` | L822–L881 | <SrcLink file="v14:nn/modules/block.py" :line="822" :end="881"/> |
| `CIB` | L884–L920 | <SrcLink file="v14:nn/modules/block.py" :line="884" :end="920"/> |
| `C2fCIB` | L923–L940 | <SrcLink file="v14:nn/modules/block.py" :line="923" :end="940"/> |
| `Attention` | L943–L996 | <SrcLink file="v14:nn/modules/block.py" :line="943" :end="996"/> |
| `PSABlock` | L999–L1033 | <SrcLink file="v14:nn/modules/block.py" :line="999" :end="1033"/> |
| `PSA` | L1036–L1076 | <SrcLink file="v14:nn/modules/block.py" :line="1036" :end="1076"/> |
| `C2PSA` | L1079–L1118 | <SrcLink file="v14:nn/modules/block.py" :line="1079" :end="1118"/> |
| `C2fPSA` | L1121–L1150 | <SrcLink file="v14:nn/modules/block.py" :line="1121" :end="1150"/> |
| `SCDown` | L1153–L1185 | <SrcLink file="v14:nn/modules/block.py" :line="1153" :end="1185"/> |
| `TorchVision` | L1188–L1233 | <SrcLink file="v14:nn/modules/block.py" :line="1188" :end="1233"/> |

## 关于 `CircularConv`

配置文件里出现的 `CircularConv`（`yolov14.yaml` L31、`yolov14-panorama.yaml` L24/L25/L27/L29/L31）**并未**作为独立 class 定义在 `block.py` 内 —— 它由 YOLO 的模块解析器在解析 YAML 时，把 `CircularConv` 映射成 `Conv` 并附加 `circular=True` 参数处理 wrap-around 填充。查找实现时，请以 conv 层（`nn/modules/conv.py`）为准。

<SourceViewer file="v14:nn/modules/block.py"/>

## 参考

- [v14 主页](../index.md)
- [主线 yolov14.yaml](./yolov14-yaml.md)
- [变体一览](./yolov14-variants.md)
- [head.py](./head-py.md) — Detect 头
- [loss.py](./loss-py.md) — GRL 与 `DomainAdaptiveLayer` 的配合
- [tal.py](./tal-py.md) — TaskAlignedAssigner
- [v14 术语表](/glossary)
