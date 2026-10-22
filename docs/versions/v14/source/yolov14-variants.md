# cfg/models/v14/*variants.yaml

v14 除了主线 `yolov14.yaml` 之外，还按"聚焦某项能力"的维度拆出 5 个专用变体配置：**Game2Real**、**MultiView**、**Adaptive**、**Deformable**、**Panorama**。它们的共同点是：都保留 80 类、同一套 scales（n/s/m/l/x）、同一套 head 结构模板；差别全部集中在 **backbone 里替换掉哪些 v14 特征模块**。理解这 5 个变体，基本就理解了 v14 每一项创新在什么场景下"值不值"。

## 5 个变体一表速览

| 变体 | 主要保留的 v14 模块 | 舍弃的模块 | 主要用途 |
| --- | --- | --- | --- |
| `yolov14-adaptive.yaml` | `ViewEmbedding` + `DomainAdaptiveLayer` + `DeformableA2C2f` | 无（是主线去掉 CircularConv 后的"通用自适应版"） | 混合视角 + 混合域的通用模型 |
| `yolov14-deformable.yaml` | `DeformableA2C2f`（P4/P5） | `ViewEmbedding` / `DomainAdaptiveLayer` / `CircularConv` | 鱼眼/广角/几何畸变场景 |
| `yolov14-game2real.yaml` | `DomainAdaptiveLayer`（P2/P4） | `ViewEmbedding` / `DeformableA2C2f` / `CircularConv`（head 用 `A2C2f`） | 游戏人物→真实人物 |
| `yolov14-multiview.yaml` | `ViewEmbedding` + `A2C2f` | `DomainAdaptiveLayer` / `DeformableA2C2f` / `CircularConv` | 无人机 / BEV / 鱼眼混合 |
| `yolov14-panorama.yaml` | `CircularConv` + `A2C2f`（含 `SphereAAttn` 可选） | `ViewEmbedding` / `DomainAdaptiveLayer` / `DeformableA2C2f` | 360° 全景 / 等距圆柱投影 |

## 各变体 SrcLink 速查

- **Adaptive** — <SrcLink file="v14:cfg_models_v14/yolov14-adaptive.yaml" :line="1" :end="58"/>
  L29–L30：`ViewEmbedding` + `DomainAdaptiveLayer` 同时保留；L36, L38：`DeformableA2C2f` 保留；入口用普通 `Conv` 而非 `CircularConv`。
- **Deformable** — <SrcLink file="v14:cfg_models_v14/yolov14-deformable.yaml" :line="1" :end="50"/>
  L28, L30：两处 `DeformableA2C2f`；无 `ViewEmbedding` / `DomainAdaptiveLayer`。
- **Game2Real** — <SrcLink file="v14:cfg_models_v14/yolov14-game2real.yaml" :line="1" :end="55"/>
  L27, L31：两处 `DomainAdaptiveLayer`；L33, L35：backbone 用 `A2C2f` 而非 `DeformableA2C2f`。
- **MultiView** — <SrcLink file="v14:cfg_models_v14/yolov14-multiview.yaml" :line="1" :end="54"/>
  L27：`ViewEmbedding [128, 6, 64]`；L32, L34：backbone 用 `A2C2f`。
- **Panorama** — <SrcLink file="v14:cfg_models_v14/yolov14-panorama.yaml" :line="1" :end="52"/>
  L24, L25, L27, L29, L31：五处 `CircularConv`（P1–P5 全部 wrap-aware）；无 `SphereAAttn` 直接调用，但 `A2C2f` 保留供 SphereAAttn 使用。

## 阅读顺序建议

1. 先看主线 [yolov14.yaml](./yolov14-yaml.md) 建立"完整能力"的心智模型。
2. 再看每个变体，对照主线**减去什么、替换成什么**：
   - Adaptive ≈ 主线 − CircularConv
   - Deformable ≈ 主线 − (ViewEmbedding + DomainAdaptiveLayer + CircularConv)
   - Game2Real ≈ 主线 − (ViewEmbedding + DeformableA2C2f + CircularConv)，把 DeformableA2C2f 降级成 A2C2f
   - MultiView ≈ 主线 − (DomainAdaptiveLayer + DeformableA2C2f + CircularConv)
   - Panorama ≈ 主线 − (ViewEmbedding + DomainAdaptiveLayer + DeformableA2C2f)，保留 CircularConv + A2C2f
3. 每个变体对应的核心实现都在 [block-py.md](./block-py.md) 中。

## 与主线的一致点

- 所有 5 个变体都是 80 类（`nc: 80`）。
- 5 档 scales 完全一致（n/s/m/l/x，`[0.5, 0.25, 1024]` → `[1.0, 1.5, 512]`）。
- Head 结构模板相同：`Upsample + Concat + DeformableA2C2f/A2C2f → Detect(P3, P4, P5)`，只是 concat 索引随 backbone 层数不同而变化。

<SourceViewer file="v14:cfg_models_v14/yolov14-adaptive.yaml"/>

<SourceViewer file="v14:cfg_models_v14/yolov14-deformable.yaml"/>

<SourceViewer file="v14:cfg_models_v14/yolov14-game2real.yaml"/>

<SourceViewer file="v14:cfg_models_v14/yolov14-multiview.yaml"/>

<SourceViewer file="v14:cfg_models_v14/yolov14-panorama.yaml"/>

## 参考

- [v14 主页](../index.md)
- [主线 yolov14.yaml](./yolov14-yaml.md)
- [nn/modules/block.py](./block-py.md) — `CircularConv` 缺失说明 / `ViewEmbedding` / `DomainAdaptiveLayer` / `DeformableA2C2f` / `A2C2f` / `SphereAAttn` 的定义
- [utils/loss.py](./loss-py.md) — 各变体对应的损失组合
- [v14 术语表](/glossary)
