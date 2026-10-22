# cfg/models/v14/yolov14.yaml

这个文件是 YOLOv14 的主线**统一检测配置**：80 类、5 个 compound-scale 档位（n/s/s/m/l/x），把 v14 论文里的四组创新——CircularConv（Panorama）、ViewEmbedding（Multi-View）、DomainAdaptiveLayer（Game2Real）、DeformableA2C2f（D-AAttn）——按 Stage 1→4-5 的顺序串进同一份 backbone，最终用 `Detect(P3, P4, P5)` 头统一输出。训练目标是注释里明示的组合损失：`L = L_det + 0.1·L_adv + 0.05·L_cross`（详见 [utils/loss.py](./loss-py.md) 里的 `v14DetectionLoss`）。所有变体（adaptive/deformable/game2real/multiview/panorama）都从这个文件派生，见 [yolov14-variants.yaml](./yolov14-variants.md)。

## 关键行段速查

| 段落 | 内容 |
| --- | --- |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="1" :end="12"/> | 头部注释：v14 统一配置声明、四大创新、`L = L_det + 0.1·L_adv + 0.05·L_cross` |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="14" :end="15"/> | `nc: 80` 类数 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="16" :end="22"/> | scales（n/s/m/l/x 五档，`[depth, width, max_channels]`） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="24" :end="28"/> | backbone 四阶段注释（scene→multi-view→game2real→deformable pyramid） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="29" :end="42"/> | backbone 主体 11 层：CircularConv→ViewEmbedding→DomainAdaptiveLayer→C3k2→DeformableA2C2f×2 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="31" :end="31"/> | 入口 P1/2：`CircularConv [64, 3, 2]`（wrap-aware） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="33" :end="34"/> | Stage 2–3：`ViewEmbedding [128, 6, 64]` + `DomainAdaptiveLayer [128]` |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="40" :end="42"/> | Stage 4–5：两处 `DeformableA2C2f`（P4 512ch area=4、P5 1024ch area=1） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="45" :end="62"/> | head 18 层：P4/P3/P4/P5 逐级上采样→Concat→DeformableA2C2f |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="62" :end="62"/> | 最终头 `[[17, 20, 23], 1, Detect, [nc]]` — Detect(P3, P4, P5) |

## 骨架与头的分工速读

- **Stage 1（L31–L32）**：`CircularConv → Conv`。P1 用 wrap-aware 卷积保全景边界连续；P2 用普通 Conv。
- **Stage 2（L33）**：`ViewEmbedding [128, 6, 64]`。6 类视角（pinhole/fisheye/panoramic/drone/bev/ground），embedding 通过 1×1 Conv 融合回通道。
- **Stage 3（L34, L38）**：两处 `DomainAdaptiveLayer`（P2/4 与 P4/16 之间），AdaIN + GRL 让 game/real 特征分布对齐。
- **Stage 4–5（L40, L42）**：P4/P5 各用 `DeformableA2C2f [512, True, 4]` 和 `[1024, True, 1]`，即 D-AAttn（可学习变形场 + Swin 风格 shifted-window）。
- **Head（L45–L62）**：P3/P4/P5 三档全部走 DeformableA2C2f，最后由 `Detect` 收口，`nc` 来自 L15。

## 训练目标（与 loss 文件的对应关系）

- `L_det` → `v14DetectionLoss` 里继承自 `v8DetectionLoss` 的 box+cls+dfl 三分量。
- `0.1·L_adv` → `DomainAdversarialLoss`（配合 `DomainAdaptiveLayer` 内部的 GRL）。
- `0.05·L_cross` → `CrossViewConsistencyLoss`（温度 τ 从 0.05 升到 0.5）。

> 注：文件顶部注释把 λ_adv 写成 0.1，实际 `v14DetectionLoss.__init__` 的默认 `lambda_adv=0.05`；调参时可覆盖。见 [loss-py.md](./loss-py.md)。

<SourceViewer file="v14:cfg_models_v14/yolov14.yaml"/>

## 参考

- [v14 主页](../index.md)
- [yolov14 变体一览](./yolov14-variants.md)
- [nn/modules/block.py](./block-py.md) — backbone 里所有积木块的实现
- [nn/modules/head.py](./head-py.md) — `Detect` 头的实现
- [utils/loss.py](./loss-py.md) — `v14DetectionLoss` 与三条辅助损失
- [utils/tal.py](./tal-py.md) — `TaskAlignedAssigner`
- [v14 术语表](/glossary)
