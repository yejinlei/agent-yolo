# YOLO 演化事件

每一次演化都是一个 `from → to` 的边，记录引入/淘汰了什么概念、层、损失、技巧，以及为什么。

## v1 → v2：从"无 anchor + 2 box"到"anchor + BN + 捷径"

| 引入 | 淘汰 |
|------|------|
| Anchor 网格 | `sqrt=1` 加速 |
| BatchNorm | 前 100k 随机 best_index |
| Passthrough / Route / Upsample 层 | — |
| `random-translate` 数据增强 | — |

- VOC2007 mAP@0.5：**63.4 → 69.4**
- FPS：**35 → 160**
- 关键理由：v1 的 grid cell 只有 2 box，无先验宽高比，定位粗糙。

## v2 → v3：多尺度 FPN + 9 anchors

- 首次一次推理出 3 种分辨率 (13×13 / 26×26 / 52×52)
- 9 anchors 通过 K-Means 从 VOC 聚类
- COCO mAP@0.5: **57.9**，51 FPS
- 关键理由：多尺度解决小目标召回问题。

## v3 → v4：技巧大合集

- Mosaic 数据增强、SPP、CIoU、DPN、Mish、WAM
- COCO mAP@0.5: **44.8** (v3 是 50.4)
- 关键理由：不是新架构，是把已知最好的技巧集齐。

## v4 → v5：换栈 PyTorch

- 从 darknet C 切到 PyTorch
- 工程化优先：pip install、开箱即用
- 引入 Focus 层 (space-to-depth)
- 精度略有回退，但易用性大幅提升

## v5 → v6 / v7：分叉

- **v6 (Megvii)**：Reparameterization + EP2P 两阶段训练
- **v7 (WongKinYiu)**：Model Scaling + EVA Attention + EPE loss

## v5 → v8：回到主线，全新范式

- 切换 **Anchor-free**
- **Decoupled Head**（分类回归分离）
- **C2f** 结构（CSP 增强）
- 引入 Varifocal + DFL 损失
- COCO mAP@0.5:0.95: **44.9**（v5x）

## v8 → v9：PGT + GELAN

- Program Guided Training 提升小目标
- GELAN 架构增强特征表达
- COCO mAP@0.5:0.95: **50.5**

## v9 → v10：NMS-free

- 用 **NWD** (Normalized Wasserstein Distance) + Assigner 替代 NMS
- FreqLoss 提升小样本
- COCO mAP@0.5:0.95: **53.8**

## v8 → v11：EMA + C3k2

- Efficient Model Architecture
- C3k2 块（融合卷积核）
- COCO mAP@0.5:0.95: **53.4**

## v11 → v12：Global Attention

- Global Attention + Spatial Attention
- 参数量减少、精度提升
- COCO mAP@0.5:0.95: **53.6** (v12) vs **53.4** (v11)

---

**共同主线**：从"如何更快"（v1→v2→v3）→"如何更准"（v3→v4→v8）→"如何更高效"（v9→v10→v11→v12）。
