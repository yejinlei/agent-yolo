# YOLO 时间线

## 2015-2019：single-shot 时代

```mermaid
timeline
  title YOLO 系列时间线 (2015-2019)
  2015 : YOLOv1 ECCV 论文<br/>35 FPS 开创 single-shot 时代
  2017 : YOLOv2 引入 anchor + BatchNorm
  2018 : YOLOv3 多尺度 FPN 首次 9 anchors
  2018 : YOLOv3-tiny 移动端部署版本
  2019 : YOLOv3-SPP 加入 SPP 模块
```

## 2020-2022：工程化与多样化

```mermaid
timeline
  title YOLO 系列时间线 (2020-2022)
  2020 : YOLOv4 技巧大合集
  2020 : YOLOv5 Ultralytics 发布 PyTorch 版
  2022 : YOLOv6 Megvii 引入 RepVDConv
  2022 : YOLOv7 WongKinYiu Model Scaling
```

## 2023-2026：回到主线 + 新范式

```mermaid
timeline
  title YOLO 系列时间线 (2023-2026)
  2023 : YOLOv8 Anchor-free + C2f + Decoupled Head
  2024 : YOLOv9 PGT + GELAN
  2024 : YOLOv10 NMS-free 端到端
  2024 : YOLOv11 EMA + C3k2
  2025 : YOLOv12 Global Attention
  2026 : 下一代 (预留)
```

## 关键年份速览

| 年份 | 版本 | 里程碑 | 精度 | FPS |
|------|------|--------|------|-----|
| 2015 | v1 | Single-shot 范式 | 63.4 (VOC) | 35 |
| 2017 | v2 | Anchor 引入 | 69.4 (VOC) | 160 |
| 2018 | v3 | 多尺度 FPN | 57.9 (COCO) | 51 |
| 2020 | v4 | 技巧集齐 | 53.4 (COCO) | 145 |
| 2020 | v5 | PyTorch 化 | 44.9 (COCO) | — |
| 2022 | v6 | 两阶段训练 | 50.4 (COCO) | 89.5 |
| 2022 | v7 | Model Scaling | 51.4 (COCO) | — |
| 2023 | v8 | Anchor-free | 44.9 (COCO) | — |
| 2024 | v9 | PGT | 50.5 (COCO) | — |
| 2024 | v10 | NMS-free | 53.8 (COCO) | — |
| 2024 | v11 | EMA | 53.4 (COCO) | — |
| 2025 | v12 | Global Attn | 53.6 (COCO) | — |

*注：COCO 数字为 mAP@0.5:0.95，VOC 为 mAP@0.5。数字来自各版本官方论文。*
