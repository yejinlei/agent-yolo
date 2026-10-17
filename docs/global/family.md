# YOLO 系列家族

YOLO 家族从 2015 年 v1 开始，迄今已超过 10 年。整个家族不是单一演进线，而是**主线 + tiny 线 + enterprise 线 + 若干分叉**并存。

## 分支谱系

```mermaid
flowchart TB
  V1[YOLOv1<br/>2015 · ECCV] --> V2[YOLOv2 · 2017]
  V2 --> V3[YOLOv3 · 2018]
  V3 --> V4[YOLOv4 · 2020]
  V3 --> V3T[YOLOv3-tiny]
  V3 --> V3S[YOLOv3-SPP]

  V4 --> V5[YOLOv5 · 2020<br/>Ultralytics]
  V4 --> V4T[YOLOv4-tiny]

  V5 --> V6[YOLOv6 · 2022<br/>Megvii]
  V5 --> V7[YOLOv7 · 2022<br/>WongKinYiu]
  V5 --> V8[YOLOv8 · 2023<br/>Ultralytics]
  V5 --> V5N[YOLOv5n/s/m/l/x<br/>tiny 线]

  V8 --> V9[YOLOv9 · 2024]
  V9 --> V10[YOLOv10 · 2024<br/>NMS-free]
  V8 --> V11[YOLOv11 · 2024<br/>EMA]
  V11 --> V12[YOLOv12 · 2025<br/>Global Attn]
```

## 四条主线

| 分支 | 代表版本 | 目标 | 主导 |
|------|---------|------|------|
| 主线 | v1→v2→v3→v4→v8→v9→v10→v11→v12 | 精度优先 | 学术界 + Ultralytics |
| Tiny 线 | v2-tiny→v3-tiny→v4-tiny→v5n→v8n→v11n | 参数量最小 | Ultralytics |
| Enterprise 线 | v9, v10 | 推理延迟优化 | CWI / THU-MIG |
| 分叉实验 | v6 (Megvii), v7 (WongKinYiu) | 特定改进 | 独立团队 |

## 贯穿家族的关键主题

- **Single-shot**：v1 定调，至今未变
- **Anchor vs Anchor-free**：v2 起用 anchor，v8 起默认 anchor-free
- **多尺度 Neck**：v3 首次 FPN，v4 PAFPN，v8 BiFPN
- **CSP 结构**：v5 起 CSPNet，v8 起 C2f
- **Decoupled Head**：v7 实验，v8 正式
- **Attention 引入**：v7 EVA、v10 全注意力、v11 SPPCSPC、v12 Global Attention
- **NMS-free**：v10 端到端消 NMS

## 与外部检测家族的收敛

```mermaid
flowchart LR
  FasterRNN[Faster R-CNN] -. anchor .-> YOLOv2
  RetinaNet[RetinaNet] -. FocalLoss .-> YOLOv3
  SSD[SSD] -. Multi-Scale .-> YOLOv3
  DETR[DETR] -. NMS-free .-> YOLOv10
  EfficientDet[BiFPN] -. Backbone .-> YOLOv8
  CSPNet[CSPNet] -. Architecture .-> YOLOv5
  RT-DETR[RT-DETR] -. End-to-End .-> YOLOv11/v12
```

详见 [演化事件](/global/evolutions) 与 [时间线](/global/timeline)。
