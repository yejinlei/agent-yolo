# YOLOv5 源码走读

## 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `models/yolo.py` | 277 | Detect 头 + Model 类 + parse_model |
| `models/common.py` | 386 | Conv/C3/Focus/SPP/Concat 等基础模块 |
| `models/yolov5s.yaml` | 48 | 网络结构配置 |
| `utils/loss.py` | 216 | compute_loss + build_targets |
| `utils/general.py` | 489 | NMS + bbox_iou + xywh 转换 |
| `utils/autoanchor.py` | 160 | autoanchor |
| `utils/datasets.py` | 876 | load_mosaic + random_perspective + letterbox |
| `utils/activations.py` | 72 | SiLU/Hardswish/Mish |
| `train.py` | 557 | 训练入口 |
| `detect.py` | 171 | 推理入口 |
| `data/hyp.scratch.yaml` | 33 | 训练超参 |

## 1. `models/yolov5s.yaml` — 网络结构配置

<SrcSnippet file="v5:models/yolov5s.yaml" :line="1" :end="48" />

- `depth_multiple: 0.33` 和 `width_multiple: 0.50` 控制模型规模
- `anchors` 定义 3 组锚点（P3/P4/P5）
- `backbone` + `head` 两段式定义，`[from, number, module, args]` 四元组

## 2. `models/common.py` — 基础模块

### Focus（L163-172）

<SrcSnippet file="v5:models/common.py" :line="163" :end="172" />

### C3（L126-138）

<SrcSnippet file="v5:models/common.py" :line="126" :end="138" />

### SPP（L149-160）

<SrcSnippet file="v5:models/common.py" :line="149" :end="160" />

## 3. `models/yolo.py` — Detect 头

### Detect 类（L24-63）

<SrcSnippet file="v5:models/yolo.py" :line="24" :end="63" />

### Model 类（L66-198）

<SrcSnippet file="v5:models/yolo.py" :line="66" :end="103" />

### parse_model（L201-252）

<SrcSnippet file="v5:models/yolo.py" :line="201" :end="252" />

## 4. `utils/loss.py` — 损失函数

### ComputeLoss 类（L88-161）

<SrcSnippet file="v5:utils/loss.py" :line="88" :end="161" />

### build_targets（L163-214）

<SrcSnippet file="v5:utils/loss.py" :line="163" :end="214" />

## 5. `utils/general.py` — NMS 与 IoU

### bbox_iou（L343-385）

<SrcSnippet file="v5:utils/general.py" :line="343" :end="385" />

### non_max_suppression（L421-493）

<SrcSnippet file="v5:utils/general.py" :line="421" :end="493" />

## 6. `utils/autoanchor.py` — 自动锚点

<SrcSnippet file="v5:utils/autoanchor.py" :line="23" :end="59" />

## 7. `utils/datasets.py` — 数据增强

### load_mosaic（L671-724）

<SrcSnippet file="v5:utils/datasets.py" :line="671" :end="724" />

### random_perspective（L851-895）

<SrcSnippet file="v5:utils/datasets.py" :line="851" :end="895" />

## 阅读路线建议

1. 先读 `yolov5s.yaml`，理解网络结构
2. 再读 `common.py` 的 Focus/C3/SPP，理解基础模块
3. 读 `yolo.py` 的 Detect 和 Model，理解检测头与模型构建
4. 读 `loss.py` 的 ComputeLoss 和 build_targets，理解损失与 anchor 分配
5. 读 `general.py` 的 bbox_iou 和 non_max_suppression，理解 IoU 与 NMS
6. 读 `autoanchor.py`，理解自动锚点
7. 读 `datasets.py` 的 load_mosaic 和 random_perspective，理解数据增强

## 延伸阅读

- [Architecture 逐层图解](/versions/v5/architecture)
- [理论 Concept](/versions/v5/concepts)
- [损失函数](/versions/v5/losses)
