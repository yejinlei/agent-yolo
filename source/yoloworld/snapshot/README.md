# YOLO-World 源码快照

本目录是 YOLO-World（开放词表目标检测模型）的**最小算法快照**，用于本项目
`docs/versions/` 文档源码页的引用与行号定位。

## 来源与 commit

- **A. Ultralytics 官方集成** — `https://github.com/ultralytics/ultralytics`
  - Commit: `e9187c12968c1242a2d4dc5f4396cc0e8bb403fb` (`ultralytics 8.1.39` — add YOLO-World training, #9268)
  - License: AGPL-3.0
- **B. AILab-CVC 原始实现** — `https://github.com/AILab-CVC/YOLO-World`
  - Commit: `4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b` (Merge PR #572)
  - License: Apache 2.0（见 `ailab-cvc/LICENSE`）

## 论文

- 标题：*YOLO-World: Open-Vocabulary Real-Time Object Detection in the Wild*
- 作者：Yuheng Ji, Zhengqi Gao, Chenxi Li, et al. (Tencent AI Lab, 中山大学)
- arXiv：[2401.17270](https://arxiv.org/abs/2401.17270)
- 会议：CVPR 2024

## 快照组织

- `ultralytics/` — Ultralytics 8.1.39 集成版（PyTorch、原生训练流程、YAML cfg）
- `ailab-cvc/` — 腾讯 AILab 原始版（基于 mmdet/mmengine 的预训练 + 微调工作流）

## 文件列表

### A. `ultralytics/` (7 文件)

| 路径 | 说明 |
|---|---|
| `nn/tasks_worldmodel.py` | 从 `ultralytics/nn/tasks.py:558-646` 抽取的 `WorldModel` 类（含 `set_classes` + text-guided `predict`） |
| `nn/modules/head.py` | `WorldDetect` / `WorldConv` / `WorldAlignMLP` — 文本引导的 head 定义 |
| `models/yolo/world/train.py` | `WorldTrainer` — 闭集微调（CLIP ViT-B/32 + `set_classes`） |
| `models/yolo/world/train_world.py` | `WorldTrainerFromScratch` — 开放词表从头训练（obj365v1 + GQA + FSSS） |
| `cfg/models/v8/yolov8-world.yaml` | YOLOv8-World v1 结构（`ImagePoolingAttn` + `C2fAttn`） |
| `cfg/models/v8/yolov8-worldv2.yaml` | YOLOv8-World v2 结构（BN 版 neck，去 `ImagePoolingAttn`） |

> **注 1**：Ultralytics 8.1.39 中并未提供独立的 `models/yolo/world/{model,val,predict}.py`；
> `WorldModel` 类实际定义于 `ultralytics/nn/tasks.py:558`，val/predict 复用 `yolo.detect.*`。
> 本快照用 `nn/tasks_worldmodel.py` 抽取该段，避免拖入 2500+ 行的 `tasks.py`。
>
> **注 2**：本版本 Ultralytics 中没有独立的 `YOLOWorldVal` 类——验证入口沿用
> `ultralytics.models.yolo.detect.DetectionValidator`，通过 `model.set_classes(["..."])`
> 传入开放词表文本。

### B. `ailab-cvc/` (10 文件)

- **模型核心** (`yolo_world/models/`)
  - `detectors/yolo_world.py` — `YOLOWorld` / `YOLOWorldV2` 检测器
  - `dense_heads/yolo_world_head.py` — `YOLOWorldHead` / `YOLOWorldV2Head`（NAC 特征对齐、类别文本嵌入）
  - `necks/yolo_world_pafpn.py` — 变体颈：VL PAN / Dual VL PAN / Efficient Neck
- **训练入口** (`tools/`)
  - `train.py` — mmengine-based 训练脚本（预训练 + 微调共用）
- **官方预训练 cfg** (`configs/pretrain_v1/`)
  - `yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py` — S 版（原文 Table 6：COCO-80 mAP 44.6）
- **官方 COCO 微调 cfg** (`configs/finetune_coco/`)
  - `yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py` — v2-S BN
  - `yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py` — v2-M
  - `yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py` — v2-L
- **元信息**
  - `LICENSE`（Apache 2.0）/ `README.md`

> **注**：AILab-CVC 仓库并不存在 `yolov8s-worldv2.yaml` 这种 YAML 命名；其 config 全部以
> `.py` 形式组织（mmengine 风格）。上述 `finetune_coco` 文件对应论文 Table 6 中的 v2-s/m/l
> 微调配方，与 Ultralytics YAML 一一对应。原始仓库亦无 `yolow/utils/mmdet_utils.py`——
> 其 mmdet 兼容层已合并入 `yolo_world/models/*` 各模块内。

## 关键类索引

| 类 | 出处 |
|---|---|
| `WorldModel` | `ultralytics/nn/tasks_worldmodel.py` |
| `WorldTrainer` | `ultralytics/models/yolo/world/train.py` |
| `WorldTrainerFromScratch` | `ultralytics/models/yolo/world/train_world.py` |
| `WorldDetect` / `WorldConv` / `WorldAlignMLP` | `ultralytics/nn/modules/head.py` |
| `YOLOWorld` / `YOLOWorldV2` | `ailab-cvc/yolo_world/models/detectors/yolo_world.py` |
| `YOLOWorldHead` / `YOLOWorldV2Head` | `ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py` |
| `YOLOV5PaFPN`（VL PAN 变体） | `ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py` |

## 未收录

- `.pt` 权重文件（`assets/` 目录）— 体积大且非算法
- `yolo_world_open` PyPI 包 — 非官方第三方实现
- `ultralytics/nn/modules/block.py`（含 `ImagePoolingAttn`）— 快照未纳入，需要时按需追加
- `ailab-cvc/yolo_world/datasets/*`（开放词表 dataset 适配器）— 快照未纳入，训练入口 `tools/train.py` 中的 import 指向它
- `ailab-cvc/yolo_world/models/assigner/task_aligned_assigner.py` — TTA 分配器（与 YOLOv8 同源，可回指 v8 快照）
- Ultralytics 中 `models/yolo/world/{model,val,predict}.py` 三文件在 8.1.39 中不存在
