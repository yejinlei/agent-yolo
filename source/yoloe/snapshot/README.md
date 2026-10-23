# YOLOE 源码快照 (Minimal Snapshot)

本目录保留了 YOLOE（Open-Vocabulary YOLO）在两条主线中的最小源码快照，用于离线阅读、代码走查和论文复现比对。

## 主来源：Ultralytics 官方仓库

### `ultralytics/models/yolo/yoloe/` —— v8.3.99

- **仓库**：<https://github.com/ultralytics/ultralytics>
- **Tag**：`v8.3.99`
- **Commit**：`2fbf7015b9e89294b59e37d78e9b5266d94d7139`
- **PR / Release note**：`ultralytics 8.3.99 — New YOLOE Open-Vocabulary Models (#19775)`（2025-03-30）
- **克隆命令**：`git clone --depth 1 --branch v8.3.99 https://github.com/ultralytics/ultralytics.git`
- 本版本是 YOLOE 首次集成进入 Ultralytics 官方库的版本，包含 PE (Prompt Ensemble)、PE-Free、VP (Visual Prompt) 三条训练路径以及 Detect / Segment 双任务。

包含：
- `models/yolo/yoloe/{__init__.py, predict.py, train.py, train_seg.py, val.py}` —— YOLOE 训练器 / 验证器 / 预测器
- `models/yolo/model.py` —— `YOLOE` 顶层 API 封装
- `nn/tasks.py` —— `YOLOEModel` / `YOLOESegModel` 结构定义
- `nn/head.py` —— `YOLOEDetect` / `YOLOESegment` / `SAVPE` 头与视觉提示嵌入模块
- `cfg/models/v8/yoloe-v8.yaml`, `yoloe-v8-seg.yaml` —— v8 系列 YOLOE 配置
- `LICENSE`（AGPL-3.0）

> 注：任务要求路径中的 `ultralytics/models/yolo/yoloe/model.py` 在 v8.3.99 里并不存在，官方把顶层 `YOLOE` 类放在 `ultralytics/models/yolo/model.py`，本快照已按实际路径落盘。
> `nn/modules/repRta.py`、`nn/modules/savpe.py` 在 v8.3.99 中亦未单独拆出：RepRTA 不在 YOLOE 中，SAVPE 位于 `nn/modules/head.py`（第 348 行）。

### `ultralytics/cfg/models/26/` —— v8.4.0

- **Tag**：`v8.4.0`
- **Commit**：`f2d3aed634a5b0e4828024718d4a61ab2f83fb19`
- **Release note**：`ultralytics 8.4.0 — YOLO26 Models Release (#23176)`（2026-01-14）
- 本版本新增 `yoloe-26` / `yoloe-26-seg` 系列（`YOLOESegment26` 头，见 v8.4.0 的 `nn/modules/head.py:1362`）。
- 快照仅保留这两个 YAML，因为 Python 侧代码与 v8.3.99 结构一致。

## 副来源：论文原始实现（可选参考）

### `thu-mig/`

- **仓库**：<https://github.com/THU-MIG/yoloe>
- **Commit**：`40cd606cabdbe2b566d6f14a6b162c89206e9a1b`（`Update README.md`，与任务指定 SHA 一致，仓库默认分支 HEAD）
- 论文：*YOLOE: Real-Time Open-Vocabulary Object Detection with Grounded DINO-Decoder*（ECCV 2024）
- 本目录只保留 `README.md` + `LICENSE`，作为论文复现入口引用。

## 快照约束

- 未落盘任何 `.pt` 权重文件
- 未落盘 `data/`、`tests/`、`docs/` 等无关目录
- `head.py` 与 `tasks.py` 保留了完整的文件（不只是 class 片段），方便直接 `grep` 与 IDE 跳转

## 文件树

```
source/yoloe/snapshot/
├── README.md
├── thu-mig/
│   ├── LICENSE
│   └── README.md
└── ultralytics/
    ├── LICENSE
    ├── cfg/
    │   └── models/
    │       ├── 26/
    │       │   ├── yoloe-26-seg.yaml
    │       │   └── yoloe-26.yaml
    │       └── v8/
    │           ├── yoloe-v8-seg.yaml
    │           └── yoloe-v8.yaml
    ├── models/
    │   └── yolo/
    │       ├── model.py
    │       └── yoloe/
    │           ├── __init__.py
    │           ├── predict.py
    │           ├── train.py
    │           ├── train_seg.py
    │           └── val.py
    └── nn/
        ├── head.py
        └── tasks.py
```
