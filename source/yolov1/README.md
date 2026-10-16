# YOLOv1 源码快照

基于 [pjreddie/darknet](https://github.com/pjreddie/darknet) 的 YOLOv1 时期快照，锁定 commit `8322a58`。

## 快照说明

| 项 | 说明 |
|----|------|
| 来源仓库 | `pjreddie/darknet` |
| 锁定 commit | `8322a58` |
| 快照路径 | `source/yolov1/snapshot/` |
| 语言 | C + CUDA |
| 构建系统 | Makefile |

> **注意**：当前仓库 `source/_repos/` 下有完整的 darknet 克隆，但其中 `cfg/yolov1.cfg` 已被 v2+ 实验值覆盖（`num=3`、`dropout=0.5`）。本快照保留了 v1 原始配置（`num=2`、`dropout=0.25`）。

## 目录结构

```
snapshot/
├── Makefile                  # 构建系统
├── LICENSE
├── README.md
├── scripts/
│   ├── voc_label.py          # VOC 标签转换
│   ├── imagenet_label.sh     # ImageNet 标签
│   ├── dice_label.sh
│   └── gen_tactic.sh
└── src/
    ├── yolo.c                # ← 端到端推理入口
    ├── yolo_demo.c           # 实时视频检测 demo
    ├── yolo_kernels.cu       # GPU 核函数
    ├── detection_layer.c    # ← 检测头：前向 + loss + NMS
    ├── detection_layer.h
    ├── box.c                # ← IoU 计算 + NMS 排序
    ├── box.h
    ├── network.c            # 网络构建与前向
    ├── network.h
    ├── image.c              # 图像预处理（resize, border, jitter）
    ├── image.h
    ├── data.c               # 数据加载与增强
    ├── data.h
    ├── convolutional_layer.c # 卷积层
    ├── maxpool_layer.c      # 池化层
    ├── connected_layer.c    # 全连接层
    ├── dropout_layer.c      # Dropout
    ├── local_layer.c        # Local 层（v1 特有）
    ├── parser.c             # cfg 文件解析
    ├── parser.h
    ├── blas.c               # GEMM / 矩阵运算
    ├── activations.c        # 激活函数
    ├── utils.c
    └── ... (其他辅助层)
```

## 关键文件速查

| 文件 | 行数 | 核心功能 | 对应本体实体 |
|------|------|---------|-------------|
| `src/yolo.c` | ~200 | 端到端推理：resize → border → predict → NMS | Model `yolo-v1` |
| `src/detection_layer.c` | ~200 | 检测头：前向 + loss 计算 + GT 分配 | Loss × 6, Trick × 3 |
| `src/box.c` | ~400 | `box_iou` + `do_nms_sort` | Trick `nms-iou05` |
| `src/image.c` | ~800 | `resize_image` + `border_image` + `random_distort` | Trick `region-jitter` |
| `src/data.c` | ~600 | 数据加载 + jitter 增强 | Trick `region-jitter` |
| `src/connected_layer.c` | ~300 | FC 层前向/反向 | Layer `fc-4096`, `fc-1470` |
| `src/parser.c` | ~500 | cfg 解析 | — |

## 如何构建

```bash
# CPU only
make

# GPU (需要 CUDA)
make GPU=1

# 运行推理
./darknet detector test cfg/yolo.cfg backup/yolo.weights data/dog.jpg
```

> 本项目不实跑 darknet——源码仅供走读分析。

## 与论文的对应关系

| 论文节 | 对应源码 |
|--------|---------|
| §2 Unified Detection | `yolo.c` + `detection_layer.c` |
| §3 Network Architecture | `cfg/yolo.cfg` + `convolutional_layer.c` |
| §3.1 Loss Function | `detection_layer.c:101-130` |
| §3.2 Training | `cfg/yolo.cfg` (lr schedule) + `data.c` (jitter) |
| §3.3 Inference | `yolo.c: detect_objects()` |
| §4 Experiments | — (结果在论文中) |

## 深入阅读

- [源码走读](./code-walkthrough.md)：4 个关键文件的逐行分析
- [VitePress 源码走读页](../../docs/versions/v1/code-walkthrough.md)：站点版速查
- [本体论 SourceFile 实例](../../ontology/instances/yolov1/sources.yaml)
