# 实验骨架

> 本目录仅存放实验脚本骨架，不实跑训练。用于记录复现路径和预期结果。

## 目录规划

```
experiments/
├── README.md                  ← 本文件
├── edge-deployment/           ← 边缘部署实验
│   ├── rpi4b/                  ← 树莓派 4B
│   │   ├── README.md           ← 实验说明
│   │   ├── convert.sh          ← PTQ 量化脚本
│   │   ├── benchmark.py        ← 基准测试
│   │   └── results.json        ← 实验结果
│   └── jetson-orin/            ← Jetson Orin Nano
│       ├── README.md
│       ├── export_onnx.py
│       └── trt_benchmark.py
└── comparison/                 ← 版本对比实验
    ├── v1_vs_v2.md             ← v1 vs v2 对比报告
    └── loss_ablation.md        ← 损失函数消融
```

## 实验 1：YOLOv1 在 Pascal VOC 上的复现

| 项 | 值 |
|----|-----|
| 目标 | 复现论文 Table 1: VOC2007 mAP=63.4 |
| 数据集 | VOC 2007+2012 trainval, 2007 test |
| 模型 | `cfg/yolo.cfg` (快照版, num=2) |
| 预训练 | ImageNet 1000 类, 前 22 层 |
| 训练 | SGD, lr=0.001, 40k iterations |
| 预期 | mAP@0.5 ≈ 63.4, FPS ≈ 35 (Titan X) |

## 实验 2：边缘部署量化对比

| 项 | 值 |
|----|-----|
| 目标 | v8n 在 Jetson Orin Nano 上 FP16 vs INT8 |
| 模型 | yolov8n.pt → ONNX → TensorRT |
| 校准集 | COCO val 128 张 |
| 预期 | FP16 mAP≈37.1, INT8 mAP≈35.8 |
| 对应案例 | `case-edge-jetson-orin` |

## 实验 3：损失函数消融

| 变体 | 修改 | 预期 mAP 变化 |
|------|------|--------------|
| baseline | 完整 6 项 loss | 63.4 |
| 去 rescore | `rescore=0` | -0.5~1.0 |
| 去 sqrt | `sqrt=0` | -1.0~2.0 |
| 去 random_best_index | 注释 73-75 行 | 训练不稳定 |
| 均匀权重 | coord=1, noobj=1 | -2.0~3.0 |

## 复现规范

每个实验目录必须包含：

1. `README.md`：实验目标、环境、步骤、预期结果
2. 脚本：可一键执行的转换/训练/测试脚本
3. `results.json`：实验结果（mAP、FPS、模型大小等）
4. 权重下载链接（不直接提交权重文件）

## 注意事项

- 本项目不实跑训练，实验骨架仅供记录复现路径
- 源码快照在 `source/yolov1/snapshot/`，构建方式见 [source/yolov1/README.md](../source/yolov1/README.md)
- 本体论 CaseStudy 实例在 `ontology/instances/yolov*/` 下的 YAML 中
- VitePress 案例页在 [docs/cases/](../docs/cases/)
