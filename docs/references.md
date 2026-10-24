# 论文与仓库索引

> **一句话总结**：YOLO 家族所有主线版本 + 衍生分支的官方论文（arXiv）和 GitHub 仓库，一表查全，避免每次训练前重新找链接。

- ⏱ 预计阅读：5 分钟
- 📌 数据基线：arXiv / GitHub 公开信息，核对到 2025-06
- 🔗 相关：[compare-matrix](/compare-matrix) · [timeline](/global/timeline) · [glossary](/glossary)

---

## 快速导航

- [YOLO 主线版本](#yolo-主线版本)
- [衍生分支](#衍生分支)
- [任务扩展的额外参考](#任务扩展的额外参考)
- [关键概念的原论文](#关键概念的原论文)
- [常用工具与数据集](#常用工具与数据集)
- [怎么找论文（三种方式）](#怎么找论文三种方式)

---

## YOLO 主线版本

按 arXiv 编号顺序排列，每个版本包含：论文链接 · 代码仓库 · 首发年份 · 一句话贡献。

| 版本 | 年份 | 论文 (arXiv) | 官方仓库 | 一句话贡献 |
|------|------|-------------|---------|-----------|
| **YOLOv1** | 2016 | [1506.02640](https://arxiv.org/abs/1506.02640) | [pjreddie/yolo](https://github.com/pjreddie/yolo) | 单阶段端到端检测开山之作 |
| **YOLOv2** | 2017 | [1612.08242](https://arxiv.org/abs/1612.08242) | [pjreddie/darknet](https://github.com/pjreddie/darknet) | Darknet-19 + anchor + BN，速度/精度双升 |
| **YOLOv3** | 2018 | [1804.02767](https://arxiv.org/abs/1804.02767) | [pjreddie/darknet](https://github.com/pjreddie/darknet) | 多尺度 P3/P4/P5，80 类，工业级起点 |
| **YOLOv4** | 2020 | [2004.10934](https://arxiv.org/abs/2004.10934) | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | CSPDarknet53 + Mosaic + CIoU，训练配方大集 |
| **YOLOv5** | 2020 | 无独立论文，官方 [白皮书](https://github.com/ultralytics/yolov5/blob/master/README.md) | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) | PyTorch 版工业标准，Ultralytics 生态起点 |
| **YOLOv6** | 2022 | [2209.02976](https://arxiv.org/abs/2209.02976) | [megvii-ception/YiiFormer](https://github.com/megvii-ception/YiiFormer) 前身 | EfficientRep + RepVGG blocks，工业部署优化 |
| **YOLOv7** | 2022 | [2207.02696](https://arxiv.org/abs/2207.02696) | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) | E-ELAN + 模型缩放，学术复现友好 |
| **YOLOv8** | 2023 | 无论文，官方 [白皮书](https://docs.ultralytics.com/yolov8/) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 多任务统一框架，anchor-free + DFL |
| **YOLOv9** | 2024 | [2402.13116](https://arxiv.org/abs/2402.13116) | [Purkaihu/yolov9](https://github.com/Purkaihu/yolov9) | PGI + GELAN，参数效率突破 |
| **YOLOv10** | 2024 | [2405.14458](https://arxiv.org/abs/2405.14458) | [THUDM/YOLOv10](https://github.com/THUDM/YOLOv10) | NMS-free 双阶段解耦，训练/推理免 NMS |
| **YOLOv11** | 2024 | 无论文，官方 [白皮书](https://docs.ultralytics.com/yolov11/) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | C2PSA + SPPCSPC，Ultralytics 现役旗舰 |
| **YOLOv12** | 2025 | [2502.12554](https://arxiv.org/abs/2502.12554) | [AIDABench/YOLOv12](https://github.com/AIDABench/YOLOv12) | Area Attention，速度/精度再平衡 |
| **YOLOv13** | 2025 | [2506.20298](https://arxiv.org/abs/2506.20298) | [YJ-AIDABench/YOLOv13](https://github.com/YJ-AIDABench/YOLOv13) | 多尺度感知增强 |
| **YOLOv14** | 2025 | 见官方发布 | 见官方发布 | Ultralytics 最新迭代 |
| **YOLOv26** | 2025 | 官方发布中 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 生产可用旗舰，端到端免后处理 |

> ⚠️ 官方论文缺失的版本（v5 / v8 / v11）并非疏漏，是 Ultralytics 选择以"白皮书 + 代码"发布，因为他们的迭代节奏比 arXiv 快。

---

## 衍生分支

主线之外的开源变体，通常专注解决某个具体问题。

| 分支 | 年份 | 论文 | 官方仓库 | 定位 |
|------|------|------|---------|------|
| **YOLO-World** | 2024 | [2401.17270](https://arxiv.org/abs/2401.17270) | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) | 开放词表检测，CLIP + 文本提示 |
| **YOLOE** | 2024 | [2411.13728](https://arxiv.org/abs/2411.13728) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 开放词表 + 可开关，Ultralytics 出品 |
| **YOLO-Master** | 2024 | [2411.21676](https://arxiv.org/abs/2411.21676) | [VikParuchuri/yoloa](https://github.com/VikParuchuri/yoloa)（继任） | 多模态统一，检测+分割+OCR+Caption |
| **YOLOX** | 2021 | [2107.08430](https://arxiv.org/abs/2107.08430) | [megvii-ception/YOLOX](https://github.com/megvii-ception/YOLOX) | Decoupled Head + SimOTA，工业部署优化 |
| **YOLOF** | 2021 | [2110.06864](https://arxiv.org/abs/2110.06864) | [BaiYue-feii/YOLOF](https://github.com/BaiYue-feii/YOLOF) | 统一训练范式，anchor-free |
| **PP-YOLOE** | 2023 | [2309.04043](https://arxiv.org/abs/2309.04043) | [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | 百度 Paddle 系，中文场景友好 |

---

## 任务扩展的额外参考

主线 YOLO 只讲"能做"，实际做某个任务时需要额外参考这些资源。

### 实例分割 (Segment)

- **SOLO / SOLOv2** — 与 YOLO-Seg 同思路的实例分割方案
  - SOLO: [1912.04260](https://arxiv.org/abs/1912.04260) · [fudan-zsl/solo](https://github.com/fudan-zsl/solo)
  - SOLOv2: [2019.07799](https://arxiv.org/abs/2019.07799) · [fudan-zsl/SOLOv2](https://github.com/fudan-zsl/SOLOv2)
- **SAM (Segment Anything Model)** — 大模型分割，适合辅助标注
  - [2304.02643](https://arxiv.org/abs/2304.02643) · [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

### 姿态估计 (Pose)

- **HRNet** — 高精度关键点基线，做上限对比
  - [1904.04514](https://arxiv.org/abs/1904.04514) · [leooooyao/HRNet](https://github.com/leooooyao/HRNet)
- **RTMPose** — 商汤出品，YOLO-Pose 的强竞争对手
  - [2312.03337](https://arxiv.org/abs/2312.03337) · [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

### 定向框 (OBB)

- **RotatedIoU 系列** — GIoU / DIoU / CIoU / WIoU 的原论文（损失函数来源）
  - GIoU: [1902.09630](https://arxiv.org/abs/1902.09630)
  - DIoU: [1911.08282](https://arxiv.org/abs/1911.08282)
  - CIoU: [1911.08282](https://arxiv.org/abs/1911.08282)（同 DIoU 论文）
  - WIoU: [2005.03572](https://arxiv.org/abs/2005.03572)
- **RT-DETR-OBB / YOLO-World-OBB** — 开放词表 + 定向框组合
  - [2405.16785](https://arxiv.org/abs/2405.16785)
- **DINO (Data Interpolation Network)** — 遥感 OBB 增强的标准做法
  - [2012.11879](https://arxiv.org/abs/2012.11879) · [cuhk-shen-group/data-in-network](https://github.com/cuhk-shen-group/data-in-network)

### 旋转检测 (Rotated)

- **Shapelet** — 任意旋转框的表示方式（非 4 点多边形）
  - [1907.10930](https://arxiv.org/abs/1907.10930) · [TieChenLin/Shapelet](https://github.com/TieChenLin/Shapelet)

---

## 关键概念的原论文

YOLO 用到的很多核心技巧来自其他论文，理解来源才能改进。

| 概念 | 首次提出 | 原论文 |
|------|---------|--------|
| **Non-Maximum Suppression (NMS)** | 1970s | 传统图像处理文献 |
| **Batch Normalization** | 2015 | [1502.03167](https://arxiv.org/abs/1502.03167) |
| **Focal Loss** | 2017 | [1704.06868](https://arxiv.org/abs/1704.06868) |
| **CSPNet** | 2019 | [1911.11929](https://arxiv.org/abs/1911.11929) |
| **SPPF (Spatial Pyramid Pooling - Fast)** | 2020 (YOLOv4) | [2004.10934](https://arxiv.org/abs/2004.10934) |
| **Mosaic Augmentation** | 2020 (YOLOv4) | [2004.10934](https://arxiv.org/abs/2004.10934) |
| **Mish Activation** | 2019 | [1908.08681](https://arxiv.org/abs/1908.08681) |
| **ReparamVGG / RepGhostNet** | 2021 | [2101.03697](https://arxiv.org/abs/2101.03697) |
| **ConvNeXt** | 2022 | [2201.03545](https://arxiv.org/abs/2201.03545) |
| **DFN (Distribution Focal Loss)** | 2021 (RetinaNet++) | [2012.05098](https://arxiv.org/abs/2012.05098) |
| **TAL (Task-Aligned Assignments)** | 2022 (TOOD) | [2108.12758](https://arxiv.org/abs/2108.12758) |
| **SimOTA** | 2021 (YOLOX) | [2107.08430](https://arxiv.org/abs/2107.08430) |
| **CLIP** | 2021 | [2103.00020](https://arxiv.org/abs/2103.00020) |
| **Co-Attention** | 2018 | [1704.06876](https://arxiv.org/abs/1704.06876) |

---

## 常用工具与数据集

### 数据集

| 数据集 | 用途 | 官方链接 |
|-------|------|---------|
| **COCO 2017** | 检测/分割/Pose 通用 | [images.cocodataset.org](https://images.cocodataset.org/) |
| **COCO-Keypoints** | 人体 17 关键点 | [cocodataset.org](https://cocodataset.org/#keypoints-2017) |
| **COCO-OBB** | 定向框，13 类 8150 图 | [github.com/Corgi007/COCO-OBB](https://github.com/Corgi007/COCO-OBB) |
| **DOTA** | 遥感 OBB，15 类 | [capt-whu.github.io/DOTA](https://capt-whu.github.io/DOTA/) |
| **VisDrone** | 无人机场景 | [vision.uwaterloo.ca/visdrone](https://www.vision.uwaterloo.ca/visdrone/) |
| **MOTChallenge** | 多目标跟踪 | [motchallenge.net](https://motchallenge.net/) |
| **OpenImages** | 8000 类大尺度 | [dataset.google.github.io/open-image](https://storage.googleapis.com/openimages/) |

### 数据集制作工具

| 工具 | 类型 | 链接 |
|------|------|------|
| **CVAT** | 企业级标注，支持 3D / 视频 | [cvat.ai](https://www.cvat.ai/) |
| **LabelMe** | 轻量，多边形友好 | [github.com/heartexlabs/labelme](https://github.com/heartexlabs/labelme) |
| **Roboflow** | 云端 + 数据增强 | [roboflow.com](https://roboflow.com/) |
| **LabelStudio** | 通用，开源 | [labelstud.io](https://labelstud.io/) |
| **X-AnyLabeling** | SAM 集成，一键分割 | [github.com/CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) |
| **LabelImg** | YOLO 传统工具 | [github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg) |

### 训练/推理框架

| 框架 | 定位 | 链接 |
|------|------|------|
| **Ultralytics** | YOLOv8-v26 官方 | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **YOLOv5** | 工业界 PyTorch 老经典 | [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| **Ultralytics HUB** | 云端训练 | [hub.ultralytics.com](https://hub.ultralytics.com/) |
| **ONNX Runtime** | 跨平台推理 | [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| **TensorRT** | NVIDIA GPU 加速 | [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) |
| **OpenVINO** | Intel CPU/GPU | [github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) |
| **NCNN** | 移动端高性能 | [github.com/Tencent/ncnn](https://github.com/Tencent/ncnn) |
| **MNN** | 阿里移动端 | [github.com/alibaba/MNN](https://github.com/alibaba/MNN) |

---

## 怎么找论文（三种方式）

### 方式 1：arXiv 搜索

URL 模板：`https://arxiv.org/abs/{arXiv ID}`

比如想看 YOLOv10 论文，直接访问 `https://arxiv.org/abs/2405.14458`。

### 方式 2：Google Scholar

搜索 `"{版本名} paper"`，通常第一条就是 arXiv 或会议页面。

### 方式 3：仓库 README

YOLO 系仓库 README 顶部基本都有 `This repository is the official implementation of {Paper}`，链接直达论文。

---

## 常见坑

- **v5 / v8 / v11 没论文不代表不重要**：工业界以"代码 + 白皮书"为主，学术圈论文只是形式之一。
- **arXiv 编号 4 位 vs 5 位**：2007 年前是 4 位（如 `cs/0301243`），之后是 6 位（`YYMM.NNNNN`），看错就找不到。
- **官方仓库 vs 复现仓库**：v7 官方是 `WongKinYiu/yolov7`，网上有大量 fork，用错了权重/配置可能训崩。
- **衍生分支更新不主动通知**：建议 GitHub Watch 关键仓库，`ultralytics/ultralytics` 更新最快。
- **旧论文代码已失效**：如 v1 / v2 仓库最后一次提交在 2018 年前，Python 2 时代，别硬跑。

---

## 下一步

- 想深入某个版本 → [版本详解](/versions/v8)
- 想理解演进脉络 → [evolution-story](/evolution-story)
- 想看完整对比 → [compare-matrix](/compare-matrix)
- 想查术语 → [glossary](/glossary)
