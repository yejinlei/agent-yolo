# YOLOv1 论文精读

**You Only Look Once: Unified, Real-Time Object Detection**

*Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi · ECCV 2016 · [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)*

---

## 1. Motivation（为什么做）

论文开篇直击当时主流检测器的两个痛点：

1. **太慢**：R-CNN 用 GPU 时 47s/img，Faster R-CNN 也要 0.14s/img，Real-Time Object Detection 仍是未解难题
2. **两阶段丢失全局上下文**：区域提议 + 分类的流水线，让每个 box 只看到自己那块区域，无法利用整图上下文

作者的判断：这两条根因可以一起解决——**把区域提议和分类合并到同一个网络里**。

## 2. 核心 Idea：Single-Shot 端到端检测

作者把目标检测重新定义为**回归问题**：

> 给定输入图像 I，网络 f 直接输出所有目标的位置、置信度和类别。

```
输入：H × W × 3
输出：N × (4 + 1 + C)
      N = grid² × num_boxes = 7² × 2 = 98
      每个预测 = (cx, cy, w, h, confidence, p_1, p_2, ..., p_C)
```

**关键**：从"分类 + 回归"两个任务被合并到一个网络里。

## 3. 网格化空间分解

### 3.1 Grid Cell

把输入图像划分成 S×S 个 grid cell。v1 使用 S=7，即 49 个 cell。

- 每个 cell 独立负责检测**中心落在该 cell 内的物体**
- 每个 cell 预测 B 个 bounding box，v1 中 B=2

### 3.2 每格 Box 数量

论文原话：

> Since YOLO performs detection and classification simultaneously, if two objects of very different aspect ratios overlap or if two objects of the same class overlap, YOLO struggles to find both of them.

作者观察到：**同格 2 box 已能学到不同宽高比的偏好**——某些 box 偏好车（宽扁），某些偏好人（高瘦）。但这也解释了"相似形状不同类物体混淆"的痛点（zebra vs horse）。

## 4. 输出维度推导

每个 box 的预测是 4+1+C=25 维：

- 4 维位置：(cx, cy, w, h)
  - (cx, cy) 相对当前 grid cell 的坐标，范围 [0, 1]
  - w, h 相对**整张图像**的宽高，范围 [0, 1]
- 1 维 confidence：**P(Object)·IoU(pred, GT)**
- 20 维类别概率

每格 2 个 box 需要输出：

```
49 cells × 2 boxes × (4 + 1 + 20) = 49 × 2 × 25 = 2450
```

但源码实际输出是 **1470**：

```
49 × (4 + 1 + 20 + 2 × 5) = 49 × 30 = 1470
```

**优化**：第二组 box 只复用 [4+1] 位置，不重复 20 类（因为同一 cell 内两个 box 的类别基本相同）。

## 5. Confidence 的独特定义

论文把 confidence 定义成：

```
confidence = P(Object) · IoU(pred, GT)
```

不是简单的 P(Object)。这个设计有两个妙处：

1. **推理时自动降权定位差的框**：即使 P(Object)=1，如果 IoU 低，最终分数也会低
2. **训练时把 IoU 信号注入 objectness 分支**：不用单独的 box loss 就能间接监督定位

但**推理时无法计算 IoU**，实际退化成了一个 learned 的 P(Object) 估计器。

## 6. 损失函数

```
L = Σ (√w·√h) · [(pred_x - t_x)² + (pred_y - t_y)²]      # coord 项
  + Σ (√w·√h) · (1 - IoU)²                                  # box 项
  + Σ (p_c - t_c)²                                           # class 项
  + Σ p_c² · [no object]                                     # no-object 项
```

**权重**：

- Coord 项：5（最难优化，需要放大）
- Box 项：1
- Class 项：1
- No-Object 项：0.5（背景格子多，不加权会淹没信号）

**sqrt=1 技巧**：

```c
if (l.sqrt) {
  box.w = pow(boxes[i].w, 0.5);
  box.h = pow(boxes[i].h, 0.5);
}
```

代码里对 w/h 做开方，让大小目标在损失里的量级接近。

## 7. 数据增强

**Region Jitter (jitter=0.2)**：训练时随机平移、缩放 20%。

```c
dx = random(0.8, 1.2)
dy = random(0.8, 1.2)
// 应用 jitter 到图像和 GT 位置
```

这是 v1 时代主要的增强手段，v4 后被 Mosaic 取代。

## 8. 训练策略

- **网络**：24 层 Conv + 2 层 FC
- **Backbone**：AlexNet 变体（第一层 7×7 stride 2）
- **预训练**：从 ImageNet 1000 类分类模型加载前 22 层
- **优化器**：SGD，momentum 0.9，weight decay 0.0005
- **Batch**：64
- **学习率**：0.001
- **轮数**：ImageNet 160 轮 + VOC 微调 23 轮

## 9. 结果

### 9.1 Pascal VOC 2007 结果

| 方法 | mAP | FPS |
|------|-----|-----|
| DPM | 22.5 | 3 |
| R-CNN | 22.2 | 0.3 |
| SPP-Net | 41.4 | 5 |
| Fast R-CNN | 52.4 | 7 |
| Faster R-CNN | 68.7 | 7 |
| **YOLO (448)** | **63.4** | **35** |
| **YOLO (224)** | **58.4** | **155** |

**分析**：

- YOLOv1 相比 Faster R-CNN 少 5.3 个 mAP 点，但快 5 倍
- 224 版本 155 FPS 是真正的实时

### 9.2 失败模式分析

论文 §7.2 列出三大错误来源：

| 错误 | 比例 | 例子 |
|------|------|------|
| 定位差 | 57% | box 位置偏 |
| 误报 | 19% | 误检 |
| 分类错 | 24% | 类别识别错 |

## 10. YOLO9000 扩展

论文 §8 展示了扩展到 9000+ 类的能力：

- **类别融合**：将 ImageNet 与 VOC 类别映射到层级结构中
- **训练策略**：先用合成图训练类别识别，再用真实图微调
- **结果**：在 VOC 2007 达到 48.3 mAP @ 50 FPS

## 11. 论文的核心结论

1. **速度不是精度与实时之间的唯一权衡**：YOLO 通过一次前向证明实时检测是可行的
2. **全局上下文重要**：整图前向让 YOLO 在识别环境/上下文相关物体（如沙滩 vs 城市）上表现更好
3. **泛化能力强**：在艺术与照片之间迁移时表现更好（因为训练数据多样）

## 12. 局限与后续工作

论文承认：

- **定位精度低**：主要错误来源
- **小目标召回差**：7×7 grid 分辨率不够
- **相似形状不同类混淆**：每格 2 box 限制

这些直接对应到 YOLOv2 的三个改进方向：

- **Anchor 网格**（v2）→ 提高定位精度
- **多尺度预测**（v3）→ 提升小目标
- **更多 anchors**（v3 的 9 anchors）→ 缓解相似形状问题

---

## 引用建议

> Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *ECCV* (pp. 804–823).

---

## 相关资源

- 论文 PDF：https://arxiv.org/pdf/1506.02640.pdf
- 论文代码：https://github.com/pjreddie/darknet (commit `8322a58`)
- PPT：https://pjreddie.com/yolo/
- 后续版本：[YOLOv2](https://pjreddie.com/darknet/yolo/) / [YOLOv3](https://pjreddie.com/darknet/yolo/)
