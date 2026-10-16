# 论文精读：You Only Look Once — Unified, Real-Time Object Detection

> **arXiv:1506.02640** · ECCV 2016
> Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
> PDF: <https://arxiv.org/pdf/1506.02640.pdf>

---

## 0. 论文定位

| 维度 | 说明 |
|------|------|
| 范式 | Single-Shot Detection（开创性） |
| 前置工作 | R-CNN (2014), Fast R-CNN (2014), Faster R-CNN (2015) |
| 后续影响 | YOLO v2-v11 + SSD + RetinaNet + DETR 系列 |
| 源码 | <https://github.com/pjreddie/darknet> commit `8322a58` |

本文档是对论文的逐节深度解读，与 VitePress 站点的 [论文精读页](../docs/versions/v1/paper.md) 互补——站点版侧重结构化速查，本版侧重推理链与批判性分析。

---

## 1. Motivation：为什么需要 YOLO

### 1.1 当时的检测器格局（2014-2015）

| 方法 | 范式 | mAP@0.5 (VOC07) | FPS |
|------|------|------------------|-----|
| R-CNN | 两阶段（Selective Search + CNN） | 66.0 | 0.03 |
| SPP-Net | 两阶段（SPP + CNN） | 41.4 | 5 |
| Fast R-CNN | 两阶段（RoI Pooling + CNN） | 52.4 | 7 |
| Faster R-CNN | 两阶段（RPN + CNN） | 69.9 | 7 |

**核心矛盾**：精度够高（Faster R-CNN 69.9），但速度不够实时（7 FPS）。

### 1.2 作者的观察

论文 §1 提出两个深层问题：

1. **两阶段流水线的结构性瓶颈**：区域提议→分类是串行的，每一步只看到局部信息，无法利用整图上下文。
2. **检测被建模为分类问题**：每个候选框独立分类，全局信息无法共享。

作者的判断：这两条根因可以一起解决——**把检测重新建模为回归问题，一次前向出所有结果**。

---

## 2. Core Idea：Single-Shot 回归

### 2.1 形式化定义

给定输入图像 $I \in \mathbb{R}^{H \times W \times 3}$，网络 $f$ 直接输出：

$$f(I) = \{(\mathbf{b}_i, c_i, \mathbf{p}_i)\}_{i=1}^{N}$$

其中：
- $\mathbf{b}_i = (x_i, y_i, w_i, h_i)$：第 $i$ 个 box 的位置和尺寸
- $c_i$：置信度（confidence）
- $\mathbf{p}_i \in \mathbb{R}^{C}$：类别概率分布
- $N = S^2 \times B = 7^2 \times 2 = 98$

### 2.2 与两阶段的关键区别

| 维度 | 两阶段（Faster R-CNN） | YOLO |
|------|------------------------|------|
| 候选区域 | RPN 生成 ~300 个 | 固定 98 个 |
| 特征提取 | 每个候选独立 RoI Pooling | 整图一次卷积 |
| 上下文 | 局部 | 全局 |
| 速度 | O(300) 次分类头 | O(1) 次回归头 |
| 损失 | 分类 + 回归 分离 | 统一回归 |

---

## 3. 网格化空间分解

### 3.1 Grid Cell 机制

将输入图像划分为 $S \times S$ 个网格（v1 取 $S=7$）。每个 grid cell 负责检测**中心落在该格内的物体**。

```
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ ● │   │   │   │   │   ← 物体中心在此格
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
 S=7 → 49 cells
```

### 3.2 每格 B 个 Box

每个 cell 预测 $B$ 个 box（v1 取 $B=2$）。论文 §2 解释为什么选 2：

> "if two objects of very different aspect ratios overlap or if two objects of the same class overlap, YOLO struggles to find both of them."

两个 box 会自然学到不同宽高比的偏好——一个偏横（车），一个偏竖（人）。但这也导致"相似形状不同类混淆"（zebra vs horse）。

### 3.3 Grid 分辨率的局限

$S=7$ 意味着每个 cell 覆盖 $64 \times 64$ 像素（448/7）。对于 < 32×32 的小目标，一个 cell 的感受野太大，定位精度不够。这是 v1 小目标召回差的直接原因。

---

## 4. 输出维度推导

### 4.1 论文描述 vs 源码实现

论文 §2 的公式暗示每格输出 $B \times (4+1+C)$：

$$\text{output} = S \times S \times (B \times (4+1) + C) = 7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30 = 1470$$

但源码 `detection_layer.c` 的实际 reshape 逻辑更复杂：

```c
// 每格输出 = [box1: x,y,w,h,conf] [box2: x,y,w,h,conf] [classes: 20]
// = 5 + 5 + 20 = 30
// 总输出 = 7 * 7 * 30 = 1470
```

**关键优化**：第二组 box 只复用 `[x,y,w,h,conf]`，不重复 20 类概率——因为同一格内两个 box 的类别相同。

### 4.2 输出张量结构

```
output[0..1469] reshape → 7×7×30

每个 30 维向量：
  [0:5]   = box1 (x, y, w, h, confidence)
  [5:10]  = box2 (x, y, w, h, confidence)
  [10:30] = class probabilities (20 类)
```

---

## 5. Confidence 的独特定义

### 5.1 定义

$$\text{confidence} = P(\text{Object}) \times \text{IoU}(\text{pred}, \text{GT})$$

### 5.2 训练时 vs 推理时

**训练时**：GT 已知，IoU 可计算。confidence 的目标值就是 IoU。

```c
// detection_layer.c:82-83
delta[l.coords + n * l.outputs] = l.object_scale * (iou - pred.conf);
```

**推理时**：GT 未知，IoU 不可计算。confidence 退化为一个 learned 的 P(Object) 估计器——网络隐式学到了"这个位置有物体的概率"。

### 5.3 设计妙处

1. **推理时自动降权定位差的框**：即使 P(Object)=1，如果定位差，最终分数也低
2. **训练时把 IoU 信号注入 objectness 分支**：不用单独的 box loss 就能间接监督定位
3. **rescore=1 修正**（源码 trick）：让 delta 更贴合当前 IoU，而非固定为 1

---

## 6. 损失函数深度分析

### 6.1 完整公式

$$L = \lambda_{\text{coord}} \cdot L_{\text{coord}} + L_{\text{box}} + L_{\text{obj}} + \lambda_{\text{noobj}} \cdot L_{\text{noobj}} + L_{\text{class}}$$

各项展开：

| 项 | 公式 | 权重 | 作用域 |
|----|------|------|--------|
| $L_{\text{coord}}$ | $\sum (\sqrt{w} \cdot \sqrt{h}) \cdot [(x - \hat{x})^2 + (y - \hat{y})^2]$ | 5 | 有目标格 |
| $L_{\text{box}}$ | $\sum (\sqrt{w} \cdot \sqrt{h}) \cdot (1 - \text{IoU})^2$ | 1 | 有目标格 |
| $L_{\text{obj}}$ | $\sum (1 - \text{conf})^2$ | 1 | 有目标格 |
| $L_{\text{noobj}}$ | $\sum \text{conf}^2$ | 0.5 | 无目标格 |
| $L_{\text{class}}$ | $\sum (p_c - \hat{t}_c)^2$ | 1 | 有目标格 |

### 6.2 权重 5/1/1/0.5 的逻辑

- **coord=5**：位置回归最难优化，梯度量级小，需放大 5 倍
- **box=1**：IoU loss 量级适中
- **obj=1**：置信度目标 [0, 1]，量级适中
- **noobj=0.5**：背景格占 49/49 的绝大多数（实际约 90%），不降权会淹没前景信号

### 6.3 sqrt=1 技巧

```c
if (l.sqrt) {
    box.w = pow(boxes[i].w, 0.5);
    box.h = pow(boxes[i].h, 0.5);
}
```

对 w/h 做开方，让大小目标在 loss 里的量级接近：
- 大目标 w=0.8 → √w=0.89
- 小目标 w=0.1 → √w=0.32

不开方：大/小 = 8:1；开方后：大/小 = 2.8:1，梯度更均衡。

### 6.4 随机 best_index（论文未提的源码 trick）

```c
if (l.count < 100000) best_index = rand() % l.n;
```

前 10 万次迭代，随机分配 GT 到某个 box，防止两个 box 抢同一个 GT 导致梯度塌缩。这是 v1 最有意思的 trick，暴露了"多 box 结构不稳定"的本质——v2 引入 anchor 后彻底废弃。

---

## 7. 网络架构

### 7.1 整体结构

24 层 Conv + 1 层 Local + 1 层 Dropout + 2 层 FC + 1 层 Detection = 29 层。

| 阶段 | 层 | 输出尺寸 | 通道 |
|------|-----|---------|------|
| Stem | Conv 7×7 s2 | 224×224 | 64 |
| Block 1 | Conv 3×3 → MaxPool | 112×112 | 192→128 |
| Block 2 | Conv 3×3 ×3 → MaxPool | 56×56 | 128→256→512 |
| Block 3 | Conv 3×3 ×5 → MaxPool | 28×28 | 512→256→512→1024 |
| Block 4 | Conv 3×3 ×2 | 14×14 | 1024 |
| Head | Local 3×3 → FC 4096 → Dropout → FC 1470 | 1×1 | 4096→1470 |
| Output | Detection Layer | 7×7×30 | — |

### 7.2 参数量估算

| 组件 | 参数量 |
|------|--------|
| Conv 7×7, 3→64 | 9,408 |
| Conv 3×3, 64→192 | 110,592 |
| Conv 3×3, 192→128 → 256 → 512 (×3) | ~3.5M |
| Conv 3×3, 512→256→512→1024 (×5) | ~35M |
| Conv 3×3, 1024 (×2) | ~18.9M |
| FC 1024×4096 | ~4.2M |
| FC 4096×1470 | ~6.0M |
| **Total** | **~70M** |

对比：ResNet-50 约 25M，YOLOv8n 约 3.6M。v1 的参数量极为臃肿。

### 7.3 全连接层的意义

v1 使用 FC 做检测头，这是历史遗留——AlexNet 的范式。FC 的好处是：
- 全局信息聚合（7×7 grid → 1470 维）
- 直接回归出绝对坐标

但缺点也是致命的：
- 输入尺寸固定（448×448）
- 参数量大（FC 占 ~10M）
- 无法用卷积的平移不变性

v2 起全部换回卷积，不再使用 FC。

---

## 8. 训练策略

### 8.1 两阶段训练

1. **Stage 1（ImageNet 分类预训练）**：前 22 层 Conv 在 ImageNet 1000 类上训练 160 轮
2. **Stage 2（VOC 检测微调）**：加 FC + Detection 头，在 VOC 2007+2012 上训练 23 轮

### 8.2 超参数

| 参数 | 值 | 备注 |
|------|-----|------|
| Optimizer | SGD | momentum=0.9 |
| Weight decay | 0.0005 | |
| Batch size | 64 | |
| Learning rate | 0.001 | step decay |
| LR schedule | 200→×2.5, 400→×2, 600→×2, 20k→×0.1, 30k→×0.1 | warmup + decay |
| Max batches | 40000 | ~160 epochs on VOC |
| Jitter | 0.2 | 20% 随机平移/缩放 |
| Dropout | 0.25 | FC 之间 |

### 8.3 数据增强

v1 仅使用 Region Jitter（随机平移/缩放 20%），没有 Mosaic、MixUp、CutMix。这是 v1 泛化能力有限的原因之一。

---

## 9. 实验结果

### 9.1 VOC 2007 主表

| 方法 | mAP@0.5 | FPS |
|------|---------|-----|
| DPM v5 | 22.5 | 3 |
| R-CNN | 66.0 | 0.03 |
| SPP-Net | 41.4 | 5 |
| Fast R-CNN | 52.4 | 7 |
| Faster R-CNN | 69.9 | 7 |
| **YOLO (448)** | **63.4** | **35** |
| **YOLO (224)** | **58.4** | **155** |

### 9.2 误差分析（论文 §7.2）

| 错误类型 | 比例 | 说明 |
|----------|------|------|
| 定位错误 | 57% | box 位置偏，IoU < 0.5 |
| 误报 | 19% | 背景误检为物体 |
| 分类错误 | 24% | 类别识别错 |

对比 Faster R-CNN：定位错误 4.0%，YOLO 19.0%。**定位是 YOLO 的最大短板**。

### 9.3 与 Faster R-CNN 的互补性

论文 §4.3 做了一个有趣的实验：YOLO + Faster R-CNN ensemble。

| 方法 | mAP@0.5 |
|------|---------|
| Faster R-CNN | 69.9 |
| YOLO | 63.4 |
| YOLO + Faster R-CNN | **75.0** |

YOLO 的误报率低（3.0% vs 6.0%），Faster R-CNN 的定位精度高。两者互补，说明 single-shot 和 two-stage 学到的特征不同。

---

## 10. YOLO9000 扩展

论文 §8 展示了 YOLO 扩展到 9000+ 类的能力：

- **WordTree**：将 ImageNet 1000 类和 VOC 20 类映射到层级结构
- **训练策略**：检测数据用 VOC，分类数据用 ImageNet，按 WordTree 节点计算损失
- **结果**：在 VOC 2007 达到 48.3 mAP @ 50 FPS，可识别 9000+ 类

---

## 11. 局限性与后续工作

### 11.1 论文承认的局限

1. **定位精度低**：主要错误来源（57%）
2. **小目标召回差**：7×7 grid 分辨率不够
3. **相似形状混淆**：每格 2 box 限制
4. **输入尺寸固定**：FC 导致无法多尺度

### 11.2 后续版本的解法

| 痛点 | v2 | v3 | v4 | v8 | v10 |
|------|-----|-----|-----|-----|-----|
| 小目标 | — | FPN 多尺度 | PAN | P2 head | — |
| 定位粗糙 | anchor | — | CIoU | DFL | — |
| 相似形状 | anchor | 9 anchors | — | — | — |
| NMS 敏感 | — | — | — | — | NMS-free |
| 输入固定 | 去FC | — | — | — | — |

### 11.3 更深层的影响

YOLO 的真正贡献不只是速度——它证明了**检测可以不依赖区域提议**。这一思想影响了：
- **SSD** (2016)：多尺度 single-shot
- **RetinaNet** (2017)：single-shot + Focal Loss
- **FCOS** (2019)：anchor-free single-shot
- **DETR** (2020)：Transformer-based single-shot

---

## 12. 批判性评述

### 12.1 被高估的部分

- **63.4 mAP 并不算高**：Faster R-CNN 69.9，差 6.5 个点。论文用速度优势掩盖了精度差距
- **35 FPS 在 Titan Black 上**：换到嵌入式设备可能只有 5-10 FPS
- **YOLO9000 的 48.3 mAP**：9000 类中大部分类别 mAP 极低，主要靠 VOC 20 类撑

### 12.2 被低估的部分

- **全局上下文的价值**：YOLO 的误报率确实低（3% vs 6%），说明全局信息有用
- **置信度定义的优雅**：P(Object)·IoU 把定位质量嵌入分数，后续很多工作沿用
- **工程化意义**：darknet 框架的简洁性让 YOLO 极易复现和改进

### 12.3 历史地位

YOLO v1 不是精度最高的，也不是速度最快的（YOLOv2+ 更快）。但它**定义了 single-shot detection 这一范式**，开启了之后 10 年的 YOLO 系列演化。论文引用量 25000+（截至 2025），是目标检测领域最高引论文之一。

---

## 引用

```bibtex
@inproceedings{redmon2016you,
  title={You only look once: Unified, real-time object detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={779--788},
  year={2016}
}
```

## 相关资源

- 论文 PDF：<https://arxiv.org/pdf/1506.02640.pdf>
- 源码：<https://github.com/pjreddie/darknet>
- 官网：<https://pjreddie.com/yolo/>
- VitePress 论文精读页：[docs/versions/v1/paper.md](../../docs/versions/v1/paper.md)
- 本体论 Paper 实例：[ontology/instances/yolov1/papers.yaml](../../ontology/instances/yolov1/papers.yaml)
