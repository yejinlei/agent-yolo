# YOLOv4 训练与推理技巧

YOLOv4 是"trick 集大成"的版本——论文把 2018-2020 年社区验证过的技巧逐项消融，最后留下一套组合拳。每个 trick 都配：**原理图、白话、源码、消融数据**。

---

## 全部 12 个 Trick 速查

| # | Trick | 类别 | 位置 | 论文消融 ΔAP |
|---|-------|------|------|-------------|
| 1 | CSP 结构 | 结构 | `yolov4.cfg` 层 3+ | +0.3 |
| 2 | Mish 激活 | 激活 | activations.c | +0.6 |
| 3 | SPP | Neck | 层 105-116 | +0.7 |
| 4 | PANet 双向聚合 | Neck | 层 140-161 | +0.9 |
| 5 | CIoU loss | 损失 | `iou_loss=ciou` | +0.5 |
| 6 | scale_x_y | 损失 | `scale_x_y=1.2/1.1/1.05` | +0.3 |
| 7 | Multi-positive | 分配 | `iou_thresh=0.213` | +1.5 |
| 8 | Mosaic 增强 | 增强 | `mosaic=1` | +0.7 |
| 9 | DropBlock | 正则化 | 代码实现 | +0.4 |
| 10 | 多尺度训练 | 增强 | `random=1` | +0.5 |
| 11 | 色彩增强 | 增强 | hue/sat/exposure | +0.3 |
| 12 | burn_in + steps | LR | `burn_in=1000` | — |

> **v3 读者路标**：v3 的 10 个 trick 在 v4 里全部保留，v4 新增 6 个（CSP/Mish/SPP/PANet/CIoU/Mosaic）+ 2 个微调（scale_x_y/multi-positive）。**没有任何 v3 trick 被删除**——v4 是纯增量。

---

## 1. CSP 结构 (`trick-csp`)

### 原理图

```mermaid
flowchart TB
    X["输入 2C"] --> SPLIT["1×1 conv 分流"]
    SPLIT --> MAIN["主路：残块×N"]
    SPLIT -.->|"旁路"| CONCAT["route concat"]
    MAIN --> CONCAT --> TRANS["1×1 conv 过渡"]
```

### 白话

> 把特征劈成两半：一半走残块（重加工），一半直连（保留原始信息），最后 concat 回来。计算量降 20%，梯度有直通路径，训练更稳。

### 源码

第一个 CSP 段（层 0-10）的分流/concat/过渡：<SrcLink file="v4:cfg/yolov4.cfg" :line="56" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="95" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="98" />。

### 消融

论文 Table 2：CSP +0.3 AP，计算量 -20%。

---

## 2. Mish 激活 (`trick-mish`)

### 原理图

```mermaid
flowchart LR
    X["x"] --> M["x · tanh(softplus(x))"]
    M --> Y["平滑非单调"]
```

### 白话

> 比 leaky 更平滑的激活：负值不是一刀切压缩成 0.1x，而是平滑过渡到接近 0 的小负值。梯度在负区间也能流动。

### 源码

激活函数注册 <SrcLink file="v4:src/activations.c" :line="50" />，前向 <SrcLink file="v4:src/activations.c" :line="143" :end="151" />，梯度 <SrcLink file="v4:src/activations.c" :line="344" :end="360" />。cfg 里骨干 72 个 mish、neck/head 35 个 leaky。

### 消融

论文 Table 3：Mish +0.6 AP（vs leaky）。

---

## 3. SPP (`trick-spp`)

### 原理图

```mermaid
flowchart TB
    X["19×19×512"] --> P5["maxpool 5×5 s1"] & P9["maxpool 9×9 s1"] & P13["maxpool 13×13 s1"] & SKIP["直连"]
    P5 & P9 & P13 & SKIP --> CAT["concat → 19×19×2048"]
```

### 白话

> 三个不同核的 maxpool 并行，同一位置用 5/9/13 三种窗口"看"——感受野扩大 3 个尺度，代价极小（maxpool 无参数）。

### 源码

三个 maxpool <SrcLink file="v4:cfg/yolov4.cfg" :line="763" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="772" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="779" />，concat <SrcLink file="v4:cfg/yolov4.cfg" :line="790" />。

### 消融

论文 Table 4：SPP +0.7 AP。

---

## 4. PANet 双向聚合 (`trick-panet`)

### 原理图

```mermaid
flowchart TB
    subgraph FPN["FPN 自顶向下"]
        D["19×19"] --> U1["upsample"] --> M1["concat 38×38"] --> U2["upsample"] --> M2["concat 76×76"]
    end
    subgraph PAN["PANet 自底向上"]
        M2 --> D1["conv s2 76→38"] --> M3["concat 38×38"] --> D2["conv s2 38→19"] --> M4["concat 19×19"]
    end
    FPN --> PAN
```

### 白话

> v3 的 FPN 只把深层语义往浅层搬；v4 加了一路下采样，把浅层的位置信息送回深层——双向奔赴。

### 源码

下采样 conv s2 <SrcLink file="v4:cfg/yolov4.cfg" :line="978" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="1065" />，concat <SrcLink file="v4:cfg/yolov4.cfg" :line="986" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="1073" />。

### 消融

论文 Table 5：PANet +0.9 AP。

---

## 5. CIoU loss (`trick-ciou`)

### 原理图

```mermaid
flowchart LR
    P["预测框"] --> IOU["IoU 项"]
    P --> D["中心距项"]
    P --> A["长宽比项"]
    IOU & D & A --> L["CIoU loss"]
```

### 白话

> 不再回归坐标偏移量，直接对 IoU 求梯度——训练目标和评测指标对齐。三项惩罚：重叠面积 + 中心距 + 长宽比。

### 源码

cfg 开关 <SrcLink file="v4:cfg/yolov4.cfg" :line="978" />，梯度计算 <SrcLink file="v4:src/box.c" :line="258" :end="476" />。详见 [losses 页](/versions/v4/losses) 和 [loss-gradient 页](/versions/v4/loss-gradient)。

### 消融

论文 Table 8：CIoU +0.5 AP（vs SSE）。

---

## 6. scale_x_y 消网格敏感 (`trick-scale-xy`)

### 原理图

```mermaid
flowchart LR
    TX["tx"] --> SIG["σ(tx)<br/>0~1"] --> SCALE["× 1.2<br/>− 0.1"] --> BX["bx ∈ (-0.1, 1.1)"]
```

### 白话

> v3 的 σ(tx) 严格在 (0,1)，每个格子只能预测自己格子内的中心点；v4 加 scale 把范围扩到 (-0.1, 1.1)，允许中心点稍微"越界"。

### 源码

cfg <SrcLink file="v4:cfg/yolov4.cfg" :line="974" />（`scale_x_y = 1.2`），前向 <SrcLink file="v4:src/yolo_layer.c" :line="337" />。

### 消融

论文 Table 8：scale_x_y +0.3 AP。

---

## 7. Multi-positive 多正样本 (`trick-multi-positive`)

### 原理图

```mermaid
flowchart TB
    GT["GT"] --> BEST["best_n 独占"]
    GT --> LOOP["同头其余 anchor<br/>IoU>0.213 也监督"]
    BEST --> AVG["delta 平均"]
    LOOP --> AVG
```

### 白话

> v3 的单 GT 单 anchor 太苛刻；v4 让同头内形状相似的 anchor 都参与监督，样本利用率翻倍。

### 源码

cfg <SrcLink file="v4:cfg/yolov4.cfg" :line="975" />（`iou_thresh=0.213`），第三遍循环 <SrcLink file="v4:src/yolo_layer.c" :line="501" :end="545" />，delta 平均 <SrcLink file="v4:src/yolo_layer.c" :line="556" />。

### 消融

论文 Table 8：Multi-positive +1.5 AP（**最大单项提升**）。

---

## 8. Mosaic 四图拼接 (`trick-mosaic`)

### 原理图

```mermaid
flowchart TB
    I1["图1"] & I2["图2"] & I3["图3"] & I4["图4"] --> X["随机十字拼接"] --> OUT["单张 608×608"]
```

### 白话

> 4 张图拼成一张，上下文多样性 4 倍。BN 一次统计 4 张图的分布，相当于隐式增大 batch。**训练时间反而更短**（22h vs v3 的 50h）。

### 源码

cfg <SrcLink file="v4:cfg/yolov4.cfg" :line="26" />（`mosaic=1`），实现 `src/data.c` L967+。

### 消融

论文 Table 8：Mosaic +0.7 AP。

---

## 9. DropBlock (`trick-dropblock`)

### 原理图

```mermaid
flowchart LR
    X["特征图"] --> D["随机挖掉连续区域块"] --> Y["输出"]
```

### 白话

> 比 Dropout 更适合卷积：随机"挖掉"连续的区域块，而不是孤立像素——强迫网络用周围信息补全。

### 源码

实现在 `src/dropout_layer.c`（DropBlock 模式）。

### 消融

论文 Table 7：DropBlock +0.4 AP。

---

## 10. 多尺度训练 (`trick-random`)

### 原理图

```mermaid
flowchart LR
    T["每 10 batch"] --> R{"random=1<br/>320~608"} --> NET["同权重"] --> E["推理任意尺寸"]
```

### 白话

> 与 v3 相同——每 10 batch 重摇输入尺寸，三尺度头在各种输入下都被锻炼。

### 源码

<SrcLink file="v4:cfg/yolov4.cfg" :line="981" />（`random=1`，仅 19×19 头配）。

---

## 11. 色彩增强三件套 (`trick-color-aug`)

### 原理图

```mermaid
flowchart LR
    IMG["训练图"] --> H["hue ±0.1"] --> S["sat ×1.5"] --> E["exp ×1.5"] --> OUT["增强图"]
```

### 白话

> 与 v3 相同——hue/saturation/exposure 全部真实生效。

### 源码

<SrcLink file="v4:cfg/yolov4.cfg" :line="14" :end="16" />。

---

## 12. burn_in + steps LR 调度 (`trick-lr-schedule`)

### 原理图

```mermaid
flowchart LR
    B0["batch 0"] -->|"burn_in: 线性升温"| B1["batch 1000<br/>lr=0.001"]
    B1 -->|"400k/450k 各 ×0.1"| END["batch 500200"]
```

### 白话

> 与 v3 相同——前 1000 iter 线性升温，40 万/45 万步各降 10 倍。

### 源码

<SrcLink file="v4:cfg/yolov4.cfg" :line="18" :end="23" />。

---

## 代码就绪但未启用的 Trick

| Trick | 状态 | 原因 |
|-------|------|------|
| SAT（自对抗训练） | `detector.c` L1717 有代码 | cfg 无 `adversarial_lr` → 默认 0 |
| DIoU-NMS | `box.c` diounms_sort 就绪 | cfg `nms_kind=greedynms` 用 greedy |
| CutMix | `data.c` 有代码 | 与 detector 不兼容（L955 报错） |
| Label Smoothing | 代码就绪 | cfg `label_smooth_eps=0` |

---

## 12 个 Trick 的谱系图

```mermaid
flowchart TB
    ROOT[YOLOv4 训练/推理技巧]
    ROOT --> CSP & MISH & SPP & PAN & CIOU & SXY & MP & MOS & DB & MS & CA & LR
    CSP --> SPP
    SPP --> PAN
    PAN --> CIOU
    MOS --> MS
```

---

## 延伸阅读

- [论文精读](/versions/v4/paper)
- [Architecture 逐层图解](/versions/v4/architecture)
- [理论 Concept](/versions/v4/concepts)
- [损失函数（CIoU 全套）](/versions/v4/losses)
- [反向传播推导](/versions/v4/loss-gradient)
- [源码走读](/versions/v4/code-walkthrough)
- [痛点追踪](/versions/v4/painpoints)
