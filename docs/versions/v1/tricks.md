# YOLOv1 训练与推理技巧

YOLOv1 有 7 个关键 knob，全部通过 `cfg/yolo.cfg` 或源码几行代码实现。每个 trick 下面都有：**原理图、白话、源码、反例/演进**。

---

## 全部 7 个 Trick 速查

| # | Trick | 类别 | 位置 | 影响 |
|---|-------|------|------|------|
| 1 | sqrt=1 加速收敛 | 损失设计 | `yolo.cfg` | 大小目标损失均衡 |
| 2 | 权重 5/1/1/0.5 | 损失设计 | `yolo.cfg` | 平衡正负样本 |
| 3 | 前 100k 随机 best_index | 训练稳定 | `detection_layer.c:136-138` | 防两个 box 抢同一个 GT |
| 4 | Region Jitter 0.2 | 数据增强 | `data.c` | 提升定位鲁棒性 |
| 5 | Dropout 0.25/0.5 | 正则化 | `yolo.cfg` | 缓解 FC 过拟合 |
| 6 | NMS IoU=0.5 | 推理后处理 | `box.c` | 去重 |
| 7 | ImageNet 预训练 | 初始化 | 权重加载 | 快速收敛 |

---

## 1. sqrt=1 加速收敛 (`trick-sqrt-one`)

### 原理图

```mermaid
flowchart LR
    subgraph RAW["直接回归 w/h"]
        A1["大目标 w=400<br/>误差 8 → L2=64"] --> BIG["梯度主导"]
        A2["小目标 w=25<br/>误差 2 → L2=4"] --> SMALL["梯度被忽略"]
    end
    subgraph SQRT["回归 √w/√h"]
        B1["√400=20<br/>误差 0.2 → L2=0.04"] --> EQ
        B2["√25=5<br/>误差 0.2 → L2=0.04"] --> EQ["梯度均衡"]
    end
    RAW -.-> SQRT
```

### 白话

> 直接让模型预测宽高，大物体的误差会把小物体按在地上摩擦。改成预测“宽高的平方根”，误差量级就拉平了。

### 源码

```cfg
# cfg/yolo.cfg
sqrt=1
```

```c
// src/detection_layer.c:108-111, 167-170
if (l.sqrt){
    out.w = out.w*out.w;      // 前向：把预测值平方得到真实 w
    out.h = out.h*out.h;
}
// 反向：对 √w 与 √gt_w 求差
l.delta[box_index+2] = l.coord_scale*(sqrt(state.truth[tbox_index + 2]) - l.output[box_index + 2]);
```

### 反例

如果把 `sqrt=0`：
- 大物体（汽车）的 L2 误差是小物体（鸟）的几十倍
- 模型优先把大物体框准，小物体 recall 进一步下降
- 预期 mAP 下降 1~2 点

### 演进

v2 引入 anchor + K-means 后，宽高由 anchor 先验吸收，不再需要 sqrt 技巧。v3+ 改用 CIoU/GIoU，几何关系显式建模。

---

## 2. 损失权重 5/1/1/0.5 (`trick-loss-weight-balance`)

### 原理图

```mermaid
flowchart TB
    subgraph WITHOUT["不加权"]
        NOOBJ_W["No-Object 项<br/>49 cells × 2 boxes = 98 项"] --> DOM["淹没其它 loss"]
        COORD_W["Coord 项<br/>值域极小"] --> DOM
    end
    subgraph WITH["加权"]
        NOOBJ_05["No-Object × 0.5"] --> BALANCE
        COORD_5["Coord × 5"] --> BALANCE["各项同量级"]
        BOX_1["Box × 1"] --> BALANCE
        OBJ_1["Obj × 1"] --> BALANCE
        CLS_1["Class × 1"] --> BALANCE
    end
    WITHOUT -.-> WITH
```

### 白话

> 背景格子太多了，不加权的话模型一天到晚只学会“这里没东西”。坐标那项又天然很小，要乘 5 才能被优化器看见。

### 源码

```cfg
# cfg/yolo.cfg
coord_scale=5
object_scale=1
noobject_scale=.5
class_scale=1
```

### 反例

如果 `coord_scale=1, noobject_scale=1`：
- no-object 项（98 项）vs coord 项（<10 项），梯度比例约 10:1
- 模型大量输出低 confidence，_coord 收敛极慢_
- 预期 mAP 下降 2~3 点

---

## 3. 前 100k 随机 best_index (`trick-random-best-index`)

### 原理图

```mermaid
flowchart TD
    START["每个有目标的 cell"] --> COUNT{"seen < 100000?"}
    COUNT -->|是| RAND["rand() % l.n<br/>随机选一个 box"]
    COUNT -->|否| IOU["选 IoU 最大 box"]
    RAND --> BOTH["两个 box 都有机会被训练"]
    IOU --> ONLY["只训练优胜 box"]
```

### 白话

> 刚开始两个 box 都是瞎猜的，如果始终让 IoU 最高的那个赢，另一个 box 永远学不到，最后变成一个 box 在干活。前 10 万次随机选，强制两个 box 都练起来。

### 源码

```c
// src/detection_layer.c:102-138
for(j = 0; j < l.n; ++j){
    // 计算 IoU 或 RMSE
    if(iou > best_iou){ best_iou = iou; best_index = j; }
}
if(1 && *(state.net.seen) < 100000){
    best_index = rand()%l.n;
}
```

### 反例

如果删掉 `rand()`：
- 一个 box 迅速学会预测常见形状，另一个 box 长期不更新
- 两个 box 退化为一个 box
- 模型对“一个格子里两种形状”束手无策

### 演进

v2 用 anchor 替代多 box 竞争，这个 trick 彻底删除。

---

## 4. Region Jitter 数据增强 (`trick-region-jitter`)

### 原理图

```mermaid
flowchart LR
    ORIG["原图 + GT"] --> JITTER["随机 crop/scale ±20%"] --> AUG["增强图"]
    AUG --> GT2["GT 同步偏移"]
    GT2 --> CELL["物体中心相对 grid cell 改变"]
    CELL --> LEARN["模型学会位置鲁棒性"]
```

### 白话

> 每次训练都把图稍微挪一挪、缩一放。物体中心和 grid cell 的相对位置跟着变，模型不能只靠死记硬背“这个 cell 出狗”。

### 源码

```cfg
# cfg/yolo.cfg
jitter=.2
```

```c
// src/image.c / data.c
// random_distort(image, hue, saturation, exposure)
// 以及 crop_image，随机平移缩放
```

### 反例

如果 `jitter=0`：
- 模型记忆绝对位置
- 测试时物体稍微偏一点，中心换 cell，直接漏检

### 演进

v4 引入 **Mosaic** 增强：把 4 张图拼成 1 张，增强幅度远超 jitter，v1 的 region jitter 基本退出历史舞台。

---

## 5. Dropout 0.25/0.5 (`trick-dropout-fc`)

### 原理图

```mermaid
flowchart LR
    FC1["FC 4096<br/>神经元"] --> DROP["Dropout<br/>随机屏蔽部分连接"] --> FC2["FC 1470"]
    FC1 -.-> FC2
```

### 白话

> FC 层参数有 2 亿多，检测数据又少，很容易对某些神经元产生依赖。训练时随机“掐掉”一些连接，模型被迫更泛化。

### 源码

```cfg
# cfg/yolo.cfg
[dropout]
probability=.5
```

```c
// src/dropout_layer.c:forward_dropout_layer
if (!state.train) return;       // 推理时不 dropout
float r = rand_uniform(0, 1);
if (r < l.probability) l.output[i] = 0;
```

### 反例

如果去掉 dropout：
- 在 VOC 这种小数据集上训练，FC 层对训练集过拟合
- 验证集 mAP 下降

### 演进

v2 去掉所有 FC，改用卷积 + BN。BN 本身有正则效果，不再需要显式 dropout。

---

## 6. NMS IoU=0.5 (`trick-nms-iou05`)

### 原理图

```mermaid
flowchart TD
    INPUT["所有 98 个框，按 confidence 排序"] --> TOP["取最高分的框 A"]
    TOP --> COMPARE["把 A 与剩余框算 IoU"]
    COMPARE --> TH{"IoU > 0.5?"}
    TH -->|是| SUP["抑制（删除）低分框"] --> TOP
    TH -->|否| KEEP["保留"] --> TOP
    TOP --> OUTPUT["最终检测结果"]
```

### 白话

> 同一物体可能被多个格子预测出多个框。NMS 的逻辑是：先保留最有信心的框，然后把和它高度重叠（IoU>0.5）的低分框删掉。

### 源码

```c
// src/box.c:do_nms_sort
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

void do_nms_sort(detection *dets, int total, int classes, float thresh){
    // 按类别分别排序、分别计算 IoU、抑制重复框
}
```

### 反例

- **IoU 阈值太高（0.7）**：相邻物体（如人群）容易被误删
- **IoU 阈值太低（0.3）**：同一物体的多个预测框去不干净，误报增多

### 演进

v10 引入 **NMS-free** 检测头，用 one-to-many / end-to-end 分配彻底不再需要 NMS。

---

## 7. ImageNet 预训练 (`trick-pretrained-from-imagenet`)

### 原理图

```mermaid
flowchart LR
    IMAGENET["ImageNet 分类<br/>1000 类<br/>120 万张图"] --> PRE["预训练前 20 conv + 1 FC"]
    PRE --> LOAD["加载权重到 YOLOv1"]
    LOAD --> FINETUNE["在 VOC 检测数据集微调"]
    FINETUNE --> CONV["检测头 + 新增 conv 随机初始化"]
```

### 白话

> 从头训练 YLOLOv1 的 backbone 太慢了。先用 ImageNet 让卷积层学会“什么是边缘、纹理、形状”，再搬到检测任务上只微调 few tens of thousands iterations。

### 源码

```c
// 训练时通过命令行加载预训练权重
// ./darknet detector train cfg/yolo.cfg darknet19_448.conv.23 data/voc.data
```

### 效果

- 无预训练：可能需要 200+ epochs 才收敛
- 有预训练：23k iterations 达到 63.4 mAP

### 演进

这是整个 YOLO 系列乃至整个 CV 界的标准做法。v1-v12 全部沿用。

---

## 7 个 Trick 的谱系图

```mermaid
flowchart TB
    ROOT[YOLOv1 训练/推理技巧]
    LOSS[损失设计] & AUG[数据增强] & REG[正则化] & INF[推理后处理] & PRE[预训练] & STABLE[训练稳定]
    ROOT --> LOSS & AUG & REG & INF & PRE & STABLE

    LOSS --> SQ[sqrt=1]
    LOSS --> WL[权重 5/1/1/0.5]
    AUG --> RJ[Region Jitter]
    REG --> DP[Dropout]
    INF --> NMS[NMS IoU 0.5]
    PRE --> IM[ImageNet 预训练]
    STABLE --> RB[随机 best_index]

    SQ -.v2 废弃.-> V2[v2]
    WL -.v2 保留.-> V2
    RJ -.v4 Mosaic 替代.-> V4[v4]
    DP -.v2 去 FC 消失.-> V2
    NMS -.v10 NMS-free.-> V10[v10]
    IM -.v1-v12 全部沿用.-> ALL[v1-v12]
    RB -.v2 anchor 后删除.-> V2
```

---

## 延伸阅读

- [论文精读](/versions/v1/paper)
- [Architecture 逐层图解](/versions/v1/architecture)
- [理论 Concept](/versions/v1/concepts)
- [损失函数](/versions/v1/losses)
- [源码走读](/versions/v1/code-walkthrough)
- [痛点追踪](/versions/v1/painpoints)
