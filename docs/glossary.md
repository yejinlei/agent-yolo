# 术语表

YOLO 系列文档共用的"人话词典"。**每个术语只给一两句白话**，看懂就够；想深挖，点后面的链接进对应页面。按主题分五组：检测基础 → 网络结构 → 训练与损失 → darknet 源码黑话 → YOLO 专属。

---

## 检测基础

### 目标检测（object detection）
在图里找出"有什么、在哪"，输出一堆框 + 类别。分类只回答"是什么"，检测还要回答"在哪里"。

### bbox（边界框）
一个矩形，用中心坐标 + 宽高 `(x, y, w, h)` 表示，圈住一个物体。

### IoU（交并比）
两个框的"交集面积 ÷ 并集面积"，0~1 之间。衡量框得准不准的尺子：>0.5 通常算"命中"。详解：[v1 IoU 概念](/versions/v1/concepts)、[v2 box.c 实现](/versions/v2/source/box#L94)。

### 置信度 / objectness
"这个框里真有物体吗"的打分。YOLO 里它同时承担"框得准不准"（训练目标是 IoU），见 [rescore](#rescore)。

### NMS（非极大值抑制）
同一物体被报了好几次时，只留分最高的框，把和它高度重叠（IoU 超阈值）的其余框压掉。详解：[v1 NMS 概念](/versions/v1/concepts)、[v2 box.c do_nms_sort](/versions/v2/source/box#L249)。

### mAP（mean Average Precision）
检测界的"总分"：每个类算一条 精确率-召回率 曲线下面积，再对所有类取平均。`mAP@0.5` 表示 IoU 阈值 0.5 下的成绩。

### 召回率（recall）
图里真有 10 只猫，你找到了几只。漏检多 = 召回低。小物体召回差是 v1/v2 共同的痛点，见 [痛点追踪](/versions/v2/painpoints)。

---

## 网络结构

### backbone（骨干网络）
检测器里负责"看图提炼特征"的前半段（一堆卷积），后半段才是检测头。v1 用 24 层 conv，v2 换成 Darknet-19。

### grid / cell（网格 / 格子）
YOLO 把输出特征图划成格子（v1 是 7×7，v2 是 13×13），**每个格子负责报出中心落在自己范围内的物体**。这是 YOLO 的核心分工方式。

### anchor（锚框 / 先验框）
v2 引入：给每个格子预备 5 个"典型形状的框"（用 K-means 从训练集聚出来），网络只预测"相对这个模板的偏移"，不用从零猜形状。详解：[v2 anchor 概念](/versions/v2/concepts)。

### K-means（K 均值聚类）
一种"自动分组"算法。v2 用它把所有真实框的宽高分成 5 组，组中心就是 anchor。换数据集要重跑。

### 感受野（receptive field）
网络某一层的一个点，"往回看"能覆盖原图多大区域。下采样越多感受野越大、看得越"全局"，但小物体细节丢得越多。

### 1×1 卷积
kernel 只有 1×1 的卷积，不看空间、只在通道维做加权和。常用于压缩通道数（v2 的检测头最后就是 1×1 → 425）。

### FC（全连接层）
每个输入连每个输出的层，参数量巨大。v1 末尾有两层（吃掉了 2 亿参数）；v2 全部去掉，改全卷积。

### maxpool（最大池化）
把 2×2 区域只留最大值，特征图宽高减半。v2 用它做了 5 次空间减半（416→13）。

### BN（批归一化 / Batch Normalization）
每层卷积后把数据"拉回均值 0、方差 1"再加一对可学习缩放参数。梯度更稳、收敛更快。v2 起成为标配，详解：[v2 BN 概念](/versions/v2/concepts)。

### passthrough / route / reorg
v2 的"细粒度特征搬运术"：`route` 取回前面高分辨率层的输出，`reorg` 把 26×26×512 **无损重排**成 13×13×2048（空间维搬进通道维），再和深层拼接。让小目标特征不被下采样吞掉。详解：[v2 架构 §3.3](/versions/v2/architecture)。

### FPN（特征金字塔）
在**多个尺度**的特征图上各自出检测头（小目标交给高分辨率头）。v2 的 passthrough 是它的雏形，v3 才真正实现。

### shortcut（残差连接）
把某一层的输入**直接加回**若干层后的输出（`output = F(x) + x`），梯度可以"抄近路"回传，深层网络不再难训。v3 的 Darknet-53 首次引入（23 个 shortcut），详解：[v3 concepts §1](/versions/v3/concepts)。

### Darknet-53
v3 的骨干网络：52 层卷积 + 23 个 shortcut 残差连接，ImageNet top-1 77.2% 与 ResNet-101 持平但快一倍。详解：[v3 架构](/versions/v3/architecture)。

### CSP（跨阶段部分连接）
Cross Stage Partial：每段把特征劈成两半，一半走残块堆栈、一半旁路直通，末尾 concat 汇合。梯度分流不重复，计算量降约 20%、精度反升。v4 骨干 CSPDarknet53 的核心，详解：[v4 concepts §1](/versions/v4/concepts)。

### Mish 激活
`x · tanh(softplus(x))`，平滑版 ReLU——负半轴允许小梯度通过，训练更稳。v4 骨干全部换成 Mish（neck/head 仍用 leaky），详解：[v4 concepts §2](/versions/v4/concepts)。

### SPP（空间金字塔池化）
Spatial Pyramid Pooling：同一特征图并联 5×5/9×9/13×13 三个 same-padding maxpool 再 concat，感受野暴力扩大而分辨率不变。v4 插在骨干与 neck 之间（层 105–116），详解：[v4 concepts §3](/versions/v4/concepts)。

### PANet（路径聚合网络）
在 FPN"深层语义往浅层送"的基础上，再加一条**自底向上**的下采样通路，把浅层定位信息回传深层。v4 的 neck = SPP + FPN + PANet，详解：[v4 concepts §4](/versions/v4/concepts)。

---

## 训练与损失

### 损失函数（loss）
"预测离真值有多远"的一个数，训练就是不断调参数把它变小。YOLO 用平方差（L2）家族，详解：[v1](/versions/v1/losses) / [v2](/versions/v2/losses)。

### 梯度 / 反向传播（backprop）
损失对每个参数的"修改建议"（导数），从最后一层一路乘链式法则传回第一层。逐行推导：[v2 反向传播](/versions/v2/loss-gradient)。

### L2 损失（平方差）
`(预测 − 目标)²`。差越大惩罚越狠（平方放大）。YOLOv1/v2 的全部损失项都是它。

### softmax
把一组数变成"和为 1 的概率"，且**互斥**（总和固定，你高我就低）。v1 没用，v2 用在类别上，v3 又改回 sigmoid。

### sigmoid / logistic
把任意数压到 0~1 的 S 形曲线。v2 用它压坐标偏移和 objectness。`logistic` 就是 sigmoid 的别名（源码里的函数名）。

### one-hot
"正确答案"的写法：80 类里第 3 类对，就是 80 维向量里只有第 3 位是 1、其余全 0。

### 学习率 / lr / decay
每次参数更新的"步长"。太大会震荡，太小会爬不动。训练越深步长要调小（decay 衰减）。

### warmup（热身）
训练刚开始时用小步长或简化规则"预热"，防止一开始梯度乱飞把网络带崩。v1 前 100k 张随机选 box，v2 前 12800 张让 anchor 学占位。

### 数据增强 / jitter
训练时随机对图做裁剪、缩放、翻转、变色，让模型见多识广、不背图。`jitter=.2` 就是几何扰动 ±20%。

### 预训练 / 微调（pretrain / finetune）
先在 ImageNet 分类任务上把 backbone 练好，再搬到检测任务上接着练（微调）。几乎所有检测器都这么做。

### logistic 激活
见 [sigmoid / logistic](#sigmoid-logistic)；v3 起类别头每类独立过 logistic（多标签场景下 Person + Woman 可同时成立），与互斥 softmax 分道扬镳，详解：[v3 concepts §4](/versions/v3/concepts)。

### CIoU loss
Complete IoU loss：坐标回归不再算 `(target − output)²`，而是直接对 IoU 求导——同时考虑**重叠面积 + 中心距 + 长宽比**。v4 坐标损失换代，详解：[v4 concepts §5](/versions/v4/concepts)。

### Mosaic 增强
训练时把 4 张图拼成一张（随机切十字），单张图里的场景更丰富、物体数量翻倍，相当于暴力版数据增强。v4 论文核心 BoF 之一，详解：[v4 concepts §8](/versions/v4/concepts)。

### scale_x_y
消除网格敏感性：对 sigmoid 后的偏移做线性放缩 `scale_x_y·σ(tx) − 0.5·(scale_x_y−1)`，让小物体在网格边缘也能被分到足够的梯度。v4 三头分别取 1.2/1.1/1.05，详解：[v4 concepts §6](/versions/v4/concepts)。

### multi-positive（多正样本）
除"形状最匹配的 anchor"外，同一头内 IoU(anchor shape, GT) > 0.213 的其余 anchor 也参与监督，再平均 delta——正样本密度显著提升。v4 新增，详解：[v4 concepts §7](/versions/v4/concepts)。

---

## darknet 源码黑话

> 读快照源码时高频出现的"方言"，集中在这里。

### cfg
darknet 的网络配置文件，一段 `[convolutional]`、`[maxpool]` 块拼出整个网络。v1 见 [yolo.cfg](/versions/v1/source/yolo-cfg)，v2 见 [yolo.cfg](/versions/v2/source/yolo-cfg)。

### delta
源码里的 `l.delta[]`：**损失对每层输出的梯度的相反数**，反向传播的"起跑信号"。约定详解：[v2 反向传播 §0](/versions/v2/loss-gradient)。

### axpy
BLAS 库函数 `y += a·x`（累加向量）。darknet 的 backward 大多只是一行 axpy：把本层 delta 累加给上一层。

### side
v1 的网格边长（`side=7`）。v2 改用 `w/h`（13×13），但 v2 的 `yolo.c` 里还残留 `l.side` 调用——见 [test_yolo bug](/versions/v2/painpoints)。

### rescore
v1/v2 都有的开关：开启后，objectness 的训练目标从硬编码 1 改成**预测框与真值的实时 IoU**，让置信度同时反映定位质量。

### bias_match
v2 开关：分配真值给 anchor 时，**只看 anchor 的形状**（宽高）与真值的 IoU，不看网络当前预测的框。避免预测干扰分配。

### DOABS
v2 快照里 `region_layer.c` 的宏（`#define DOABS 1`）：把框的宽高多除一次网格尺寸做归一。数学自洽但极易误读，属早期坐标系 hack，见 [v2 痛点 §6](/versions/v2/painpoints)。

### coord / object / noobject / class scale
损失四项的权重（v1: 5/1/0.5/1，v2: 5/1/1/1）。coord 乘 5 是因为坐标项数值天然小，不放大就被其他项淹没。

---

## YOLO 专属

### P(Object) × IoU
v1 对置信度的定义：框里有物体的概率 × 框与真值的重合度。一个分数同时说"有没有"和"准不准"。

### √w√h 回归
v1 的宽高回归在**平方根空间**进行：大框小框的误差量级被拉平，小物体不被大物体"按在地上摩擦"。v2 起被 log 空间取代。

### log 空间回归
v2 的宽高回归方式：网络输出 `tw`，解码时 `宽 = e^tw × anchor宽`。输出的是"相对 anchor 的对数缩放"。详解：[v2 log 回归概念](/versions/v2/concepts)。

### 多尺度训练
全卷积网络不挑输入尺寸，训练时随机换分辨率（320~608），让模型对各种大小的物体都鲁棒。v2 引入。

### WordTree / YOLO9000
v2 论文的另一半：把 ImageNet 和 WordNet 的类别组织成一棵树，联合训练出能认 **9000+ 类** 的检测器。工程复杂、后续版本没沿用，见 [v2 痛点 §7](/versions/v2/painpoints)。

### Darknet-19
v2 的 backbone 名字：19 层卷积 + BN，因去掉 FC 得名。也是 v2 预训练权重文件名（`darknet19_448.conv.23`）的来源。

---

## 还有不懂的？

- 先从 [v1 概览](/versions/v1/index) 的"3 分钟电梯演讲"读起，按版本顺序走
- 遇到源码行号引用（如 `region_layer.c:168`），点一下就能跳到对应文件的高亮行
- 全站支持搜索（页面顶部搜索框），术语、函数名都能搜
