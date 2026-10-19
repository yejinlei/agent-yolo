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
Spatial Pyramid Pooling：同一特征图并联 5×5/9×9/13×13 三个 same-padding maxpool 再 concat，感受野暴力扩大而分辨率不变。v4 插在骨干与 neck 之间（层 105–116），v5 沿用同一公式但实现改成 `nn.MaxPool2d`，详解：[v4 concepts §3](/versions/v4/concepts) / [v5 concepts §3](/versions/v5/concepts)。

### PANet（路径聚合网络）
在 FPN"深层语义往浅层送"的基础上，再加一条**自底向上**的下采样通路，把浅层定位信息回传深层。v4 的 neck = SPP + FPN + PANet（手写 route/shortcut）；v5 改用 yaml `Concat` 声明，更易改。详解：[v4 concepts §4](/versions/v4/concepts) / [v5 concepts §4](/versions/v5/concepts)。

---

### RepVGG 结构重参数化
训练时用"3×3 卷积 + 1×1 卷积 + 恒等映射"三条并联分支求和堆容量，部署前用数学等价变换把三条分支**融成一个 3×3 卷积**——训练更猛、推理不慢。v6 全线采用，详解：[v6 RepVGG 概念](/versions/v6/concepts)。

### RepBlock / BottleRep
v6 的重复堆叠单元：BottleRep 是带 alpha 加权残差的双卷积块，RepBlock 把 n 个 RepVGGBlock（或 BottleRep）串起来。训练多分支、推理单路。

### BepC3 / CSPStackRep
v6 给中大模型准备的 CSP 版 Rep 块：三个 1×1 卷积 + 一串双卷积残差子块 + 跨阶段拼接。大模型再用纯单路 RepBlock 参数会爆炸，论文里 M/L 就换这个。

### SimSPPF / CSPSPPF
v6 把 v5 的 SPPF 再简化：串 3 个 MaxPool 后 concat，激活次数减半；CSPSPPF 是加 CSP 分支的版本，v6 骨干尾部使用。

### BiFusion
v6 Neck 的三路特征融合模块：当前尺度 + 上采样来的高级特征 + 下采样来的低级特征，三路 concat 再卷积。RepBiFPANNeck 的核心积木。

### fuse_P2
v6 骨干的可选开关：把 160×160 的 P2 高分辨率特征也输出给 Neck（v5 只出 P3/P4/P5），给小目标特征多一层保险；yolov6s 默认开启。

### EffiDeHead
v6 的高效解耦头：分类/回归各一条轻量分支、**没有 objectness 分支**、anchor-free 直接回归 ltrb 距离。蒸馏续训时加 proj_conv 做 DFL 解码。

### ELAN / E-ELAN（高效层聚合）
ELAN 把两条 1×1 旁路和多条 3×3 深链**每隔一个取一个** concat 到一起，梯度有短路可抄、不押注在一条深链上；E-ELAN（Extended ELAN）再加 expand/shuffle/merge（分组卷积 + 通道洗牌）增加路径多样性，是 v7 训练更大网络的骨架。快照里的落地形态就是 yaml 中的 4 路/6 路 Concat 块，详解：[v7 concepts §2](/versions/v7/concepts)。

### RepConv / RepConvN（计划重参数化）
v7 的训练期三分支卷积（3×3 + 1×1 + 恒等映射），部署前融成单个 3×3；RepConvN 是"条件恒等"版——只在 `c2==c1 且 stride==1` 时才保留 identity 分支，残差/Concat 邻接的 conv 一律禁用 identity，这就是论文 planned re-param 的规则。详解：[v7 concepts](/versions/v7/concepts)、[RepConv 源码](/versions/v7/source/common-py)。

### compound scaling（复合缩放）
放大 concat 型网络时，computational block 只缩 depth、transition 层按同样变化量缩 width——比 v4/v5 的 depth/width 各缩各的更匹配多路 concat 结构。v7 消融：51.7 → 52.9 AP，详解：[v7 concepts §3](/versions/v7/concepts)。

### partial auxiliary head（部分辅助头）
v7 的辅助监督头只接在**部分层**上（不逐层全接），且 aux 分支不带隐式知识、损失统一 ×0.25——只在训练期存在，推理时直接丢弃。详解：[v7 concepts](/versions/v7/concepts)、[IAuxDetect 源码](/versions/v7/source/yolo-py)。

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
Complete IoU loss：坐标回归不再算 `(target − output)²`，而是直接对 IoU 求导——同时考虑**重叠面积 + 中心距 + 长宽比**。v4 坐标损失换代，v5 沿用同一公式但用 PyTorch autograd 自动求导（消除了 v4 手写梯度的 bug 风险）。详解：[v4 concepts §5](/versions/v4/concepts) / [v5 losses](/versions/v5/losses)。

### Mosaic 增强
训练时把 4 张图拼成一张（随机切十字），单张图里的场景更丰富、物体数量翻倍，相当于暴力版数据增强。v4 论文核心 BoF 之一，v5 沿用并在 `load_mosaic` 里实现为 NumPy 切片，详解：[v4 concepts §8](/versions/v4/concepts) / [v5 tricks §5](/versions/v5/tricks)。

### scale_x_y
消除网格敏感性：对 sigmoid 后的偏移做线性放缩 `scale_x_y·σ(tx) − 0.5·(scale_x_y−1)`，让小物体在网格边缘也能被分到足够的梯度。v4 三头分别取 1.2/1.1/1.05，详解：[v4 concepts §6](/versions/v4/concepts)。

### multi-positive（多正样本）
除"形状最匹配的 anchor"外，同一头内 IoU(anchor shape, GT) > 0.213 的其余 anchor 也参与监督，再平均 delta——正样本密度显著提升。v4 新增，详解：[v4 concepts §7](/versions/v4/concepts)。v5 改成 wh 比例 + 邻居偏移，思路一致但匹配规则不同，见 [v5 concepts §8](/versions/v5/concepts)。

### Focus
v5 的 stem 模块：把输入按 `(行, 列)` 各切 2 份得 4 个空间位置，concat 到通道维（3→12），再接一个 Conv——无损下采样，参数量比 stride-2 conv 低，但部署端不友好。v6.0 起被 6×6 Conv s2 替代，详解：[v5 concepts §1](/versions/v5/concepts)。

### C3
v5 的 CSP Bottleneck 模块：cv1 + cv2 两路分流（cv1 走 n 个 Bottleneck，cv2 直通），concat 后过 cv3——比 v4 的 BottleneckCSP 少一次 BN/激活，参数量更省，详解：[v5 concepts §2](/versions/v5/concepts)。

### SiLU（Swish）
`x · sigmoid(x)`，平滑非单调激活。v5 全程启用（替代 v4 的 Mish/Leaky），详解：[v5 concepts §4](/versions/v5/concepts)。

### autoanchor
v5 的 anchor 自动学习：先 k-means 按训练集 wh 聚类 9 个 anchor，再用遗传算法（GA）进化 1000 代；训练前若 BPR < 0.98 就自动重算并替换进模型——anchor 从"手工调参"变成"全自动"，详解：[v5 concepts §9](/versions/v5/concepts)。

### EMA（指数移动平均）
训练时同时维护一份"平滑版"参数（`ema = decay × ema + (1−decay) × model`），eval/推理用这份平滑参数——比裸模型泛化更稳。v5 标配，v4 无。

### AMP（自动混合精度）
`torch.cuda.amp` 自动选择 fp16/fp32：matmul/conv 用 fp16 加速，loss/softmax 用 fp32 保精度——训练速度 ×2-3，显存减半。v5 标配。

### yaml 驱动架构
v5 的网络配置不再用 cfg 逐层手写，而是 `models/yolov5s.yaml` 用 `[from, number, module, args]` 四元组 + `depth_multiple`/`width_multiple` 两个缩放系数描述整个网络——s/m/l/x 同一份代码只改两个数。详解：[v5 architecture](/versions/v5/architecture)。

---

### SimOTA
YOLOX 提出的动态标签分配：按"分类 cost + IoU cost"动态决定每个 GT 分几个正样本。YOLOv6 初版用它，技术报告评测后被 TAL 替代。

### TAL（Task Alignment Learning）
源自 TOOD 的任务对齐分配：对齐度量 = 分类分数^α × IoU^β，每个 GT 取 top-k（v6 是 13）当正样本，并用归一化度量当**分类软标签**——分类准的还必须定位准。详解：[v6 损失 §4](/versions/v6/losses)。

### ATSS
自适应训练样本选择：按 GT 与各层 anchor 中心的距离统计定阈值、再按 IoU 挑正样本。v6 保留它作可选 warmup 分配器（默认关闭）。

### VarifocalLoss（VFL）
非对称分类损失：负样本按预测分降权（p^γ），正样本用 IoU 软标签当目标且不降权。v6 用它取代 BCE，从此不再需要 objectness 分支。

### DFL（Distribution Focal Loss）
把"框边到网格点的距离"建模成 n 个离散 bin 上的分布，损失只打在真值两侧最近的 bin 上（线性加权交叉熵）。v6 里 reg_max=16，主要服务于**框回归自蒸馏**——标准训练默认不开。

### Python dict 配置
v6 把 v5 的 yaml 换成可执行 Python（model / solver / data_aug 三段 dict）：import 即校验、能写注释和逻辑，代价是失去语言中立性。

### trainable bag-of-freebies（可训练免费午餐）
v7 的方法论：所有涨点手段（E-ELAN、compound scaling、planned re-param、双头标签分配、EMA……）都只花**训练期**成本，部署模型与基线同速——"免费涨点"。详解：[v7 论文精读](/versions/v7/paper)、[v7 训练技巧](/versions/v7/tricks)。

### lead head guided label assigner（引导头标签分配）
v7 的标签分配器：先用 dynamic-k（cost = 分类 + 3·(−log IoU)）给 lead head 挑好正样本，再把这份分配结果**引导**给 aux head——两个头看同一批 GT，但粗细粒度不同。详解：[v7 损失 §4](/versions/v7/losses)。

### coarse-to-fine 标签分配（coarse-for-auxiliary / fine-for-lead）
同一个 GT，lead 头用**细粒度**正样本（邻域 3 点采样、半径 g=0.5），aux 头用**粗粒度**正样本（5 点采样、g=1.0）：粗样本帮网络"先看清大概"，细样本再精修定位。详解：[v7 损失](/versions/v7/losses)、[find_5_positive 源码](/versions/v7/source/loss-py)。

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
