# YOLOv2 架构与层速查

本页按**层类型**速查 v2 的骨架；逐层 shape、cfg 行号与参数量明细见 [Architecture 逐层图解](/versions/v2/architecture)。

快照 cfg 只有 **5 种层**：`convolutional`（22 个）、`maxpool`（5 个）、`route`（2 个）、`reorg`（1 个）、`region`（1 个）。共 31 层。

> 与 v1 相比，v2 **去掉了 FC / dropout**，新增 `route` / `reorg`（passthrough）与 `region`（替代 `detection`）。所有卷积统一 **BN + leaky**。

## 1. convolutional — 特征提取的全部主力（22 个）

<SrcSnippet file="v2:cfg/yolo.cfg" :start="20" :end="26" />

关键要素：`filters`、`size`、`stride`、`pad`，以及 **`batch_normalize=1`**（v1 没有）。激活统一 `leaky`。

- 通道规律：32 → 64 → 128(64→128) → 256(128→256) → 512(256→512) → 1024(512→1024)。
- **1×1 卷积**大量用于通道压缩（类似 GoogLeNet bottleneck）。
- 检测头最后是 **1×1 → 425**（<SrcLink file="v2:cfg/yolo.cfg" :line="220" />），全卷积，无 FC。

实现：BN 融合在 <SrcLink file="v2:src/convolutional_layer.c" /> 的 `forward_convolutional_layer` 中；解析见 <SrcLink file="v2:src/parser.c" />。

## 2. maxpool — 5 次空间减半

<SrcSnippet file="v2:cfg/yolo.cfg" :start="28" :end="30" />

位置：<SrcLink file="v2:cfg/yolo.cfg" :line="28" />、<SrcLink file="v2:cfg/yolo.cfg" :line="40" />、<SrcLink file="v2:cfg/yolo.cfg" :line="68" />、<SrcLink file="v2:cfg/yolo.cfg" :line="96" />、<SrcLink file="v2:cfg/yolo.cfg" :line="140" />。空间共减半 5 次：416/2⁵ = **13**。实现：<SrcLink file="v2:src/maxpool_layer.c" />。

## 3. route — 特征拼接（passthrough 的起点）

route 层把某几层的输出直接拼接（或取单层）：

<SrcSnippet file="v2:cfg/yolo.cfg" :start="203" :end="210" hl="203,209-210" />

- 层 26 `<route layers=-9>`：取**往回数第 9 层**的输出（即层 17，26×26×512），作为 reorg 的输入。
- 层 28 `<route layers=-1,-3>`：拼接倒数第 1（reorg 结果 13×13×2048）与倒数第 3（层 25，13×13×1024），得到 13×13×3072。

> `layers` 是**相对索引**（负号往回数），不是绝对层号。实现：<SrcLink file="v2:src/route_layer.c" />。

## 4. reorg — 空间重排（无损升通道）

<SrcSnippet file="v2:cfg/yolo.cfg" :start="206" :end="207" />

```
输入: 26×26×512
输出: 13×13×2048   // stride=2，每 2×2 邻域搬进通道
```

- **不改变数据总量**，只是把空间维换成通道维，保住了小目标的细粒度。
- 是 passthrough 链的中间环节：`route(-9) → reorg → route(-1,-3)`。
- 实现：<SrcLink file="v2:src/reorg_layer.c" />。

## 5. region — v2 的检测/损失头（替代 detection）

<SrcSnippet file="v2:cfg/yolo.cfg" :start="227" :end="244" />

实现：<SrcLink file="v2:src/region_layer.c" />。职责边界（容易误会的点）：

1. **前向**：把 1×1 conv 的 425 维按 `coords(4) → obj(1) → classes(80)` 解释；`softmax=1` 时对类头做 softmax，`objectness` 做 sigmoid。
2. **训练 loss/delta**：只在 `state.train` 时执行（<SrcLink file="v2:src/region_layer.c" :line="177" />），四项损失全写在这一个前向函数里。
3. **不负责 NMS**：NMS 在推理入口 <SrcLink file="v2:src/yolo.c" :line="189" /> 调用 <SrcLink file="v2:src/box.c" :line="249" /> 完成，region 层不参与。

头参数语义：`anchors`（5 组 K-means 先验）、`bias_match=1`、`num=5`、`coords=4`、`classes=80`、`softmax=1`、`rescore=1`、`jitter=.2`、四个 `*_scale` 损失权重。

## 6. 输出张量布局

```
13×13 × 5 × 85
       └──────── 每 anchor：tx,ty,tw,th + objectness + 80 类
```

逻辑切分（每 cell 5 组）：

| 段 | 含义 | 激活 |
|----|------|------|
| tx, ty | 网格内偏移 | sigmoid |
| tw, th | 相对 anchor 的对数缩放 | 线性 |
| objectness | P(Object)，每 anchor 独立 | sigmoid |
| 80 类 | 类别概率 | softmax |

## 7. 参数量一览

| 段 | 参数量（约） |
|----|-------------|
| 卷积合计 | ≈ 67 M |
| route / reorg / region | 0（无参数） |
| **合计** | **≈ 67 M** |

对比：v1 约 272 M（其中 FC 占 2 亿+）。v2 去 FC 后锐减。

## 延伸阅读

- [Architecture 逐层图解](/versions/v2/architecture) — 31 行逐层明细表
- [损失函数](/versions/v2/losses) / [反向传播推导](/versions/v2/loss-gradient)
- [训练技巧](/versions/v2/tricks)
- [源码走读](/versions/v2/code-walkthrough)
- [痛点追踪](/versions/v2/painpoints)
