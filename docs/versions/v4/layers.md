# YOLOv4 架构与层速查

本页按**层类型**速查 v4 的骨架；逐层 shape、cfg 行号与参数量明细见 [Architecture 逐层图解](/versions/v4/architecture)。

快照 cfg 有 **6 种层**：`convolutional`（110 个）、`shortcut`（23 个）、`route`（21 个）、`maxpool`（3 个）、`upsample`（2 个）、`yolo`（3 个）。共 162 层。

> **v3 读者路标**：v4 保留了 v3 的 shortcut/route/upsample/yolo，新增 `maxpool`（SPP 用，v3 消失）和**大量 conv**（CSP 分流 + PANet 双向聚合）。骨干激活从 leaky 换成 **mish**。所有卷积统一 **BN + mish（骨干）/ leaky（neck/head）**（末层 1×1→255 是 linear）。

## 1. convolutional — 绝对主力（110 个）

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="30" :end="38" />

关键要素：`filters`、`size`、`stride`、`pad`、`batch_normalize=1`、`activation=mish/leaky/linear`。

- **下采样专用 stride-2 conv ×7**：层 1/11/24/55/86（骨干 5 次，608/2⁵ = 19）+ 层 141/152（PANet 双向，76→38→19）
- **CSP 分流/过渡 1×1 conv**：每段开头分流减半（骨干 64→32、128→64、256→128、512→256、1024→512），concat 后 1×1 conv 过渡恢复
- **残块内 1×1/3×3 交替**：1×1 降维 + 3×3 升回，`shortcut from=-3` 相加——与 v3 同构，但多了 CSP 分流包装
- 检测头末层 **1×1 → 255（linear）**：<SrcLink file="v4:cfg/yolov4.cfg" :line="950" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="1037" /> / <SrcLink file="v4:cfg/yolov4.cfg" :line="1124" />——三个头各自一份

实现：BN 融合在 `src/convolutional_layer.c`；解析见 `src/parser.c`。

## 2. shortcut — 残差相加（23 个）

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="59" :end="61" />

- `from=-3`：把**往回数第 3 层**的输出加到当前层——跳过"1×1 + 3×3"两个卷积。
- 23 个 shortcut = 5 段残块数之和（1+2+8+8+4）——与 v3 **完全相同**（v4 没有改变残块内部的 shortcut 结构，只在外层加了 CSP 分流）
- 实现在 <SrcLink file="v4:src/shortcut_layer.c" />（两输入逐元素相加，activation=linear）

## 3. route — 特征取用/拼接（21 个）

v4 的 route 比 v3 多了 **17 个**（v3 只有 4 个），因为 CSP 分流/拼接占了 10 个，PANet 双向聚合又加了 6 个：

| 用途 | route 语义 | 示例 | 产出 |
|------|-----------|------|------|
| CSP 分流 | `layers=-2` | 层 3/13/26/57/88 | 取回输入特征作为旁路 |
| CSP concat | `layers=-1,-N` | 层 9/22/53/84/103 | 主路 + 旁路拼接 |
| SPP 取中间层 | `layers=-2/-4` | 层 109/111 | 取回原始输入给大核 maxpool |
| SPP concat | `layers=-1,-3,-5,-6` | 层 113 | 4 份拼接 |
| FPN 回连 | `layers=85/54` | 层 119/129 | 绝对层号取骨干半路特征 |
| FPN 拼接 | `layers=-1,-3` | 层 121/131 | 上采样结果 + 回连特征拼接 |
| PANet 拼接 | `layers=-1,-16/-37` | 层 142/153 | 下采样结果 + 中途特征拼接 |

> 规则：**负数是相对索引**（从当前层往回数），**正数是绝对层号**（0-indexed）。v3 首次引入绝对层号，v4 大量使用（FPN 回连 + PANet 拼接）。

实现：<SrcLink file="v4:src/route_layer.c" />（多输入 concat）。

## 4. maxpool — SPP 专用（3 个）

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="763" :end="766" />

- **stride=1 + size=5/9/13**：same padding，特征图尺寸不变（19×19），只扩感受野
- **v3 没有 maxpool 层**（v3 的下采样全用 stride-2 conv），v4 只在 SPP 段用 maxpool
- 3 个 maxpool + 1 个直连 + 3 个 route 取回 → 层 113 concat 成 2048 通道
- 实现：<SrcLink file="v4:src/maxpool_layer.c" />

## 5. upsample — 最近邻上采样（2 个）

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="818" :end="819" />

- `stride=2`：空间放大 2 倍（19→38、38→76），通道不变
- 位置：层 118、128，都在 route 拼接**之前**——把深层语义放大到与浅层同尺寸再 concat
- 与 v3 同构：2 个 upsample、最近邻复制、无参数
- 实现：<SrcLink file="v4:src/upsample_layer.c" />

## 6. yolo — 检测/损失头（3 个）

<SrcSnippet file="v4:cfg/yolov4.cfg" :start="967" :end="982" />

实现：<SrcLink file="v4:src/yolo_layer.c" />。职责边界：

1. **前向激活**：训练前向最开始统一做 logistic——tx/ty 一段 + objectness+80 类一段（<SrcLink file="v4:src/yolo_layer.c" :line="336" :end="339" />）；**scale_x_y 在 logistic 后应用**（L337）。tw/th 保持线性。
2. **训练 loss/delta**：只在 `state.train` 时执行（<SrcLink file="v4:src/yolo_layer.c" :line="346" />），**三遍循环**分配 GT（见 [concepts §7](/versions/v4/concepts#_7-multi-positive-多正样本分配-concept-multi-positive)）。
3. **不负责 NMS**：NMS 仍由推理入口 <SrcLink file="v4:src/yolo.c" /> 调 <SrcLink file="v4:src/box.c" /> 完成。

头参数语义：`mask`（本头认领的 anchor 序号）、`anchors`（18 个 = 9 组共享）、`num=9`、`classes=80`、`jitter=.3`、`ignore_thresh=.7`、`truth_thresh=1`、`scale_x_y=1.2/1.1/1.05`、`iou_thresh=0.213`、`iou_loss=ciou`、`nms_kind=greedynms`、`beta_nms=0.6`。

> ⚠️ **四项 `*_scale` 权重仍不存在**：与 v3 一致，全部硬编码进 `yolo_layer.c`。新增 `iou_normalizer=0.07`、`cls_normalizer=1.0` 控制 CIoU 项和类别项的归一化。

## 7. 输出张量布局（三个头同构）

```
76×76 / 38×38 / 19×19 × 3 anchors × 85
                          └──────── 每 anchor：tx,ty,tw,th + objectness + 80 类
```

逻辑切分（每 anchor 85 维）：

| 段 | 含义 | 激活 |
|----|------|------|
| tx, ty | 格子内偏移 | sigmoid + scale_x_y 缩放平移 |
| tw, th | 相对 anchor 的对数缩放 | 线性 |
| objectness | P(Object)，每 anchor 独立 | sigmoid |
| 80 类 | 类别概率 | 独立 logistic（多标签） |

## 8. 参数量一览

| 段 | 参数量（约） |
|----|-------------|
| 骨干（层 0–104） | ≈ 26.6 M |
| SPP + 过渡（层 105–116） | ≈ 3.8 M |
| FPN + 头A（层 117–138） | ≈ 8.9 M |
| PANet + 头B（层 140–149） | ≈ 5.1 M |
| PANet + 头C（层 151–160） | ≈ 4.5 M |
| shortcut / route / maxpool / upsample / yolo | 0（无参数） |
| **合计** | **≈ 64.1 M** |

对比：v3 ≈ 62 M、v2 ≈ 67 M、v1 ≈ 272 M。CSP 分流让骨干参数略增（多了分流/过渡 conv），但被 SPP + PANet 吃掉——总量与 v3 持平。

## 延伸阅读

- [Architecture 逐层图解](/versions/v4/architecture) — 162 层逐段明细表
- [损失函数（CIoU 全套）](/versions/v4/losses)
- [训练技巧](/versions/v4/tricks)
- [反向传播推导](/versions/v4/loss-gradient)
- [源码走读](/versions/v4/code-walkthrough)
- [痛点追踪](/versions/v4/painpoints)
