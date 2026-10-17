# YOLOv3 架构与层速查

本页按**层类型**速查 v3 的骨架；逐层 shape、cfg 行号与参数量明细见 [Architecture 逐层图解](/versions/v3/architecture)。

快照 cfg 只有 **5 种层**：`convolutional`（75 个）、`shortcut`（23 个）、`route`（4 个）、`upsample`（2 个）、`yolo`（3 个）。共 107 层。

> **v2 读者路标**：v2 的 `maxpool`（5 个）和 `reorg`（1 个）在 v3 里**消失了**——下采样全靠 stride-2 卷积，上采样全靠 `upsample`；新增 `shortcut`（残差相加）。检测头从 `region` 改名 `yolo`。所有卷积统一 **BN + leaky**（末层 1×1→255 是 linear）。

## 1. convolutional — 主力（75 个）

<SrcSnippet file="v3:cfg/yolov3.cfg" :start="25" :end="33" />

关键要素：`filters`、`size`、`stride`、`pad`、`batch_normalize=1`、`activation=leaky`。

- **下采样专用 stride-2 conv ×5**：层 1/5/12/37/62（kernel 全是 3×3），416/2⁵ = 13。v2 的 maxpool 全被它们替代。
- **残块内 1×1/3×3 交替**：1×1 降维一半、3×3 升回，`shortcut from=-3` 相加。
- 检测头末层 **1×1 → 255（linear）**：<SrcLink file="v3:cfg/yolov3.cfg" :line="599" /> / <SrcLink file="v3:cfg/yolov3.cfg" :line="685" /> / <SrcLink file="v3:cfg/yolov3.cfg" :line="772" />——三个头各自一份。

实现：BN 融合在 <SrcLink file="v3:src/convolutional_layer.c" /> 的 `forward_convolutional_layer` 中；解析见 <SrcLink file="v3:src/parser.c" />。

## 2. shortcut — 残差相加（23 个，v3 新增）

<SrcSnippet file="v3:cfg/yolov3.cfg" :start="455" :end="457" />

- `from=-3`：把**往回数第 3 层**的输出加到当前层——正好跳过"1×1 + 3×3"两个卷积。
- 23 个 shortcut = 5 段残块数之和（1+2+8+8+4）；两个输入通道必须一致（残块设计的约束）。
- 这是 v2 快照里没有的层类型，实现在 <SrcLink file="v3:src/shortcut_layer.c" />（两输入逐元素相加，activation=linear）。

## 3. route — 特征取用/拼接（4 个）

v3 的 route 出现了**相对索引和绝对层号混用**，这是初读 cfg 最容易踩的坑：

<SrcSnippet file="v3:cfg/yolov3.cfg" :start="618" :end="619" />

| 层号 | cfg | 语义 | 产出 |
|------|-----|------|------|
| 83 | `layers = -4` | 相对：往回数 4 层（=层 79） | 13×13×512 |
| 86 | `layers = -1, 61` | 相对 -1（上采样结果）+ **绝对 61**（段4 末） | 26×26×768 |
| 95 | `layers = -4` | 相对：=层 91 | 26×26×256 |
| 98 | `layers = -1, 36` | 相对 -1 + **绝对 36**（段3 末） | 52×52×384 |

> 规则：**负数是相对索引**（从当前层往回数），**正数是绝对层号**（0-indexed）。v2 只用过负数，v3 首次用绝对层号跨头回连。实现：<SrcLink file="v3:src/route_layer.c" />（多输入 concat）。

## 4. upsample — 最近邻上采样（2 个，v3 新增）

<SrcSnippet file="v3:cfg/yolov3.cfg" :start="629" :end="630" />

- `stride=2`：空间放大 2 倍（13→26、26→52），通道不变。
- 位置：层 85、97，都在 route 拼接**之前**——把深层语义放大到与浅层同尺寸再 concat。
- 实现：<SrcLink file="v3:src/upsample_layer.c" />（简单最近邻复制，无参数）。

与 v2 reorg 的方向对比：reorg 把浅层**压小**送进单头；upsample 把深层**放大**送回浅层——v2 是"把细节搬上来"，v3 是"把语义搬下去"。

## 5. yolo — 检测/损失头（3 个，替代 region）

<SrcSnippet file="v3:cfg/yolov3.cfg" :start="607" :end="615" />

实现：<SrcLink file="v3:src/yolo_layer.c" />。职责边界（与 v2 region 一致的误会点）：

1. **前向激活**：训练前向最开始统一做 logistic——tx/ty 一段、objectness+80 类一段（<SrcLink file="v3:src/yolo_layer.c" :line="141" :end="143" />）；tw/th 保持线性。
2. **训练 loss/delta**：只在 `net.train` 时执行（<SrcLink file="v3:src/yolo_layer.c" :line="149" />），两遍循环分配 GT（见 [concepts §7](/versions/v3/concepts#_7-gt-分配两遍循环-ignore-thresh-concept-gt-assignment)）。
3. **不负责 NMS**：NMS 仍由推理入口 <SrcLink file="v3:examples/yolo.c" /> 调 <SrcLink file="v3:src/box.c" /> 完成。

头参数语义：`mask`（本头认领的 anchor 序号）、`anchors`（9 组共享）、`num=9`、`classes=80`、`jitter=.3`、`ignore_thresh=.5`、`truth_thresh=1`（实际不触发，见下）、`random=1`。

> ⚠️ **四个 `*_scale` 权重没了**：v2 cfg 的 `coord_scale/object_scale/noobject_scale/class_scale` 在 v3 的 [yolo] 段不存在，权重硬编码进 yolo_layer.c（坐标权重还是动态的 `2−w·h`，见 [losses 页](/versions/v3/losses)）。

## 6. 输出张量布局（三个头同构）

```
13×13 / 26×26 / 52×52 × 3 anchors × 85
              └──────── 每 anchor：tx,ty,tw,th + objectness + 80 类
```

逻辑切分（每 anchor 85 维）：

| 段 | 含义 | 激活 |
|----|------|------|
| tx, ty | 格子内偏移 | sigmoid |
| tw, th | 相对 anchor 的对数缩放 | 线性 |
| objectness | P(Object)，每 anchor 独立 | sigmoid |
| 80 类 | 类别概率 | **独立 logistic（多标签）** |

## 7. 参数量一览

| 段 | 参数量（约） |
|----|-------------|
| 骨干（层 0–74） | ≈ 40.5 M |
| 检测头 A（层 75-81） | ≈ 16.1 M |
| 检测头 B（层 87-93） | ≈ 3.1 M |
| 检测头 C（层 99-105） | ≈ 1.1 M |
| shortcut / route / upsample / yolo | 0（无参数） |
| **合计** | **≈ 61.8 M** |

对比：v2 ≈ 67 M、v1 ≈ 272 M。层数翻 3 倍、参数反而略降——残差结构把"深度"换成了"效率"。

## 延伸阅读

- [Architecture 逐层图解](/versions/v3/architecture) — 107 层逐段明细表
- [损失函数](/versions/v3/losses) / [反向传播推导](/versions/v3/loss-gradient)
- [训练技巧](/versions/v3/tricks)
- [源码走读](/versions/v3/code-walkthrough)
- [痛点追踪](/versions/v3/painpoints)
