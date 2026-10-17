# YOLOv1 架构与层速查

本页按**层类型**速查 v1 的骨架；逐层 shape、cfg 行号与参数量明细见 [Architecture 逐层图解](/versions/v1/architecture)。

<DiagramArch />

快照 cfg 只有 **5 种层**：`convolutional`（24 个）、`maxpool`（4 个）、`connected`（2 个）、`dropout`（1 个）、`detection`（1 个）。

> 勘误：早期资料（含本页旧版）称 v1 有一个 `[local]` 层。快照 <SrcLink file="cfg/yolo.cfg" /> 中**不存在任何 `[local]` 段**，`local_layer.c` 虽在源码树里但未被 yolo.cfg 使用。

## 1. convolutional — 特征提取的全部主力

<SrcSnippet file="cfg/yolo.cfg" :start="24" :end="29" />

四要素：`filters`（输出通道）、`size`（kernel）、`stride`、`pad`；激活统一为 `leaky`（斜率 0.1，实现见 <SrcLink file="src/activations.c" />）。注意快照 cfg **没有 `batch_normalize`**——v1 卷积不带 BN（BN 是 v2 的标志性改进，解析逻辑在 <SrcLink file="src/parser.c" />）。

两种下采样卷积：首层 7×7/s2（448→224，<SrcLink file="cfg/yolo.cfg" :line="24" />）与 head 前的 3×3/s2（14→7，<SrcLink file="cfg/yolo.cfg" :line="189" />）。1×1 卷积在 @28 段大量用于通道压缩（256↔512 交替）。

## 2. maxpool — 4 次空间减半

<SrcSnippet file="cfg/yolo.cfg" :start="31" :end="33" />

位置：<SrcLink file="cfg/yolo.cfg" :line="31" />、<SrcLink file="cfg/yolo.cfg" :line="42" />、<SrcLink file="cfg/yolo.cfg" :line="74" />、<SrcLink file="cfg/yolo.cfg" :line="148" />。加上 2 次 stride=2 卷积，空间共减半 6 次：448/2⁶ = 7。实现：<SrcLink file="src/maxpool_layer.c" />。

## 3. connected — 参数巨怪

两层全连接接在 7×7×1024 特征之后：

<SrcSnippet file="cfg/yolo.cfg" :start="209" :end="218" hl="210,214,217" />

- FC 4096：50,176 → 4096，**205,524,992 个参数**（占全网约 76%）
- FC 1470：4096 → 1470，约 6.02 M 参数
- 两层之间 dropout=0.5（快照值；论文写 0.25）
- 实现：<SrcLink file="src/connected_layer.c" />，dropout：<SrcLink file="src/dropout_layer.c" />（非训练态直接透传）

1470 的正确分解：**49 × (2×5 + 20) = 49 × 30 = 1470**——两个 box 各 5 维、20 类两框共享。

## 4. detection — 前向即损失的杂合检测头

<SrcSnippet file="cfg/yolo.cfg" :start="221" :end="234" hl="224-228,231-234" />

实现：<SrcLink file="src/detection_layer.c" />。职责边界（容易误会的点）：

1. **前向**：把 FC 1470 按 `classes(20) → conf×2 → coords×8` 的内存顺序解释；`softmax=0` 时**不做任何激活**（没有 sigmoid/反 sigmoid）。
2. **训练 loss/delta**：只在 `state.train` 时执行（<SrcLink file="src/detection_layer.c" :line="60" />），四项加权 L2 全写在这一个前向函数里。
3. **不负责 NMS**：NMS 在推理入口 <SrcLink file="src/yolo.c" :line="346" /> 调用 <SrcLink file="src/box.c" :line="249" /> 完成，detection_layer 不参与。

头参数语义：`side=7, num=2`（49 格 × 2 框 = 98 候选）、`coords=4`、`sqrt=1`（宽高 √ 空间回归）、`rescore=1`（conf 目标取 IoU）、四个 `*_scale` 是损失权重。

## 5. 输出张量布局

<DiagramTensor30 />

## 6. 参数量一览（修正口径）

| 段 | 参数量（约） |
|----|-------------|
| 卷积合计 | 60 M |
| FC 4096 | 205.5 M |
| FC 1470 | 6.0 M |
| **合计** | **≈ 272 M** |

旧版资料流传的「v1 约 70M、FC4096 约 47M」与快照结构对不上（50,176×4096 本身就超过 2 亿），本页以 cfg 实际结构计算为准。

## 延伸阅读

- [Architecture 逐层图解](/versions/v1/architecture) — 32 行逐层明细表
- [损失函数](/versions/v1/losses) / [反向传播推导](/versions/v1/loss-gradient)
- [训练技巧](/versions/v1/tricks)
- [源码走读](/versions/v1/code-walkthrough)
- [痛点追踪](/versions/v1/painpoints)
