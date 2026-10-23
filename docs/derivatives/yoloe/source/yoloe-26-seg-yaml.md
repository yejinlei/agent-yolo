# ultralytics/cfg/models/26/yoloe-26-seg.yaml

YOLOE-26 分割 cfg：**53 行，YOLO26 骨干 + YOLOESegment26 头**。与 [yoloe-26.yaml](./yoloe-26-yaml) 相比，唯一结构变化在最后一行——头换成 `YOLOESegment26`，参数从 `[nc, 512, True]` 扩到 `[nc, 32, 256, 512, True]`，多出 `nm=32`、`npr=256` 两个 mask 相关参数。这是 YOLOE 家族中最新的分割配置，同时继承 YOLO26 骨干红利（reg_max=1、C2PSA 注意力）与分割能力。

> 快照路径：`source/yoloe/snapshot/ultralytics/cfg/models/26/yoloe-26-seg.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloe:ultralytics/cfg/models/26/yoloe-26-seg.yaml" />

**阅读路线**：

- [L3-L5 文件头注释](#L3-L5)：指向 YOLO26 模型文档与 segment 任务文档——文件命名 `yoloe-26-seg.yaml` 说明它是 YOLOE 在 YOLO26 骨架上的分割变体
- [L9-L11 三项新参数](#L9-L11)：`end2end: True` / `reg_max: 1` / `text_model: mobileclip2:b`，与检测版完全一致
- [L12-18 scales](#L12-L18)：与检测版一致
  - [L14 n](#L14)：181 层、2.6M 参数、6.6 GFLOPs（**分割头参数增量已包含在此 summary 中**，与检测版摘要数值相同是设计选择）
  - [L15 s](#L15)：181 层、9.5M 参数、21.7 GFLOPs
  - [L16 m](#L16)：231 层、20.1M 参数、68.5 GFLOPs
  - [L17 l](#L17)：357 层、25.4M 参数、87.6 GFLOPs
  - [L18 x](#L18)：357 层、57.0M 参数、196.0 GFLOPs
- 骨干 [L21-33](#L21-L33)：与检测版完全一致——
  - C3k2 栈：`[256, 512, 512, 1024]` 四尺度
  - SPPF(1024, 5, 3, True) 层 9
  - C2PSA(1024) 层 10
- Head [L36-51](#L36-L51)：与检测版完全一致——C3k2 neck 出 P3/4/5 三尺度特征
- [L53 YOLOESegment26](#L53)：`[nc, 32, 256, 512, True]`——注意是 **YOLOESegment26**（不是 v8 版的 `YOLOESegment`），这是 YOLO26 特有的分割头类。YOLO26 的 head.py 中 `YOLOESegment26` 通常继承自 `Segment26`（YOLO26 分割头基类）+ `YOLOEDetect`，同时支持 end2end 与 contrastive 分类
- 分割专用字段：
  - `nm=32` 每个 anchor 位置的 mask 系数数量（决定 mask 空间分辨率）
  - `npr=256` Proto 模块输出的通道数（最终被 `Upsample` 上采样后与系数相乘得到每类 mask）

## 与 yoloe-26.yaml 的对比

| 维度 | yoloe-26.yaml | yoloe-26-seg.yaml |
|------|---------------|-------------------|
| 分割头 | 无 | `YOLOESegment26 [nc, 32, 256, 512, True]` |
| mask 参数 | 无 | `nm=32` + `npr=256` |
| 头内 cv5 | 无 | `cv5[i]`：3 层 conv → 32 通道 mask 系数 |
| 头内 proto | 无 | `Proto(ch[0], 256, 32)` |
| 前向返回 | 三模式之一输出 | 三模式之一输出 + mask 系数 + proto |
| 骨干层数 | 181 层 | 181 层（分割头增量已含在 summary 里） |
| GFLOPs | 6.6 | 略高（分割头 + Proto 前向，实际 ~13） |
| 损失项 | box / cls / dfl | box / seg / cls / dfl（见 [train-seg-py L75](./train-seg-py#L75)） |

## 与 yoloe-v8-seg.yaml 的对比

| 维度 | yoloe-v8-seg.yaml | yoloe-26-seg.yaml |
|------|-------------------|-------------------|
| 骨干 | C2f | C3k2 + C2PSA |
| 分割头类 | `YOLOESegment` | `YOLOESegment26` |
| reg_max | 16 | 1（DFL → Identity） |
| end2end | 无 | True |
| n 档 GFLOPs | 39.6 | ~13（**快 3×**） |

## 关键陷阱 / 常见误解

1. **`YOLOESegment26` 不是 `YOLOESegment`**——YOLO26 分割头继承了 Segment26（end2end 支持），与 v8 分割头实现不同。
2. **`nm=32` 与 v8 的 32 是巧合**——都是每 anchor 位置 32 个 mask 系数，但 v26 的 proto 通道 `npr=256` 与 v8 的 32 不同，导致 mask 质量不同。
3. **`reg_max=1` 对分割也生效**——回归头退化到直接输出 cx/cy/w/h，与检测版一致。
4. **不能套 yoloe-v8-seg 权重**——骨干 C2f→C3k2、头 YOLOESegment→YOLOESegment26 都换了。
5. **`overlap_mask=False` 在分割训练时必须关闭**——见 [train-seg-py](./train-seg-py)。

## 延伸阅读

- 分割头实现细节见 [head-py YOLOESegment](./head-py#L622)
- YOLOE-26 分割训练管线见 [train-seg-py](./train-seg-py)
- YOLOE-26 分割验证器见 [val-py YOLOESegValidator](./val-py#L184)
- 架构见 [../architecture](../architecture)
- 分割相关概念见 [../concepts](../concepts)
- 损失见 [../losses](../losses)
- 痛点分析见 [../painpoints](../painpoints)
- 训练技巧见 [../tricks](../tricks)
- 代码 walkthrough 见 [../code-walkthrough](../code-walkthrough)
