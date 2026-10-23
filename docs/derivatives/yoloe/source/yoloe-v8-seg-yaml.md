# ultralytics/cfg/models/v8/yoloe-v8-seg.yaml

YOLOE v8 版分割 cfg：**45 行、C2f 骨干 + YOLOESegment 收尾**。检测头的最后一个 `nn.Conv2d(c3, self.nc, 1)` 换成了 `nn.Conv2d(c3, embed, 1)`——分类头不再输出固定类数，而是投影到 512 维嵌入空间，与文本 embedding 做对比学习；同时新增 `cv5` 分支产出 32 维 mask 系数、`Proto` 模块产出 256 维 mask prototype。相比检测版，参数增量约 +500K（n 档 148 → 161 层，19.5 → 39.6 GFLOPs）。

> 快照路径：`source/yoloe/snapshot/ultralytics/cfg/models/v8/yoloe-v8-seg.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloe:ultralytics/cfg/models/v8/yoloe-v8-seg.yaml" />

**阅读路线**：

- [L4 nc=80](#L4)——与检测版一致，「单图文本上限」语义
- [L5-11 scales](#L5-L11)：与检测版参数结构一致，但每档层数更多（分割头多 cv5 分支 + Proto）
  - [L7 n](#L7)：161 层、4.2M 参数、39.6 GFLOPs（相比检测 n 档 +13 层、+509K 参数）
  - [L8 s](#L8)：161 层、13.4M 参数、71.5 GFLOPs
  - [L9 m](#L9)：201 层、29.1M 参数、131.4 GFLOPs
  - [L10 l](#L10)：241 层、47.6M 参数、225.6 GFLOPs
  - [L11 x](#L11)：241 层、73.7M 参数、330.8 GFLOPs
- 骨干 [L14-25](#L14-L25)：完全复用 v8 C2f 栈，层号 0-9
- Head [L28-43](#L28-L43)：与检测版一致的 FPN/PAN，输出 P3/4/5 三尺度特征
- [L45 YOLOESegment](#L45)：`[nc, 32, 256, 512, True]`——
  - `nc` 类数（同上，单图文本上限）
  - `nm=32` 个 mask 系数（每个 anchor 位置的 mask 系数数量，决定 mask 空间分辨率）
  - `npr=256` 个 proto 通道（`Proto` 模块输出特征图的通道数，最终上采样后与系数相乘得到每类 mask）
  - `embed=512` 文本/视觉嵌入维度
  - `True` 开 BNContrastiveHead
- 分割专用字段解读：
  - `nm=32`：mask 系数数量，与 anchor 位置数无关，是每个位置共享的 mask 字典维度
  - `npr=256`：Proto 输出通道数，决定 mask 语义分辨率
  - 分割头实现见 [head-py L622 YOLOESegment](./head-py#L622)

## 与 yoloe-v8.yaml 的对比

| 维度 | yoloe-v8.yaml | yoloe-v8-seg.yaml |
|------|---------------|-------------------|
| 检测头 | `YOLOEDetect [nc, 512, True]` | `YOLOESegment [nc, 32, 256, 512, True]` |
| mask 相关参数 | 无 | `nm=32` + `npr=256` |
| 头内 cv5 | 无 | `cv5[i]`：3 层 conv → 32 通道（mask 系数） |
| 头内 proto | 无 | `Proto(ch[0], 256, 32)` |
| 前向返回 | `(x, mc, p)` 训练态：仅检测输出 | `(x, mc, p)` 训练态：检测输出 + mask 系数 + proto |
| 骨干层数（n 档） | 148 层 | 161 层（+13 层 mask 分支） |
| GFLOPs（n 档） | 19.5 | 39.6（近 2×） |
| 损失项 | box / cls / dfl | box / seg / cls / dfl（见 [train-seg-py L75](./train-seg-py#L75)） |
| 数据要求 | COCO 检测数据 | LVIS / COCO 实例分割数据 |

## 分割头前向流程

```
x (P3, P4, P5 特征)
├── proto = Proto(x[0])  # 256 通道 P3 特征图
├── mc = concat(cv5[i](x[i]) for i in nl)  # mask 系数
├── x = YOLOEDetect.forward(x, text)  # 检测头输出（三模式之一）
└── 返回 (x, mc, proto)
```

## 关键陷阱 / 常见误解

1. **`overlap_mask=False` 是必需的**——开放集下 mask 重叠消解会破坏 prompt-to-mask 对齐，见 [train-seg-py](./train-seg-py)。
2. **`task: segment` 而非 `task: detection`**——这是 cfg 头里的关键字段，与 [yoloe-v8-yaml](./yoloe-v8-yaml) 的差异。
3. **proto 输出 256 通道**——比 v8 的 32 通道多得多，因为开放集 mask 需要更强的原型空间。
4. **mask 系数 `mc` 与检测头共享**——`mc = concat(cv5[i](x[i]) for i in nl)` 复用 P3/P4/P5 三尺度特征。
5. **与 yolov8-seg.yaml 的差异**：head 从 `Segment` 换成 `YOLOESegment`，其余骨干/neck 完全一致。

## 延伸阅读

- 分割头实现细节见 [head-py YOLOESegment](./head-py#L622)
- YOLOE-26 分割版见 [yoloe-26-seg-yaml](./yoloe-26-seg-yaml)
- 分割训练管线见 [train-seg-py](./train-seg-py)
- 分割验证器见 [val-py YOLOESegValidator](./val-py#L184)
- 架构总览见 [../architecture](../architecture)
- 损失公式见 [../losses](../losses)
- 概念体系见 [../concepts](../concepts)
- 痛点见 [../painpoints](../painpoints)
- 代码 walkthrough 见 [../code-walkthrough](../code-walkthrough)
