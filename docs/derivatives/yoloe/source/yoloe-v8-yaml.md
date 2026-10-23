# ultralytics/cfg/models/v8/yoloe-v8.yaml

YOLOE v8 版检测 cfg：**45 行、C2f 骨干 + FPN/PAN 融合 + YOLOEDetect 收尾**。整份文件与 [yolov8.yaml](/versions/v8/source/yolov8-yaml) 逐行相似，唯一实质差异在最后一行：`Detect` → `YOLOEDetect`，参数从 `[nc]` 扩成 `[nc, 512, True]`——512 是文本/视觉 embedding 的维度，`True` 强开 BNContrastiveHead（Contrastive Head 的 BN 版本，见 [head-py L462](./head-py#L462)）。骨干与 neck 完全复用 v8 结构，把「开放词表检测」的复杂性完全下沉到检测头里，是「头换、骨不换」的代表性配置。

> 快照路径：`source/yoloe/snapshot/ultralytics/cfg/models/v8/yoloe-v8.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloe:ultralytics/cfg/models/v8/yoloe-v8.yaml" />

**阅读路线**：

- [L4 nc=80](#L4)——**YOLOE 的 nc 语义特殊**：不是数据集类别数，而是「单图文本上限」，见 [train-py L42-L47](./train-py#L42)；训练时若数据集 `data.nc > 80`，会被截断为 80
- [L5-11 scales 复合缩放](#L5-L11)：五档 n/s/m/l/x，`[depth, width, max_channels]` 三元组——n/s 通道上限 1024、m 768、l/x 512；每档带参数/GFLOPs 摘要
  - [L7 n](#L7)：148 层、3.7M 参数、19.5 GFLOPs
  - [L8 s](#L8)：148 层、12.8M 参数、51.0 GFLOPs
  - [L9 m](#L9)：188 层、28.4M 参数、110.5 GFLOPs
  - [L10 l](#L10)：228 层、46.8M 参数、204.5 GFLOPs
  - [L11 x](#L11)：228 层、72.9M 参数、309.3 GFLOPs
- 骨干 [L14-25](#L14-L25)：Conv s2 ×2 → C2f(128) → P3/8 → C2f(256) 重复 6 → P4/16 → C2f(512) 重复 6 → P5/32 → C2f(1024) → [L25 SPPF(1024, 5)](#L25)
  - C2f 的 `True` 参数表示开 shortcut，与 v8 完全一致
  - 层号 0-9 与 v8 对齐，便于对照
- Head [L28-45](#L28-L45)：
  - FPN 上行：[L29 Upsample](#L29) + [L30 Concat 骨干 P4](#L30) → C2f(512) → [L33-35 降到 P3/8](#L33-L35) 出小目标特征
  - PAN 下行：[L37 Conv s2](#L37) + [L38 Concat head P4](#L38) → C2f(512) → [L41-43 降到 P5/32](#L41-L43) 出大目标特征
  - head 内的 C2f 不传 `True`，shortcut 默认 False（与骨干不同）
- [L45 YOLOEDetect](#L45)：`[nc, 512, True]`——
  - 第一个 512 是 head 内 embedding 维度（cv3 最后一层 Conv2d 输出到 512 而非 nc）
  - 第二个 True 强开 BNContrastiveHead（对比头）
  - 三尺度 from 索引 `15, 18, 21` 与 v8 完全一致
- **anchor-free 结构证据**：yaml 通篇找不到 `anchors` 字段，`reg_max=16` 由 [head-py L41](./head-py#L41) 内建，DFL 输出 4×16=64 维

## 与 yolov8.yaml 的对比

| 维度 | yolov8.yaml | yoloe-v8.yaml |
|------|-------------|---------------|
| 行数 | 46 行 | 45 行 |
| backbone | 层 0-9（C2f×4 + SPPF） | 层 0-9（同上，完全一致） |
| neck | 三尺度 C2f | 三尺度 C2f（同上） |
| 检测头 | `Detect [nc]` | `YOLOEDetect [nc, 512, True]` |
| 头内组件 | cv2 + cv3（cv3 输出到 nc） | cv2 + cv3（输出到 embed=512）+ cv4 + reprta + savpe |
| anchor | 无 | 无 |
| 分类机制 | 直接 softmax 到 nc 类 | Contrastive：`logit_scale · (f · t)/‖t‖`，nc 变成动态（一张图内文本数） |
| 训练数据 | 标准 YOLO 检测数据 | + LVIS 多模态数据（可选） |
| 推理输入 | 图像 | 图像 + 文本/视觉提示 |

## 三个 yaml 的家族关系

| cfg | 骨干 | 头 | 用途 |
|-----|------|-----|------|
| [yolov8.yaml](/versions/v8/source/yolov8-yaml) | v8 C2f | `Detect` | 原始 v8 检测 |
| **yoloe-v8.yaml**（本页） | v8 C2f | `YOLOEDetect` | YOLOE 检测（v8 骨架） |
| [yoloe-v8-seg.yaml](./yoloe-v8-seg-yaml) | v8 C2f | `YOLOESegment` | YOLOE 分割（v8 骨架） |
| [yoloe-26.yaml](./yoloe-26-yaml) | v26 C3k2 + C2PSA | `YOLOEDetect` | YOLOE 检测（v26 骨架） |
| [yoloe-26-seg.yaml](./yoloe-26-seg-yaml) | v26 C3k2 + C2PSA | `YOLOESegment26` | YOLOE 分割（v26 骨架） |

## 关键陷阱 / 常见误解

1. **`nc=80` 不是数据集类别数**——它是「单图文本上限」；训练时若 `data.nc > 80`，会被截断为 80。见 [train-py L42-L47](./train-py#L42)。
2. **head 参数 `[nc, 512, True]`**——第二个 512 是 embedding 维度，第三个 True 强开 BNContrastiveHead；切勿与骨干里的 512 混淆。
3. **头内 C2f 不传 `True`**——head 内的 C2f shortcut 默认 False，与骨干不同，这是 v8 系列的统一约定。
4. **anchor-free**——yaml 通篇没有 `anchors` 字段，`reg_max=16` 由 head 内建。
5. **与 v26 骨架不兼容**——本 cfg 使用 v8 C2f 骨干，不能直接套 v26 的 C3k2/C2PSA 权重。

## 延伸阅读

- YOLOE 头实现细节见 [head-py YOLOEDetect](./head-py#L438)
- YOLOEModel 建模器见 [tasks-py L789](./tasks-py#L789)
- 训练管线见 [train-py](./train-py)
- YOLO 入口 `YOLOE` 类见 [model-py](./model-py)
- 架构总览见 [../architecture](../architecture)
- 概念体系见 [../concepts](../concepts)
- 损失见 [../losses](../losses)
- 训练技巧见 [../tricks](../tricks)
- 痛点分析见 [../painpoints](../painpoints)
- 代码 walkthrough 见 [../code-walkthrough](../code-walkthrough)
