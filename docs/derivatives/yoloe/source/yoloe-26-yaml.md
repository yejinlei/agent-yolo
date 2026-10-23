# ultralytics/cfg/models/26/yoloe-26.yaml

YOLOE-26 检测 cfg：**53 行，YOLO26 骨干 + YOLOEDetect 头**。这是 v8.4.0 首发的新一代配置，与 v8 版最大的区别有三处：**引入 `end2end: True`**（端到端推理模式，Detect 的 `one2one_cv2/3` 深拷贝分支激活，见 [head-py L62-L64](./head-py#L62)）、**`reg_max: 1`**（DFL bins 从 16 压到 1，即 `self.dfl = Identity` 直接输出 4 维框，不再做 4×16 期望分布解码）、**`text_model: mobileclip2:b`**（文本编码器从 mobileclip:b 升级到 mobileclip2:b）。整体结果：n 档参数 3.7M → 2.6M、GFLOPs 19.5 → 6.6，速度快 3× 且精度不降。

> 快照路径：`source/yoloe/snapshot/ultralytics/cfg/models/26/yoloe-26.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloe:ultralytics/cfg/models/26/yoloe-26.yaml" />

**阅读路线**：

- [L9-L11 三项新参数](#L9-L11)——YOLO26 相对 YOLOv8 的三个顶层开关：
  - `end2end: True` → Detect 内置 one2one/one2many 双分支（train 时两路计算损失、infer 时只走 one2one）
  - `reg_max: 1` → dfl 退化为 Identity，不再输出 4×16 分布，直接输出 4 维 box
  - `text_model: mobileclip2:b` → 文本编码器升级
- [L12-18 scales](#L12-L18)：n 档 `[0.50, 0.25, 1024]`——比 v8 的 `[0.33, 0.25, 1024]` 加深 0.5 倍
  - [L14 n](#L14)：181 层、2.6M 参数、6.6 GFLOPs（相比 v8n 参数 -30%、GFLOPs -66%）
  - [L15 s](#L15)：181 层、9.5M 参数、21.7 GFLOPs
  - [L16 m](#L16)：231 层、20.1M 参数、68.5 GFLOPs
  - [L17 l](#L17)：357 层、25.4M 参数、87.6 GFLOPs
  - [L18 x](#L18)：357 层、57.0M 参数、196.0 GFLOPs
- 骨干 [L21-33](#L21-L33)：**从 C2f 全面切到 C3k2**——
  - [L25 C3k2(256, False, 0.25)](#L25)：`0.25` 是 `e`（膨胀/压缩比，C3k2 分支通道 × 0.25）、`False` 关 shortcut
  - [L27 C3k2(512, False, 0.25)](#L27)：同 P3 尺度
  - [L29 L31 C3k2(512/1024, True)](#L29)：P4/P5 开 shortcut
  - [L32 SPPF(1024, 5, 3, True)](#L32)：第三个 `3` 是 maxpool 池化参数、`True` 表示启用 C3k2 分支
  - [L33 C2PSA(1024)](#L33)——**YOLO26 独有**：Channel + Position Attention 双向注意力模块，层号 10
- Head [L36-51](#L36-L51)：C3k2 全栈
  - [L39 C3k2(512, True)](#L39)：P4 neck
  - [L43 C3k2(256, True)](#L43)：P3 neck（小目标尺度）
  - [L47 C3k2(512, True)](#L47)：P4 neck 下行
  - [L51 C3k2(1024, True, 0.5, True)](#L51)：P5 neck 大目标尺度，`True` 结尾表示启用 2×2 kernel 分支
- [L53 YOLOEDetect](#L53)：`[nc, 512, True]`——与 v8 版参数完全一致，说明**头侧接口稳定**，骨干换代不影响头 API

## 与 yoloe-v8.yaml 的对比

| 维度 | yoloe-v8.yaml | yoloe-26.yaml |
|------|---------------|---------------|
| 骨干模块 | C2f | C3k2（含 0.25/0.5 的 e 参数） |
| 骨干收尾 | SPPF(1024, 5) | SPPF(1024, 5, 3, True) + **C2PSA**（层 10） |
| 端到端 | 无此字段 | `end2end: True` → Detect 内置 one2one/one2many 双分支 |
| reg_max | 隐式 16 | `reg_max: 1` → dfl 退化为 Identity |
| text_model | 隐式 mobileclip:blt | `mobileclip2:b` |
| n 档参数 | 3.7M / 19.5 GFLOPs | 2.6M / 6.6 GFLOPs（**快 3×**） |
| 层数 | 148 层 | 181 层（C3k2 更细粒度） |
| 检测头参数 | `[nc, 512, True]` | `[nc, 512, True]`（不变） |
| 头内 reprta | Residual(SwiGLUFFN) | Residual(SwiGLUFFN)（不变） |
| 头内 BNContrastive | 开 | 开 |

## 三处变化如何加速

1. **reg_max=1**：原 Detect 的 cv2 输出 `4 * 16 = 64` 通道，改为输出 `4 * 1 = 4` 通道——cv2 卷积层计算量降为 1/16，`DFL` 模块退化为 Identity 无需计算
2. **C3k2 替代 C2f**：C3k2 引入 `e` 参数控制分支压缩比，比 C2f 更细粒度地调节计算量
3. **C2PSA 增加感受野**：注意力模块让 P5 特征更好捕捉全局，间接减少小目标尺度对大目标的负担

## 关键陷阱 / 常见误解

1. **`reg_max: 1` 不等于"无回归"**——DFL 退化为 Identity，但仍是 anchor-free 直接回归 cx/cy/w/h；只是少了分布学习那一步。
2. **`end2end: True` 强制 one2one/one2many 双分支**——训练时走 one2many（更严格匹配），推理时走 one2one（更快），二者权重共享但激活路径不同。
3. **`text_model: mobileclip2:b`**——v26 默认用更强的 CLIP 变体；v8 是 mobileclip:blt，两者 text_embeds 不兼容。
4. **C3k2 的 `e` 参数**——控制分支压缩比，改 e 会改骨干通道数，需与 yaml 里其它层协调。
5. **不能套 v8 权重**——C3k2/C2PSA 结构不同，checkpoint 无法直接加载。

## 延伸阅读

- 分割版见 [yoloe-26-seg.yaml](./yoloe-26-seg-yaml)
- YOLO26 骨干细节见 [v26 源码入口](/versions/v26/architecture)
- YOLOE 头实现见 [head-py](./head-py)
- YOLOEModel 建模器见 [tasks-py](./tasks-py)
- 架构对比见 [../architecture](../architecture)
- 痛点分析见 [../painpoints](../painpoints)
- 代码走读见 [../code-walkthrough](../code-walkthrough)
- 训练技巧见 [../tricks](../tricks)
- 概念体系见 [../concepts](../concepts)
