# ultralytics/nn/head.py

`YOLOEDetect` 是 YOLOE 的核心 head，把标准 `Detect` 扩展成**同时支持文本 prompt、视觉 prompt、Prompt-Free 三条路径**。同文件还提供 `YOLOESeg`（分割 head）和 `DetectionBlock` / `ContrastiveBlock` 两个基础块。

Source: <SourceViewer file="yoloe:ultralytics/nn/head.py" />

## `YOLOEDetect.__init__`
[#L330-L465](#L330-L465)

- [#L355 起](#L355)：初始化 `nm`（每个 scale 的 anchor 数）、`reg_max`、`no`（输出通道）。YOLOE 的 `no = nc + 4 * reg_max`，但 `nc` 由 prompt 决定，`__init__` 时置 0，`forward` 时按 `txt_embeds.shape[0]` 动态算。
- [#L408](#L408)：`self.cv4 = nn.ModuleList(...)` 三个 `cv3` 分支，负责 cv3（cls）+ cv4（reg）双头；每个 `cv3` 是 `CV3` 块，末端 BN 已就位。
- [#L410](#L410)：`self.bn = nn.BatchNorm1d(...)`——**BNContrastiveHead 的核心**。文本 embedding 经过 BN 后再与视觉特征做点积，得到对比 logits：`logit_scale * (f @ t^T) / ||t||`。
- [#L460](#L460)：`self.text_project = RepRTA(embed, embed)` 是文本嵌入的特征变换；见下方术语纠偏。
- [#L464](#L464)：**`self.reprta = Residual(SwiGLUFFN(embed, embed))`**——这就是 RepRTA 的真实实现。
- [#L504](#L504)：`self.prompt_free = LRPCHead(...)`——Prompt-Free 推理专用子 head。

**⚠️ 术语纠偏**：RepRTA **不是** RealNVP，也**不是** Normalizing Flow。RepRTA 的实现是 [L464 `Residual(SwiGLUFFN(embed, embed))`](#L464)——一层 `SwiGLU` 前馈网络 + 恒等残差连接，作用是对文本嵌入做一次非线性"表征变换"。名字里的 "R" 是 **R**epresentation（表征）或 **R**esidual（残差），与 V（Vortex）无关，与 RealNVP 那个 V 更没有关系。切勿把 RepRTA 讲成 flow-based 生成模型。

## `YOLOEDetect.preprocess`（文本/视觉 prompt 预处理）
[#L475-L520](#L475-L520)

- [#L478](#L478)：`text_project` / `reprta` 只作用于文本 embedding——把 CLIP text token 拉进视觉特征的同维度空间。
- [#L511-L512](#L511-L512)：Prompt-Free 模式下 `del self.reprta; self.reprta = nn.Identity()`——彻底跳过文本变换。
- [#L517](#L517)：`F.normalize(self.reprta(tpe), dim=-1, p=2)`——对变换后的文本嵌入做 L2 归一化，保证后续点积只反映方向、不反映模长。

## `YOLOEDetect.loss` / `forward`
[#L530-L654](#L530-L654)

- `forward`：`txt_embeds` 为空 → 走 `LRPCHead`（Prompt-Free）；非空 → 走 `BNContrastiveHead` 分支。
- 视觉 prompt 模式额外触发 `spatial_aware_visual_prompt_embed`（SAVPE），把 prompt 的空间分布也送入 head。
- `loss`：对比损失 `InfoNCE`，正例是 `txt_embeds[i]` 与匹配 GT 类别的视觉特征，负例是 batch 内其他类别。

## `ContrastiveBlock` / `DetectionBlock`
[#L588-L650](#L588-L650)

- [#L588 `SwiGLUFFN`](#L588)：SiLU + GLU 门控 FFN，`x = SiLU(gate(x)) * up(x)`。
- [#L605 `Residual`](#L605)：`y = y + f(x)`，恒等残差。
- 两个模块组合起来就是 RepRTA 的实体——**Residual(SwiGLUFFN)**，没有其它花活。

## YOLOE 特有细节一览

| 细节 | 位置 | 作用 |
| --- | --- | --- |
| `BNContrastiveHead` | [#L410](#L410) | BN 归一化 text / visual，再点积 |
| `logit_scale` | [#L413](#L413) | 温度参数，可学习 |
| RepRTA = `Residual(SwiGLUFFN)` | [#L464](#L464) | 文本嵌入的非线性变换 |
| SAVPE | [#L470](#L470) | 视觉 prompt 的空间感知嵌入 |
| `LRPCHead` | [#L504](#L504) | Prompt-Free 推理专用 |
| `spatial_aware_visual_prompt_embed` | [#L495](#L495) | 视觉 prompt 前处理 |

## 阅读路线

1. `YOLOEDetect.__init__`（[#L330-L465](#L330-L465)）：看清所有子模块的构造顺序。
2. RepRTA 定义（[#L464](#L464)）：**再次确认它是 `Residual(SwiGLUFFN)`，不是 flow**。
3. `preprocess`（[#L475-L520](#L475-L520)）：文本 / 视觉 prompt 的预处理路径。
4. `forward`（[#L530-L654](#L530-L654)）：三 prompt 模式的分发。
5. `SwiGLUFFN` / `Residual`（[#L588](#L588) / [#L605](#L605)）：RepRTA 的构件。

## 与 YOLOv8 `Detect` 的对比

| 维度 | YOLOv8 `Detect` | YOLOE `YOLOEDetect` |
| --- | --- | --- |
| 类别数 | 静态 `nc` | 动态 `txt_embeds.shape[0]` |
| cls head | `cv3` | `cv3` + `BNContrastiveHead` |
| prompt 支持 | 无 | text / visual / prompt-free 三模式 |
| 损失 | BCE + L1 + DFL | 追加 InfoNCE 对比损失 |

## 延伸阅读

- [../architecture](../architecture)：head 在整体链路中的位置
- [../concepts](../concepts)：BNContrastiveHead 与 RepRTA 的概念解释
- [../losses](../losses)：对比损失 InfoNCE 的推导
- [../painpoints](../painpoints)：RepRTA 为什么不是 flow
- [../tricks](../tricks)：`logit_scale` 的温度调度
