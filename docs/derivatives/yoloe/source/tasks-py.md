# ultralytics/nn/tasks.py（YOLOEModel 部分）

`YOLOEModel` 继承 `DetectionModel`，是 `YOLOEDetect` 的容器类；`YOLOESegModel` 再继承 `YOLOEModel + SegmentationModel`，处理分割任务。本文件还提供 `YOLOEWrapper`，负责把训练集/推理输入包装成 head 需要的 `(img, txt_embeds)` 对。

Source: <SourceViewer file="yoloe:ultralytics/nn/tasks.py" />

**⚠️ 术语纠偏**：`YOLOEDetect.reprta`（见 [head-py L464](./head-py#L464)）实现是 `Residual(SwiGLUFFN(embed, embed))`——**是 Residual + SwiGLU FFN，不是 RealNVP、不是 Normalizing Flow**。本文件里的 `without_reprta` 参数正是控制这条 Residual(SwiGLU) 变换是否参与，与 flow 无关。

## `DetectionModel` 基类的 `set_classes`
[#L702](#L702)

- 基类实现：把 `text → embeddings`（走 CLIP），缓存到 model 上，之后 predict 时直接查表。
- `YOLOEModel.set_classes` 会**覆盖**这个基类方法。

## `YOLOEModel.__init__`
[#L789-L803](#L789-L803)

- 复用 `DetectionModel` 的初始化。
- 记录 `task` 字段；当 task 包含 `detect` / `segment` 时，允许走 open-set 分支。
- 保留 `names`（闭集类别名）与 `embeddings`（闭集类别嵌入）两个属性。

## `YOLOEModel.get_text_pe`
[#L805-L838](#L805-L838)

- **入参**：`text` 字符串（逗号分隔）或 `text_embeds` 张量；`batch=80`；`cache_clip_model=True`；`without_reprta=False`。
- **流程**：
  1. `tokenize` → `token_to_id`：把词切成 CLIP token id。
  2. 查 `text_embeds` 缓存表；命中则跳过 CLIP 推理。
  3. [#L830 `if without_reprta`](#L830)：`True` 时**直接返回原始 CLIP 文本嵌入**，不做 RepRTA 变换；`False` 时走 `head.reprta`（`Residual(SwiGLUFFN)`）做表征变换。
- **返回**：`(H, D)` 的 text prompt embedding，H 是 prompt 数。

## `YOLOEModel.get_visual_pe`
[#L839-L851](#L839-L851)

- 视觉 prompt 模式入口。
- 输入 `img`（prompt 图）和 `visual`（已保存的 embedding 路径或 CLIP 特征）。
- 走 CLIP image encoder 得到视觉特征；如提供预计算 embedding，则直接加载。

## `YOLOEModel.set_vocab`
[#L852 起](#L852)

- 一次性把闭集词表写入 model——后续每次 predict 免 token 化。
- 内部调用 `get_text_pe(names)`（[#L896](#L896)）得到 embeddings，再调 `set_classes(names, embeddings)`（[#L897](#L897)）写入。

## `YOLOEModel.set_classes`
[#L907-L1008](#L907-L1008)

- 覆盖基类：不仅缓存 `embeddings`，还**触发 fuse**——把 `BN + Conv + 文本嵌入` 折叠成单层 Conv，推理时省掉 BN 与 matmul。
- 详见 [../painpoints](../painpoints)：fuse 后 `BNContrastiveHead` 的 `bn` + `cv3` + `text_embeds` 变成单矩阵，速度提升约 30%。

## `YOLOESegModel`
[#L1009 起](#L1009)

- `class YOLOESegModel(YOLOEModel, SegmentationModel)`。
- 复用 `get_text_pe` / `get_visual_pe` / `set_vocab` / `set_classes`。
- 增加 mask 头 `YOLOESeg`（见 [head-py](./head-py)）的 mask prediction 分支。

## `YOLOEWrapper`
[#L1040 起](#L1040)

- 训练/验证阶段的包装器：把 `txt_embeds` 注入 forward 输入。
- `forward(img, txt_embeds)`：把两个输入分别送入 backbone 与 head。
- 训练时 `txt_embeds` 冻结；推理时可为 `None`（Prompt-Free 模式）。

## YOLOEModel 五组 API 速查

| API | 定义位置 | 用途 |
| --- | --- | --- |
| `get_text_pe(text, batch, cache_clip_model, without_reprta)` | [#L805](#L805) | 文本 → embedding，可选 RepRTA 变换 |
| `get_visual_pe(img, visual)` | [#L839](#L839) | 视觉 prompt → embedding |
| `set_vocab(vocab, names)` | [#L852](#L852) | 一次性写入闭集词表 |
| `set_classes(names, embeddings)` | [#L907](#L907) | 更新类别名 + 触发 fuse |
| `forward(img, txt_embeds)` | `DetectionModel` 基类 | 三 prompt 模式统一前向 |

## 阅读路线

1. `YOLOEModel.__init__`（[#L789](#L789)）：容器类基础。
2. `get_text_pe`（[#L805](#L805)）：**重点看 `without_reprta` 分支**（[#L830](#L830)），理解 RepRTA 如何被旁路。
3. `get_visual_pe`（[#L839](#L839)）：视觉 prompt 路径。
4. `set_vocab` → `set_classes`（[#L852](#L852) / [#L907](#L907)）：闭集词表写入 + fuse 折叠。
5. `YOLOESegModel`（[#L1009](#L1009)）：分割扩展。
6. `YOLOEWrapper`（[#L1040](#L1040)）：训练侧输入包装。

## 与 YOLOv8 `DetectionModel` 的对比

| 维度 | YOLOv8 | YOLOE |
| --- | --- | --- |
| `set_classes` | 仅缓存 embeddings | 额外触发 fuse 折叠 |
| `get_text_pe` | 无 | 有，支持 `without_reprta` 开关 |
| `set_vocab` | 无 | 有，一次性词表注入 |
| `forward` 签名 | `(img)` | `(img, txt_embeds)` |
| 容器基类 | `DetectionModel` | `YOLOEModel` |

## 延伸阅读

- [../architecture](../architecture)：`YOLOEModel` 在整个链路中的位置
- [../concepts](../concepts)：`without_reprta` 参数的动机
- [../losses](../losses)：`BNContrastiveHead` 训练目标
- [../painpoints](../painpoints)：fuse 折叠的实现细节
- [../tricks](../tricks)：`set_vocab` 的性能收益
