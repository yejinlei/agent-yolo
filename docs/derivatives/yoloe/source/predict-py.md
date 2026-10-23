# ultralytics/models/yolo/yoloe/predict.py

**YOLOE 推理管线**：`YOLOEVPDetectPredictor` 是唯一非平凡类——`DetectionPredictor` 的子类，把视觉提示的三件套（`set_prompts` / `pre_transform` / `inference`）接到原生的预处理-前向-后处理流水线上。`YOLOEVPSegPredictor` 只是 `YOLOEVPDetectPredictor + SegmentationPredictor` 的空壳。文件 170 行，覆盖了三 prompt 模式中最复杂的 Visual Prompt 模式。

Source: <SourceViewer file="yoloe:ultralytics/models/yolo/yoloe/predict.py" />

## `YOLOEVPDetectPredictor.setup_model`
[#L32-L41](#L32)

- [#L41 `done_warmup=True`](#L41)：跳过 warmup——视觉提示模式不需要热身，因为每张图的 prompt 都不同。
- 其余走 `DetectionPredictor.setup_model`：加载权重、选 device、编译。

## `YOLOEVPDetectPredictor.set_prompts`
[#L43-L51](#L43)

- 把外部传入的 `prompts` 字典挂到 `self.prompts`。
- **契约**（见 [model-py L244-L268](./model-py#L244)）：必须含 `cls`；可选含 `bboxes` / `masks`。
- `cls` 是类别索引数组；`bboxes` 是 `(N, 4)`；`masks` 是 `(N, H, W)`。

## `YOLOEVPDetectPredictor.pre_transform`
[#L53-L95](#L53)

- 单/多图像分支：
  - [#L70-L72 `pop bboxes/masks/cls`](#L70)：从字典里取出三个字段。
  - [#L73-L75 单图分支](#L73)：调 `_process_single_image`，把结果 `unsqueeze` 到 `(1, N, H, W)`。
  - [#L76-L93 多图分支](#L76)：当前只支持 `bboxes`，`assert bboxes is not None`；[#L83-L85 `category` 也必须是 `np.ndarray` 列表](#L83)；[#L89-L92 `pad_sequence` 对齐](#L89) 到最大 N。

## `YOLOEVPDetectPredictor._process_single_image`
[#L97-L132](#L97)——**核心几何变换**

- [#L114-L122 bboxes 缩放 + letterbox 偏移](#L114)：
  ```
  gain = min(dst_h/src_h, dst_w/src_w)
  bboxes *= gain
  bboxes[..., 0::2] += round((dst_w - src_w*gain)/2 - 0.1)
  ```
  减 0.1 是防止 `round` 上溢。
- [#L123-L127 masks 分支](#L123)：调 `super().pre_transform(masks)` 上采样；[#L127 `masks[masks == 114] = 0`](#L127) 把 padding 值 114 归零（114 是 YOLOE 的默认 pad 值）。
- [#L132 `LoadVisualPrompt().get_visuals`](#L132)：把 bboxes/masks 转成 `(N, H, W)` 的视觉提示张量。

## `YOLOEVPDetectPredictor.inference`
[#L134-L146](#L134)

- [#L146 `super().inference(im, vpe=self.prompts)`](#L146)：把 prompts 作为 `vpe` 参数传给 `YOLOEModel.predict`（见 [tasks-py L943](./tasks-py#L943)）。

## `YOLOEVPDetectPredictor.get_vpe`
[#L148-L164](#L148)——**从参考图抽 vpe**

- [#L161 只支持单图](#L161)。
- [#L164 `return_vpe=True`](#L164)：让模型只返回 vpe 而跳过检测，供 [YOLOE.predict L285-L286](./model-py#L285) 的 `refer_image` 流程使用。

## `YOLOEVPSegPredictor`
[#L167-L170](#L167)

- 双继承空壳：`class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor)`。
- 复用 detect 侧的所有 prompts 处理逻辑，只把 `model` 换成 `YOLOESegModel`。

## 三 prompt 模式入口对照

| 模式 | Predictor | prompts 形态 | cls_pe 来源 |
| --- | --- | --- | --- |
| Text prompt | `DetectionPredictor` | 无（模型已 set_classes） | `get_text_pe` |
| Visual Prompt | **YOLOEVPDetectPredictor** | `{bboxes, cls}` 或 `{masks, cls}` | `get_visual_pe` + `pre_transform` |
| Prompt-Free | `DetectionPredictor` | 无（模型已 lrpc + fuse） | 无（走 `forward_lrpc`） |

## YOLOE 推理特有细节

| 细节 | 位置 | 原因 |
| --- | --- | --- |
| `done_warmup=True` | [#L41](#L41) | 每张图 prompt 不同，warmup 无意义 |
| `bboxes *= gain` + letterbox 偏移 | [#L114-L122](#L114) | 视觉提示必须与图像几何对齐 |
| `masks[masks == 114] = 0` | [#L127](#L127) | 114 是 YOLOE pad 值，必须归零 |
| 多图仅支持 bboxes | [#L76-L93](#L76) | masks 多图 pad 复杂，先不做 |
| `return_vpe=True` | [#L164](#L164) | refer_image 模式下只返回 vpe |

## 阅读路线

1. `setup_model`（[#L32](#L32)）：warmup 关闭的原因。
2. `set_prompts`（[#L43](#L43)）：prompts 字典契约。
3. `pre_transform`（[#L53](#L53)）：单/多图像分支。
4. `_process_single_image`（[#L97](#L97)）：bboxes 缩放 + masks pad 处理。
5. `inference`（[#L134](#L134)）：vpe 参数传递。
6. `get_vpe`（[#L148](#L148)）：refer_image 支持。

## 延伸阅读

- [../architecture](../architecture)：三 prompt 推理链路总览
- [../concepts](../concepts)：Visual Prompt 的几何对齐
- [../painpoints](../painpoints)：`masks == 114` 的陷阱
- [../tricks](../tricks)：`gain` 与 letterbox 的数值经验
- [../losses](../losses)：推理阶段无损失，训练见 [train-py](./train-py)
- [../code-walkthrough](../code-walkthrough)：端到端 walkthrough
