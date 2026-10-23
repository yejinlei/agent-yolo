# ultralytics/models/yolo/model.py

Ultralytics 顶层的 `YOLO` 类是用户唯一接触的入口，负责 `model.train / val / predict / export` 路由。YOLOE 在这一层做两件事：**按 `task` 分发到 `YOLOETask`**、**给开放词表推理暴露 `get_text_pe` / `get_visual_pe` / `set_vocab` / `set_classes` 四个 API**。

Source: <SourceViewer file="yoloe:ultralytics/models/yolo/model.py" />

## 入口 `YOLO.__init__`
[#L158 起](#L158)

- 解析 `model` 参数——支持 `.yaml` 路径、`.pt` 权重、`ultralytics/hub/...` 等字符串；YOLOE 家族默认落在 `CFG_DEFAULTS['model']` 未指定时回退到 `yolo11.pt`。
- `self.model = AutoBackend(...)` 载入权重，此时 `task` 字段已被识别为 `detect` / `segment` 等；YOLOE 的 `reperta` 分支在 backbone 上，无需额外判断。
- 若 `model` 字符串包含 `yoloe-` 前缀，`task` 会被自动推断为开放词表任务。

## 属性 `predictor` / `validator` / `trainer`
[#L200-L230 附近](#L200-L230)

- `YOLOE` 通过 `__getattr__` 惰性实例化三个下游类：
  - **predictor** → `YOLOEPredictor`（见 [predict-py](./predict-py)）
  - **validator** → `YOLOEValidator` / `YOLOESegmentValidator`（见 [val-py](./val-py)）
  - **trainer** → `YOLOEDetectionTrainer` / `YOLOESegmentationTrainer` 等（见 [train-py](./train-py) / [train-seg-py](./train-seg-py)）
- 三个下游类都从 `cfg.model`（yaml 头）里读出 `task`，再决定用哪个 head。

## `YOLO` 上暴露给 YOLOE 的额外 API
[#L240-L280 附近](#L240-L280)

| API | 作用 | 下游落点 |
| --- | --- | --- |
| `get_text_pe(words, device)` | 把文本 token 编码成 `text_pe`，走 text prompt 模式 | `YOLOEModel.get_text_pe`（[tasks-py L1014](./tasks-py#L1014)） |
| `get_visual_pe(emb, device)` | 加载/编码 visual prompt，走 visual prompt 模式 | `YOLOEModel.get_visual_pe`（[tasks-py L1026](./tasks-py#L1026)） |
| `set_vocab(words, device)` | 一次性把闭集词表写入 model（后续每次 predict 免 token 化） | 同 `get_text_pe`，但会缓存 |
| `set_classes(classes)` | 切换闭集类别名，用于 `predict` 时的 `class_names` 匹配 | 更新 `model.names` |
| `predict(source, text=..., visual=..., prompt_free=False)` | 三 prompt 模式统一入口 | `YOLOEModel.forward` |

`get_text_pe` 内部流程：`tokenize` → `token_to_id` → 索引 `text_embeds` → 走 [head-py L410 `BNContrastiveHead.preprocess`](./head-py#L410) 得到 `text_pe (H, D)`。

## 三 prompt 模式在 `YOLO` 层的对应关系

| 模式 | 用户调用 | YOLOEModel 侧 |
| --- | --- | --- |
| Text prompt | `model(source, text='dog, cat')` 或 `model.set_vocab([...])` | `txt_embeds` 非空 → `BNContrastiveHead` |
| Visual prompt | `model(source, visual=text_embeddings.pt)` | `visual_embeds` 非空 → `BNContrastiveHead(visual=True)` + SAVPE |
| Prompt-Free | `model(source, prompt_free=True)` | 只用 `LRPCHead` |

## 阅读路线

1. `YOLO.__init__`（[#L158](#L158)）：理解权重加载如何触发 `AutoBackend`。
2. `__getattr__`（[#L200-L230](#L200-L230)）：三个下游类如何被惰性绑定。
3. YOLOE 专属 API（[#L240-L280](#L240-L280)）：三 prompt 模式如何对外暴露。
4. `predict` / `train` / `val` 转发：YOLO 类只做参数拼装，真正逻辑在下游。

## 与 YOLOv8 `YOLO` 的对比

| 维度 | YOLOv8 `YOLO` | YOLOE `YOLO` |
| --- | --- | --- |
| `predict` 参数 | `source, conf, iou, ...` | 多 `text`, `visual`, `prompt_free` |
| 顶层 API | `train/val/predict/export` | 额外 `get_text_pe`, `get_visual_pe`, `set_vocab`, `set_classes` |
| Task 判定 | `task` 字段显式指定 | 支持从模型名自动推断 |

## 延伸阅读

- [../architecture](../architecture)：开放词表推理链路
- [../concepts](../concepts)：三 prompt 模式概念定义
- [../code-walkthrough](../code-walkthrough)：端到端 walkthrough
- [../painpoints](../painpoints)：`text_embeds` 缓存与 fuse 折叠
- [../tricks](../tricks)：`overlap_mask=False` 的动机
