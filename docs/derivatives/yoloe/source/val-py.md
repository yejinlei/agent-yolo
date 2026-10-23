# ultralytics/models/yolo/yoloe/val.py

**YOLOE 验证管线**：`YOLOEValidator` 是 `DetectionValidator` 的扩展，负责 open-set 验证——把 `txt_embeds` 与每张图的 GT 类别对齐，再走 `BNContrastiveHead` 的对比度量。同文件提供 `YOLOESegmentValidator`（分割版）和 `YOLOEValidatorS`（S 系列小尺寸模型版）。

Source: <SourceViewer file="yoloe:ultralytics/models/yolo/yoloe/val.py" />

## 类继承图

```
DetectionValidator
  ├── YOLOEValidator                    ← 检测侧验证
  │     └── YOLOEValidatorS             ← S 系列小尺寸变体
  └── (SegmentationValidator + YOLOEValidator)
        └── YOLOESegmentValidator       ← 分割侧验证
```

## `YOLOEValidator.__init__`
[#L20-L60](#L20)

- 复用 `DetectionValidator.__init__`。
- **关键差异**：`self.text_embeds` 从 `args.text_embeddings_path` 加载，若路径为空则现场调用 `CLIP` 编码 `self.names`。
- `self.clip_model` 保留引用，用于动态追加类别时重新编码。
- `self.names` 与 `self.text_embeds` 一一对应，禁止修改顺序。

## `YOLOEValidator.get_dataloader`
[#L60-L100](#L60)

- 复用基类数据加载，但**关闭 mosaic / mixup / copy_paste**（这些 augment 在验证阶段本就应关闭）。
- open-set 数据需要 `open_set=True` 标记，`_pre_transform` 里据此决定是否送入 `txt_embeds`。

## `YOLOEValidator._pre_transform` / `pre_transform`
[#L100-L140](#L100)

- 把 `txt_embeds` 注入 batch：`batch["txt_embeds"] = self.text_embeds.to(device)`。
- 与训练侧 `YOLOEWrapper`（见 [tasks-py L1040](./tasks-py#L1040)）共享这套 batch 结构。
- Prompt-Free 验证时 `txt_embeds=None`，走 `LRPCHead` 分支。

## `YOLOEValidator.postprocess`
[#L140-L187](#L140)

- 复用 `DetectionValidator.postprocess` 的 NMS / IoU 计算。
- **open-set 差异**：类别分数来自 `BNContrastiveHead` 的点积 logits，而非静态 `nc` 分类输出。
- 阈值 `conf` 在 `YOLOEValidator` 中默认略低（0.25 而非 0.5），因为开放集分数分布更分散。

## `YOLOEValidatorS`（S 系列变体）
[#L60-L100 附近的子类](#L60)

- 针对 `yoloe-s` 系列小尺寸模型（CSPNeXt-S 骨干 + `reg_max: 1` + `end2end: True`）。
- `conf` 阈值更低、`iou` 阈值更严，因为小模型在开放集下更容易误检。
- 其它逻辑复用 `YOLOEValidator`。

## `YOLOESegmentValidator`
[#L100-L187 附近的分割子类](#L100)

- 双继承 `SegmentationValidator + YOLOEValidator`。
- 增加 mask 后处理：`segment_postprocess` 把预测的 mask 与 GT 做 Dice 计算。
- `overlap_mask=False`——开放集下不做 mask 重叠消解，避免类别混淆。

## YOLOE 验证特有细节

| 细节 | 位置 | 原因 |
| --- | --- | --- |
| `text_embeds` 加载 | `__init__` | open-set 验证必需 |
| batch 注入 `txt_embeds` | `_pre_transform` | head 需要 text prompt |
| `conf` 默认 0.25 | `__init__` | 开放集分数分布更分散 |
| `iou` 默认 0.6 | `__init__` | 开放集下误检率高，收紧 NMS |
| `overlap_mask=False` | `YOLOESegmentValidator` | 防止开放集 mask 拼接混淆 |
| 关闭 copy_paste | `get_dataloader` | 验证阶段禁用所有破坏对齐的 augment |

## 阅读路线

1. `YOLOEValidator.__init__`（[#L20](#L20)）：`text_embeds` 加载。
2. `get_dataloader`（[#L60](#L60)）：augment 关闭策略。
3. `_pre_transform`（[#L100](#L100)）：`txt_embeds` 注入 batch。
4. `postprocess`（[#L140](#L140)）：open-set 阈值策略。
5. `YOLOEValidatorS`：小尺寸模型的阈值调整。
6. `YOLOESegmentValidator`：分割验证与 `overlap_mask=False`。

## 与 YOLOv8 `DetectionValidator` 的对比

| 维度 | YOLOv8 | YOLOE |
| --- | --- | --- |
| `text_embeds` | 无 | 有 |
| `conf` 默认 | 0.5 | 0.25 |
| `iou` 默认 | 0.7 | 0.6 |
| `overlap_mask` | 开 | 关 |
| batch 注入 | 无 | `txt_embeds` |

## 延伸阅读

- [../architecture](../architecture)：验证链路总览
- [../concepts](../concepts)：open-set 阈值策略
- [../painpoints](../painpoints)：开放集分数分布陷阱
- [../tricks](../tricks)：`conf` / `iou` 调参经验
- [../losses](../losses)：训练损失与验证指标的关系
