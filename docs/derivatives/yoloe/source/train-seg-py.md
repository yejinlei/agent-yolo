# ultralytics/models/yolo/yoloe/train_seg.py

本文件定义 4 个分割侧 Trainer 类。检测侧的 5 个 Trainer 在 [train-py](./train-py)；两文件合计 9 个。分割侧同样通过 `YOLOETrainer`（`yoloe/train.py`）+ `SegmentationTrainer` 组合，得到 `YOLOESegTrainer`。

Source: <SourceViewer file="yoloe:ultralytics/models/yolo/yoloe/train_seg.py" />

## 类继承图

```
SegmentationTrainer
  └── YOLOESegTrainer                       ← YOLOETrainer + SegmentationTrainer 组合
        ├── YOLOEPESegTrainer                ← PEFT 分割微调
        └── YOLOESegTrainerFromScratch       ← 从零训练
              └── YOLOESegVPTrainer          ← Visual-Prompt 分割训练
```

## `YOLOESegTrainer.__init__`
[#L14-L42](#L14)

- 组合继承：`YOLOETrainer, SegmentationTrainer`——前者提供 YOLOE 的 `text_embeds` 加载，后者提供 `SegmentationTrainer` 的 mask 处理。
- `super().__init__` 按 MRO 依次调用两个父类的 `__init__`。
- 关键属性：`self.text_embeds`、`self.clip_model`、`self.model.names`。

## `YOLOESegTrainer.get_model`
[#L43-L67](#L43)

- 返回 `YOLOESegModel`（见 [tasks-py L1009](./tasks-py#L1009)）而非 `YOLOESeg`。
- 若 `weights` 提供，走 `AutoBackend` 加载；否则按 yaml 从零构建。
- yaml 头里 `task: segment` 决定 head 类型（`YOLOESeg` 而非 `YOLOEDetect`）。

## `YOLOESegTrainer.get_validator`
[#L68-L80](#L68)

- 返回 `YOLOESegmentValidator`（见 [val-py](./val-py)），保证分割侧验证用同一套 `text_embeds`。
- 与检测侧的 `YOLOEValidator` 共享 `_get_text_embeds` 逻辑。

## `YOLOEPESegTrainer`（PEFT 分割微调）
[#L81-L131](#L81)

- 继承 `SegmentationTrainer` + `YOLOEPESegTrainer` 局部逻辑。
- 冻结 backbone/neck，只训练 `YOLOESeg.bn` / `logit_scale` / `cv3` / `cv4` 末端几层。
- 场景：已有 YOLOE-seg 权重、快速适配新类别。

## `YOLOESegTrainerFromScratch`
[#L132-L137](#L132)

- 组合继承：`YOLOETrainerFromScratch, YOLOESegTrainer`。
- 无预训练权重；mask 头 `cv4` 从零初始化。
- `_setup_train` 里 `BNContrastiveHead.bn` 用 KL 分布、`logit_scale = 1/0.07`。

## `YOLOESegVPTrainer`
[#L138-L141](#L138)

- 组合继承：`YOLOEVPTrainer, YOLOESegTrainerFromScratch`。
- Visual-Prompt 分割训练：每张图配一张"prompt 图"（ROI 或 mask）。
- 走 `spatial_aware_visual_prompt_embed`（SAVPE）+ `BNContrastiveHead(visual=True)` 分支。

## YOLOE 分割特有细节

| 细节 | 位置 | 原因 |
| --- | --- | --- |
| `overlap_mask=False` | `SegmentationTrainer` 默认关闭 | 开放集 mask 拼接不做重叠消解，避免类别混淆 |
| mask head 输出维度 | `YOLOESeg` | `nm * nc`——nc 由 prompt 动态决定 |
| `text_embeds` 与 mask 对齐 | `_setup_train` | 每个 GT mask 匹配一个 prompt embedding |
| `cv4` 初始化 | `YOLOESegTrainerFromScratch` | 用 `kaiming_uniform` + bias 0，防止 mask 全空 |
| Prompt-Free 分割 | `YOLOESegVPTrainer` | 走 `LRPCHead` 分支 |

## 阅读路线

1. `YOLOESegTrainer.__init__`（[#L14](#L14)）：组合继承的 MRO。
2. `get_model`（[#L43](#L43)）：`YOLOESegModel` 的构造。
3. `get_validator`（[#L68](#L68)）：分割验证器。
4. `YOLOEPESegTrainer`（[#L81](#L81)）：PEFT 冻结策略。
5. `YOLOESegTrainerFromScratch`（[#L132](#L132)）：从零训练。
6. `YOLOESegVPTrainer`（[#L138](#L138)）：Visual-Prompt 分割。

## 与 YOLOv8 `SegmentationTrainer` 的对比

| 维度 | YOLOv8 | YOLOE |
| --- | --- | --- |
| mask 头 | `Segment` | `YOLOESeg` |
| `text_embeds` | 无 | 有 |
| `overlap_mask` | 默认开 | open-set 关闭 |
| mask loss | BCE + Dice | 追加对比损失 |
| 训练参数 | 全参 | 部分冻结 |

## 延伸阅读

- [../architecture](../architecture)：分割链路
- [../concepts](../concepts)：`overlap_mask=False` 的动机
- [../losses](../losses)：mask loss 与 contrastive loss 的组合
- [../painpoints](../painpoints)：mask 拼接的开放集陷阱
- [../tricks](../tricks)：`cv4` 初始化的经验值
