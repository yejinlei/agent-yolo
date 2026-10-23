# ultralytics/models/yolo/yoloe/train.py

本文件定义 5 个检测侧 Trainer 类（seg 版 4 类在 [train-seg-py](./train-seg-py)，两文件合计 9 个 Trainer）。核心是 `YOLOETrainer`（继承标准 `DetectionTrainer`）——它做三件事：加载 `text_embeds`、按 task 分发到 PE / FromScratch / Prompt-Free / Visual-Prompt 变体。

Source: <SourceViewer file="yoloe:ultralytics/models/yolo/yoloe/train.py" />

## 类继承图

```
DetectionTrainer
  ├── YOLOETrainer                    ← 通用父类（YOLOE detect 入口）
  │     └── YOLOETrainerFromScratch   ← 从权重从零训练
  │           ├── YOLOEPEFreeTrainer          ← Prompt-Free（PE + FromScratch）
  │           └── YOLOEVPTrainer              ← Visual-Prompt 模式
  └── YOLOEPETrainer                  ← PEFT 微调（LoRA/Adapter）
```

## `YOLOETrainer.__init__`
[#L20-L84](#L20)

- 复用 `DetectionTrainer.__init__`。
- **关键差异**：`self.text_embeds` 从 `cfg.text_embeddings_path` 加载（若存在），否则从 `CLIP` 现场推理 `self.names`。
- `self.clip_model` 保留引用，用于 `set_classes` 时重新编码新增类别。
- `self.text_projector` 是 `head.text_project`（RepRTA 分支），训练时冻结权重、只学 `logit_scale`。

## `YOLOETrainer._setup_train` / `_setup_dataloader`

- **`_setup_train`**：调用 `self.model.set_classes(self.names, self.text_embeds)` 写入闭集类别，触发 `BN + Conv + text_embeds` 的 fuse 折叠。
- **`_setup_dataloader`**：调用 `self._close_dataloader_mosaic()`——open-set 模式下 last 10% 关闭 mosaic；mixup 全程关闭（会破坏 text-image 对齐）。
- **`get_validator`**：返回 `YOLOEValidator` / `YOLOEValidatorS`（见 [val-py](./val-py)），保证 train/val 用同一套 `text_embeds`。

## `YOLOEPETrainer`（PEFT 微调）
[#L85-L130](#L85)

- 继承 `DetectionTrainer` + `YOLOETrainer` 的部分逻辑。
- 只训练 `BNContrastiveHead` 的 BN / `logit_scale` / `cv3` 末端几层，backbone / neck 全部冻结。
- 适合"已有 YOLOE 权重、想快速适配新类别"的场景。

## `YOLOETrainerFromScratch`
[#L131-L298](#L131)

- 完全从零训练（无预训练权重）。
- `_setup_train` 额外初始化：`BNContrastiveHead.bn` 用 KL 分布初始化、`logit_scale` 初始化为 `1/0.07`（CLIP 温度）。
- 训练时长明显更长——通常 300 epoch 起。

## `YOLOEPEFreeTrainer`
[#L299-L318](#L299)

- 组合继承：`YOLOEPETrainer` + `YOLOETrainerFromScratch`。
- 目标：Prompt-Free 训练——只训练 `LRPCHead`（[head-py L504](./head-py#L504)），不加载任何 `text_embeds`。
- `_setup_train` 里 `self.text_embeds = None`，走 `head.reprta = nn.Identity()` 分支（[head-py L511-L512](./head-py#L511)）。

## `YOLOEVPTrainer`
[#L319-L355](#L319)

- Visual-Prompt 训练：`get_visual_pe` 路径。
- 额外要求数据集每张图片配一个"prompt 图"（如裁剪的目标 ROI 或 mask 图）。
- 走 `spatial_aware_visual_prompt_embed`（SAVPE）+ `BNContrastiveHead(visual=True)` 分支。

## YOLOE 特有工程细节

| 细节 | 位置 | 原因 |
| --- | --- | --- |
| `text_embeds` 加载 | `_setup_train` | open-set 训练必需 |
| `text_projector` 冻结 | `build_optimizer` | 只学 BN / logit_scale |
| mixup 全程关闭 | `_setup_dataloader` | 文本-图像必须 1-to-1 对齐 |
| mosaic last 10% 关闭 | `_close_dataloader_mosaic` | 避免拼接伪影影响 prompt 匹配 |
| `logit_scale = 1/0.07` 初值 | `YOLOETrainerFromScratch` | 沿用 CLIP 温度 |
| Prompt-Free 时 `reprta = Identity` | `YOLOEPEFreeTrainer` | 彻底跳过文本变换 |

## 阅读路线

1. `YOLOETrainer.__init__`（[#L20](#L20)）：`text_embeds` 加载逻辑。
2. `_setup_train`：`set_classes` 触发 fuse。
3. `YOLOEPETrainer`（[#L85](#L85)）：PEFT 微调冻结策略。
4. `YOLOETrainerFromScratch`（[#L131](#L131)）：从零训练的初始化。
5. `YOLOEPEFreeTrainer`（[#L299](#L299)）：Prompt-Free 分支。
6. `YOLOEVPTrainer`（[#L319](#L319)）：Visual-Prompt 分支。

## 与 YOLOv8 `DetectionTrainer` 的对比

| 维度 | YOLOv8 | YOLOE |
| --- | --- | --- |
| 加载 `text_embeds` | 否 | 是 |
| `BNContrastiveHead` 冻结 | 无此模块 | BN / logit_scale 可训 |
| mixup | 默认开 | open-set 关闭 |
| mosaic 关闭时机 | last 10% | open-set 更早 |
| 目标函数 | box + cls + dfl | 追加 InfoNCE |

## 延伸阅读

- [../architecture](../architecture)：训练-推理双链路
- [../concepts](../concepts)：三 prompt 模式的训练策略
- [../losses](../losses)：`BNContrastiveHead` 的对比损失
- [../painpoints](../painpoints)：为什么冻结 `text_embeds`
- [../tricks](../tricks)：mosaic 关闭时机的经验值
