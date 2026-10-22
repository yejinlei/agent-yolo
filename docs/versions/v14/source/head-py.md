# nn/modules/head.py

这个文件是 YOLO 历代检测头的集合地：从 `Detect` / `Segment` / `Pose` / `OBB` 到 v10 的 `v10Detect`（无 anchor head），再到 RT-DETR 的 `RTDETRDecoder`，最后到 v14 的 `WorldDetect`（多视角/世界模型变体）。**v14 在此文件的唯一新增是 `WorldDetect`**——一个继承自 `Detect` 的 head，支持视角标签预测与跨视角一致性；其余 head（`Detect`、`Segment`、`Pose`、`OBB`、`Classify`、`v10Detect`、`RTDETRDecoder`）都直接沿用 v13 版本。

## 关键行段速查

### v14 新增

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `WorldDetect` | L307–L359 | <SrcLink file="v14:nn/modules/head.py" :line="307" :end="359"/> |

### 主流 YOLO head（v14 沿用 v13）

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `Detect` | L21–L172 | <SrcLink file="v14:nn/modules/head.py" :line="21" :end="172"/> |
| `Detect.__init__` | L34–L62 | <SrcLink file="v14:nn/modules/head.py" :line="34" :end="62"/> |
| `Detect.forward` | L64–L74 | <SrcLink file="v14:nn/modules/head.py" :line="64" :end="74"/> |
| `Detect.forward_end2end` | L76–L98 | <SrcLink file="v14:nn/modules/head.py" :line="76" :end="98"/> |
| `Detect._inference` | L100–L131 | <SrcLink file="v14:nn/modules/head.py" :line="100" :end="131"/> |
| `Detect.bias_init` | L133–L144 | <SrcLink file="v14:nn/modules/head.py" :line="133" :end="144"/> |
| `Detect.decode_bboxes` | L146–L148 | <SrcLink file="v14:nn/modules/head.py" :line="146" :end="148"/> |
| `Detect.postprocess` | L150–L172 | <SrcLink file="v14:nn/modules/head.py" :line="150" :end="172"/> |
| `Segment` | L175–L197 | <SrcLink file="v14:nn/modules/head.py" :line="175" :end="197"/> |
| `OBB` | L200–L227 | <SrcLink file="v14:nn/modules/head.py" :line="200" :end="227"/> |
| `Pose` | L230–L279 | <SrcLink file="v14:nn/modules/head.py" :line="230" :end="279"/> |
| `Classify` | L282–L304 | <SrcLink file="v14:nn/modules/head.py" :line="282" :end="304"/> |

### 变体 / 特殊 head

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `RTDETRDecoder` | L361–L589 | <SrcLink file="v14:nn/modules/head.py" :line="361" :end="589"/> |
| `v10Detect` | L592–L625 | <SrcLink file="v14:nn/modules/head.py" :line="592" :end="625"/> |

## `WorldDetect`：v14 唯一的新 head

`WorldDetect` 继承 `Detect`，在标准 `box+cls+dfl` 之外追加一条**视角分类分支**：

```
y = forward(x)  # x: 来自 DomainAdaptiveLayer 的融合特征
    stride, decoded_bboxes, class_scores = super().forward(x)
    view_logits = self.view_head(x)     # 新增：视角分类 logits
    return y, view_logits
```

`view_head` 是一个简单的 `Conv(chn, chn // 2) → Conv(chn // 2, num_views)`，与 `DomainAdaptiveLayer` 中的 `domain_logits` **不是同一个概念**：
- `DomainAdaptiveLayer.domain_logits` 用于 DANN 域对抗损失（Section 3.3），训练时反向传播通过 GRL；
- `WorldDetect.view_logits` 是显式的视角预测（Section 3.4），直接参与 cross-view consistency 损失。

## v14 主线用哪个 head？

- **主线 `yolov14.yaml`** 使用 `Detect`（不是 `WorldDetect`）。`WorldDetect` 只在 YOLO-Omni 论文的多视角训练配置里启用。
- 因此读 v14 主线源码时，`Detect` 的行为与 v13 完全一致；`WorldDetect` 是理解论文 Section 3.4 的入口。

## 参考

- [v14 主页](../index.md)
- [主线 yolov14.yaml](./yolov14-yaml.md) — 头层配置
- [block.py](./block-py.md) — `DomainAdaptiveLayer` / `ViewEmbedding` 与 head 的对接
- [loss.py](./loss-py.md) — `v14DetectionLoss` 消费 head 的输出
- [tal.py](./tal-py.md) — 正样本分配器
- [v14 术语表](/glossary)
