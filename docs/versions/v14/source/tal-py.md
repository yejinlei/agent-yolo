# utils/tal.py

这个文件是 YOLO 的**任务对齐分配器（Task-Aligned Assigner, TAL）**定义地，也是 box 距离变换与 anchor 生成的工具集所在。**v14 没有改动这个文件**——`TaskAlignedAssigner` 的 `topk=13, alpha=1.0, beta=6.0` 默认值（构造函数）与 [loss.py `v8DetectionLoss`](./loss-py.md) 里 `TaskAlignedAssigner(topk=tal_topk=10, alpha=0.5, beta=6.0)` 的调用参数，都沿用 YOLOv8/v13 的设置。v14 的领域适应、多视角一致性、偏移正则化都发生在**损失组合层**（[loss-py.md](./loss-py.md) 的 `v14DetectionLoss`）与**特征提取层**（[block-py.md](./block-py.md) 的 `DomainAdaptiveLayer` / `ViewEmbedding`），分配器本身保持不动。

## 关键行段速查

### 分配器

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `TaskAlignedAssigner` | L14–L295 | <SrcLink file="v14:utils/tal.py" :line="14" :end="295"/> |
| `TaskAlignedAssigner.__init__`（默认 `topk=13, alpha=1.0, beta=6.0`） | L29–L37 | <SrcLink file="v14:utils/tal.py" :line="29" :end="37"/> |
| `TaskAlignedAssigner.forward` | L39–L80 | <SrcLink file="v14:utils/tal.py" :line="39" :end="80"/> |
| `TaskAlignedAssigner._forward` | L82–L118 | <SrcLink file="v14:utils/tal.py" :line="82" :end="118"/> |
| `TaskAlignedAssigner.get_pos_mask` | L120–L130 | <SrcLink file="v14:utils/tal.py" :line="120" :end="130"/> |
| `TaskAlignedAssigner.get_box_metrics` | L132–L151 | <SrcLink file="v14:utils/tal.py" :line="132" :end="151"/> |
| `TaskAlignedAssigner.select_topk_candidates` | L157–L190 | <SrcLink file="v14:utils/tal.py" :line="157" :end="190"/> |
| `TaskAlignedAssigner.get_targets` | L192–L238 | <SrcLink file="v14:utils/tal.py" :line="192" :end="238"/> |
| `TaskAlignedAssigner.select_candidates_in_gts` | L240–L262 | <SrcLink file="v14:utils/tal.py" :line="240" :end="262"/> |
| `TaskAlignedAssigner.select_highest_overlaps` | L264–L295 | <SrcLink file="v14:utils/tal.py" :line="264" :end="295"/> |
| `RotatedTaskAlignedAssigner` | L298–L330 | <SrcLink file="v14:utils/tal.py" :line="298" :end="330"/> |

### Anchor 与坐标变换工具

| Func | 行号 | SrcLink |
| --- | --- | --- |
| `make_anchors` | L333–L345 | <SrcLink file="v14:utils/tal.py" :line="333" :end="345"/> |
| `dist2bbox` | L348–L357 | <SrcLink file="v14:utils/tal.py" :line="348" :end="357"/> |
| `bbox2dist` | L360–L363 | <SrcLink file="v14:utils/tal.py" :line="360" :end="363"/> |
| `dist2rbox` | L366–L385 | <SrcLink file="v14:utils/tal.py" :line="366" :end="385"/> |

## v14 是否改动？

**结论：完全未改动。** 具体证据：

1. 构造函数（L29）的默认参数仍为 `topk=13, alpha=1.0, beta=6.0`。
2. 训练态由 [loss-py.md `v8DetectionLoss.__init__` L178](./loss-py.md) 覆盖为 `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)`，`v14DetectionLoss` 继承自它并未重写。
3. 没有 v14 特有的 assigner 变体（无 `V14TaskAlignedAssigner`、无 domain-aware / view-aware 分配变体）。
4. `RotatedTaskAlignedAssigner`（L298–L330）为 OBB 分支保留，v14 主线未启用。

因此 TAL 在 v14 中属于**基础设施层**，其正样本分配规则仍是：

```
align_metric = cls_score^α · IoU^β
```

α=0.5, β=6.0，取每个 GT 的 top-10 anchor，然后按 mask_pos 归一化 target_scores。

## 参考

- [v14 主页](../index.md)
- [loss.py](./loss-py.md) — `v14DetectionLoss` / `v8DetectionLoss` 调用 TAL
- [head.py](./head-py.md) — Detect 输出对接 TAL
- [block.py](./block-py.md) — v14 特征层模块
- [主线 yolov14.yaml](./yolov14-yaml.md)
- [v14 术语表](/glossary)
