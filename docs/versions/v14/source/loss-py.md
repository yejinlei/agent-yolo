# utils/loss.py

这个文件是 v14 的**训练目标文件**：把 v14 的四条损失（`L_det` + `λ_reg·L_offset` + `λ1·L_adv` + `λ2·L_cross`）在 `v14DetectionLoss`（继承自 `v8DetectionLoss`）里组合起来，同时保留了 `VarifocalLoss` / `FocalLoss` / `DFLoss` / `BboxLoss` / `RotatedBboxLoss` / `KeypointLoss` 等 YOLO 历代沉淀的原子损失。三条 v14 特化的辅助损失 `DomainAdversarialLoss`（Section 3.3）、`CrossViewConsistencyLoss`（Section 3.4）、`OffsetRegularizationLoss`（Section 3.2）都定义在此文件。GRL 的实现在 [block.py `GradReverse`](./block-py.md) 中，本文件只调用它。

## 关键行段速查

### 基础 / 原子损失（v14 沿用）

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `VarifocalLoss` | L18–L39 | <SrcLink file="v14:utils/loss.py" :line="18" :end="39"/> |
| `FocalLoss` | L42–L64 | <SrcLink file="v14:utils/loss.py" :line="42" :end="64"/> |
| `DFLoss` | L67–L90 | <SrcLink file="v14:utils/loss.py" :line="67" :end="90"/> |
| `BboxLoss` | L93–L115 | <SrcLink file="v14:utils/loss.py" :line="93" :end="115"/> |
| `RotatedBboxLoss` | L118–L139 | <SrcLink file="v14:utils/loss.py" :line="118" :end="139"/> |
| `KeypointLoss` | L142–L156 | <SrcLink file="v14:utils/loss.py" :line="142" :end="156"/> |

### YOLO v8 系列检测头损失

| Class | 行号 | SrcLink |
| --- | --- | --- |
| `v8DetectionLoss` | L159–L262 | <SrcLink file="v14:utils/loss.py" :line="159" :end="262"/> |
| `v8SegmentationLoss` | L265–L445 | <SrcLink file="v14:utils/loss.py" :line="265" :end="445"/> |
| `v8PoseLoss` | L448–L600 | <SrcLink file="v14:utils/loss.py" :line="448" :end="600"/> |
| `v8ClassificationLoss` | L603–L611 | <SrcLink file="v14:utils/loss.py" :line="603" :end="611"/> |
| `v8OBBLoss` | L614–L727 | <SrcLink file="v14:utils/loss.py" :line="614" :end="727"/> |
| `E2EDetectLoss` | L730–L745 | <SrcLink file="v14:utils/loss.py" :line="730" :end="745"/> |

### v14 特化损失（Section 3.2 / 3.3 / 3.4 / 3.7）

| Class | 行号 | SrcLink | 关键超参 |
| --- | --- | --- | --- |
| `CrossViewConsistencyLoss` | L748–L813 | <SrcLink file="v14:utils/loss.py" :line="748" :end="813"/> | `τ_min=0.05, τ_max=0.5, β=0.01` |
| `DomainAdversarialLoss` | L816–L860 | <SrcLink file="v14:utils/loss.py" :line="816" :end="860"/> | `λ_domain=0.1`（默认；v14 中由 `v14DetectionLoss` 传入 0.05） |
| `OffsetRegularizationLoss` | L863–L909 | <SrcLink file="v14:utils/loss.py" :line="863" :end="909"/> | `λ_reg: 0.01 → 0.001` over 50 epochs |
| `v14DetectionLoss` | L912–L983 | <SrcLink file="v14:utils/loss.py" :line="912" :end="983"/> | `λ_adv=0.05, λ_cross=0.05, λ_reg=0.01` |

## 四条 schedule（在 `v14DetectionLoss.set_epoch` 里驱动）

对应 L933–L950 的 `set_epoch()` 与 L952–L982 的 `__call__()`：

1. **`λ_reg` 线性退火**（Section 3.2）：`0.01 → 0.001`，50 epoch 完成，见 `OffsetRegularizationLoss.current_lambda`（L891–L897）。
2. **GRL `λ` 由 0 升到 1**（Section 3.3）：DANN 公式 `λ = 2/(1+exp(-10p))-1`，100 epoch 完成；实现在 [block.py `grl_lambda_schedule`](./block-py.md)，本文件 L948 通过 `m.grl_lambda = grl_lambda_schedule(epoch, max_epochs=100)` 下发到每个 `DomainAdaptiveLayer`。
3. **Cross-view 温度 τ 指数下降**（Section 3.4）：`τ = τ_min + (τ_max − τ_min)·exp(−β·t)`，`τ_min=0.05, τ_max=0.5, β=0.01`；每 iteration 由 L980 的 `self.cross_loss.step_update()` 推进。
4. **λ_adv / λ_cross 常数**：`λ_adv=0.05`（DomainAdversarialLoss 内部权重）、`λ_cross=0.05`（在 L977 显式乘上）。

## `v14DetectionLoss.__call__` 的组合顺序（L952–L982）

```
det_loss, det_items = super().__call__(preds, batch)   # L_det (box+cls+dfl)
aux = 0
aux += offset_loss(offset, offset_init)               # λ_reg · L_offset (若 batch 里给了 offset)
aux += adv_loss(domain_logits, domain_labels)[1]       # λ1 · L_adv    (若给了 domain_labels)
aux += λ_cross · cross_loss(features, view_ids, cls)   # λ2 · L_cross (若给了 features+view_ids)
cross_loss.step_update()                              # τ 推进
return det_loss + aux, det_items
```

## GRL 与 DomainAdversarialLoss 的关系

- `GradReverse`（GRL）实现在 [block.py L39–L61](./block-py.md)，由 `DomainAdaptiveLayer.forward`（[block.py L1939–L1969](./block-py.md)）在计算 `domain_logits` 前调用。
- `DomainAdversarialLoss`（L816–L860）本身只做标准 `nn.CrossEntropyLoss`——**梯度反转已经在 GRL 里做了**，因此这里不需要像旧 DANN 那样翻转标签。

## 参考

- [v14 主页](../index.md)
- [主线 yolov14.yaml](./yolov14-yaml.md) — 训练目标注释
- [block.py](./block-py.md) — `GradReverse` / `grl_lambda_schedule` / `DomainAdaptiveLayer` / `ViewEmbedding`
- [head.py](./head-py.md) — Detect 输出对接
- [tal.py](./tal-py.md) — `TaskAlignedAssigner` 由 `v8DetectionLoss.__init__` 实例化
- [v14 术语表](/glossary)
