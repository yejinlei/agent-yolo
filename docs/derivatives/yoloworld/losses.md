# 损失函数

YOLO-World 的损失函数在结构上**完全沿用 YOLOv8**（`loss_cls` + `loss_bbox` + `loss_dfl` 三项，权重 `box=7.5 / cls=0.5 / dfl=1.5`），但在开放词表场景下新增了**`txt_masks` 加权机制**——防止随机采样的 1203 类负样本污染 `loss_cls`。本页把这三项损失 + TAL 分配 + 分布式补偿 + txt_masks 加权全部落到源码级。

- **主类**：AILab-CVC 的 `YOLOWorldHead.loss_by_feat`（<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="428" :end="577"/>）——150 行代码集中处理所有损失；
- **Ultralytics 版**：直接复用 v8 的 `DetectionModel.init_criterion` + `BCEWithLogitsLoss` + `v8ClassificationLoss` + `DFL`——**没有开放词表加权**（这是 Ultralytics 集成的简化）；
- **训练时 batch 大小**：S 档 4×8 GPU × 16 batch = 512 total；L 档常见 4×8 × 8 = 256 total——影响 `avg_factor` 与 `world_size` 补偿。

## 目录

- [一、总览：三项损失 + 权重 + 归一化](#一总览三项损失--权重--归一化)
- [二、loss_cls：文本嵌入加权交叉熵](#二loss_cls文本嵌入加权交叉熵)
- [三、loss_bbox：IoU Loss](#三loss_bbioiou-loss)
- [四、loss_dfl：Distribution Focal Loss](#四loss_dfldistribution-focal-loss)
- [五、TAL 分配结果如何注入损失](#五tal-分配结果如何注入损失)
- [六、txt_masks 加权机制详解](#六txt_masks-加权机制详解)
- [七、world_size 分布式补偿](#七world_size-分布式补偿)
- [八、优化器与 paramwise_cfg](#八优化器与-paramwise_cfg)
- [九、Ultralytics 集成版的差异](#九ultralytics-集成版的差异)
- [十、常见调试场景](#十常见调试场景)

---

## 一、总览：三项损失 + 权重 + 归一化

YOLO-World 的最终损失公式与 YOLOv8 完全一致：

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{cls}}_{\text{CE 类交叉熵}} \cdot w_{cls} + \underbrace{\mathcal{L}_{bbox}}_{\text{IoU 定位}} \cdot w_{box} + \underbrace{\mathcal{L}_{dfl}}_{\text{DFL 分布}} \cdot w_{dfl}$$

**默认权重**（Ultralytics v8 `DetectionModel.init_criterion`）：

| 分量 | 权重 | 说明 |
|------|------|------|
| `box` | **7.5** | IoU Loss 权重 |
| `cls` | **0.5** | CE 类权重 |
| `dfl` | **1.5** | Distribution Focal Loss 权重 |
| `giou` | 0.0 | 不使用（YOLOv8 用 `IoULoss` 而非 `GIoULoss`） |

**YOLO-World 相对 v8 的差异**：

1. `cls` 从 `nc=80` 变成 `K=1203`（预训练时）或 `K=80`（微调时）——**CE 的类别维度是动态的**；
2. `loss_cls` 内部引入 `txt_masks` 加权——只有正类类别计算梯度；
3. `avg_factor` 从 `total_num_pos` 变成 `assigned_scores_sum * num_imgs * world_size`——**分布式补偿 + 正样本数补偿**；
4. 损失分量仍返回 `dict(loss_cls=..., loss_bbox=..., loss_dfl=...)` 三项，**格式不变**。

---

## 二、loss_cls：文本嵌入加权交叉熵

### 2.1 公式

传统 CE 是：

$$\mathcal{L}_{cls} = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

YOLO-World 里 `K = 1203`（LVIS 类数），但一张图上只有 1~10 个正类。如果直接把 1203 类都当负类计算 CE，梯度会被**大量随机采样的假负类**污染。

**txt_masks 加权 CE**：

$$\mathcal{L}_{cls} = -\frac{1}{|M_{pos}|} \sum_{i,j} M_{ij} \cdot y_{ij} \log \hat{p}_{ij}$$

其中 `M_{ij}` 是 `txt_masks`——只有被 TAL 分配为正样本 anchor 对应的类别位置才计入损失。

### 2.2 源码（AILab-CVC 版）

<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="520" :end="529"/>

```python
# loss_by_feat 里的 loss_cls 计算
# cls_score: [B, K, H*W], y_true: [B, K, H*W], fg_mask: [B, H*W]
flatten_cls_preds = cls_preds.permute((0, 2, 1)).reshape(-1, K)  # [N, K]
y_true = y_true.permute((0, 2, 1)).reshape(-1, K)                 # [N, K]
fg_mask = fg_mask.reshape(-1)                                     # [N]

num_pos = fg_mask.sum() * world_size  # 分布式正样本总数
assigned_scores = torch.gather(cls_preds, 1, y_true_argmax)      # 每个 anchor 的 GT 类分数
assigned_scores = assigned_scores[fg_mask]                        # 只取正样本
assigned_scores_sum = max(assigned_scores.sum(), 1)

# ★ 核心：txt_masks 加权
cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(-1, flatten_cls_preds.shape[1], -1)
loss_cls = self.loss_cls(
    flatten_cls_preds * cls_weight,     # ← 假负类权重 0
    y_true * cls_weight,                # ← 假负类目标 0
    weight=cls_weight,                  # ← 假负类权重 0
    avg_factor=num_pos,
) / assigned_scores_sum                 # ← 用正类分数之和归一
```

**关键三处加权**：

1. **`flatten_cls_preds * cls_weight`**：预测 logits 里假负类乘 0（CE 里 `-log(p)` 项为 0）；
2. **`y_true * cls_weight`**：目标标签里假负类乘 0；
3. **`weight=cls_weight`**：`BCEWithLogitsLoss` 的逐元素权重，进一步屏蔽假负类。

### 2.3 Ultralytics 版的 CE

Ultralytics 集成版**没有 txt_masks 加权**，用的是普通 CE：

```python
# ultralytics/loss/v8ClassificationLoss.py
class v8ClassificationLoss:
    def __call__(self, preds, targets):
        # preds: [B, K, H*W]
        # targets: [B, n_gt, 5]  (idx, x, y, w, h)
        ...
        loss = F.cross_entropy(preds, targets, reduction='none', ignore_index=0)
        ...
```

**为什么 Ultralytics 版不需要 txt_masks？** 因为 Ultralytics 的 `WorldTrainer.preprocess_batch` 里直接 tokenize + encode **当前 batch 里的所有类**（不是随机采样 1203 类），所以类别维度天然等于正类数，没有假负类问题。

见 <SrcLink file="yoloworld:ultralytics/models/yolo/world/train.py" :line="81" :end="91"/>。

---

## 三、loss_bbox：IoU Loss

### 3.1 公式

YOLO-World 沿用 v8 的 **CIoU Loss**（Complete IoU Loss）：

$$\mathcal{L}_{CIoU}(b_{pred}, b_{gt}) = 1 - \text{IoU}(b_{pred}, b_{gt}) + \frac{\rho^2(b_{pred}, b_{gt})}{c^2} - v$$

其中：
- `ρ²(b_pred, b_gt)` 是两框中心距离的欧氏距离平方；
- `c²` 是最小外接矩形对角线距离平方；
- `v` 是纵横比一致性项，只在两框有交集时生效。

**为什么用 CIoU 而不是 IoU / GIoU / DIoU？** CIoU 是 v8 的默认选择，兼顾重叠度、中心距离、纵横比三个几何因子，训练收敛更快。

### 3.2 源码

<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="548" :end="550"/>

```python
# pred_bboxes_pos: [N_pos, 4]
# assigned_bboxes_pos: [N_pos, 4]
# bbox_weight: [N_pos]  ← 由 assigned_scores 反算而来
loss_bbox = self.loss_bbox(
    pred_bboxes_pos,
    assigned_bboxes_pos,
    weight=bbox_weight,
    avg_factor=num_pos,
) / assigned_scores_sum
```

**关键点**：

- `bbox_weight` 是**每个正样本 anchor 的权重**，等于 `assigned_scores`（对应 GT 类的分数）；
- `avg_factor=num_pos`：用正样本数做归一化；
- 最后再除以 `assigned_scores_sum`——**双重归一化**，确保不同 batch 之间的 loss 尺度稳定。

### 3.3 Ultralytics 版

Ultralytics 用 `v8DetectionLoss`（<SrcLink file="yoloworld:ultralytics/models/yolo/world/train.py" :line="55" :end="60"/> 引用），公式完全一致——只是没有开放词表加权，因此 `bbox_weight = assigned_scores * fg_mask`（fg_mask 是布尔数组）。

---

## 四、loss_dfl：Distribution Focal Loss

### 4.1 公式

DFL 是 YOLOv8 引入的**分布式焦点损失**，把回归任务转成 16 个 bucket 的分布学习：

$$\mathcal{L}_{DFL} = -\frac{1}{C} \sum_{k=1}^{4} \sum_{i=0}^{n-1} y_{k,i} \log p_{k,i}$$

其中：
- `n = reg_max = 16`：分布 bucket 数；
- `C = 4`：bbox 4 个边界（l, t, r, b）；
- `y_{k,i}`：目标分布（把连续坐标离散化到 16 个 bucket，采用双线性插值）；
- `p_{k,i}`：预测分布（sigmoid 之前）。

**为什么用 DFL 而不是 MSE / IoU？** DFL 假设 bbox 边界位置是一个**分布**而非点估计，能显著改善小目标和长尾目标的定位精度。

### 4.2 源码

<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="561" :end="566"/>

```python
if self.use_dfl:
    # pred_dist_pos: [N_pos, 4 * reg_max]  ← 4 个边界 × 16 bucket
    # assigned_ltrb_pos: [N_pos, 4]        ← 目标 LTRB
    loss_dfl = self.loss_dfl(
        pred_dist_pos.reshape(-1, self.reg_max),          # [-1, 16]
        assigned_ltrb_pos.reshape(-1),                     # [-1]
        weight=bbox_weight.expand(-1, 4).reshape(-1),     # 每个 anchor 权重复制 4 次
        avg_factor=num_pos,
    ) / assigned_scores_sum
else:
    loss_dfl = torch.zeros_like(loss_bbox)  # DFL 关闭时置 0
```

**关键点**：

- `reg_max=16` 是**硬编码**（v8 默认），对应 `cv2` 输出 `4 × 16 = 64` 通道；
- `weight` 是**每个 anchor 4 个边界的统一权重**，用 `bbox_weight.expand(-1, 4)` 复制；
- `avg_factor=num_pos` 与 `loss_bbox` 相同——三者共享归一化因子；
- 最终返回的 `loss_dfl` 与 `loss_bbox` 尺度可比。

### 4.3 Ultralytics 版

Ultralytics 用 `ultralytics.utils.loss.DFL`（<SrcLink file="yoloworld:ultralytics/nn/modules/head.py" :line="31" :end="41"/> 引用）：

```python
class DFL(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.c = c
        self.x = nn.arange(self.c)

    def forward(self, x):
        return F.linear(F.softmax(x, 1), self.x.to(x.device, x.dtype))
```

推理时把 16-bucket 分布**期望值化**，训练时反向传播。

---

## 五、TAL 分配结果如何注入损失

### 5.1 TAL 分配的核心返回

TAL（Task-Aligned Assigner）在 v8 与 YOLO-World 里是同一份实现（`ultralytics.utils.tal.TAL` / `mmyolo.ops.alAssigner`），返回：

| 变量 | Shape | 说明 |
|------|-------|------|
| `assigned_bboxes` | `[B, n_gt, topk, 4]` | 每个 GT 分配的 topk 个 anchor 坐标 |
| `assigned_scores` | `[B, n_gt, topk]` | 每个分配的对齐分数（$T = C^{\alpha} \cdot I^{\beta}$） |
| `fg_mask_pre_prior` | `[B, H*W]` | 每个 anchor 是否是正样本（True 表示是） |
| `indices` | `[B, n_gt, topk]` | 每个 GT 分配的 anchor 索引 |

### 5.2 分配后扁平化

<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="509" :end="518"/>

```python
# 从 TAL 结果提取正样本
fg_mask = fg_mask_pre_prior.reshape(-1)     # [N_anchors]
pred_bboxes_pos = pred_bboxes.reshape(-1, 4)[fg_mask]       # [N_pos, 4]
pred_dist_pos = pred_dists.reshape(-1, 4 * self.reg_max)[fg_mask]  # [N_pos, 64]
assigned_bboxes_pos = assigned_bboxes.reshape(-1, 4)[fg_mask]      # [N_pos, 4]
assigned_ltrb_pos = self._ltrb_from_bbox(assigned_bboxes_pos)      # [N_pos, 4]
```

**关键点**：

- `fg_mask` 是**分配后的正样本掩码**（每个 anchor 是否被某个 GT 选中）；
- `pred_bboxes_pos / pred_dist_pos / assigned_bboxes_pos` 都通过 `fg_mask` 索引，把负样本全部过滤掉；
- `assigned_ltrb_pos` 是**从 bbox 反算的 LTRB**（left, top, right, bottom 到 anchor 中心的距离），供 DFL 使用。

### 5.3 TAL 参数

YOLO-World 完全沿用 v8 参数：

```python
assigner = dict(
    type='TaskAlignedAssigner',
    topk=10,       # ★ 每个 GT 分配 10 个 anchor
    num_classes=num_training_classes,  # 80 或 1203
    alpha=0.5,     # TAL 打分：T = C^0.5 * I^6.0
    beta=6.0,
    eps=1e-9,
)
```

**α=0.5 / β=6.0 的直觉**：分类分数权重低、IoU 权重大——**优先选中 IoU 高的 anchor**，而不是分类分数高的。

---

## 六、txt_masks 加权机制详解

### 6.1 为什么需要 txt_masks

在开放词表预训练时，每个 batch 会随机采样 1203 类作为**候选文本集合**：

```
batch_texts = RandomLoadText(
    all_classes=1203,
    num_neg_samples=(1203, 1203),  # 从 1203 类里采样 1203 个负类
    max_num_samples=80,             # 最多 80 个正类
)
```

一张图里可能只有 5 个正类 GT（比如 3 个 "person"、1 个 "dog"、1 个 "car"），但 1203 类里其它 1198 个类别都是**随机采样的假负类**。如果不加权，CE 会把 1198 个假负类的 loss 也算进去——**梯度被稀释 100+ 倍**。

### 6.2 txt_masks 的构造

```python
# 假设 batch_texts[i] 是第 i 张图的文本集合（80 类）
# gt_classes[i] 是第 i 张图的 GT 类别索引

batch_text_masks = []
for i in range(B):
    mask = torch.zeros(K, dtype=torch.bool)   # K=80
    for c in gt_classes[i]:
        if c in batch_texts[i]:
            mask[c] = True    # 只有 GT 类里出现在 batch_texts 里的才 True
    batch_text_masks.append(mask)
batch_text_masks = torch.stack(batch_text_masks)  # [B, K]
```

**逻辑**：

- 一张图里 GT 是 "person"、"dog"、"car"；
- batch_texts 里包含这 3 个正类 + 77 个随机负类（从 1203 类采样）；
- `txt_masks[i, class_id] = True` 仅当 `class_id ∈ GT ∩ batch_texts`；
- 因此 `txt_masks` 里最多有 3 个 True（GT 类数），其余 77 个 False。

### 6.3 txt_masks 在哪里计算

在 `YOLOWorldHead.loss_by_feat` 内部，从 `batch_input_metas` 中提取：

```python
def loss_by_feat(self, *args, batch_input_metas, **kwargs):
    # batch_input_metas[i]['texts'] 是当前 batch 的文本列表
    # batch_input_metas[i]['batch_gt_instances'] 是 GT
    # 内部构造 batch_text_masks
    ...
```

引用：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="428" :end="450"/>

### 6.4 三种加权方式

同一 `txt_masks` 在 `loss_cls` 里被用在**三处**：

| 位置 | 作用 |
|------|------|
| `flatten_cls_preds * cls_weight` | 屏蔽假负类 logits（数值为 0，log 后为 -∞，但 BCE 内部处理） |
| `y_true * cls_weight` | 屏蔽假负类目标（目标值也置 0） |
| `weight=cls_weight` | `BCEWithLogitsLoss` 的逐元素权重，进一步抑制假负类梯度 |

**为什么三重加权？** 单靠 `weight` 参数可能不够（因为 BCE 对 0 目标仍然会算 log(1-p)），三重加权确保假负类**完全不影响**梯度。

---

## 七、world_size 分布式补偿

### 7.1 问题

分布式训练时，每个 GPU 只计算**自己 batch 的 loss**，然后 `all_reduce` 平均。但如果 `avg_factor` 是**当前 GPU 的 num_pos**，各 GPU 归一化后 loss 尺度不同，平均后会引入偏差。

### 7.2 补偿公式

<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="570" :end="576"/>

```python
# num_pos 是每个 GPU 的正样本数
# world_size 是总 GPU 数
num_pos = fg_mask.sum() * world_size

# 三项 loss 都除以 assigned_scores_sum
loss_cls = loss_cls / assigned_scores_sum
loss_bbox = loss_bbox / assigned_scores_sum
loss_dfl = loss_dfl / assigned_scores_sum

# 最后乘回 world_size（补偿 DDP 平均）
loss_cls = loss_cls * num_imgs * world_size
loss_bbox = loss_bbox * num_imgs * world_size
loss_dfl = loss_dfl * num_imgs * world_size

return dict(
    loss_cls=loss_cls,
    loss_bbox=loss_bbox,
    loss_dfl=loss_dfl,
)
```

**关键点**：

- `num_pos = fg_mask.sum() * world_size`：**假装** num_pos 是全局的，让 `avg_factor` 稳定；
- 最后乘以 `num_imgs * world_size`：把 loss 乘回 DDP 平均因子；
- 最终返回的 loss 与单机训练**尺度一致**。

### 7.3 Ultralytics 版

Ultralytics 用 `torch.distributed.ReduceOp.SUM` 而非 `MEAN`，所以在 loss 里不做显式 `world_size` 补偿，只在 `DDP` wrapper 里做一次全局平均。

---

## 八、优化器与 paramwise_cfg

### 8.1 AdamW 默认配置

<SrcLink file="yoloworld:ailab-cvc/configs/pretrain_v1/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py" :line="158" :end="172"/>

```python
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-3,
        weight_decay=0.05 / 2,   # ★ 0.025，比常规小一半
        batch_size_per_gpu=16,
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # ★ CLIP 学习率 1%
            'logit_scale': dict(weight_decay=0.0),       # ★ 温度不衰减
        },
    ),
    constructor='YOLOWv5OptimizerConstructor',
)
```

### 8.2 paramwise_cfg 逐项解释

| 键 | 值 | 作用 |
|----|----|------|
| `lr` | 2e-3 | 主学习率 |
| `weight_decay` | 0.025 | 常规权重衰减（比默认 0.05 小） |
| `batch_size_per_gpu` | 16 | S 档每卡 16 batch |
| `bias_decay_mult` | 0.0 | **bias 不衰减**（BN bias、Conv bias） |
| `norm_decay_mult` | 0.0 | **norm 层不衰减**（BN 权重） |
| `backbone.text_model.lr_mult` | **0.01** | CLIP 学习率降到主 lr 的 1%（虽然冻结但仍保留微调） |
| `logit_scale.weight_decay` | 0.0 | **温度系数不衰减**（避免 temperature 被拉向 0） |

### 8.3 为什么 CLIP lr_mult=0.01？

CLIP 是**冻结**的（`frozen_modules=['all']`），理论上不应参与优化器更新。但保留 `lr_mult=0.01` 是**历史遗留**——早期实验里 CLIP 也参与微调，后来完全冻结，配置没删干净。**实际效果**：由于 `requires_grad_(False)`，无论 lr 是多少 CLIP 都不会更新。

### 8.4 为什么 logit_scale 不衰减？

`logit_scale` 是温度系数，初值 `-1.0`（BNContrastiveHead）或 `log(1/0.07)≈2.659`（ContrastiveHead）。如果对它做 weight_decay，会把 temperature 拉向 0，导致 `exp(τ)` 变成极小值——**logits 全部压平**，模型无法区分正负类。**因此必须 weight_decay=0**。

---

## 九、Ultralytics 集成版的差异

Ultralytics 的 `WorldTrainer` 直接复用 v8 的 `DetectionLoss`：

<SrcLink file="yoloworld:ultralytics/models/yolo/world/train.py" :line="31" :end="91"/>

```python
class WorldTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        texts = [b['text'] for b in batch['txt']]
        token = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            txt_feats = self.text_model.encode_text(token)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        batch['txt_feats'] = txt_feats
        return batch
```

**关键差异**：

| 维度 | AILab-CVC 版 | Ultralytics 版 |
|------|-------------|----------------|
| 类别数 | 1203（预训练） / 80（微调） | **80 硬编码** |
| txt_masks | ✅ 三重加权 | ❌ 无加权（类别维度 = 正类数） |
| world_size 补偿 | ✅ 显式 | ❌ DDP wrapper 处理 |
| avg_factor | `assigned_scores_sum * num_pos` | `max(total_num_pos, 1)` |
| 归一化 | `loss / assigned_scores_sum` | 无二次归一化 |

**为什么 Ultralytics 版不用 txt_masks？**

因为 Ultralytics 的 `WorldTrainer.preprocess_batch` 里**只对当前 batch 的类做编码**——不是从 1203 类里随机采样。因此类别维度天然等于正类数，没有假负类问题。

**代价**：无法用 Ultralytics 版训练**全 1203 类开放词表**（`nc=min(self.data["nc"], 80)` 硬编码）。要训练全 LVIS 类，必须用 AILab-CVC 原版。

---

## 十、常见调试场景

### 10.1 loss_cls 突然爆炸

**症状**：训练到某个 epoch 后 `loss_cls` 从 2.0 突然飙到 20+。

**可能原因**：
- `txt_masks` 构造错误（比如 GT 类 ID 与 batch_texts 索引不一致）；
- `assigned_scores_sum` 过小（正样本数少，除以小数）；
- `world_size` 补偿错（多卡训练时忘记乘）；
- `logit_scale` 被衰减（温度拉向 0，logits 压平）——**检查 paramwise_cfg**。

**排查**：
```python
# 检查 txt_masks
print(batch_text_masks.sum(dim=-1))  # 每张图应该 >= GT 类数

# 检查 assigned_scores_sum
print(assigned_scores.sum(), fg_mask.sum())  # 应该有合理正样本数
```

### 10.2 loss_bbox 不下降

**症状**：`loss_bbox` 稳定在 0.7 附近不下降。

**可能原因**：
- TAL 分配错误（`fg_mask` 全是 False）——检查 `assigned_scores_sum > 0`；
- `bbox_weight` 太小（assigned_scores 都很小）——检查分类分数是否收敛；
- `pred_bboxes_pos` 数值溢出——检查坐标范围。

### 10.3 loss_dfl 一直是 0

**症状**：训练日志里 `loss_dfl = 0.0`。

**可能原因**：
- `self.use_dfl = False`（配置错误）；
- `pred_dist_pos` 或 `assigned_ltrb_pos` 为 None；
- `reg_max <= 1`（DFL 关闭）。

### 10.4 loss 尺度不一致

**症状**：单机训练和 8 卡训练 loss 数值不同。

**原因**：`avg_factor` 没做 `world_size` 补偿。

**修复**：
```python
num_pos = fg_mask.sum() * world_size  # ← 补偿
loss_cls = loss_cls / assigned_scores_sum
# 最后乘回
loss_cls = loss_cls * num_imgs * world_size
```

### 10.5 Ultralytics 版 loss_cls 偏高

**症状**：Ultralytics 训练的 `loss_cls` 比 AILab-CVC 高 0.3~0.5。

**原因**：Ultralytics 版**没有 txt_masks 加权**，用普通 CE——类别维度更大时 loss 自然更大。**这是设计差异，不是 bug**。

---

## 小结

YOLO-World 的损失函数**在数学上完全沿用 YOLOv8**（三项 CE + IoU + DFL），但工程实现上做了三处适配：

1. **类别维度动态化**：从 `nc=80` 变成 `K=1203`（预训练）或 `K=80`（微调）；
2. **txt_masks 三重加权**：屏蔽 1203 类里的 1198 个随机负类，让 `loss_cls` 只关注真正的 GT 类；
3. **world_size 分布式补偿**：多卡训练时保证 loss 尺度与单机一致。

这三处适配让 YOLO-World 在**不改动 v8 骨干**的前提下，能稳定训练出 47.1 AP 的开放词表检测器。

## 深入阅读

- [损失函数的理论背景](./concepts) · [架构与层速查](./layers) · [训练技巧](./tricks) · [源码走读](./code-walkthrough)
