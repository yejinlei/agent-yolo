# YOLOv6 反向传播推导

::: tip 只想带走三件事
1. 梯度全部交给 PyTorch `autograd`，但 **VFL / GIoU / DFL 都有干净的解析梯度**，可以不看黑盒直接推；
2. **DFL 的梯度最优雅**：左右 bin 加权 CE 等价于"两格软标签分布"的交叉熵，$\partial L/\partial z_k = p_k - \hat y_k$；
3. **TAL 分配不可导**：`@torch.no_grad()` + `detach()`，分配结果当常量，梯度不穿过标签分配器。
:::

## 0. 梯度来源与记号

| | darknet (v1-v4) | PyTorch (v5/v6) |
|--|-----------------|-----------------|
| 梯度计算 | 手写 `delta_xxx` | `autograd` 自动 |
| 梯度存储 | `layer.delta` 数组 | `tensor.grad` |
| 标签分配梯度 | 不涉及 | 显式 `no_grad` + `detach` 切断 |
| 多正样本聚合 | `averages_yolo_deltas` | 自动求和 |

本页记号：

- $s = \sigma(a)$：`pred_score`，head 已过 sigmoid（<SrcLink file="v6:yolov6/models/effidehead.py" :line="85" />），损失用 `F.binary_cross_entropy`（非 with_logits），故 $\frac{\partial s}{\partial a} = s(1-s)$
- $t$：`gt_score`，TAL 归一化对齐分数（常量）；$y \in \{0,1\}$：one-hot label（常量）
- $z_k$：DFL 的 logits，$p_k = \mathrm{softmax}(z)_k$

## 1. VarifocalLoss 的梯度 (`grad-vfl`)

$$L_{vfl} = w(s) \cdot \mathrm{BCE}(s, t), \qquad w = \alpha s^\gamma (1-y) + t\cdot y$$

（实现见 <SrcLink file="v6:yolov6/models/losses/loss.py" :line="201" :end="211" />，$\alpha=0.75,\ \gamma=2.0$）

### 1.1 正样本（$y = 1$）：软目标回归

此时 $w = t$ 与 $s$ 无关，BCE 目标也是 $t$：

$$L = t\left[-t\ln s - (1-t)\ln(1-s)\right]$$

$$\frac{\partial L}{\partial s} = t\cdot\frac{s-t}{s(1-s)} \quad\Longrightarrow\quad \frac{\partial L}{\partial a} = \frac{\partial L}{\partial s}\cdot s(1-s) = t\,(s - t)$$

- 梯度把 $s$ 拉向 $t$（而非 1.0）——**质量感知**：IoU 一般的正样本不会把分类分往 1 硬拽
- 梯度整体按 $t$ 缩放：高质量样本话语权大，低质量样本学得温和

### 1.2 负样本（$y = 0,\ t = 0$）：Focal 式降权

此时 $L = \alpha s^\gamma \cdot (-\ln(1-s))$，**权重本身是 $s$ 的函数**，autograd 会把权重的导数也计入：

$$\frac{\partial L}{\partial s} = \alpha\gamma s^{\gamma-1}(-\ln(1-s)) + \frac{\alpha s^{\gamma}}{1-s}$$

$$\boxed{\ \frac{\partial L}{\partial a} = \alpha s^{\gamma}\Big[\gamma(1-s)\big(-\ln(1-s)\big) + s\Big]\ }$$

- 常见近似（把 weight 当常数）：$\frac{\partial L}{\partial a} \approx \alpha s^{\gamma+1}$
- 无论精确式还是近似式，$s \to 0$（易分负样本）时梯度都以 $s^\gamma$ 速度消失——这正是 VFL 压制海量负样本的机制
- 两式在 $s \to 0$ 时行为一致，$s$ 较大时精确式多出的 $\gamma(1-s)(-\ln(1-s))$ 项让难负样本获得更陡的修正

## 2. GIoU 的梯度 (`grad-giou`)

记预测框 $B$（面积 $A$）、GT 框 $B^\*$（面积 $A^\*$）、交集 $I$、并集 $U = A + A^\* - I$、最小包围框面积 $C$：

$$\mathrm{GIoU} = \frac{I}{U} - \frac{C - U}{C}, \qquad L = 1 - \mathrm{GIoU} = 1 - \frac{I}{U} + 1 - \frac{U}{C}$$

对任意预测框坐标 $x$（GT 与 $C$ 中属 GT 的量为常量）：

$$\frac{\partial L}{\partial x} = -\frac{U \frac{\partial I}{\partial x} - I \frac{\partial U}{\partial x}}{U^2} - \frac{C \frac{\partial U}{\partial x} - U \frac{\partial C}{\partial x}}{C^2}, \qquad \frac{\partial U}{\partial x} = \frac{\partial A}{\partial x} - \frac{\partial I}{\partial x}$$

以左边界 $x_1$ 为例（$I = w_i h_i$，$w_i = [\min(x_2, x_2^\*) - \max(x_1, x_1^\*)]_+$，其余三条边对称）：

$$\frac{\partial I}{\partial x_1} = -h_i \cdot \mathbb{1}\!\left[x_1^\* < x_1 < \min(x_2, x_2^\*)\right], \quad \frac{\partial A}{\partial x_1} = -(y_2 - y_1), \quad \frac{\partial C}{\partial x_1} = -c_h \cdot \mathbb{1}\!\left[x_1 < x_1^\*\right]$$

- 第一项（IoU 部分）只在**发生重叠的边界**上有梯度；第二项（$(C-U)/C$ 部分）在**完全不重叠**时仍提供指向 GT 的方向梯度——这正是 GIoU 修 CIoU/IOU"零梯度死区"的地方
- YOLOv6 里这些全部由 autograd 完成：`IOUloss` 的 GIoU 分支是纯可导 tensor 运算（<SrcLink file="v6:yolov6/utils/figure_iou.py" :line="49" :end="63" />），`loss = 1.0 - iou`（<SrcLink file="v6:yolov6/utils/figure_iou.py" :line="93" />）
- **权重与归一化都是常量**：`bbox_weight = target_scores.sum(-1)` 来自 no_grad 的分配器，`/target_scores_sum` 同理——只是缩放梯度，不新增梯度路径（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="234" :end="241" />）

### 2.1 到 ltrb 距离的链式传导

预测框由 `dist2bbox` 从 anchor point $(p_x, p_y)$ 与 ltrb 距离解出：$x_1 = p_x - l,\ y_1 = p_y - t,\ x_2 = p_x + r,\ y_2 = p_y + b$，因此

$$\frac{\partial L}{\partial l} = -\frac{\partial L}{\partial x_1}, \quad \frac{\partial L}{\partial r} = \frac{\partial L}{\partial x_2}, \quad \frac{\partial L}{\partial t} = -\frac{\partial L}{\partial y_1}, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y_2}$$

即 xyxy 梯度的简单符号翻转，autograd 自动完成（<SrcLink file="v6:yolov6/utils/general.py" :line="32" :end="43" />）。

## 3. DFL 的梯度：左右 bin 加权 CE (`grad-dfl`)

每个方向 17 个 bin（`reg_max=16`），$p_k = \mathrm{softmax}(z)_k$，预测距离 $\hat d = \sum_k p_k \cdot k$（`proj` 投影，<SrcLink file="v6:yolov6/models/losses/loss.py" :line="194" :end="198" />）。目标距离 $t \in [0, 16)$，令 $l = \lfloor t \rfloor$，$r = l + 1$，$w_l = r - t$，$w_r = 1 - w_l$：

$$L_{dfl} = w_l \cdot \mathrm{CE}(z, l) + w_r \cdot \mathrm{CE}(z, r), \qquad \mathrm{CE}(z, m) = -\ln p_m$$

利用 $\frac{\partial \mathrm{CE}(z,m)}{\partial z_k} = p_k - \mathbb{1}[k=m]$：

$$\frac{\partial L_{dfl}}{\partial z_k} = w_l\big(p_k - \mathbb{1}[k=l]\big) + w_r\big(p_k - \mathbb{1}[k=r]\big) = p_k - \hat y_k$$

$$\boxed{\ \hat y_k = w_l\,\mathbb{1}[k=l] + w_r\,\mathbb{1}[k=r], \qquad \sum_k \hat y_k = 1\ }$$

- **DFL = 对"两格软标签分布"的交叉熵**：目标分布把全部质量放在最近的左右整数 bin 上，按距离分配；梯度在 $p = \hat y$ 时归零
- $t$ 恰为整数时 $w_l = 1, w_r = 0$，退化为普通 CE——DFL 是 CE 的连续插值推广
- 对 logits 的梯度简洁到不需要第二行；对 $t$ 的依赖只通过 $\hat y$ 体现（$t$ 本身是常量，`bbox2dist` 还会 `clip(0, reg_max-0.01)`，<SrcLink file="v6:yolov6/utils/general.py" :line="46" :end="52" />）

### 3.1 GIoU 路径穿过分布期望的梯度（use_dfl=True 时）

$\hat d = \sum_k p_k \cdot proj_k$ 是 softmax 的加权和，GIoU 损失也会沿它反传：

$$\frac{\partial \hat d}{\partial z_k} = p_k\,(proj_k - \hat d)$$

即"高于当前期望的 bin 收到正梯度、低于期望的 bin 收到负梯度"，把分布的期望往增减方向推。**但 `configs/yolov6s.py` 默认 `use_dfl=False`**（<SrcLink file="v6:configs/yolov6s.py" :line="32" :end="33" />），此时 `pred_distri` 直接是 ltrb 标量，第 2 节的链式传导直达；DFL 分支 `loss_dfl = pred_dist.sum() * 0.` 不产生任何梯度（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="258" :end="259" />）。

## 4. TAL 分配：不可导的"常量" (`grad-assign`)

```python
@torch.no_grad()
def forward(self, pd_scores, pd_bboxes, ...): ...

# 调用处
self.formal_assigner(
    pred_scores.detach(),
    pred_bboxes.detach() * stride_tensor,
    ...)
```

- `TaskAlignedAssigner.forward` 整体 `@torch.no_grad()`（<SrcLink file="v6:yolov6/assigners/tal_assigner.py" :line="21" :end="28" />），调用处再叠加 `.detach()`（<SrcLink file="v6:yolov6/models/losses/loss.py" :line="96" :end="103" />）——**双重保险**
- 于是 `target_labels / target_bboxes / target_scores / fg_mask` 全是常量：VFL 的软标签 $t$、GIoU 的 `bbox_weight`、DFL 的目标距离，都不携带梯度
- 数学上这不可回避：top-k 选择、`select_highest_overlaps` 的 argmax、`mask_in_gts` 的 0/1 判定都是**离散决策**，本身不可导
- 工程上这是正确近似：类似 EM 算法的 E 步——"先假定分配固定，优化预测；分配下次迭代再更新"。把对齐度量 $s^\alpha \cdot \mathrm{IoU}^\beta$ 当作常量，梯度只负责让**已被选中的样本**预测得更准

## 5. 全流程串起来

```
Input → EfficientRep + RepBiFPAN → EffiDeHead
  → cls_output = sigmoid(cls_logits)      [s]
  → reg_output = reg_logits               [use_dfl=False: 直接 ltrb]
                 softmax → proj → 期望距离 [use_dfl=True]
  → dist2bbox → pred_bboxes
  → TAL (no_grad, detach) → target_scores / target_bboxes / fg_mask   ← 常量
  → loss_cls = VFL(s, t, y)               → 梯度 t(s−t) / αs^γ[...]
  → loss_iou = GIoU(pred, gt) × w        → 梯度链式到 ltrb
  → loss_dfl = 左右 bin 加权 CE           → 梯度 p_k − ŷ_k（默认关）
  → loss = 1.0*loss_cls + 2.5*loss_iou + 0.5*loss_dfl
  → loss.backward() → autograd 累加到 cls_preds / reg_preds 及上游
```

（总损失组装 <SrcLink file="v6:yolov6/models/losses/loss.py" :line="171" :end="177" />）

## 6. 与 v5 的梯度对比

| 维度 | v5 | v6 |
|------|----|----|
| 分类梯度 | BCEWithLogits，target=`iou.detach()` | VFL：正样本 $t(s-t)$，负样本 $\alpha s^\gamma[\gamma(1-s)(-\ln(1-s))+s]$ |
| 回归梯度 | CIoU（autograd） | GIoU（autograd）+ 对齐分数加权 |
| 分布回归 | 无 | DL 梯度 $p_k - \hat y_k$（`use_dfl` 默认关） |
| objectness | 独立 BCE，`detach` 单向隔离 | 无 objectness 分支，质量信息内嵌 VFL 软标签 |
| 标签分配 | `build_targets`（autograd 之外） | TAL `@no_grad` + `detach` 双重切断 |
| 归一化 | ×batch_size | ÷ `target_scores_sum`（软分数和） |

## 与其他页的链接

- [损失函数](/versions/v6/losses)
- [理论 Concept](/versions/v6/concepts)
- [v5 反向传播推导（对比）](/versions/v5/loss-gradient)
