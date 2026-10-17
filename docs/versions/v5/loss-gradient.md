# YOLOv5 反向传播推导

::: tip 只想带走三件事
1. v5 的梯度**不再手写**，全部交给 PyTorch `autograd`；
2. CIoU 的梯度路径与 v4 相同，但**自动求导消除了 v4 的 1/(1+ar²) 因子缺失 bug**；
3. 多正样本的 delta 平均也交给 autograd，无需手动 `averages_yolo_deltas`。
:::

## 0. 与 darknet 的本质区别

| | darknet (v1-v4) | PyTorch (v5) |
|--|-----------------|--------------|
| 梯度计算 | 手写 `delta_xxx` | `autograd` 自动 |
| 梯度存储 | `layer.delta` 数组 | `tensor.grad` |
| 反向传播 | `backward_xxx` | `loss.backward()` |
| 多正样本平均 | `averages_yolo_deltas` | 自动求和 |

- v4 的 CIoU 梯度在 `box.c` 中手推了 200 多行，v5 只需 `bbox_iou` 返回可导 tensor
- v4 的 `delta_yolo_box` 需要手动处理 `(2−truth.w·truth.h)` 小框权重，v5 直接用 CIoU 的数学性质

## 1. CIoU 的自动求导路径

```python
iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
lbox += (1.0 - iou).mean()
```

`bbox_iou` 内部：

```python
v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
alpha = v / (v - iou + (1 + eps))
return iou - (rho2 / c2 + v * alpha)
```

- `v` 包含 `atan` 和 `pow`，`alpha` 包含除法——全部可导
- `autograd` 会自动计算 $\frac{\partial (1 - \text{CIoU})}{\partial (x, y, w, h)}$

## 2. 从 pbox 到 raw output 的梯度链

```
raw output (tx, ty, tw, th)
  → sigmoid → pxy = sigmoid(tx)*2 - 0.5
  → pwh = (sigmoid(tw)*2)^2 * anchor
  → pbox = [pxy, pwh]
  → bbox_iou(pbox, tbox, CIoU=True)
  → iou
  → lbox = (1 - iou).mean()
```

每一环都是可导的 PyTorch 操作，autograd 自动链式求导。

## 3. Objectness 的梯度隔离

```python
tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0)
```

- `iou.detach()` 切断梯度：box 回归的梯度不会通过 objectness 反向传播
- 但 objectness 的 BCE 梯度会正常传播到 `pi[..., 4]`
- 这是"单向依赖"：box 好 → obj 高，但 obj 高不会反过来影响 box

## 4. 多正样本的梯度聚合

v5 的 `build_targets` 返回多个正样本位置：

```python
for i, pi in enumerate(p):
    b, a, gj, gi = indices[i]
    ps = pi[b, a, gj, gi]
    ...
```

- 每个正样本独立计算 loss，autograd 自动求和
- 无需 v4 的 `averages_yolo_deltas`（手动平均 delta）
- 梯度自动累加到对应的 `pi[b, a, gj, gi]` 位置

## 5. 全流程串起来

```
Input image
  → Focus → Conv → C3 → ... → Detect
  → raw output [bs, 3, 85, ny, nx]
  → sigmoid → decode → pbox
  → bbox_iou → CIoU → lbox
  → BCEobj → lobj
  → BCEcls → lcls
  → loss = lbox + lobj + lcls
  → loss.backward()
  → autograd 自动计算所有参数的梯度
```

## 6. 与 v4 的梯度对比

| 维度 | v4 | v5 |
|------|----|----|
| CIoU 梯度 | 手写 200 行，缺 1/(1+ar²) | autograd，完整 |
| 多正样本平均 | `averages_yolo_deltas` | autograd 自动 |
| 坐标编码梯度 | 手写 scale_x_y | 自动 |
| 小框权重 | `(2−truth.w·truth.h)` | CIoU 隐含 |
| 代码量 | ~500 行 | ~50 行 |

## 与其他页的链接

- [损失函数](/versions/v5/losses)
- [理论 Concept](/versions/v5/concepts)
- [v4 反向传播推导（对比）](/versions/v4/loss-gradient)
