# YOLOv8 反向传播推导

::: tip 只想带走三件事
1. **BCEWithLogits 的合并梯度还是那一行**：\(\frac{\partial L}{\partial z} = \sigma(z) - y\)——v8 没有 pos_weight，但目标换成了软标签 \(q \cdot y\)：正类通道梯度 \(p - q\)，背景通道就是 \(p\)；
2. **DFL 的最优分布期望恰为目标值**：\(p^* = w_l e_{t_l} + w_r e_{t_l+1}\) 时 \(\sum_k k\,p^*_k = t\)——「回归分布」在 DFL 极小点上等价于「回归期望」；实现全交给 `F.cross_entropy` 的 autograd，本页给物理直觉；
3. **TAL 分配不可微且被双重切断**：`@torch.no_grad()` 装饰器 + 调用处 `.detach()`，topk / argmax / scatter 全是离散决策——分配结果作为常量进入损失，梯度只回传网络输出。
:::

## 0. 梯度来源与记号

| | darknet (v1-v4) | PyTorch (v5-v8) |
|--|-----------------|-----------------|
| 梯度计算 | 手写 `delta_xxx` | `autograd` 自动 |
| 标签分配梯度 | 不涉及 | 分配结果在 autograd 之外（v6 显式 `no_grad` + `detach`，v7 离散索引隐式常量，v8 装饰器 + detach 双保险） |

本页记号：

- \(z\)：分支输出的 logit，\(p = \sigma(z)\)；分类与 DFL 共用这套记号
- \(y\)：one-hot 类别目标（TAL scatter 产出，<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="235" :end="238" />）
- \(q\)：软标签质量，分类目标 \(t = q \cdot y\)，\(q \in (0, 1]\) 由 align_metric 归一化而来（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="120" :end="125" />），是 detach 常量
- \(s_k, k = 0..15\)：某条 ltrb 边 16 个 bin 的分布 logit；\(p_k = \mathrm{softmax}(s)_k\)
- \(t\)：目标距离（feature 像素），\(t_l = \lfloor t \rfloor\)、\(t_r = t_l + 1\)、\(w_l = t_l + 1 - t\)、\(w_r = 1 - w_l\)
- CIoU 里 box1=预测、box2=GT（BboxLoss 的调用顺序，<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="38" />）
- \(\Sigma_q = \max(\sum \text{target\_scores},\ 1)\)：box / cls / dfl 三项共同的归一化分母（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="214" />）

## 1. BCEWithLogits 的合并梯度 (`grad-bce`)

v8 的分类是 `nn.BCEWithLogitsLoss(reduction='none')`（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="146" />），sigmoid 与 BCE 一起求导。逐元素损失（目标 \(y \in \{0, 1\}\)）：

$$L = -\Big[y\ln p + (1-y)\ln(1-p)\Big], \qquad p = \sigma(z)$$

$$\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}, \qquad \frac{\partial p}{\partial z} = p(1-p)$$

$$\boxed{\ \frac{\partial L}{\partial z} = \sigma(z) - y\ }$$

- **没有 pos_weight**：v7 的 `cls_pw / obj_pw` 在 v8 检测损失里没有对应物，合并梯度不带 \((1-t)p - w_+ t(1-p)\) 的权重项，干净到 \(p - y\)
- **数值稳定**：logit 与 BCE 融合成 log-sum-exp 一体计算，不会出现先 sigmoid 再取对数的溢出路径
- **目标其实是软的**：把 \(y\) 换成 \(q \cdot y\) 就是 v8 的实际目标——梯度逐通道变成 \(p - q\)（正类）与 \(p\)（背景），完整推导见[第 4 节](#_4-软标签下的分类梯度-grad-soft-label)；损失装配见 [分类 BCE 与软标签](/versions/v8/losses#_4-分类-bce-与软标签-loss-cls)
- **背景通道就是 \(p\)**：负样本海量（8400 位置只有少数正样本），分类头的主要梯度是把无目标位置的分数压向 0——v8 无 objectness 分支，这份工作由 cls 兼职

## 2. CIoU 的回归梯度 (`grad-ciou`)

回归损失（逐正样本加权，<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="39" />）：

$$L_{iou} = \frac{\sum_i (1 - \mathrm{CIoU}_i)\, q_i}{\Sigma_q}$$

[CIoU loss](/glossary#ciou-loss) 的实现不在 loss.py——`from .metrics import bbox_iou`（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="7" />），定义整段在 <SrcLink file="v8:ultralytics/yolo/utils/metrics.py" :line="75" :end="129" />：

$$\mathrm{CIoU} = \mathrm{IoU} - \frac{\rho^2}{c^2} - \alpha v, \qquad v = \frac{4}{\pi^2}\left(\arctan\frac{w_2}{h_2} - \arctan\frac{w_1}{h_1}\right)^2, \qquad \alpha = \frac{v}{v - \mathrm{IoU} + 1}$$

（\(c^2\) 最小包围框对角线平方 <SrcLink file="v8:ultralytics/yolo/utils/metrics.py" :line="118" :end="120" />、\(v\) 与 \(\alpha\) <SrcLink file="v8:ultralytics/yolo/utils/metrics.py" :line="122" :end="125" />）对预测框参数 \(\theta\)（GT 侧全是常量）：

$$\frac{\partial L}{\partial \theta} \propto -\frac{\partial \mathrm{IoU}}{\partial \theta} + \frac{\partial}{\partial \theta}\frac{\rho^2}{c^2} + \alpha\frac{\partial v}{\partial \theta}$$

- **\(\alpha\) 在 `torch.no_grad()` 下计算**（<SrcLink file="v8:ultralytics/yolo/utils/metrics.py" :line="123" :end="124" />）→ 是常量权重，不产生 \(\partial \alpha / \partial \theta\) 项；IoU 与 \(\rho^2/c^2\) 是纯 tensor 运算，**实现全部走 autograd**，页面只给一阶物理直觉（与 v7 同一口径）
- 真正值得手写的只有 \(v\) 的派生项。记 \(\theta_i = \arctan\frac{w_i}{h_i}\)，则 \(v = \frac{4}{\pi^2}(\theta_2 - \theta_1)^2\)，只有 \(\theta_1\) 依赖预测：

$$\frac{\partial v}{\partial w_1} = \frac{8}{\pi^2}(\theta_1 - \theta_2)\cdot\frac{h_1}{w_1^2 + h_1^2}, \qquad \frac{\partial v}{\partial h_1} = \frac{8}{\pi^2}(\theta_2 - \theta_1)\cdot\frac{w_1}{w_1^2 + h_1^2}$$

$$\boxed{\ \alpha\,\partial v\ \text{把预测宽高比拉向 GT 宽高比，与面积、位置无关}\ }$$

### 2.1 链式传导到分布参数

预测框不是 sigmoid 形变，而是 **ltrb 期望距离的线性函数**（`dist2bbox`，<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="261" :end="270" />）：

$$b_1\_{x_1} = a_x - d_l, \qquad b_1\_{x_2} = a_x + d_r, \qquad d_l = \sum_{k=0}^{15} k\,p_k$$

- \(\partial \mathrm{box} / \partial d = \pm 1\)——**没有 v5/v7 的 \(2\sigma(1-\sigma)\) 自我限幅**；再往里一层，\(d\) 是 softmax 期望，对 \(s_k\) 的梯度经 softmax 雅可比传播（见[第 3 节](#_3-dfl-的期望值梯度-grad-dfl)）
- **权重即话语权**：\(\times q_i / \Sigma_q\)——对齐质量高的正样本对回归梯度贡献大（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="37" :end="39" />）；匹配在像素域、回归在 feature 域，只差一个常量 stride（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="222" />）

## 3. DFL 的期望值梯度 (`grad-dfl`)

[DFL](/glossary#dfl-distribution-focal-loss) 对一条边取相邻两 bin 的 [softmax](/glossary#softmax) 交叉熵插值（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="59" :end="60" />）。设分布 \(p = \mathrm{softmax}(z)\)，目标 \(t \in [0, 14.99]\)：

$$L = w_l\,\mathrm{CE}(z, t_l) + w_r\,\mathrm{CE}(z, t_r), \qquad t_l = \lfloor t \rfloor,\ \ t_r = t_l + 1,\ \ w_l = t_l + 1 - t,\ \ w_r = 1 - w_l$$

交叉熵对 logit 的导数是标准结果（\(\mathrm{CE}(z, j) = -z_j + \log\sum_k e^{z_k}\)）：

$$\frac{\partial\, \mathrm{CE}(z, j)}{\partial z_k} = p_k - \delta_{[k=j]}$$

于是：

$$\boxed{\ \frac{\partial L}{\partial z_k} = w_l\,(p_k - \delta_{[k=t_l]}) + w_r\,(p_k - \delta_{[k=t_r]})\ }$$

由 \(w_l + w_r = 1\) 可化简为 \(\frac{\partial L}{\partial z_k} = p_k - w_l\,\delta_{[k=t_l]} - w_r\,\delta_{[k=t_r]}\)——梯度把分布质量往相邻两根「针」上押，押的比例就是插值权重。

- **最优解与期望恒等式**：令梯度为零，得 \(p^* = w_l\, e_{t_l} + w_r\, e_{t_r}\)，此时期望

$$\sum_k k\,p^*_k = w_l\, t_l + w_r\,(t_l + 1) = t_l + w_r = t$$

  **分布回归等价于回归期望**：DFL 的极小点恰好是「期望值等于目标」的分布——均值被钉死，方差完全自由；这也解释了为什么推理端可以用 softmax 期望积分无损解码（[DFL 距离分布框表示](/versions/v8/concepts#_4-dfl-距离分布框表示-concept-dfl)）
- **实现口径**：\(F.cross\_entropy\) 自带 log-sum-exp，梯度交给 autograd；`.mean(-1)` 对 ltrb 四条边平均、再 \(\times q\)、\(\div \Sigma_q\)（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="44" :end="45" />）
- **边界安全**：target 由 `bbox2dist` clamp 到 \([0, 14.99]\)（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="276" />）→ \(t_l \le 14 < 15\)，插值恒有定义

## 4. 软标签下的分类梯度 (`grad-soft-label`)

v8 的分类目标是软标签 \(t = q \cdot y\)（\(y\) one-hot，\(q \in (0, 1]\)）。由[第 1 节](#_1-bcewithlogits-的合并梯度-grad-bce)的一般形式 \(\partial L / \partial z = p - t\)：

$$\boxed{\ \frac{\partial L}{\partial z} = \sigma(z) - q\,y = \underbrace{q\,y\,\big(\sigma(z) - y\big)}_{\text{质量加权目标项}} + \underbrace{(1 - q\,y)\,\sigma(z)}_{\text{背景压制项}}\ }$$

- **按通道拆开**：正类通道（\(y=1\)）梯度 \(p - q\)——logit 的平衡点从 1 降到 q，**分类分数的天花板就是对齐质量**；背景通道（\(y=0\)）梯度 \(p\)，纯压制
- **质量加权视角**：把上式第一项单独看，正是「对 hard one-hot 的 BCE 逐项乘 q」的梯度 \(q\,(\sigma(z) - y)\)——[VarifocalLoss](/glossary#varifocalloss-vfl) 的正样本走的就是这一式（weight = gt_score·label，<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="20" />）。v8 没有采用它（VFL 留档，见 [VarifocalLoss 存而未用](/versions/v8/losses#_6-varifocalloss-存而未用-loss-vfl)），而是把 q 写进目标本身：VFL 缩放推力、零点不动；v8 软标签直接移动零点
- **q 是常量**：q 由 TAL 在 `@torch.no_grad()` 里算出（align_metric 归一化，<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="120" :end="125" />），梯度不会经目标「作弊」回流——与 v7 把 \(\overline{\mathrm{IoU}}\) detach 后写进 obj 目标是同一思想，只是搬到了 cls 分支
- **全局归一化再稀释**：损失 \(\div \Sigma_q\)（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="214" />，\(\Sigma_q\) 为 detach 常量）——目标项与背景项一起被 \(\Sigma_q\) 缩放，相当于按 batch 内「总质量」而非「样本数」平均

## 5. TAL 分配的不可微边界 (`grad-assign`)

```python
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
```

（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="210" :end="212" />）

- **双重切断**：assigner forward 整体 `@torch.no_grad()`（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="82" :end="83" />）+ 调用处预测侧 `.detach()`——v6 显式切断、v7 离散索引隐式常量，v8 两道锁一起上
- **离散决策清单**：`torch.topk` 选候选（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="180" />）、`argmax` 冲突消解（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="45" />、<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="53" />）、`scatter_add_` 计数与 [one-hot](/glossary#one-hot) 写入（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="191" />、<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="238" />）、`masked_fill_` 过滤（<SrcLink file="v8:ultralytics/yolo/utils/tal.py" :line="194" />）、布尔掩码——这些操作没有导数可言
- **常量进入损失**：`target_bboxes / target_scores / fg_mask` 在三分量里都是常量，autograd 图只包含网络输出路径——**梯度只对 `pred_distri / pred_scores` 回传**；TAL 分配进入损失计算的装配见 [TAL 分配进入损失计算](/versions/v8/losses#_5-tal-分配进入损失计算-loss-tal)
- **梯度的实际通路**：BboxLoss 的 `pred_bboxes[fg_mask]` 布尔索引等价 gather（<SrcLink file="v8:ultralytics/yolo/utils/loss.py" :line="38" :end="44" />），box/dfl 梯度只流向被选中的正样本；cls BCE 铺满全部 8400 位置（<SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="218" />），负样本位置经背景项收到 \(p\) 的压制梯度
- **通行做法**：类 EM 的 E 步——先固定分配优化预测，分配下个迭代再更新；v6 的 TAL、v7 的 SimOTA、v8 的 TAL 三代分配器都把分配挡在 autograd 之外，这是检测标签分配的行业惯例

## 6. 全流程串起来

```
Input → C2f 骨干 → PAN 聚合 → Detect 头（8400 位置 = 6400+1600+400，出口 64+nc 维）
  → split → pred_distri (8400×64) / pred_scores (8400×nc)
  → TAL 分配（@no_grad + detach → 常量）
      mask_in_gts × topk(align_metric = score^0.5 · CIoU^6, topk=10) × mask_gt
      → select_highest_overlaps 冲突消解 → one-hot → × 归一化 q
  → lcls = BCE(z_cls, q·y).sum / Σq
        梯度 p − qy = qy(σ−y) + (1−qy)σ
  → liou = (1 − CIoU)·q，/ Σq
        梯度 −∂IoU + ∂(ρ²/c²) + α∂v，链式 ±1 到期望距离（线性，无 sigmoid 限幅）
  → ldfl = [wl·CE(z,tl) + wr·CE(z,tl+1)].mean(4边)·q，/ Σq
        梯度 wl(p−δtl) + wr(p−δtl+1)，最优分布的期望恰为 t
  → loss = 7.5·lbox + 0.5·lcls + 1.5·ldfl → × batch_size
  → loss.backward() → autograd 累加回 Detect 头 / 骨干
```

（总装 <SrcLink file="v8:ultralytics/yolo/v8/detect/train.py" :line="186" :end="230" />）

## 7. 与 v6/v7 的梯度对比

| 维度 | v6 | v7 | v8（本快照） |
|------|----|----|--------------|
| 分类梯度 | VFL：正 \(t(s-t)\)、负 \(\alpha s^\gamma\) | BCEWithLogits，\(p - t\) | BCE 软标签：\(p - qy\)（正通道 \(p-q\)，背景 \(p\)） |
| 回归梯度 | GIoU + 对齐分数加权 | CIoU，链式 ×\(2\sigma(1-\sigma)\) | CIoU ×q；链式 ±1（期望距离线性），分布侧经 softmax 雅可比 |
| DFL | 可选 | 无 | 默认；\(w_l(p_k-\delta_{t_l}) + w_r(p_k-\delta_{t_r})\)，最优期望 = t |
| objectness | 无 | \(p - t\)，\(t = \overline{\mathrm{IoU}}\) 经 detach | 无——cls 软标签兼职 |
| 分配梯度 | TAL `@no_grad` + `detach` | 离散索引隐式切断 | `@no_grad` 装饰器 + 调用处 `detach` 双保险 |
| 归一化 | ÷ target_scores_sum | × batch_size | ÷ Σq（三项统一）再 × batch_size |

## 与其他页的链接

- [损失函数](/versions/v8/losses)
- [理论 Concept](/versions/v8/concepts)
- [v7 反向传播推导（对比）](/versions/v7/loss-gradient)
- [v6 反向传播推导（对比）](/versions/v6/loss-gradient)
