# YOLOv7 反向传播推导

::: tip 只想带走三件事
1. **BCEWithLogits 的合并梯度干净到一行**：sigmoid 与 BCE 一起求导，$\frac{\partial L}{\partial z} = p - t$——v7 的 cls/obj 全走这条式；
2. **obj 目标是 detach 的 IoU 比例**且快照里 $gr \equiv 1.0$：正样本把 obj logit 拉向 $\overline{\mathrm{IoU}}$（常量），梯度不会经目标"作弊"回流回归分支；
3. **标签分配不可导**：SimOTA 的 topk/argmin/argmax 是离散决策，梯度只经被选中的正样本回传；aux 头三项各 ×0.25，共享层里约占两成话语权。
:::

## 0. 梯度来源与记号

| | darknet (v1-v4) | PyTorch (v5/v6/v7) |
|--|-----------------|---------------------|
| 梯度计算 | 手写 `delta_xxx` | `autograd` 自动 |
| 梯度存储 | `layer.delta` 数组 | `tensor.grad` |
| 标签分配梯度 | 不涉及 | 分配结果在 autograd 之外（v6 显式 `no_grad` + `detach`，v7 离散索引天然不可导） |
| 多正样本聚合 | `averages_yolo_deltas` | 自动求和 / mean |

本页记号：

- $z$：head 输出的 logit（未过 sigmoid），$p = \sigma(z)$
- $t$：BCE 目标——cls 用 $c_p = 1 - 0.5\varepsilon$ / $c_n = 0.5\varepsilon$（<SrcLink file="v7:utils/loss.py" :line="11" :end="13" />，快照 $\varepsilon = 0$）；obj 用 $(1-gr) + gr\cdot\overline{\mathrm{IoU}}$（<SrcLink file="v7:utils/loss.py" :line="472" />）
- 解码：$x = 2\sigma(t_x) - 0.5$、$y = 2\sigma(t_y) - 0.5$、$w = a_w(2\sigma(t_w))^2$、$h = a_h(2\sigma(t_h))^2$（网格 / anchor 单位，<SrcLink file="v7:utils/loss.py" :line="465" :end="466" />）
- $(w_1, h_1)$：预测框宽高；$(w_2, h_2)$：GT 框宽高（CIoU 里 box1=预测、box2=GT）

## 1. BCEWithLogits：sigmoid 与 BCE 的合并梯度 (`grad-bce`)

v7 的分类与 objectness 都是 `nn.BCEWithLogitsLoss(pos_weight=...)`（<SrcLink file="v7:utils/loss.py" :line="430" :end="431" />），目标填入见 <SrcLink file="v7:utils/loss.py" :line="476" :end="479" />。逐元素损失（含正权重 $w_+$）：

$$L = -\Big[w_+\,t\ln p + (1-t)\ln(1-p)\Big], \qquad p = \sigma(z)$$

对 logit 求导，链式一乘就是著名的"p 减 t"：

$$\frac{\partial L}{\partial p} = \frac{(1-t)p - w_+ t(1-p)}{p(1-p)}, \qquad \frac{\partial p}{\partial z} = p(1-p)$$

$$\boxed{\ \frac{\partial L}{\partial z} = (1-t)\,p - w_+\,t\,(1-p) \ \xrightarrow{\ w_+ = 1\ }\ p - t\ }$$

快照 `cls_pw = obj_pw = 1.0`（<SrcLink file="v7:data/hyp.scratch.p5.yaml" :line="10" :end="12" />），于是处处 $\frac{\partial L}{\partial z} = p - t$：

- 正样本（$t = c_p = 1$）：梯度 $p - 1 < 0$，把 logit 往上推；负样本（$t = c_n = 0$）：梯度 $p > 0$，往下推；$p = t$ 时归零
- **label smoothing 只改 $t$ 不改式子**：$\varepsilon > 0$ 时正目标变 $1 - 0.5\varepsilon$，梯度零点随之下移，不会把正样本 logit 推到无穷
- 对比 v6：v6 头先过 sigmoid、用 `F.binary_cross_entropy`（非 with_logits），要自己乘 $s(1-s)$；v7 保持 v5 口径的 with-logits，数值稳定且梯度就是 $p - t$
- `FocalLoss` 包装（<SrcLink file="v7:utils/loss.py" :line="121" />）会在 $p_t^\gamma$ 上再乘调制因子并计入权重导数，但 `fl_gamma: 0.0` 默认关（<SrcLink file="v7:utils/loss.py" :line="437" :end="439" />）——本页按裸 BCE 推

## 2. CIoU 的梯度 (`grad-ciou`)

回归损失 $L_{box} = 1 - \mathrm{CIoU}$（<SrcLink file="v7:utils/loss.py" :line="468" :end="469" />）：

$$L_{box} = 1 - \mathrm{IoU} + \frac{\rho^2}{c^2} + \alpha v, \qquad v = \frac{4}{\pi^2}\left(\arctan\frac{w_2}{h_2} - \arctan\frac{w_1}{h_1}\right)^2, \qquad \alpha = \frac{v}{v - \mathrm{IoU} + 1}$$

（实现 <SrcLink file="v7:utils/general.py" :line="378" :end="382" />；$c^2$ 为最小包围框对角线、$\rho^2$ 为中心距）对任意预测框参数 $\theta$（GT 侧全是常量）：

$$\frac{\partial L_{box}}{\partial \theta} = -\frac{\partial \mathrm{IoU}}{\partial \theta} + \frac{\partial}{\partial \theta}\frac{\rho^2}{c^2} + \alpha\frac{\partial v}{\partial \theta}$$

- 前两项（IoU 与 DIoU 部分）由 autograd 对纯 tensor 运算直接求出，无需手推
- **$\alpha$ 在 `torch.no_grad()` 下计算**（<SrcLink file="v7:utils/general.py" :line="380" :end="381" />）→ $\alpha$ 是常量权重，不产生 $\frac{\partial \alpha}{\partial \theta}$ 项；真正值得手写的是 $v$ 的派生项

记 $\theta_i = \arctan\frac{w_i}{h_i}$（$i=1$ 预测、$i=2$ GT），则 $v = \frac{4}{\pi^2}(\theta_2 - \theta_1)^2$，只有 $\theta_1$ 依赖预测：

$$\frac{\partial v}{\partial w_1} = \frac{8}{\pi^2}(\theta_1 - \theta_2)\cdot\frac{h_1}{w_1^2 + h_1^2}, \qquad \frac{\partial v}{\partial h_1} = \frac{8}{\pi^2}(\theta_2 - \theta_1)\cdot\frac{w_1}{w_1^2 + h_1^2}$$

$$\boxed{\ \alpha\frac{\partial v}{\partial \theta}\ \text{把宽高比}\ \frac{w_1}{h_1}\ \text{拉向}\ \frac{w_2}{h_2}\ \text{，与面积、位置无关}\ }$$

- 直觉：$v$ 只惩罚"长宽比不一致"，预测框比 GT 更扁或更瘦都会吃到 $\alpha v$ 的修正；两框长宽比一致时 $\theta_1 = \theta_2$、该项梯度归零
- $\alpha = \frac{v}{v - \mathrm{IoU} + 1}$ 的作用是让这项随 IoU 自适应：重叠越低，$\alpha$ 越大，长宽比修正越激进

### 2.1 链式传导到解码参数

预测框不是直接回归 $(x, y, w, h)$，而是 sigmoid 形变（<SrcLink file="v7:utils/loss.py" :line="465" :end="466" />）：

$$x = 2\sigma(t_x) - 0.5, \qquad w = 4a_w\,\sigma(t_w)^2$$

$$\frac{\partial x}{\partial t_x} = 2\sigma(t_x)\big(1 - \sigma(t_x)\big), \qquad \frac{\partial w}{\partial t_w} = 8a_w\,\sigma(t_w)\big(1 - \sigma(t_w)\big)$$

于是 $t_x$ 的最终梯度 = CIoU 对 $x$ 的梯度 × $2\sigma(1-\sigma)$，其余三个参数同构：

- sigmoid 因子 $\sigma(1-\sigma) \le 0.25$ 天然压缩梯度：预测饱和（$\sigma \to 0/1$）时梯度趋零——这是 sigmoid 解码的自我限幅
- `ComputeLossOTA` 里做匹配的框是**像素域**（解码后 `× stride`，<SrcLink file="v7:utils/loss.py" :line="692" :end="694" />），但那只喂给 cost（常量路径）；损失本体里的框仍是网格/anchor 单位
- `lbox = (1 - iou).mean()` 对 $n$ 个正样本取均值（<SrcLink file="v7:utils/loss.py" :line="469" />），再乘 `hyp['box'] = 0.05`（<SrcLink file="v7:utils/loss.py" :line="492" />）——回归梯度整体再被压 20 倍量级

## 3. obj 软目标下的梯度 (`grad-obj`)

objectness 对全图每个 (层, anchor, 网格) 位置做 BCE（<SrcLink file="v7:utils/loss.py" :line="485" :end="486" />），目标矩阵只在正样本位置写软值：

$$t_{obj} = \begin{cases} (1 - gr) + gr\cdot\overline{\mathrm{IoU}} & \text{正样本位置} \\ 0 & \text{其余位置} \end{cases}$$

（<SrcLink file="v7:utils/loss.py" :line="472" />；$\overline{\mathrm{IoU}} = \mathrm{CIoU}.\mathrm{detach}().\mathrm{clamp}(0)$）由第 1 节：

$$\frac{\partial L_{obj}}{\partial z_{obj}} = p - t_{obj}$$

- **负海量位置**：$t = 0$，梯度就是 $p$——obj 头的主要工作是把无目标位置压向 0
- **正样本位置**：$t = (1-gr) + gr\cdot\overline{\mathrm{IoU}}$，快照 `model.gr = 1.0` 写死（<SrcLink file="v7:train.py" :line="288" />，退火代码被注释 <SrcLink file="v7:train.py" :line="343" />）→ $t = \overline{\mathrm{IoU}}$，**obj logit 被拉向 IoU 比例而非 1.0**：定位越准、obj 分越高，两者语义对齐
- **`detach` 的关键作用**：$t$ 是常量，obj 损失的梯度**不会**经 $\overline{\mathrm{IoU}}$ 流回回归分支——否则目标自己会跟着预测动，出现"抬高 IoU 目标来降 obj 损失"的作弊路径；回归分支只从 `lbox` 收梯度
- **权重只是缩放**：`balance[i] ∈ [4.0, 1.0, 0.4]`（<SrcLink file="v7:utils/loss.py" :line="442" />）与 `hyp['obj'] = 0.7`（<SrcLink file="v7:utils/loss.py" :line="493" />）不改变 $p - t$ 的方向与零点，只改 P3/P4/P5 各层的步长

## 4. aux 头 ×0.25 的梯度占比 (`grad-aux`)

`ComputeLossAuxOTA` 对 aux 头的三项损失各乘 0.25（<SrcLink file="v7:utils/loss.py" :line="1259" />、<SrcLink file="v7:utils/loss.py" :line="1269" />、<SrcLink file="v7:utils/loss.py" :line="1271" :end="1273" />）。记 lead 头对参数 $\theta$ 的梯度为 $g$，aux 头同构梯度为 $g_{aux}$：

$$\frac{\partial L}{\partial \theta_{aux}} = 0.25\, g_{aux}, \qquad \frac{\partial L}{\partial \theta_{shared}} = g_{lead} + 0.25\, g_{aux}$$

- **aux 头自身**：收到的梯度只有同构 lead 头的四分之一——它是"学个粗略定位"的辅助塔，不会跟 lead 抢话语权
- **共享骨干**：若两头梯度量级相当，aux 路径约占 $\frac{0.25}{1 + 0.25} = 20\%$ 的更新量——coarse 目标（`find_5_positive`，`g = 1.0`）以小权重持续给骨干注入"宽视野"信号，防 lead 过拟合细粒度分配
- **0.25 三处一致**：lbox / lcls / lobj 全部同比例，aux 的整体"音量"被统一压低，不改变三项之间的平衡
- aux 头只在训练期存在，推理只输出 lead 头——0.25 买到的是正则化，不花推理成本

## 5. 标签分配：cost 匹配不可导 (`grad-assign`)

```python
fg_pred = pi[b, a, gj, gi]                # 注意：没有 .detach()
...
cost = (
    pair_wise_cls_loss
    + 3.0 * pair_wise_iou_loss
)
...
_, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
...
_, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
```

- `find_3_positive` / `find_5_positive` 产出的是**整数索引** (b, a, gj, gi)（布尔掩码 → `t[j]` → `.long()` → `clamp_`，<SrcLink file="v7:utils/loss.py" :line="1646" :end="1697" />）；SimOTA 匹配的 `topk` / `torch.min(..., dim=0)` / `argmax` 也全是**离散决策**（<SrcLink file="v7:utils/loss.py" :line="744" :end="757" />）——这些操作本身没有导数可言
- **v7 没有显式 `detach`**（候选预测 `fg_pred` 是裸的 <SrcLink file="v7:utils/loss.py" :line="687" />），但 cost 计算出的量只被用来生成整数索引与常量：`matching_matrix` 是 `zeros_like` 上的 1.0 写入（<SrcLink file="v7:utils/loss.py" :line="742" :end="748" />），cost 张量随即 `del` 释放（<SrcLink file="v7:utils/loss.py" :line="735" />、<SrcLink file="v7:utils/loss.py" :line="750" />）——**autograd 图从未与最终 loss 连通**，效果等价于常量
- 类比 EM 算法的 E 步："先假定分配固定，优化预测；分配下一迭代再更新"。cost 里的 $\sigma(p_{cls})\cdot\sigma(p_{obj})$ 与 IoU 被当作常量，梯度只负责让**已被选中的样本**预测得更准
- **梯度的实际通路**：`ps = pi[b, a, gj, gi]` 的 gather（<SrcLink file="v7:utils/loss.py" :line="462" />）——`lbox` / `lcls` 的梯度只流向被选中的正样本位置；`lobj` 的 BCE 铺满全图（<SrcLink file="v7:utils/loss.py" :line="485" :end="486" />），负样本位置只经 obj 收到 $p$ 的压制梯度
- 与 v6 的对照：v6 是 `@torch.no_grad()` + `.detach()` 的显式双重切断；v7 是"离散索引 + 用后即弃"的隐式切断——殊途同归，分配结果都是常量

## 6. 全流程串起来

```
Input → E-ELAN / MPConv 骨干 → RepConv 头（lead p[0:3]，aux p[3:6] 可选）
  → 标签分配（离散索引，常量）
      ComputeLoss:      anchor_t 宽高比 + g=0.5 五点偏移
      ComputeLossOTA:   find_3_positive 预筛 → cost = BCEcls + 3.0×(−log IoU) → dynamic-k
      ComputeLossAuxOTA: build_targets(g=0.5)→lead，build_targets2(g=1.0)→aux，均用 lead 预测匹配
  → lbox = (1 − CIoU).mean()             → 梯度 −∂IoU + ∂(ρ²/c²) + α∂v，链式 ×2σ(1−σ) 到 (tx ty tw th)
  → lobj = BCEobj(z_obj, (1−gr)+gr·Īou) → 梯度 p − t（t 为 detach 常量，gr=1.0）
  → lcls = BCEcls(z_cls, cp/cn)          → 梯度 p − t
  → [aux 三项各 ×0.25]
  → loss = 0.05·lbox + 0.7·lobj + 0.3·lcls，× batch_size
  → loss.backward() → autograd 累加到头、骨干（共享层 g_lead + 0.25·g_aux）
```

（总损失组装 <SrcLink file="v7:utils/loss.py" :line="492" :end="498" />；aux 版 <SrcLink file="v7:utils/loss.py" :line="1279" :end="1285" />）

## 7. 与 v5/v6 的梯度对比

| 维度 | v5 | v6 | v7（本快照） |
|------|----|----|--------------|
| 分类梯度 | BCEWithLogits，$p - t$ | VFL：正 $t(s-t)$、负 $\alpha s^\gamma[\cdots]$ | BCEWithLogits，$p - t$（cp/cn，smoothing 默认关） |
| 回归梯度 | CIoU（autograd + $\alpha v$ 派生项） | GIoU（autograd）+ 对齐分数加权 | CIoU，同 v5；链式 × $2\sigma(1-\sigma)$ 到 $(t_x, t_y, t_w, t_h)$ |
| objectness 梯度 | $p - t$，$t$ 经 `detach` | 无 obj 分支 | $p - t$，$t = \overline{\mathrm{IoU}}$ 经 `detach`，$gr \equiv 1.0$ |
| 标签分配 | 静态 `build_targets`（autograd 之外） | TAL `@no_grad` + `detach` 双重切断 | 离散索引隐式切断（topk/argmin/argmax 不可导） |
| 多头梯度 | 无 | 无 | 共享层 $g_{lead} + 0.25\,g_{aux}$，aux 约占两成 |
| 归一化 | × batch_size | ÷ `target_scores_sum` | × batch_size |

## 与其他页的链接

- [损失函数](/versions/v7/losses)
- [理论 Concept](/versions/v7/concepts)
- [v6 反向传播推导（对比）](/versions/v6/loss-gradient)
