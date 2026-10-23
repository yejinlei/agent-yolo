# ultralytics/nn/modules/moe/loss.py

MoE 平衡损失与 Z-loss 模块：**393 行**，是 Master 训练管线中 MoE 相关所有辅助损失的实现。核心类 **MoELoss**（[L97](#L97)）继承 `nn.Module`，被 `DetectionModel.init_criterion` 挂在 v8DetectionLoss 之外作为附加损失。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/moe/loss.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/moe/loss.py" />

**阅读路线**：

- [L16 gshard_balance_loss](#L16)：**GShard 风格平衡损失**——`CE(dispatching, uniform) * CE(uniform, dispatching)`，两个交叉熵的乘积，鼓励路由权重在专家上均匀分布
- [L29 weighted_gshard_balance_loss](#L29)：按 token 权重加权的 GShard
- [L54 differentiable_balance_loss](#L54)：可微版本的平衡损失——用 top-k 掩码替代 top-k 选择，梯度可以反向传播
- **MoELoss**（核心类）[L97-L393](#L97)：
  - [L110-L155 __init__](#L110-L155)：`balance_loss_coeff=0.01 / z_loss_coeff=0.001 / ddp_reduce=True`；DDP 环境下通过 `all_reduce_mean` 做跨卡同步
  - [L157-L200 _flatten_router_tensor / _flatten_expert_indices](#L157-L200)：把 [B, E, H, W] 展平为 [N, E]
  - [L202-L245 _get_global_statistics](#L202-L245)：DDP 环境下 all_reduce 聚合每张卡的 expert 使用率
  - [L247-L360 forward](#L247-L360)：
    - 收集所有路由权重与路由分布
    - 计算 gshard_balance_loss 与 router Z-loss（`logsumexp(logits)^2`）
    - [L289-L310 温度退火集成](#L289-L310)：与 Top-K Scheduler 联动
    - [L369-L379 NaN Guard](#L369-L379)：**关键防御**——`if not torch.isfinite(total_loss).all(): total_loss = torch.nan_to_num(total_loss)`，防止 MoE 训练早期梯度爆炸导致 checkpoint 丢失
  - [L380-L393 属性](#L380-L393)：`balance_loss / z_loss / aux_loss` 属性分别暴露各分量

## 与 v8/v12 loss 的对比

| 维度 | v8 v8DetectionLoss | Master MoELoss |
|------|--------------------|-----------------|
| 主损失 | box + cls + dfl | 无主损失（辅助） |
| 平衡损失 | 无 | GShard + differentiable |
| Z-loss | 无 | 路由 logits 的 logsumexp² |
| DDP 同步 | — | `all_reduce_mean` 跨卡 |
| NaN 防御 | 无 | [L369-L379 NaN Guard](#L369-L379) |

**为什么需要 NaN Guard**：MoE 在训练早期 top_k 变化剧烈、路由权重可能爆炸、balance loss 可能产生 NaN——**Master 训练 10 epoch 内出现一次 NaN 就丢全部进度**，这是很多开源 MoE 实现没做的关键防御。

**训练集成**：`DetectionModel.init_criterion` 里 MoELoss 被拼接到 v8DetectionLoss 之外，最终总损失 = v8DetectionLoss + `0.01 * MoELoss.balance_loss` + `0.001 * MoELoss.z_loss`（[痛点 8 §损失集成](../painpoints#8-moeloss-是-无主-辅助损失-pain-loss-integration)）。

**Top-K 退火联动**：MoELoss 与 [topk_anneal.py](./moe-topk-anneal-py) 的 `DynamicTopKScheduler` 通过 `anneal_mot_temperature` 联动——训练后期 top_k 从 3 退火到 2，balance loss 系数同步衰减。

MoE 主模块见 [modules.py 源码页](./moe-modules-py)，路由见 [routers.py 源码页](./moe-routers-py)，Top-K 退火见 [topk_anneal.py 源码页](./moe-topk-anneal-py)，训练集成见 [train.py 源码页](./train-py)。损失全景见 [losses 页](../losses)。

## 常见问题

**Q：MoELoss 的默认系数是多少？**
A：`balance_loss_coeff=0.01`、`z_loss_coeff=0.001`——balance loss 权重远大于 z_loss。这是 GShard 论文推荐的默认值。

**Q：DDP 环境下如何同步 expert 使用率？**
A：`_get_global_statistics`（[L202-L245](#L202-L245)）用 `all_reduce_mean` 跨卡聚合——**每张卡的 expert 使用率先本地统计，再 all_reduce 求均值**，最后计算 balance loss。这是 Master 相比单机 MoE 实现的必需改动。

**Q：NaN Guard 会隐藏什么问题？**
A：NaN Guard 只是**防止训练崩溃**，不解决根本原因。如果频繁触发 `_nan_guard_hits`，说明：
1. 学习率过高（balance loss 梯度爆炸）
2. top_k 退火太快（[topk_anneal.py](./moe-topk-anneal-py)）
3. 训练数据有问题（全 NaN 输入）

**Q：为什么 MoELoss 不进入 v8DetectionLoss？**
A：因为 MoE 是**独立组件**，不属于检测头。v8DetectionLoss 只管 box/cls/dfl 三分量，MoE 的辅助损失单独挂接。这也是为什么 [train.py L208 loss_names 只有三个](./train-py)——用户看不到 MoE 的 balance/z-loss 数值。

**Q：Z-loss 的公式是什么？**
A：`logsumexp(logits, dim=expert_dim) ^ 2` 的均值——限制 logits 的幅度，避免 softmax 数值爆炸。这是 MoE 训练的**标配损失**。
