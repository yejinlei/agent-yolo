# ultralytics/nn/modules/moe/topk_anneal.py

Top-K 退火调度器：**205 行**，用于在训练过程中动态调整 ES_MOE 的 `top_k` 值——**从 3 退火到 2**，实现「训练初期用更多 expert 表达、后期用更少 expert 保精度」的稀疏策略。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/moe/topk_anneal.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/moe/topk_anneal.py" />

**阅读路线**：

- [L23 apply_top_k](#L23)：把给定的 k 值写回所有 `ES_MOE.top_k` / `routing.top_k`
- **TopKAnnealConfig**（dataclass）[L37-L50](#L37)：
  - `enabled: bool = False`（默认关闭，用户显式开启）
  - `init_top_k: int = 2`
  - `min_top_k: int = 1`
  - `window: int = 3`（观察窗口的 epoch 数）
  - `map_eps: float = 1e-3`（mAP 增益阈值）
  - `gini_ceiling: float = 0.30`（Gini 系数阈值——越低表示 expert 分布越均匀）
  - `hold: int = 2`（连续几个 epoch mAP 增益低于阈值才退火）
  - `rollback_drop: float = 5e-3`（mAP 回退阈值——退火后 mAP 掉太多则回滚）
  - `ema_momentum: float = 0.8`（EMA 平滑）
- **TopKAnnealState** [L51-L70](#L51)：当前状态——current_top_k / history / last_map / rollback
- **DynamicTopKScheduler**（核心）[L71-L170](#L71)：
  - [L71 __init__](#L71)：接收 config + model
  - [L100-L130 on_epoch_end](#L100-L130)：
    - 取当前 val mAP
    - 计算与上个 epoch 的增量
    - **mAP 增益 < map_eps 且连续 hold 个 epoch** → top_k -= 1
    - **退火后 mAP 掉 > rollback_drop** → 回滚 top_k
  - [L131-L155 Gini 系数计算](#L131-L155)：从 expert_usage_counts 计算分布不均度
- [L171 _val_map_from_trainer](#L171)：从 trainer 提取 val mAP
- [L181 create_topk_anneal_callback](#L181)：把 Scheduler 挂到 trainer.callbacks

## 与其他调度的对比

| 维度 | v8/v12 | Master topk_anneal |
|------|--------|---------------------|
| Top-K 调度 | 无 | 动态退火（3→2→1） |
| 学习率调度 | Cosine | Cosine（不变） |
| mAP 感知 | 无 | 基于 val mAP 增益 |
| 回滚机制 | 无 | 退火失败自动回滚 |
| Gini 感知 | 无 | expert 分布不均度判断 |

**退火触发条件**（三个同时满足）：

1. 最近 `hold` 个 epoch 的 mAP 增益均 < `map_eps`；
2. 当前 `current_top_k > min_top_k`；
3. Gini 系数 < `gini_ceiling`（expert 分布已经足够均衡，减少 top_k 不会伤精度）。

**回滚机制**：退火后若下一个 epoch 的 mAP 下降超过 `rollback_drop`，自动把 top_k 加回去。这是 Master 相比其他 MoE 训练框架的显著改进——**退火是有条件的、可撤销的**，不会盲目追求稀疏。

**默认关闭**：`enabled: bool = False`，用户需要在 cfg 里显式开启 `topk_anneal: enabled=true`。这也是 [痛点 9 §MoE 训练开关分散](../painpoints#9-moe-训练开关分散-pain-config-spread) 的一个体现。

Top-K 退火依赖 [ES_MOE top_k 属性](./moe-modules-py) 与 [_MoTRouter 温度退火](./mot-router-py) 联动。架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)。

## 常见问题

**Q：Top-K 退火默认关闭吗？**
A：是——`TopKAnnealConfig.enabled = False`。用户必须在 cfg 里显式 `topk_anneal: enabled=true` 才启用。这也是为什么 Master 论文报告 top_k=3 训练，但推理时是 top_k=2 的原因——**训练时用退火**。

**Q：init_top_k=2 意味着什么？**
A：从 top_k=2 开始训练——**已经是稀疏状态**，不是常见的「top_k=num_experts → 逐步减少」。Master 选择「一开始就稀疏」是出于收敛速度的考虑。

**Q：Gini 系数的物理意义？**
A：Gini 系数 0 表示 expert 完全均匀使用，1 表示完全集中。**gini_ceiling=0.30** 意味着「当 expert 分布已经足够均衡（Gini < 0.3）时，减少 top_k 不会导致 expert 崩塌」。

**Q：与学习率退火的关系？**
A：两者独立——学习率退火由 cosine schedule 控制，Top-K 退火由 mAP 感知触发。**两者可能在同一 epoch 触发**，这是 Master 训练后期的典型现象。
