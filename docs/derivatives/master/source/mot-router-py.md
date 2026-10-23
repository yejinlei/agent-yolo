# ultralytics/nn/modules/mot/router.py

MoT（Mixture of Transformers）路由层与辅助损失：**396 行**——与 [MoE routers.py](./moe-routers-py) 平行实现，专为 Transformer Expert 设计。核心类 **_MoTRouter**（[L67](#L67)）继承 `FP32RouterMixin`，避免 FP16 下 softmax 数值不稳定。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/mot/router.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/mot/router.py" />

**阅读路线**：

- [L31 differentiable_balance_loss](#L31)：MoT 专属的可微平衡损失（与 [MoE loss.py](./moe-loss-py) 的同名函数**独立实现**）
- **_MoTRouter**（核心）[L67-L344](#L67)：
  - [L67 __init__](#L67)：`dim / num_experts=3 / top_k=2 / use_spatial=True / temperature=1.0 / exploration_eps=0.02 / scene_aware=False / scene_hidden_dim=None / scene_inference_mode="dynamic" / export_masked=True`
  - **签名差异**：与 [DynamicRoutingLayer](./moe-routers-py) 不同，_MoTRouter 增加了 `use_spatial / scene_aware / export_masked` 三个参数——为 Transformer Expert 的「场景感知路由」服务
  - [L100-L160 场景感知分支](#L100-L160)：`scene_aware=True` 时启用额外 MLP 计算场景描述子，作为路由额外特征
  - [L161-L220 主路由](#L161-L220)：`global_pool` → Conv → LogSoftmax（temperature 缩放）
  - [L221-L280 Top-K + exploration](#L221-L280)：Gumbel-Softmax 探索，`exploration_eps` 控制探索率
  - [L281-L340 forward 完整流程](#L281-L340)：路由权重 + 掩码 + 场景融合
- **_mot_router_aux_loss** [L345-L382](#L345)：MoT 路由器辅助损失——**与 [MoE MoELoss](./moe-loss-py) 完全独立**，包含平衡损失 + 路由熵正则
- **_mot_router_z_loss** [L371-L380 附近](#L371)：路由 logits 的 Z-loss
- **anneal_mot_temperature** [L383-L396](#L383)：温度退火——训练后期 temperature 从 1.0 降到 0.5，让路由更确定性

## 与 MoE 路由的对比

| 维度 | MoE DynamicRoutingLayer | MoT _MoTRouter |
|------|-------------------------|-----------------|
| 输入维度 | [B, C, H, W] | [B, C, H, W] |
| 输出维度 | [B, E, H, W] | [B, E, H, W] |
| 精度 | 混合精度 | **FP32 强制** |
| 场景感知 | 无 | scene_aware 可选 |
| 探索机制 | 无 | Gumbel-Softmax |
| Temperature 退火 | 无 | anneal_mot_temperature |
| 辅助损失 | [MoELoss](./moe-loss-py) | [_mot_router_aux_loss](#L345) |

**FP32RouterMixin 的意义**：Transformer 的 Softmax 在 FP16 下容易出现数值爆炸（softmax 输出可能为 NaN 或 0），FP32 强制路由层用全精度计算——**MoT 路由的数值稳定性比 MoE 更重要**，因为 Transformer Expert 的加权组合对权重精度敏感。

**Temperature 退火的物理意义**：`logits / temperature` 中，temperature 大 → 分布更平缓（探索更多），temperature 小 → 分布更尖锐（更确定性）。训练后期降低 temperature，让模型学到「每个 token 只由 1-2 个 expert 处理」的稀疏模式。

**独立损失实现**：_MoTRouter 不复用 [MoELoss](./moe-loss-py)——两份独立的辅助损失代码，这也导致 [痛点 9 §MoE/MoT 训练开关分散](../painpoints#9-moe-训练开关分散-pain-config-spread)。

MoT Expert 见 [experts.py 源码页](./mot-experts-py)，MoE 路由见 [routers.py 源码页](./moe-routers-py)，架构总览见 [architecture 页](../architecture)。

## 常见问题

**Q：FP32RouterMixin 是什么？**
A：一个 mixin 类，重写路由器的 `forward` 让 logits 用 FP32 计算——即使主模型是 FP16，路由层的 softmax/logsumexp 也强制用 FP32。避免数值爆炸。

**Q：scene_aware=True 有什么用？**
A：启用「场景感知路由」——额外计算一个场景描述子（image-level embedding），作为路由的额外特征。适合「不同场景用不同 expert 组合」的应用，如无人机场景识别。

**Q：exploration_eps=0.02 是什么意思？**
A：Gumbel-Softmax 的探索率——**2% 的概率随机选一个 expert**，避免路由过早收敛。训练后期应降低此值。

**Q：为什么 MoT 需要独立的辅助损失？**
A：MoT 的路由分布与 MoE 不同——MoT 用 LogSoftmax（每个 token 唯一 expert），MoE 用 Sigmoid（可多激活）。**balance loss 的公式不能复用**，需要单独实现 `_mot_router_aux_loss`。

**Q：temperature 退火如何调度？**
A：[anneal_mot_temperature](#L383) 通过 `create_topk_anneal_callback` 注册到 trainer——按 `start_epoch / end_epoch` 线性从 1.0 降到 0.5。

**Q：export_masked=True 是什么意思？**
A：导出 ONNX/TensorRT 时保留 masked routing 分支——某些部署平台需要显式的 top-k mask 才能正确执行稀疏路由。
