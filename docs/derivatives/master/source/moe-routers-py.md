# ultralytics/nn/modules/moe/routers.py

MoE 路由层集合：**543 行、7 个 Router 类**，服务于 ES_MOE / UltraOptimizedMoE / A2C2fMoE 等上层模块。核心是 [L430 DynamicRoutingLayer](#L430)——ES_MOE 直接依赖的路由器，其他 6 个用于不同 MoE 变体。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/moe/routers.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/moe/routers.py" />

**阅读路线**：

- 工具函数 [L16 _get_router_in_channels](#L16)：从 router 模块递归取输入通道；[L34 _validate_router_input](#L34)：通道不匹配时抛 ValueError
- **UltraEfficientRouter** [L59-L168](#L59)：极简路由——`AdaptiveAvgPool2d` + 两层 Conv1×1 + Sigmoid；用于 UltraOptimizedMoE
- **BaseRouter** [L169-L268](#L169)：抽象基类——`forward(x) -> routing_weights [B, num_experts, H, W]`，含 `get_routing_snapshot` / `get_routing_diagnostics`
- **EfficientSpatialRouter** [L269-L312](#L269)：BaseRouter 子类，加入 top_k 支持
- **AdaptiveRoutingLayer** [L313-L338](#L313)：动态 kernel 大小路由
- **LocalRoutingLayer** [L339-L372](#L339)：局部窗口路由（用于 spatial expert）
- **AdvancedRoutingLayer** [L373-L429](#L373)：多尺度特征融合路由
- **DynamicRoutingLayer**（核心）[L430-L543](#L430)：
  - [L430 __init__](#L430)：`in_channels / num_experts=3 / reduction=8 / top_k=None`
  - [L438-L444 网络结构](#L438-L444)：`AdaptiveAvgPool2d(1)` → Conv2d(C→C/reduction) → GELU → Conv2d(C/reduction→num_experts)
  - **8× 下采样特性**：`AdaptiveAvgPool2d(1)` 把 H×W 全压成 1×1——**每张图一个路由决定**，与 v8 的 per-token 路由截然不同
  - [L445-L447 routing_temperature / top_k 属性](#L445-L447)：Temperature annealing 接口
  - [L452-L465 forward](#L452-L465)：池化 → 全局特征 → sigmoid → 可选 top-k 掩码
  - [L466-L494 路由权重广播](#L466-L494)：把 [B, num_experts, 1, 1] 广播回 [B, num_experts, H, W]

## 与 v12 路由机制的对比

| 维度 | v12（无 MoE） | Master routers.py |
|------|---------------|-------------------|
| 路由算法 | 无 | 7 种 Router 变体 |
| 主路由 | — | DynamicRoutingLayer |
| 下采样倍率 | — | **8×（`AdaptiveAvgPool2d(1)`）** |
| Top-K 支持 | — | DynamicRoutingLayer top_k + Sparse forward |
| Temperature | — | 支持 anneal（[_MoTRouter](./mot-router-py) 同款） |
| DDP 同步 | — | 通过 [loss.py](./moe-loss-py) 的 all_reduce_mean |

**Master 路由的三个显著特征**：

1. **全局池化 8× 下采样**：与 Switch Transformer / GShard 的 per-token router 不同，DynamicRoutingLayer 每张图只算**一次**路由决定——把 H×W 全压成 1×1，节省大量路由计算；
2. **`reduction=8`**：中间通道被压到 1/8，进一步降低 router 计算量；
3. **top_k 掩码而非重排**：Sparse 模式不重排样本，直接 mask——DDP 无 all_to_all，简化实现。

**Expert 侧见 [experts.py 源码页](./moe-experts-py)，损失见 [loss.py 源码页](./moe-loss-py)。** 架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)，Top-K 退火见 [topk_anneal.py 源码页](./moe-topk-anneal-py)。

## 常见问题

**Q：DynamicRoutingLayer 与 MoT _MoTRouter 有什么关系？**
A：两个都是「全局池化 + 线性层」的路由器，但**完全独立的实现**：
- DynamicRoutingLayer 用 Sigmoid（不 softmax，多个 expert 可以同时激活）
- _MoTRouter 用 LogSoftmax（每个 token 有唯一主 expert）
- _MoTRouter 有 `temperature` / `exploration_eps` 等 MoE 没有的参数

**Q：为什么所有 router 都用 AdaptiveAvgPool2d(1)？**
A：这是 Master 的核心设计——**「每张图一个路由决定」**而非「每个 token 一个路由决定」。相比 Switch Transformer 的 per-token 路由，节省 H×W 倍的路由计算。代价是「同一张图内的空间特征用同一组 expert」，可能损失局部差异。

**Q：reduction 参数是什么意思？**
A：路由网络的中间通道缩小比例——`reduction=8` 意味着 `C → C/8 → num_experts`，节省 router 计算量。默认 8 是 Master 团队测出的平衡点。

**Q：LocalRoutingLayer 与 DynamicRoutingLayer 有什么区别？**
A：LocalRoutingLayer 是「局部窗口路由」——用 3×3 卷积代替全局池化，输出 [B, E, H, W]（空间相关），而 DynamicRoutingLayer 是「全局池化路由」——每张图一个决定。Master 只用 DynamicRoutingLayer，其他 router 是为其他 MoE 变体服务。
