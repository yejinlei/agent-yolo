# YOLO-Master 理论 Concept

> Master 的三大理论贡献是 **ES-MoE（Efficient Sparse Mixture-of-Experts）**、**MoT（Mixture-of-Transformers）** 与 **MoA（Mixture-of-Attention）**——分别对应「稀疏路由」「多 Transformer 融合」「多尺度注意力」三个方向。三大组件都建立在 **Ultralytics YOLOv12 骨干**之上，**共用同一套损失、同一套分配器、同一套 head**——理论层面的增量**全部集中在 `moe/` 与 `mot/` 两个新包**。

> **v8~v13 读者路标**：v8 读者会看到 Master 引入 MoE（Mixture of Experts）这一 LLM 里的概念——**ES-MoE 是首次把 MoE 从 LLM 迁移到 YOLO 通用检测**；v11/v12 读者会看到 Master 保留了 A2C2f / C3k2 / Detect / v8DetectionLoss，只在骨干里追加稀疏路由；v13 读者会看到 Master 与 v13 是同期两条并行路线——**Master 走 MoE 稀疏、v13 走超图门控**，两者都宣称 Nano 档位 +0.8~1.0 AP。

## 1. 只想带走三件事

1. **ES-MoE 是"3 个 DW 专家 + 1 个 CNN Router + Top-K 稀疏"**——**Expert 0/1/2 用 3×3/5×5/7×7 深度可分离卷积**（<SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="119" :end="180"/>），**Router 是 1×1 + DWConv 3×3 + SE + Softmax + Top-K**（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="65" :end="409"/>）；训练时全 3 专家计算 + balance loss，**推理时只算 Top-2 专家**——**"compute-on-demand"**；
2. **MoT 三个专家是"局部 + 窗口 + 可变形"的 Transformer 组合**——`_LocalConvTransformerExpert`（DWConv 局部）/ `_WindowTransformerExpert`（Swin 窗口）/ `_DeformableTransformerExpert`（Deformable DETR 可变形）（<SrcLink file="master:ultralytics/nn/modules/mot/experts.py" :line="84" :end="530"/>），**由 `_MoTRouter` 内容感知路由**（<SrcLink file="master:ultralytics/nn/modules/mot/router.py" :line="67" :end="380"/>）；
3. **MoA 是"1×1 卷积 router + 三组 attention"**——CNN-native 的多尺度注意力融合：`Local`（DW-3×3）/ `Regional`（池化 stride=2）/ `Global`（linear-attention），**只在 v0_1 变体里启用**，v0 系列不用。

## 2. ES-MoE 机制详解

### 2.1 从 LLM MoE 到视觉检测

**MoE 的起源**：LLM 领域里，**Switch Transformer**（Fedus et al., 2021）与 **GShard**（Lepikhin et al., 2020）最早引入"稀疏专家路由"——**同一个 Transformer 层，根据输入内容路由到不同子层**。这样做的核心收益是"**总参数很大、每次前向只用一小部分**"——模型容量大幅提升，计算量接近常数。

**视觉检测的挑战**：

- **LLM 的 router 用 Linear**（token embedding → expert logits），**视觉检测的输入是 2D 图像**——不同空间位置可能需要不同专家；
- **LLM 的专家是 Transformer Block**，**视觉检测的专家是 Conv Block**——卷积更适合空间局部性；
- **LLM 的稀疏是"token 级别"**，**视觉检测的稀疏是"图像/token 双级别"**——同一张图里不同区域激活不同专家；
- **LLM 通常用 GPU/TPU 集群**，**视觉检测要跑在边缘设备**——稀疏推理必须能真正减少计算（不能只是权重裁剪）。

**Master 的选择**：

- **Router 用 CNN**（不是 Linear）——DWConv 3×3 提供空间感知；
- **专家用深度可分离卷积**（不是 Transformer Block）——保留视觉检测的空间归纳偏置；
- **稀疏是"图像级别"**——同一张图内所有 token 共享一套 Top-K 选择（简化实现）；
- **推理时 Top-K 稀疏**——只计算 top_k=2 个专家，实际减少 33% 计算量。

### 2.2 三个专家：不同感受野的深度可分离卷积

**ES-MoE 的核心创新**：**专家不是"同一结构的不同权重"，而是"不同感受野的三种卷积"**——每个专家有一个**不同核大小的深度可分离卷积**：

| 专家 | 核大小 | 感受野 | 擅长 |
|-----|-------|-------|------|
| Expert 0 | 3×3 | 局部 | 边缘、纹理、细粒度 |
| Expert 1 | 5×5 | 中等 | 中等尺度对象、局部结构 |
| Expert 2 | 7×7 | 较大 | 大尺度对象、远距离上下文 |

**为什么是深度可分离**：

- 传统 Conv 3×3：`C×3×3 = 9C²` 参数
- DWConv 3×3：`C×3×3 = 9C` 参数（省 1/9）
- **DWConv + PWConv（1×1）≈ 10C²**——比传统 Conv 省 33%，保留通道混合能力

**为什么是三个不同核大小**：

- 一个 3×3 卷积的感受野是 3×3 像素；
- 一个 5×5 卷积的感受野是 5×5 像素；
- 一个 7×7 卷积的感受野是 7×7 像素；
- **同一张图里，不同区域的对象尺度不同——局部纹理用 3×3、中等对象用 5×5、大对象用 7×7**；
- **Router 学出「这个位置该激活哪些核大小」**——这就是 Master 论文里说的 "compute-on-demand"。

**代码**：`default_kernel_sizes = [3, 5, 7]`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="487"/>），`max_kernel_size=15` 是硬上限（保证核大小不超过 15×15）。

### 2.3 DynamicRoutingLayer：轻量 CNN Router

**Router 的任务**：给定输入特征图 `x ∈ [B, C, H, W]`，输出每个位置激活哪些专家的权重 `w ∈ [B, num_experts, 1, 1]`。

**DynamicRoutingLayer 结构**（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="65" :end="409"/>）：

```
1. Conv 1×1: C → C/reduction (reduction=8, 降维 8 倍)
2. DWConv 3×3: C/reduction → C/reduction (空间聚合)
3. SE 层: 
   - AdaptiveAvgPool2d(1) → Conv 1×1 (降维) → SiLU → Conv 1×1 → Sigmoid
   - 通道重要性建模
4. Conv 1×1: C/reduction → num_experts (3)
5. Softmax(dim=1) (跨专家归一化)
6. Top-K 选择:
   - 训练：全 3 专家参与前向，权重用于 balance loss
   - 推理：只保留 top_k=2 个专家，其余置 0
```

**关键设计**：

- **Router 是 CNN，不是 Linear**——**DWConv 3×3 提供空间上下文感知**，`SE` 提供通道重要性建模；
- **Router 输出形状 `[B, num_experts, 1, 1]`**——**同一张图内所有空间位置共享一套 Top-K 选择**（图像级别路由，不是 token 级别）；
- **reduction=8**——Router 的降维系数，控制 Router 的参数量；
- **Top-K 用 `torch.topk`**——返回索引与权重，非 Top-K 位置的权重被置 0。

### 2.4 Top-K 稀疏推理

**核心逻辑**（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="534" :end="600"/>）：

```python
def forward(self, x):
    routing_weights = self.routing(x)  # [B, 3, 1, 1]
    
    if self.training:
        # 训练：所有 3 个专家都算
        expert_outputs = [expert(x) for expert in self.experts]
        weighted_output = sum(w * out for w, out in zip(routing_weights, expert_outputs))
        # balance loss 基于全 3 专家的使用频率
    else:
        # 推理：只算 top_k=2 个专家
        if self.use_sparse_inference:
            _, top_k_indices = torch.topk(routing_weights, k=self.top_k, dim=1)
            selected_experts = [self.experts[i] for i in top_k_indices.flatten()]
            expert_outputs = [expert(x) for expert in selected_experts]
            # 只有 top_k 个专家参与前向
        else:
            # 稠密推理：所有 3 专家都算
            ...
    
    output = self.norm(weighted_output)
```

**动态阈值 `dynamic_threshold=0.4`**：推理时，如果某专家的权重 < 0.4，直接置 0（不参与加权）——**避免"激活一个专家但权重很低"这种浪费计算的情况**。

**训练时为什么不稀疏**：

- **balance loss 需要全专家的使用频率**——只算 top_k=2 个专家时，第三个专家的 `usage` 恒为 0，balance loss 会强制前 2 个专家也平均使用；
- **稠密计算时 Router 才能学到有意义的分布**——如果只算 2 个，Router 只会学出"top-2 平均分配"，学不到"3 个专家各有所长"；
- **推理时才启用稀疏**——平衡"训练时学得全面"与"推理时算得快"。

### 2.5 Balance Loss 与 Z-Loss

**Balance Loss（GShard 风格）**（<SrcLink file="master:ultralytics/nn/modules/moe/loss.py" :line="16" :end="96"/>）：

$$L_{\text{balance}} = N \cdot \sum_{i=1}^{N} \text{importance}_i \cdot \text{usage}_i$$

- $N$ = num_experts = 3
- $\text{importance}_i$ = router 的平均概率（**保留梯度**）
- $\text{usage}_i$ = 实际被激活的比例（**detach**）
- **均匀分布时 = 1.0**；某个专家过度使用则 loss 上升

**为什么 importance 保留梯度、usage detach**：

- importance 保留梯度 → 梯度可以回传 Router 的 logits → **Router 学会"避免让某个专家过度激活"**；
- usage detach → 不让梯度穿过离散选择（top_k 索引不可微） → **避免梯度爆炸**。

**Z-Loss（路由稳定性）**：

$$L_z = \mathbb{E}_x\left[\log\left(\sum_i \exp(\text{logits}_i)\right)^2\right]$$

- 惩罚 Router 输出 logits 过大（softmax 输入过大 → inf → NaN）；
- **训练数值稳定**——社区 issue 里记录的 "Router NaN 风险" 主要就靠 z-loss 抑制。

**总 MoE 辅助损失**（<SrcLink file="master:ultralytics/nn/modules/moe/loss.py" :line="97" :end="180"/>）：

$$L_{\text{moe}} = 0.01 \cdot L_{\text{balance}} + 0.01 \cdot L_z$$

**权重 0.01** 是论文披露的默认值——**balance loss 权重过大会压制主任务（检测 loss）**，0.01 是经验最优。

### 2.6 MoE 与 Ultralytics MoE 家族的关系

Master 的 `moe/` 包里不仅实现了 ES_MOE，还实现了大量变体——**这些变体大多未在主表里出现，是腾讯内部迭代的历史产物**：

| 变体 | 类 | 用途 |
|-----|-----|-----|
| **ES_MOE** | `ES_MOE`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="410" :end="787"/>） | **主表用的核心 MoE**（3×3/5×5/7×7 专家） |
| UltraOptimizedMoE | `UltraOptimizedMoE`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="121" :end="398"/>） | 优化版 MoE（GroupNorm 稳定小 batch） |
| A2C2fMoE | `A2C2fMoE`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="1301" :end="1770"/>） | A2C2f + MoE 融合（v0_1 用） |
| AdaptiveCapacityMoE | `AdaptiveCapacityMoE`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="321" :end="408"/>） | 复杂度自适应（未在主表） |
| MOE 别名 | `MOE = ES_MOE`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="1775"/>） | 历史兼容别名 |

**Router 家族**（<SrcLink file="master:ultralytics/nn/modules/moe/routers.py" :line="59" :end="380"/>）：

| Router | 用途 |
|--------|------|
| `UltraEfficientRouter`（<SrcLink file="master:ultralytics/nn/modules/moe/routers.py" :line="59" :end="168"/>） | 轻量 router（1×1 + SE + sigmoid） |
| `BaseRouter`（<SrcLink file="master:ultralytics/nn/modules/moe/routers.py" :line="169" :end="268"/>） | Router 基类 |
| `EfficientSpatialRouter`（<SrcLink file="master:ultralytics/nn/modules/moe/routers.py" :line="269" :end="380"/>） | 空间感知 router |
| `DynamicRoutingLayer`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="65" :end="409"/>） | **ES_MOE 用的默认 router** |

**Expert 家族**（<SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="73" :end="200"/>）：

| Expert | 用途 |
|--------|------|
| `SimpleExpert`（<SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="73" :end="92"/>） | 朴素 1×1 双卷积（GroupNorm） |
| `SpatialExpert`（<SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="94" :end="117"/>） | 1×1 + DW 3×3 + 1×1（空间感知） |
| `GhostExpert`（<SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="119" :end="180"/>） | Ghost 稀疏（primary + cheap op） |
| `DepthwiseSEparableConv`（<SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="27" :end="63"/>） | **ES_MOE 用的默认专家** |

## 3. MoT 详解：三个 Specialized Transformer 专家

> **v8~v13 读者路标**：论文标题里的 "Specialized Transformers" 实际由 **MoT（Mixture-of-Transformers）** 实现——**三个 Transformer 专家 + 一个内容感知 Router**。MoT 只在 **v0_1 变体**里启用，v0 系列不使用。

### 3.1 三个专家

**_LocalConvTransformerExpert**（<SrcLink file="master:ultralytics/nn/modules/mot/experts.py" :line="84" :end="183"/>）：

- **局部深度卷积 Transformer**——QKV 前先用 DWConv 3×3 做局部混合，再走 attention；
- **适合**：纹理、边缘、细粒度特征；
- **内部**：`dw_mix (DW 3×3) → qkv (1×1) → attention → proj (1×1) → FFN (GLU)`；
- **FFN 是 GLU（Sigmoid gate × value）**——不是标准 FFN 也不是 SwiGLU。

**_WindowTransformerExpert**（<SrcLink file="master:ultralytics/nn/modules/mot/experts.py" :line="184" :end="361"/>）：

- **Swin 风格窗口 Transformer**——把 2D 特征图切成窗口，每个窗口内做 attention；
- **适合**：中等尺度对象、局部区域模式；
- **窗口大小**：默认 7×7（可配置）；
- **窗口间移动**：奇偶窗口交替，模拟 Swin 的 shifted window attention。

**_DeformableTransformerExpert**（<SrcLink file="master:ultralytics/nn/modules/mot/experts.py" :line="362" :end="530"/>）：

- **可变形采样 Transformer**——参考 Deformable DETR，每个 query 只采样少数几个点；
- **适合**：稀疏采样、遮挡对象、不规则形状；
- **采样点数量**：默认 4 个（可配置）。

### 3.2 _MoTRouter：内容感知路由

**_MoTRouter 结构**（<SrcLink file="master:ultralytics/nn/modules/mot/router.py" :line="67" :end="180"/>）：

```
_MoTRouter(dim, num_experts=3, top_k=2, use_spatial=True):
    if use_spatial:
        # token-level routing（每个空间位置独立路由）
        router = Conv 1×1: dim → num_experts
    else:
        # image-level routing（整张图共享一套路由）
        router = AdaptiveAvgPool2d(1) + Conv 1×1: dim → num_experts
    
    logits = router(x)                    # [B, 3, H, W]
    weights = Softmax(dim=1)(logits / T)  # T = temperature
    top_k_mask = top-k-mask(weights)      # Top-K 掩码
    final_weights = renormalize(weights * top_k_mask)
```

**关键设计**：

- **默认 `use_spatial=True`**——token-level routing，每个位置独立路由；
- **Top-K mask 而非硬选择**——`final_weights = renormalize(weights * top_k_mask)`——**保留梯度流**（硬选择会切断梯度）；
- **export_masked=True**——导出 ONNX/TorchScript 时用可 trace 的算子重建 Top-K 权重，**保证静态图数值一致**。

### 3.3 _mot_router_aux_loss：MoT 的辅助损失

**_mot_router_aux_loss**（<SrcLink file="master:ultralytics/nn/modules/mot/router.py" :line="345" :end="400"/>）：

$$L_{\text{MoT}} = \lambda \cdot L_z^{\text{MoT}}$$

- 只用 **z-loss**（不像 MoE 用 balance + z-loss）——MoT 的路由是 soft 的（不做硬选择），balance loss 意义不大；
- **z-loss 保证 Router 输出 logits 稳定**——防止 softmax 输入爆炸。

### 3.4 为什么 "Specialized Transformers" 不是具名类

**Master 论文标题**里的 "Specialized Transformers" 是一个**描述性概念**，不是代码里的具体类——**实现完全通过 MoT 的 `_LocalConvTransformerExpert` / `_WindowTransformerExpert` / `_DeformableTransformerExpert` 三个专家 + `_MoTRouter` 完成**。

**读者按论文标题搜代码会找不到**：

- 没有 `SpecializedTransformer` 类；
- 没有 `MasterDetect` 类（Master 用 v11 的 Detect）；
- 没有 `YOLOMasterModel` 类（Master 用 v8 的 YOLO）；
- **只有 `mot/` 包里的 `_LocalConv` / `_Window` / `_Deformable` 三个下划线前缀的专家类**。

**为什么用下划线前缀**：Python 惯例里下划线前缀表示"内部实现"——**Master 作者不希望读者直接构造这些类，只通过 MoT 的顶层 API 使用**。

## 4. MoA 详解：多尺度注意力融合

> **v8~v13 读者路标**：**MoA（Mixture-of-Attention）** 只在 v0_1 变体里启用——v0 系列（论文主表）**不使用 MoA**。MoA 的机制是 **1×1 卷积 router 把 token 软分配到 Local / Regional / Global 三组 attention 头**。

### 4.1 MoA 的机制

```
MoA(dim, num_heads, temperature):
    router = Conv 1×1: dim → 3  (三组 attention 的权重)
    
    weights = Softmax(dim=1)(router(x) / temperature)
    # weights: [B, 3, H, W]
    
    local_out = DWConv 3×3 (x)              # 局部注意力
    regional_out = Pooling stride=2 (x)     # 区域注意力（pooling 提供中等尺度上下文）
    global_out = LinearAttention (x)        # 全局注意力（linear attention，非二次复杂度）
    
    output = weights[0] * local_out + weights[1] * regional_out + weights[2] * global_out
```

**三组 attention 的语义**：

| Attention | 感受野 | 计算复杂度 | 用途 |
|-----------|-------|-----------|------|
| Local（DWConv 3×3） | 局部 | O(C·HW) | 纹理、边缘 |
| Regional（Pool stride=2） | 中等 | O(C·HW/4) | 中等尺度对象 |
| Global（Linear Attention） | 全局 | O(C²·H + C²·W)（近线性） | 远距离上下文、跨尺度依赖 |

**为什么叫 "CNN-native"**：

- MoA 的 router 是 **1×1 卷积**（不是 Linear），输出空间分布与输入对齐；
- **三组 attention 都是 CNN 风格**（DWConv / Pool / Conv）——不使用标准的 self-attention QKV 投影；
- 保留卷积的**空间归纳偏置**——比纯 Transformer 更适合视觉检测。

### 4.2 MoA 与 A2C2f 的关系

**Master 沿用 v12 的 A2C2f**（骨干 #8/#11）——A2C2f 是 v12 引入的注意力模块（**A**ttention-**A**ware **C**2**f**），内部是**局部 + 全局**两路 attention 融合（`a2=True` 时启用）。

**MoA 与 A2C2f 的差异**：

- **A2C2f** 是**固定融合**（局部 + 全局两路加权，权重是学习的常数）；
- **MoA** 是**动态融合**（三路 attention，权重是路由网络按 token 内容动态计算）；
- **A2C2f** 只在骨干 P4/P5 出现（v12 保留 2 处），**MoA** 只在 v0_1 变体出现（v0 不用）。

## 5. Phased Routing：Top-K 退火调度

> **v8~v13 读者路标**：**Phased Routing** 是 Master 论文里提出的训练策略——**不是两阶段训练**（不是先训 router、再训 expert），而是**一次端到端训练，在训练中动态调整 Top-K 值**。

### 5.1 为什么需要 Phased Routing

**LLM MoE 常见做法**：两阶段训练——先训 router（fixed experts）、再训 expert（fixed router）——**避免 Router 与 Expert 相互拉锯**。

**Master 的选择**：端到端训练 + **Top-K 退火**——

- **早期**：`top_k=3`（所有专家都激活）——**Router 有充分的梯度信号学出分布**；
- **中期**：`top_k=2`（默认）——**平衡"专家多样性"与"稀疏性"**；
- **晚期**：`top_k=1`（可选）——**极致稀疏，最大化推理加速**。

### 5.2 TopKAnnealConfig 与 DynamicTopKScheduler

**TopKAnnealConfig**（<SrcLink file="master:ultralytics/nn/modules/moe/topk_anneal.py" :line="37" :end="49"/>）：

```python
@dataclass
class TopKAnnealConfig:
    enabled: bool = False          # 默认关闭（保持向后兼容）
    init_top_k: int = 2            # 初始 top_k
    min_top_k: int = 1             # 最小 top_k
    window: int = 3                # 滑动窗口（验证 epoch 数）
    map_eps: float = 1e-3          # mAP 增益阈值（饱和判定）
    gini_ceiling: float = 0.30     # Gini 上限（负载均衡判定）
    hold: int = 2                  # 连续满足的 epoch 数
    rollback_drop: float = 5e-3    # mAP 下降触发回滚
    ema_momentum: float = 0.8      # EMA 动量
```

**DynamicTopKScheduler**（<SrcLink file="master:ultralytics/nn/modules/moe/topk_anneal.py" :line="71" :end="168"/>）：

**公式**（每个验证 epoch 评估一次）：

$$\text{saturated}_t = \max(\text{mAP}_{t-w+1:t+1}) - \max(\text{mAP}_{t-2w+1:t-w+1}) < \text{map\_eps}$$

$$\text{balanced}_t = \text{ema\_gini}_t \leq \text{gini\_ceiling}$$

**决策**：

- 若 `saturated` 且 `balanced` 连续 `hold` 个 epoch 且 `top_k > min_top_k` → **退火：`top_k -= 1`**；
- 若刚退火后 mAP 下降 > `rollback_drop` → **回滚：`top_k += 1`** 并冻结。

**关键设计**：

- **验证 mAP 饱和**（不再提升）+ **负载均衡**（Gini 系数低于阈值）——**双条件都满足才退火**；
- **回滚保护**——退火后如果 mAP 显著下降，立即回滚；
- **EMA 平滑**——避免单次 mAP 波动影响决策；
- **默认关闭**（`enabled=False`）——保持向后兼容，需要显式启用。

### 5.3 训练流程

```
Epoch 0-100: top_k=3（Router 学习）
    ↓
Epoch 100-300: 验证 mAP 稳定在 42.0-42.2，Gini=0.25
    ↓
Epoch 300: saturated + balanced → top_k=2（默认）
    ↓
Epoch 300-500: mAP 稳定在 42.3-42.4
    ↓
Epoch 500: saturated + balanced → top_k=1
    ↓
Epoch 500-600: mAP 稳定在 42.4（几乎无损）
```

**训练时** top_k 动态变化不影响权重加载——所有专家的权重都保留，只是"哪些参与前向"在变。

## 6. 与 GShard / Switch Transformer 的对比

| 维度 | GShard（LLM） | Switch Transformer（LLM） | **ES-MoE（Master）** |
|-----|---------------|--------------------------|---------------------|
| Router | Linear | Linear | **1×1 + DW 3×3 + SE（CNN）** |
| Expert | Transformer Block | Transformer Block | **DWConv 3×3/5×5/7×7** |
| Top-K | 全激活（GShard） | top_k=1（Switch） | **top_k=2（默认）** |
| 稀疏 | 无（稠密） | Token 级稀疏 | **图像级稀疏** |
| 训练 | 端到端 | 两阶段 | **端到端 + Phased Routing** |
| Balance Loss | N·Σ(imp·usage) | N·Σ(imp·usage) | **同款** |
| 稀疏推理 | 无 | 无 | **有（use_sparse_inference=True）** |
| 参数量 | 大（100B+） | 大（100B+） | **小（3M 级别）** |
| 部署 | TPU 集群 | TPU 集群 | **GPU / 边缘设备** |

**核心差异**：

- **Router 用 CNN vs Linear**——视觉检测里 CNN router 能感知空间分布，比 Linear 更适合；
- **专家用 DWConv vs Transformer Block**——卷积更适合空间局部性，深度可分离更适合边缘设备；
- **稀疏推理 vs 稠密**——Master 推理时只算 top_k=2 个专家，实际减少 33% 计算；
- **Phased Routing vs 两阶段**——Master 端到端训练，通过 Top-K 退火动态调整稀疏度，避免了"先训 router 再训 expert"的复杂性。

## 7. Master 的理论定位

**Master 属于「稀疏 MoE 家族」在视觉检测的首次应用**——它证明了三件事：

1. **MoE 迁移到视觉检测可行**——LLM MoE 的 Sparse Routing 概念可以直接迁移到 YOLO 通用检测，不需要从头设计；
2. **CNN-native 的 Router 与 Expert 更适合视觉检测**——Router 用 CNN、Expert 用 DWConv，比纯 Transformer 方案更适合 YOLO；
3. **Phased Routing 是可行的训练策略**——端到端训练 + Top-K 退火，避免两阶段训练的复杂性，同时保持稀疏推理的收益。

**Master 与同期工作的对比**：

- **v13**（2025-06）：**超图消息传递 + 门控连接**——高阶关联路线，慢但精度提升明显；
- **Master**（2025-12）：**MoE 稀疏路由 + Specialized Transformers**——稀疏加速路线，快且精度提升；
- **两者同期 2025-2026 都是学术 fork**，**都在 Nano 档位宣称 +0.8~1.0 AP**，但工程落点完全不同。

## 8. 参考

- <SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="27" :end="63"/> — DepthwiseSEparableConv
- <SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="65" :end="409"/> — DynamicRoutingLayer
- <SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="410" :end="787"/> — ES_MOE
- <SrcLink file="master:ultralytics/nn/modules/moe/modules.py" :line="1301" :end="1770"/> — A2C2fMoE
- <SrcLink file="master:ultralytics/nn/modules/moe/routers.py" :line="59" :end="380"/> — UltraEfficientRouter / BaseRouter / EfficientSpatialRouter
- <SrcLink file="master:ultralytics/nn/modules/moe/experts.py" :line="73" :end="200"/> — SimpleExpert / SpatialExpert / GhostExpert
- <SrcLink file="master:ultralytics/nn/modules/moe/loss.py" :line="16" :end="180"/> — gshard_balance_loss / MoELoss
- <SrcLink file="master:ultralytics/nn/modules/moe/topk_anneal.py" :line="37" :end="205"/> — TopKAnnealConfig / DynamicTopKScheduler
- <SrcLink file="master:ultralytics/nn/modules/mot/experts.py" :line="84" :end="530"/> — MoT 三个专家
- <SrcLink file="master:ultralytics/nn/modules/mot/router.py" :line="67" :end="400"/> — _MoTRouter / _mot_router_aux_loss
- [YOLO-Master 总览](./index) · [论文精读](./paper) · [Architecture](./architecture) · [架构与层速查](./layers) · [损失函数](./losses) · [训练技巧](./tricks) · [痛点追踪](./painpoints) · [源码走读](./code-walkthrough)
- arXiv:2512.23273 Lin et al. — YOLO-Master 论文
- arXiv:2101.03961 Lepikhin et al. — GShard
- arXiv:2101.03961 Fedus et al. — Switch Transformer
