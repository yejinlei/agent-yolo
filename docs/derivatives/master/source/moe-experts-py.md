# ultralytics/nn/modules/moe/experts.py

MoE 专家（Expert）模块集合：**311 行、9 个 Expert 类**——为 ES_MOE / UltraOptimizedMoE / AdaptiveCapacityMoE 等提供可插拔的卷积单元。**所有 Expert 都基于 Depthwise-Separable Conv（[L280 DepthwiseSeparableConv](#L280)）+ GroupNorm（非 BatchNorm）**，兼顾稀疏计算与训练稳定性。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/moe/experts.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/moe/experts.py" />

**阅读路线**：

- **OptimizedSimpleExpert** [L13-L37](#L13)：最基础的 3×3 DW-Conv + GroupNorm + SiLU + 1×1 Conv + GroupNorm + SiLU 组合
- **FusedGhostExpert** [L38-L72](#L38)：Ghost module 融合的专家——用深度卷积生成线性变换，节省参数
- **SimpleExpert** [L73-L93](#L73)：无 GroupNorm 的极简版本，兼容老 checkpoint
- **SpatialExpert** [L94-L118](#L94)：显式引入 depthwise 3D 特征（用于 3D 输入或 spatial MoE）
- **GhostExpert** [L119-L151](#L119)：Ghost module 单独实现
- **InvertedResidualExpert** [L152-L182](#L152)：MobileNetV2 风格——膨胀率可配置，用于 inverted 类型
- **SharedInvertedExpertGroup** [L183-L279](#L183)：多个 expert 共享膨胀卷积权重，节省参数
- **DepthwiseSEparableConv** [L280-L298](#L280)：**其他 Expert 的基础砖**——DW-Conv(k) + GroupNorm + SiLU + Pointwise
- **EfficientExpertGroup** [L299-L311](#L299)：**ES_MOE 直接使用的 Expert 组**——DepthwiseSeparableConv + 一个 kernel_size 参数

## 与 v12 卷积模块的对比

| 维度 | v12 Conv/C3k2/A2C2f | Master Expert 家族 |
|------|----------------------|---------------------|
| 归一化 | BatchNorm2d | **GroupNorm**（8 groups 默认） |
| 卷积模式 | 普通 Conv | DW-Separable（DW + PW） |
| Kernel 大小 | 固定 3 | 可配置 3/5/7/9 |
| 参数共享 | 无 | SharedInvertedExpertGroup |
| Ghost 融合 | 无 | FusedGhostExpert / GhostExpert |

**为什么全部用 GroupNorm**：MoE 的稀疏前向使得 BN 的 batch statistics 不稳定（每个 expert 只处理部分样本，BN 的统计量在极小 batch 下方差极大）——GroupNorm 是 per-sample 的，天然规避此问题。这也是为什么 [ES_MOE.norm 用 BN](./moe-modules-py)（在所有 expert 混合之后，batch statistics 稳定了）而 Expert 内部用 GroupNorm。

**Kernel 尺寸策略**：ES_MOE 的默认 `expert_kernel_sizes = [3, 5, 7]`（[modules.py 源码页](./moe-modules-py)）——不同 expert 处理**不同感受野**的图像区域，这是 Master 论文核心创新之一：让每个 expert 专精一个尺度。

架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)，路由见 [routers.py 源码页](./moe-routers-py)，损失见 [loss.py 源码页](./moe-loss-py)。

## 常见问题

**Q：为什么 Master 不用 BatchNorm？**
A：MoE 的稀疏前向使得 BN 的 batch statistics 极不稳定——**每个 expert 只处理部分样本，batch 尺寸可能小到 1**，BN 的方差估计失效。GroupNorm 是 per-sample 的，规避此问题。

**Q：EfficientExpertGroup 与其他 Expert 有什么区别？**
A：EfficientExpertGroup 是**简化版**——只用 DepthwiseSeparableConv 一个组件，而其他 Expert 组合多个组件（Conv + Ghost + ...）。ES_MOE 直接用 EfficientExpertGroup，说明 Master 认为「DW-Conv + GroupNorm + SiLU + 1×1 Conv」的组合已够表达力。

**Q：GhostExpert 的原理是什么？**
A：Ghost module 用「便宜的深度卷积」生成特征图，再与主卷积的输出拼接——**大幅减少参数量**。FusedGhostExpert 进一步融合了主 conv 和 ghost conv 的计算。

**Q：kernel_size 3/5/7 的物理意义？**
A：**不同感受野**——3×3 处理细粒度边缘、5×5 处理中等尺度结构、7×7 处理大目标轮廓。**每个 expert 专精一个尺度**，这是 Master 论文的核心创新。

**Q：GroupNorm 的 groups 参数默认值？**
A：GroupNorm 默认 groups=8——`num_channels / 8` 必须整除。ES_MOE 通过 [EfficientExpertGroup](#L299) 把 channels 强制对齐到 8 的倍数，避免维度冲突。

**Q：SharedInvertedExpertGroup 何时启用？**
A：仅 AdaptiveCapacityMoE 使用——多个 expert 共享 DW-Conv 权重以节省参数。**ES_MOE 不使用此组件**，因为 Master 团队认为独立权重更利于 expert 特化。

**Q：Expert 类的输出通道与输入通道关系？**
A：Expert 一般保持 `in_channels = out_channels`（same channels）——路由层的加权求和前提是各 expert 输出维度一致。若通道不同，ES_MOE 内部会强制对齐。

**Q：MoE 稀疏计算如何避免显存抖动？**
A：ES_MOE 的 top_k 默认 3，`num_experts=4`——**每步只用 3/4 expert**，显存占用固定；不同 expert 复用同一份激活缓冲区，避免 OOM。
