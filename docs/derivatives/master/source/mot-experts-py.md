# ultralytics/nn/modules/mot/experts.py

MoT（Mixture of Transformers）专家模块：**548 行、3 个 Transformer Expert 类**——与 MoE 的卷积专家不同，MoT 的专家是**基于 Transformer 注意力**的，为 Master 提供更强的语义建模能力。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/mot/experts.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/mot/experts.py" />

**阅读路线**：

- [L21 _roll_via_cat](#L21)：无内存复制的循环位移——用 `torch.cat` 沿指定维度滚动窗口
- [L40 _sdpa](#L40)：Scaled Dot-Product Attention 封装——优先用 `torch.nn.functional.scaled_dot_product_attention`（更快），fallback 到手写
- **_LocalConvTransformerExpert** [L84-L183](#L84)：
  - 局部卷积 + 短程注意力的组合——**每个 expert 只关注局部区域**
  - 3×3 DW-Conv 预处理 → SDPA（短程）→ 残差融合
- **_WindowTransformerExpert** [L184-L361](#L184)：
  - **窗口注意力（Windowed Attention）**——把 H×W 分成 `window_size × window_size` 的窗口，每个窗口内独立做 SDPA
  - 用 [L21 _roll_via_cat](#L21) 让窗口之间产生位移重叠，避免窗口边界丢失信息
  - 适合大分辨率特征图（H×W > 阈值时启用，SDPA 复杂度 O(n²)）
- **_DeformableTransformerExpert** [L362-L548](#L362)：
  - **可变形注意力（Deformable Attention）**——每个 query 学习一组可变形采样点
  - 参考 [L40 _sdpa](#L40) 的 fallback 逻辑
  - 与 `nn/modules/transformer.py` 的 [DeformableTransformerDecoderLayer](./transformer-py) 类似但更精简

## 与 v12 Transformer 组件的对比

| 维度 | v12 TransformerBlock | Master MoT Expert |
|------|----------------------|-------------------|
| 注意力范围 | 全局 | Local / Window / Deformable |
| 复杂度 | O((H·W)²) | O(window²) 或 O(sample_points) |
| 位置编码 | 无 | 隐式（窗口索引） |
| 与卷积结合 | 无 | LocalConv 混合 |

**为什么 MoE 之外还有 MoT**：Master 的 ES_MOE 用**卷积专家**处理**空间特征**，MoT 用**注意力专家**处理**语义关联**。虽然官方 yaml（[yolo-master-n.yaml](./yolo-master-n-yaml)）里没有出现 MoT，但代码里完整实现了——**论文里有、yaml 没启用**，这是 [痛点 4 §MoT 未落地](../painpoints#4-mot-未落地-pain-mot-unused) 的具体体现。

**三种 Expert 的适用场景**：

- `_LocalConvTransformerExpert`：局部语义融合（浅层）
- `_WindowTransformerExpert`：中等分辨率的全局-局部平衡（中层）
- `_DeformableTransformerExpert`：跨尺度语义关联（深层）

MoT 路由见 [router.py 源码页](./mot-router-py)，Transformers 相关实现见 [transformer.py 源码页](./transformer-py)，架构总览见 [architecture 页](../architecture)。

## 常见问题

**Q：MoT 在官方 yaml 里启用了吗？**
A：**没有**——`yolo-master-n.yaml` / `yolo-master-l.yaml` 里都只有 ES_MOE，没有 MoT 相关模块。MoT 代码存在但**未被官方模型使用**，这也是 [痛点 4](../painpoints#4-mot-未落地-pain-mot-unused) 的核心。

**Q：MoT Expert 与 ES_MOE Expert 的关键差异？**
A：
- **计算范式**：MoE Expert 用卷积（局部感受野），MoT Expert 用 Transformer 注意力（全局）
- **组合方式**：MoE 是「路由权重 × expert 输出」加权求和；MoT 是「每个 expert 独立前向，路由决定组合」
- **参数规模**：MoT Expert 参数量比 MoE Expert 大 10× 以上

**Q：为什么 Master 要写 MoT 但不用？**
A：推测是**实验性代码**——Master 团队可能在做「MoE + MoT 混合」的实验但没走通。用户若要在自己的模型里启用 MoT，需自己写 yaml 分支。

**Q：_DeformableTransformerExpert 与 MSDeformAttn 的关系？**
A：[MSDeformAttn](./transformer-py) 是 Ultralytics 上游的 Multi-Scale Deformable Attention，_DeformableTransformerExpert 是 Master 自己实现的**精简版**——为 MoT Expert 单独设计，参数更少、更适合稀疏计算。
