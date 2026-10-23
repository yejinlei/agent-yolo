# ultralytics/nn/modules/transformer.py

Transformer 相关组件集合：**~800 行**，Master 直接从 Ultralytics 上游继承，为 Detect/A2C2f/Deformable Transformer Decoder 提供底层 Transformer 组件。核心类是 **AIFI**（[L171](#L171)，Area Attention）——A2C2f 的核心积木。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/transformer.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/transformer.py" />

**阅读路线**：

- **TransformerEncoderLayer** [L32-L170](#L32)：基础 encoder layer——Multi-Head Self-Attention + FFN + 残差 + LayerNorm
- **AIFI**（核心）[L171-L246](#L171)：**Area Attention 变体**——
  - 与普通 TransformerEncoderLayer 的区别：把 H×W tokens 分成 `area × area` 的 patch，每个 patch 内独立做注意力
  - [L171 __init__](#L171)：`d_model / nhead / dropout / area / mlp_ratio`
  - [L200-L235 forward](#L200-L235)：patch 切分 → 每个 patch 独立 SDPA → 拼接
  - 用于 A2C2f 的 `a2=True` 分支
- **TransformerLayer** [L247-L277](#L247)：Decoder 使用的单层（Q/K/V 分别来自不同输入）
- **TransformerBlock** [L278-L323](#L278)：Encoder 多层堆叠
- **MLPBlock / MLP** [L324-L418](#L324)：FFN 变体——含 SwiGLU
- **LayerNorm2d** [L419-L461](#L419)：2D LayerNorm（对 HWC 归一化，保留 HW 维度）
- **MSDeformAttn** [L462-L587](#L462)：Multi-Scale Deformable Attention——每个 query 学习多尺度采样点
- **DeformableTransformerDecoderLayer** [L588-L714](#L588)：Decoder 层——用于 YOLOE/RTDETR
- **DeformableTransformerDecoder** [L715-L800](#L715)：Decoder 主体

## 与 v12 transformer.py 的对比

| 维度 | v12 transformer.py | Master transformer.py |
|------|---------------------|------------------------|
| 行数 | ~750 | ~800 |
| AIFI | 有 | **完全一致** |
| MSDeformAttn | 有 | 完全一致 |
| MoT 相关 | 无 | 无（MoT 在 `mot/` 目录独立实现） |

**Master 没有修改 transformer.py**——A2C2f 的 AIFI 直接从 Ultralytics 上游继承。这也是为什么 [痛点 5 §A2C2f 未真正改造](../painpoints#5-a2c2f-未真正改造-pain-a2c2f-unchanged)：Master 只在自己的 `moe/` 目录里加了 ES_MOE，A2C2f 本身完全来自 v12。

**AIFI 的关键点**：

- **Area 参数**：`area=4` 意味着把 H×W 分成 4×4=16 个 patch，每个 patch 独立做注意力——复杂度从 O((H·W)²) 降到 O(16 × (H·W/16)²)
- **patch 之间无交互**：这是 AIFI 的代价——patch 边界的语义会丢失，但换来巨大的 FLOPs 节省
- **A2C2f 里 `area=4` 与 `area=1` 两种配置**（[yolo-master-n.yaml L20 与 L23](./yolo-master-n-yaml)）：中深层用 area=1（全分辨率），浅层用 area=4（省 FLOPs）

**为什么 Master 保留 A2C2f**：A2C2f 是 v12 的核心积木（相比 v11 的 C2f 提速 30%+），Master 保留它作为 backbone 的「加速分支」，MoE 则作为「表达分支」插入在 C3k2 之后——**两个正交的优化方向**。

Detect 头见 [head.py 源码页](./head-py)，MoT Expert 见 [mot-experts-py 源码页](./mot-experts-py)，架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)。

## 常见问题

**Q：AIFI 与标准 Transformer Encoder 有什么差别？**
A：AIFI 是「Area Attention」变体——把 H×W tokens 分成 `area × area` patch，每个 patch 内独立做注意力。标准 Transformer 是全局注意力，AIFI 是**局部注意力**。

**Q：A2C2f 里的 area 参数怎么理解？**
A：`area=4` 意味着 H×W 分成 4×4=16 patch，每个 patch 独立注意力；`area=1` 就是全分辨率——**等价于标准 Transformer**。Master yaml 里 area=4 与 area=1 并存（[yolo-master-n.yaml L20 与 L23](./yolo-master-n-yaml)）。

**Q：Master 为什么没改 transformer.py？**
A：A2C2f 是 v12 的核心积木，Master 保留它作为「加速分支」。Master 只在自己的 `moe/` 目录里加了 ES_MOE，**transformer.py 是继承来的**。这也是为什么 [痛点 5](../painpoints#5-a2c2f-未真正改造-pain-a2c2f-unchanged) 存在。

**Q：DeformableTransformerDecoder 有什么用？**
A：用于 YOLOE / RTDETR——**Open-Vocabulary 检测头**。Master 未启用，但代码存在，方便未来扩展。

**Q：LayerNorm2d 与标准 LayerNorm 差别？**
A：标准 LayerNorm 对 features 维度归一化，LayerNorm2d 保留 H×W 空间维度——**每个空间位置独立归一化**，适合 2D backbone 中间层。

**Q：TransformerEncoderLayer 与 AIFI 复用吗？**
A：不复用——AIFI 独立实现，虽然名字都叫 encoder，但 AIFI 用了 patch 切分逻辑。这也是 AIFI 更慢于 A2C2f 的原因。
