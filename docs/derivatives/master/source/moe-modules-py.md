# ultralytics/nn/modules/moe/modules.py

ES_MOE 主模块文件：**1810 行、~5000 行 MoE 家族的收束点**。文件里同时包含 UltraOptimizedMoE 家族、ES_MOE（Master 的公开招牌类）、A2C2fMoE（A2C2f + MoE 联合）、以及一批历史别名（`MOE = ES_MOE`）。[L1 顶部注释](#L1) 明确「Modified by Tencent on 2026/02/13」，[L11-14](#L11-L14) 说明该文件被拆分后仍作为 `_helpers` / `gated` 的聚合出口——`from .modules import X` 依旧可用。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/moe/modules.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/moe/modules.py" />

**阅读路线**：

- 文件头 [L1-L15](#L1-L15)：Tencent 修改时间戳、AgPL-3.0 许可、拆分说明
- 依赖注入 [L17-L60](#L17-L60)：`nn` / `FlopsUtils` / 全部 experts / 全部 routers / ABlock+A2C2f+C3k / MoELoss / 调度器 / 路由协议
- 二次导出 [L62-L110](#L62-L110)：把 `_helpers` 和 `gated` 里定义的符号全部再导出——**用户 `from ultralytics.nn.modules.moe.modules import X` 的入口**
- **UltraOptimizedMoE** [L121-L260 附近](#L121)：Master 之前的「优化 MoE 原型」——UltraEfficientRouter + 三类 expert + shared expert + 条件计算
- **ES_MOE**（核心）[L410-L660](#L410)：
  - [L413-L424 __init__ 签名](#L413-L424)：`in_channels / out_channels / num_experts=4 / reduction=8 / top_k=2 / use_sparse_inference=True / dynamic_threshold=0.4 / max_kernel_size=15 / expert_kernel_sizes=None`
  - [L442-L457 参数校验](#L442-L457)：8 处 ValueError（top_k 范围、reduction、dynamic_threshold、max_kernel_size 强制奇数）
  - [L471-L472 DynamicRoutingLayer](#L471-L472)：路由层注入
  - [L478-L493 内核尺寸决策](#L478-L493)：默认 `[3, 5, 7]`；`expert_kernel_sizes` 允许 pruned checkpoint 恢复时指定
  - [L493 experts 组](#L493)：`EfficientExpertGroup` × `num_experts`
  - [L496-L499 norm 输出归一化](#L496-L499)：BN + SiLU（**注意**：与 experts.py 里默认 GroupNorm 不同）
  - [L504-L505 non-persistent buffers](#L504-L505)：`load_balancing_loss` / `expert_usage_counts`
  - [L511-L533 _ensure_compat_attrs](#L511-L533)：老 checkpoint 属性修补
  - [L535-L583 forward](#L535-L583)：路由 → load balancing → dense 或 sparse 前向 → norm
  - [L656-L663 _dense_forward](#L656-L663)：训练态全专家加权求和
  - [L665-L699 _sparse_forward](#L665-L699)：推理态 Top-K + dynamic_threshold 剪枝 + 权重归一化
- **A2C2fMoE** [L1301-L1361](#L1301)：继承 A2C2f，把 `self.m` 替换成 `ABlockMoE`（内含 MoE 的 MLP），[L1332-L1341 aux_loss 聚合](#L1332-L1341)
- **HyperUltimateMoE** [L1368-L1500](#L1368)：通道切分 + MatMulFusedExperts + AdaptiveBalanceController + Progressive Sparsity
- **别名收束** [L1772-L1781](#L1772-L1781)：
  - [L1775 `MOE = ES_MOE`](#L1775)：**yaml 里写 `MOE` 与 `ES_MOE` 完全等价**
  - [L1776 `EfficientSpatialRouterMoE = OptimizedMOE`](#L1776)
  - [L1780-L1781 老权重升级 fallback](#L1780-L1781)

## 与其他版本的对比

| 维度 | Ultralytics v8 | Ultralytics v12 | Master modules.py |
|------|----------------|-----------------|-------------------|
| MoE 支持 | 无 | 无 | 4×ES_MOE + A2C2fMoE + UltimateMoE |
| 路由算法 | — | — | DynamicRoutingLayer + Top-K + 稀疏推理 |
| Expert 家族 | — | — | SimpleExpert / Ghost / Spatial / Inverted |
| 别名机制 | — | — | MOE = ES_MOE（yaml 兼容） |
| 平衡损失 | — | — | MoELoss（[loss.py](./moe-loss-py)） |
| Top-K 退火 | — | — | DynamicTopKScheduler（[topk_anneal.py](./moe-topk-anneal-py)） |

**yaml 里看到什么就是什么**：Master 的 `yolo-master-n.yaml` 写 `ES_MOE`，`parse_model` 走 ES_MOE 独立分支；老 checkpoint 里可能写 `MOE`，`__init__.py` 的 `MOE = ES_MOE` 保证两条路都走通。A2C2fMoE 与 yaml 里直接写 A2C2f 是**两个不同的类**——[痛点 10 §A2C2f 与 A2C2fMoE 的隐性差异](../painpoints#10-taskspy-特例的隐性耦合-pain-tasks-coupling)。

路由层见 [routers.py 源码页](./moe-routers-py)，专家见 [experts.py 源码页](./moe-experts-py)，损失见 [loss.py 源码页](./moe-loss-py)，Top-K 退火见 [topk_anneal.py 源码页](./moe-topk-anneal-py)。架构总览见 [architecture 页](../architecture)，方法学见 [concepts 页](../concepts)。

## 常见问题

**Q：ES_MOE 的 default num_experts 是 4 还是 3？**
A：`ES_MOE.__init__` 里默认 `num_experts=4`（[L417](#L417)），但 [DynamicRoutingLayer 默认 num_experts=3](./moe-routers-py)。yaml 里 ES_MOE 只传 `in_channels`——**实际运行时 num_experts=4**（ES_MOE 默认）。若需要 3 个 expert，需显式传入 `num_experts=3`。

**Q：expert_kernel_sizes 什么时候必须传？**
A：当 pruned checkpoint 加载后再 `train()` 时——[L478-L493](#L478-L493) 的 `expert_kernel_sizes` 让模型按 pruned 时的 kernel 尺寸重建，而不是回退到默认 `[3, 5, 7]`。这是 Master 支持的「prune → fine-tune 恢复」路径。

**Q：ES_MOE 与 MOE 别名怎么用的？**
A：[L1775 `MOE = ES_MOE`](#L1775)——yaml 里写 `MOE` 或 `ES_MOE` 完全等价。老 checkpoint（.pt 文件）里如果存的类名是 `MOE`，加载时会自动解析为 ES_MOE。

**Q：为什么文件里还有 HyperUltimateMoE？**
A：这是 Master 早期的「最优化 MoE」实验版本，未在官方 yaml 中启用——**代码存在但不推荐使用**。用户应使用 ES_MOE。
