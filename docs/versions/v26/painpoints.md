# YOLO26 痛点追踪

YOLO26（Ultralytics 官方仓库 commit `2e241d0`，2026 版，AGPL-3.0，arXiv:2606.03748）是 YOLO 系列**第一个「6 任务方向全部首发官方权重」的版本**——Detection / Segmentation / Pose / Depth / Semantic / OBB / Classification 全部有 `yolo26*.pt` 直接下载。但代价是：**v26 的痛点结构从「结构层静默决策」变成了「README 与代码偏差 / 训练策略文档缺失」**。本页追踪 10 个痛点，每个都配 v1~v14 路标。

## 痛点全景

> 已读过 [v14 痛点页](/versions/v14/painpoints)？v14 的 10 个痛点集中在「README 声称但源码 grep 不到」与「训练侧静默失效」；**v26 的痛点换了一批**——**E2E 权重退火 README 未描述**（`E2ELoss.decay` 0.8→0.1）、**`E2EDetectLoss` vs `E2ELoss` 两套并存但 README 只提一个**、**`flow_model` 兜底 `_rle_zero` 是隐藏 workaround**、**`reg_max: 1` DFL 退化机制 README 描述与代码不一致**、**Depth SILog λ=1.0 实际是「中心化方差」而非「纯 scale-invariant」**、**Semantic `persistent=False` buffer README 未提**、**`A2C2f` / `AAttn` 类保留但 yaml 0 处实例化**、**`SAVPE` / `RealNVP` / `Proto26` 三个新模块只在特定任务 yaml 用**、**TAL `topk=13 α=1.0` 类默认与 `topk=10 α=0.5 β=6.0` 显式覆盖并存**、**AGPL-3.0 许可沿用**。

| 痛点 | 严重度 | v1~v14 处理 | v26 处理 | 后续影响 |
| --- | --- | --- | --- | --- |
| [1. E2E 权重退火 README 未描述](#1-e2e-权重退火-readme-未描述-pain-e2e-decay-undocumented) | **High** | v1~v14 无 E2E | `E2ELoss.decay` 0.8→0.1 是 v26 核心机制，README 只说 `end2end: True` | 复现训练结果与官方对不上 |
| [2. `E2EDetectLoss` vs `E2ELoss` 两套并存](#2-e2ededectloss-vs-e2eloss-两套并存-pain-e2e-two-implementations) | **High** | v1~v14 无 E2E | 代码里两套实现，README 只提 `E2ELoss` | 选择哪个版本影响结果 |
| [3. `reg_max: 1` DFL 退化机制 README 与代码不一致](#3-reg_max-1-dfl-退化机制-readme-与代码不一致-pain-dfl-degenerate) | **High** | v8~v14 DFL 明确 reg_max=16 或 1 | v26 reg_max=1 走 DFL 分支但 bias=1 返回 0 | 复现 DFL 相关代码会误解 |
| [4. `flow_model` 兜底 `_rle_zero` 是隐藏 workaround](#4-flow_model-兜底-_rle_zero-是隐藏-workaround-pain-flow-model-workaround) | High | v1~v14 无 RLE | `pred_kpt[..., :0].sum() + sum(p[..., :0].sum())` 让 flow 参数进入计算图 | 调试时看不到 flow 梯度 |
| [5. Depth SILog λ=1.0 实际是「中心化方差」](#5-depth-silog-λ10-实际是中心化方差-pain-silog-centering) | Medium | v1~v14 无 Depth | README 说「scale-invariant」，代码里减去 `log_diff.mean()` | 论文公式与代码公式不完全等价 |
| [6. Semantic `persistent=False` buffer README 未提](#6-semantic-persistentfalse-buffer-readme-未提-pain-persistent-false) | Medium | v1~v14 无语义分割 | Cityscapes 19 类权重不进 state_dict | checkpoint 加载时权重被丢弃 |
| [7. `A2C2f` / `AAttn` 类保留但 yaml 0 处实例化](#7-a2c2f--aattn-类保留但-yaml-0-处实例化-pain-a2c2f-unused) | Medium | v11/v12/v13 用 A2C2f | 类定义在 block.py，yaml 里 0 处 | 「继承 v11」是名义而非结构 |
| [8. `SAVPE` / `RealNVP` / `Proto26` 三个新模块只在特定 yaml 用](#8-savpe--realnvp--proto26-三个新模块只在特定-yaml-用-pain-new-modules-partial) | Medium | v1~v14 无这三个 | block.py 定义，只有 pose/seg yaml 用 | 检测 yaml 加载不到新模块 |
| [9. TAL 类默认 vs 显式覆盖并存](#9-tal-类默认-vs-显式覆盖并存-pain-tal-default-vs-explicit) | Medium | v8~v14 显式覆盖 | `topk=13 α=1.0` 类默认 vs `topk=10 α=0.5 β=6.0` 显式 | 直接实例化 TaskAlignedAssigner 用默认值会不同 |
| [10. AGPL-3.0 许可沿用](#10-agpl-30-许可沿用-pain-agpl) | Low | v8~v14 混合 | v26 沿用 AGPL-3.0 | 商用/闭源用户受限 |

## 1. E2E 权重退火 README 未描述 (`pain-e2e-decay-undocumented`)

> **v1~v14 读者路标**：v1~v14 都没有 E2E 机制，v26 首次引入——**但 README 只说 `end2end: True`，没提 `E2ELoss.decay` 的 0.8→0.1 曲线**。

**README 描述**：

> `end2end: True # whether to use end-to-end mode`
>
> YOLO26 introduces an end-to-end training mode that eliminates the need for NMS during inference.

**代码实现**（<SrcLink file="v26:utils/loss.py" :line="1283" :end="1315" />）：

```python
class E2ELoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.o2m = 0.8  # 初始 one2many 权重
        self.o2o = 0.2  # 初始 one2one 权重
        self.o2m_copy = 0.8  # 退火起点
        self.final_o2m = 0.1  # 退火终点
    
    def decay(self, x):
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m
```

**偏差详情**：
- **README 未描述 0.8→0.1 曲线**：读者无法预知训练后期的 one2many 权重会降到 0.1；
- **README 未描述 `decay` 是 per-epoch 而非 per-iteration**：与 v14 的 τ per-iter 不同；
- **README 未描述推理时只跑 one2one**：读者可能以为推理时也跑两分支。

**影响**：
- **复现训练结果与官方对不上**：如果用户手动设置 `w_o2m=0.5` 固定值，会得不到官方的 NMS-free 精度；
- **调试时无法理解 loss 曲线**：训练早期的 loss 会明显低于后期（因为 one2many 权重高）。

## 2. `E2EDetectLoss` vs `E2ELoss` 两套并存 (`pain-e2e-two-implementations`)

> **v1~v14 读者路标**：v1~v14 无 E2E，v26 首次引入——**但代码里两套实现并存，README 只提 `E2ELoss`**。

**代码位置**：
- `E2EDetectLoss`：<SrcLink file="v26:utils/loss.py" :line="1263" :end="1280" />（等权相加）
- `E2ELoss`：<SrcLink file="v26:utils/loss.py" :line="1283" :end="1315" />（权重退火）

**差异对比**：

| 维度 | `E2EDetectLoss` | **`E2ELoss`** |
|------|-----------------|---------------|
| 权重 | 等权（0.5 + 0.5） | **退火（0.8→0.1 + 0.2→0.9）** |
| 推荐用于生产 | ❌ | **✅** |
| README 描述 | ❌ | **✅** |
| COCO mAP | 未报告 | **官方报告** |

**偏差详情**：
- **README 只提 `E2ELoss`**：用户不知道还有 `E2EDetectLoss`；
- **两套实现参数不同**：`E2EDetectLoss` 无 `decay` 方法，`E2ELoss` 有；
- **训练时选择影响结果**：如果用 `E2EDetectLoss`，训练末期 one2many 权重仍是 0.5，模型偏向「多分支共识」而非「单分支最优」。

**影响**：
- **复现时可能选错版本**：用户如果用 `E2EDetectLoss`，得到的 NMS-free 精度会比官方低；
- **调试时无法对比**：两套实现没有明确的使用场景说明。

## 3. `reg_max: 1` DFL 退化机制 README 与代码不一致 (`pain-dfl-degenerate`)

> **v1~v14 读者路标**：v8 用 reg_max=16，v11~v14 用 reg_max=1；v26 也用 reg_max=1——**但 README 说「DFL 已退化」，代码里仍走 DFL 分支**。

**README 描述**：

> `reg_max: 1 # DFL bins`
>
> YOLO26 uses reg_max=1 to simplify the distribution focal loss, eliminating the need for the DFL head.

**代码实现**（<SrcLink file="v26:utils/loss.py" :line="90" :end="112" />）：

```python
def __init__(self, reg_max=16):
    self.reg_max = reg_max

def forward(self, x):
    if self.reg_max <= 1:  # v26: reg_max=1
        # 判断路径：ch[i] * reg_max > 1 → True（因为 reg_max=1 时 ch[i] 已经很大）
        # 实际走 DFL 分支，但 bias 长度为 1，F.linear(x, [0]) 返回 0
        # 等价于 Identity
```

**偏差详情**：
- **README 说「DFL 已退化」**：暗示不走 DFL 分支；
- **代码实际走 DFL 分支**：只是 bias 长度为 1 导致 `F.linear` 返回 0；
- **判断路径 `ch[i] * reg_max > 1` 为 True**：因为 reg_max=1 时 ch[i] 很大（比如 4），4*1=4>1。

**影响**：
- **复现 DFL 相关代码会误解**：如果用户想「关闭 DFL」，改 `reg_max=0` 会让判断路径为 False，走 Identity——**但这不是官方用法**；
- **调试 DFL 分支时困惑**：为什么 DFL 分支明明走了但梯度为 0？

## 4. `flow_model` 兜底 `_rle_zero` 是隐藏 workaround (`pain-flow-model-workaround`)

> **v1~v14 读者路标**：v1~v14 无 RLE，v26 首次引入——**`flow_model` 兜底是隐藏的 BaseTrainer 兼容性 workaround**。

**代码实现**（<SrcLink file="v26:utils/loss.py" :line="810" :end="932" />）：

```python
class PoseLoss26(v8PoseLoss):
    def __init__(self, model, kpt_shape, flow_model=None):
        super().__init__(model, kpt_shape)
        self.flow_model = flow_model  # 可选 RealNVP
    
    def __call__(self, preds, batch):
        det_loss, det_items = super().__call__(preds, batch)
        # RLE 损失
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        error = error.clamp(-100, 100)
        valid_mask = ~(isnan(error) | isinf(error)).any(dim=-1)
        rle_loss = self.rle_loss(pred_sigma, pred_phi, gt_coords, target_weights)
        
        # 兜底：让 flow_model 参数进入计算图
        _rle_zero = pred_kpt[..., :0].sum() + sum(p[..., :0].sum() for p in self.flow_model.parameters())
        rle_loss = rle_loss + _rle_zero
        
        return det_loss + rle_loss, det_items
```

**偏差详情**：
- **README 未提 `_rle_zero`**：用户不知道 flow_model 参数通过空切片进入计算图；
- **`[..., :0]` 是空切片**：不影响数值，但让 flow_model 参数进入计算图；
- **BaseTrainer 期望所有可训练参数有梯度**：这个 workaround 是为了避免 `RuntimeError: element 0 of tensors does not require grad`。

**影响**：
- **调试时看不到 flow 梯度**：如果用户想看 flow_model 的梯度，需要知道这个 workaround；
- **BaseTrainer 兼容性**：如果未来 BaseTrainer 改变，这个 workaround 可能失效。

## 5. Depth SILog λ=1.0 实际是「中心化方差」 (`pain-silog-centering`)

> **v1~v14 读者路标**：v1~v14 无 Depth，v26 首次引入——**README 说「scale-invariant」，代码里减去 `log_diff.mean()`**。

**README 描述**：

> YOLO26 uses a scale-invariant logarithmic RMSE (SILog) loss for depth estimation, which is invariant to depth scaling.

**代码实现**（<SrcLink file="v26:utils/loss.py" :line="1196" :end="1211" />）：

```python
def silog_loss(self, pred_depth, gt_depth, silog_lambda=1.0):
    log_diff = torch.log(pred_depth) - torch.log(gt_depth)
    m = log_diff.mean()  # 中心化
    silog = torch.sqrt(
        ((log_diff - m) ** 2).mean()  # 中心化方差
        + (1 - silog_lambda) * m**2    # 绝对偏差项
        + 1e-6
    )
    return silog
```

**偏差详情**：
- **README 说「scale-invariant」**：暗示 `λ=1.0` 时是「纯 scale-invariant」；
- **代码里 `λ=1.0` 时减去 `m²` 项**：`(1-λ)*m² = 0`，但 `log_diff - m` 是中心化方差——**不是「纯 scale-invariant」而是「中心化方差」**；
- **真正的「scale-invariant」需要 `λ=1.0` 且 m=0**：但 m 是数据决定的，不是 0。

**影响**：
- **论文公式与代码公式不完全等价**：SILog 论文定义 `L = sqrt(mean((log_d - log_d_pred)²) + λ*mean(log_d - log_d_pred)²)`，代码实现略有差异；
- **消融实验时需注意**：如果用户想用 `λ=0.0`（log-RMSE），代码里 `(1-λ)*m² = m²` 会让梯度依赖绝对偏差。

## 6. Semantic `persistent=False` buffer README 未提 (`pain-persistent-false`)

> **v1~v14 读者路标**：v1~v14 无语义分割，v26 首次引入——**`persistent=False` buffer 是隐藏的 checkpoint 优化**。

**代码实现**（<SrcLink file="v26:utils/loss.py" :line="1430" :end="1442" />）：

```python
class SemanticSegmentationLoss:
    def __init__(self, nc, data_name=""):
        self.nc = nc
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"}
        if self.use_cityscapes_weight:
            weight = CITYSCAPES_WEIGHT  # 19 类硬编码权重
            self.register_buffer("weight", weight, persistent=False)  # 非持久化
```

**偏差详情**：
- **README 未提 `persistent=False`**：用户不知道 Cityscapes 权重不进 state_dict；
- **`persistent=False` 避免重复序列化**：不同 GPU 上的权重加载冲突被避免；
- **checkpoint 大小优化**：19 类权重不进 checkpoint，减少文件体积。

**影响**：
- **checkpoint 加载时权重被丢弃**：如果用户手动加载 checkpoint，需要知道权重是硬编码的；
- **消融实验时需注意**：如果用户想修改 Cityscapes 权重，需要改代码而非加载 checkpoint。

## 7. `A2C2f` / `AAttn` 类保留但 yaml 0 处实例化 (`pain-a2c2f-unused`)

> **v1~v14 读者路标**：v11/v12/v13 用 A2C2f 作为骨干深段融合块，v14 用 DeformableA2C2f；**v26 完全回归 C3k2，A2C2f / AAttn 类保留但 yaml 0 处实例化**。

**代码位置**：
- `A2C2f`：<SrcLink file="v26:nn/modules/block.py" :line="1804" />
- `AAttn`：<SrcLink file="v26:nn/modules/block.py" :line="1447" />

**yaml 使用情况**：
- `yolo26.yaml`：0 处 A2C2f，0 处 AAttn
- 其他 10 份 yaml：0 处 A2C2f，0 处 AAttn

**偏差详情**：
- **类定义在 block.py**：v26 保留了 v11/v12 的 A2C2f / AAttn 类；
- **yaml 里 0 处实例化**：v26 完全回归 C3k2 融合块；
- **「继承 v11」是名义而非结构**：v26 骨干与 v11 结构相同（Conv + C3k2 + SPPF + C2PSA），但 v11 用 C3k2，v26 也用 C3k2。

**影响**：
- **代码冗余**：A2C2f / AAttn 类保留但不用，增加代码体积；
- **读者困惑**：为什么 v26 保留 v11 的模块但不用？

## 8. `SAVPE` / `RealNVP` / `Proto26` 三个新模块只在特定 yaml 用 (`pain-new-modules-partial`)

> **v1~v14 读者路标**：v1~v14 无这三个新模块，v26 首次引入——**只在特定任务 yaml 用**。

**代码位置**：
- `SAVPE`：<SrcLink file="v26:nn/modules/block.py" :line="1930" />
- `RealNVP`：<SrcLink file="v26:nn/modules/block.py" :line="2045" />
- `Proto26`：<SrcLink file="v26:nn/modules/block.py" :line="1998" />

**yaml 使用情况**：

| 模块 | 用途 | 用在哪些 yaml |
|------|------|--------------|
| `SAVPE` | 空间注意力视角感知池化 | **0 处**（保留但不用） |
| `RealNVP` | Normalizing Flow | `yolo26-pose.yaml` |
| `Proto26` | 实例分割掩膜头 | `yolo26-seg.yaml`, `yolo26-ed-seg.yaml` |

**偏差详情**：
- **`SAVPE` 保留但不用**：类似 A2C2f，类定义在但 yaml 0 处；
- **`RealNVP` 只在 pose yaml 用**：其他任务加载不到；
- **`Proto26` 只在 seg yaml 用**：其他任务加载不到。

**影响**：
- **新模块测试覆盖度低**：只在特定任务用，其他任务无法验证；
- **推理时加载失败**：如果用户在 Detection 模型里加载 Proto26，会失败。

## 9. TAL 类默认 vs 显式覆盖并存 (`pain-tal-default-vs-explicit`)

> **v1~v14 读者路标**：v8 首次留下这个隐性坑，v11~v14 没修；v26 继续沿用——**类默认与显式覆盖并存**。

**代码位置**：
- `TaskAlignedAssigner` 类默认：<SrcLink file="v26:utils/tal.py" :line="29" />（`topk=13, α=1.0`）
- `v8DetectionLoss` 显式覆盖：<SrcLink file="v26:utils/loss.py" :line="159" />（`topk=10, α=0.5, β=6.0`）

**偏差详情**：
- **类默认 `topk=13, α=1.0`**：tal.py L29 的类定义；
- **显式覆盖 `topk=10, α=0.5, β=6.0`**：loss.py L159 显式传参；
- **v8 以来七版没修**：这个隐性坑从未被修复。

**影响**：
- **直接实例化 `TaskAlignedAssigner` 用默认值会不同**：用户如果绕过 `v8DetectionLoss` 直接实例化 TAL，会用 `topk=13, α=1.0`；
- **调试时困惑**：为什么同一个类有两种不同的参数？

## 10. AGPL-3.0 许可沿用 (`pain-agpl`)

> **v1~v14 读者路标**：v8~v12 是 AGPL-3.0，v13 是 Apache-2.0，v14 回切 AGPL-3.0；v26 沿用 AGPL-3.0。

**许可证说明**：
- **AGPL-3.0**：Copyleft 许可证，商用需要开源修改后的代码；
- **与 Apache-2.0 的区别**：Apache-2.0 允许闭源商用，AGPL-3.0 不允许；
- **v26 沿用 AGPL-3.0**：与官方 Ultralytics 一致。

**影响**：
- **商用/闭源用户受限**：如果用 v26 开发商用产品，需要遵守 AGPL-3.0；
- **与 v13 对比**：v13 是 Apache-2.0，商用更友好；v26 回切 AGPL-3.0，商用受限。

## 痛点优先级建议

| 优先级 | 痛点 | 影响范围 | 修复建议 |
|--------|------|---------|---------|
| **P0** | 1. E2E 权重退火 README 未描述 | 所有 E2E 训练用户 | README 补充 `decay` 公式 |
| **P0** | 2. `E2EDetectLoss` vs `E2ELoss` 两套并存 | 所有 E2E 训练用户 | README 明确推荐 `E2ELoss` |
| **P1** | 3. `reg_max: 1` DFL 退化机制 | 调试 DFL 用户 | README 补充「判断路径 vs 实际行为」 |
| **P1** | 4. `flow_model` 兜底 | 调试 flow 用户 | README 补充 `_rle_zero` workaround |
| **P2** | 5. Depth SILog λ=1.0 | Depth 训练用户 | README 补充「中心化方差 vs scale-invariant」 |
| **P2** | 6. Semantic `persistent=False` | Semantic 训练用户 | README 补充「权重不进 checkpoint」 |
| **P3** | 7. `A2C2f` / `AAttn` 类保留但不用 | 代码维护者 | 考虑删除未用类 |
| **P3** | 8. `SAVPE` / `RealNVP` / `Proto26` 部分使用 | 代码维护者 | 文档说明每个模块的使用场景 |
| **P4** | 9. TAL 类默认 vs 显式覆盖 | 调试 TAL 用户 | 修复类默认与显式覆盖不一致 |
| **P4** | 10. AGPL-3.0 许可 | 商用用户 | 无修复（许可决定） |

## 想快速对上位读者的话

- **v8 读者**：v8 时代 README 与代码基本对齐，v26 出现 5 处 README 与代码偏差——**建议读代码而非 README**；
- **v11/v12 读者**：v11/v12 用 A2C2f，v26 保留 A2C2f 类但 yaml 不用——**「继承 v11」是名义而非结构**；
- **v13 读者**：v13 是 Apache-2.0，v26 回切 AGPL-3.0——**商用受限**；
- **v14 读者**：v14 的痛点是「README 声称但代码 grep 不到」，v26 的痛点是「README 与代码偏差」——**v26 的偏差更隐蔽**，因为代码确实存在，只是 README 没说清楚；
- **想复现 v26 官方结果**：用 `E2ELoss`（退火版），不要用 `E2EDetectLoss`（等权版）；`reg_max=1` 是必须的（DFL 退化）；`end2end=True` 是必须的（双分支）。

## 参考

- [v26 架构总览](/versions/v26/architecture)
- [v26 概念详解](/versions/v26/concepts)
- [v26 损失函数](/versions/v26/losses)
- [v14 痛点追踪](/versions/v14/painpoints)
- [v13 痛点追踪](/versions/v13/painpoints)
- [v8 痛点追踪](/versions/v8/painpoints)
- <SrcLink file="v26:utils/loss.py" :line="90" :end="112" />（DFL 退化）
- <SrcLink file="v26:utils/loss.py" :line="810" :end="932" />（PoseLoss26 flow_model 兜底）
- <SrcLink file="v26:utils/loss.py" :line="1169" :end="1211" />（DepthLoss26 SILog）
- <SrcLink file="v26:utils/loss.py" :line="1263" :end="1315" />（E2EDetectLoss / E2ELoss）
- <SrcLink file="v26:utils/loss.py" :line="1389" :end="1442" />（SemanticSegmentationLoss persistent=False）
- Ultralytics YOLO26 官方文档：https://docs.ultralytics.com/models/yolo26
