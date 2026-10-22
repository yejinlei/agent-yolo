# YOLO26 架构与层速查

> 首发载体：Ultralytics 官方 `ultralytics` 仓库 commit `2e241d0`（2026 版），对应 arXiv:2606.03748。以 [yolo26.yaml](/versions/v26/source/yolo26-yaml) 为基准（**cfg 目录 `cfg_models_26/`**；5 档 scales：n/s/m/l/x，每档都是「depth × width × max_channels」三元组，见 [yaml L11-L17](file:///f:/src/agent-yolo/source/yolo26/snapshot/cfg_models_26/yolo26.yaml#L11-L17)），本页通道数为**缩放前基准值**（width=1.0）。总览页见 [YOLO26 架构总览](/versions/v26/architecture)。

> **v1~v14 读者路标**：这张 yaml 是 [yolov14.yaml](/versions/v14/layers) 的替代谱系——**骨干 11 层（v14 是 12 层）回归经典 Conv + C3k2 + SPPF + C2PSA 组合，不再有任何新引入的锚点模块**；**Head 13 层**（v14 也是 13 层，规模一致但结构不同）——**用 4 个 C3k2 融合块替代 v14 的 3 处 DeformableA2C2f + 1 处 C3k2**。总层数 24（v14 是 25）——**v26 更简单、更保守，但用 Detect(end2end=True) 把 NMS 干掉了**。

## 1. 全网络层一览表（yolo26.yaml L20-L52，基准宽度 @640 输入）

> **v1~v14 读者路标**：这张表可与 [v14 逐层图解 §1](/versions/v14/layers) 逐层对照——**骨干 12 层缩到 11 层，主要差在 v14 的 ViewEmbedding/DomainAdaptiveLayer/CircularConv 三处锚点全部撤出**；**Head 结构一致（FPN 上行 + PAN 下行 + Detect）但融合块从 DeformableA2C2f 换回 C3k2**。Detect 位置在 yaml L52。

### 1.1 Backbone（yaml L20-L32，11 层）

| 行号 | 层号 | 层类型 | 参数（yaml 原文） | 输出（基准 @640） | 说明 |
|-----|-----|-----------|--------------------|--------------------|------|
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="22" /> | 0 | `Conv` | 64, 3, 2 | 64×320×320 | **P1/2 stem** · 标准 stride-2 3×3 卷积 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="23" /> | 1 | `Conv` | 128, 3, 2 | 128×160×160 | P2/4 · 标准 stride-2 卷积 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="24" /> | 2 | `C3k2` | 256, **False**, 0.25 ×2 | 128×160×160 | P2 融合块 · C3k2 e=0.25 · no BN 短接 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="25" /> | 3 | `Conv` | 256, 3, 2 | 256×80×80 | P3/8 · 下采样 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="26" /> | 4 | `C3k2` | 512, **False**, 0.25 ×2 | 256×80×80 | P3 融合块 · C3k2 e=0.25 · no BN 短接 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="27" /> | 5 | `Conv` | 512, 3, 2 | 512×40×40 | P4/16 · 下采样 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="28" /> | 6 | `C3k2` | 512, **True** ×2 | 512×40×40 | **P4 融合块** · C3k2 with Bottleneck（e=0.5） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="29" /> | 7 | `Conv` | 1024, 3, 2 | 1024×20×20 | P5/32 · 下采样 |
| <SrcLink file="v26:cfg_models_26\yolo26.yaml" :line="30" /> | 8 | `C3k2` | 1024, **True** ×2 | 1024×20×20 | **P5 融合块** · C3k2 with Bottleneck |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="31" /> | 9 | `SPPF` | 1024, 5, 3, True | 1024×20×20 | **空间金字塔池化** · 5 分支 maxpool 3×3（3/5/7/9/11 感受野） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="32" /> | 10 | `C2PSA` | 1024 ×2 | 1024×20×20 | **v11 保留** · Cross-Stage Partial Attention（P5 语义聚合） |

**骨干关键节奏**：

- **11 层标准 Conv + C3k2 + SPPF + C2PSA 组合**——**v8/v10/v11/v12/v13 都有的骨干范式**，v14 的 3 处新锚点（ViewEmbedding / DomainAdaptiveLayer / CircularConv）在 v26 全部撤出；
- **两处浅层 C3k2（#2/#4）走 `no BN` 短接**（参数第 3 位 0.25 显式指定 `e=0.25`，同时 `shortcut=False`）——**降低浅层参数量**；
- **两处深段 C3k2（#6/#8）走 `with Bottleneck`**（第 2 参数 True，默认 e=0.5）——**深段用 C3k2 Bottleneck 表达力更强**；
- **#9 SPPF 是 v11~v13 的经典组合**：`C3k2 → SPPF → C2PSA` 三层「池化-注意力」收尾；
- **#10 C2PSA 是 v11/v12/v13/v14 保留的跨阶段注意力**：4-Head MultiheadAttention + Reshape + Conv 1×1，v26 保留原样；
- **无新引入模块**：v26 骨干是**「v11 骨干的官方快照」**——所有 block 都是 ultralytics 早就有的模块。

### 1.2 Head（yaml L35-L52，14 层含 Detect）

| 行号 | 层号 | 层类型 | 参数（yaml 原文） | 输出（基准 @640） | 说明 |
|-----|-----|-----------|--------------------|--------------------|------|
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="36" /> | 11 | `nn.Upsample` | None, 2, nearest | 1024×40×40 | P5 上采样到 P4 尺度 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="37" /> | 12 | `Concat` | `[-1, 6]` | 1536×40×40 | cat backbone P4（512）+ 上采样 P5（1024） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="38" /> | 13 | `C3k2` | 512, **True** ×2 | 512×40×40 | **FPN P4 融合块** · C3k2 with Bottleneck |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="40" /> | 14 | `nn.Upsample` | None, 2, nearest | 512×80×80 | P4 上采样到 P3 尺度 |
| <SrcLink file="v26:cfg_models_26\yolo26.yaml" :line="41" /> | 15 | `Concat` | `[-1, 4]` | 768×80×80 | cat backbone P3（256）+ 上采样 P4（512） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="42" /> | 16 | `C3k2` | 256, **True** ×2 | 256×80×80 | **P3/8-small 检测输入** |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="44" /> | 17 | `Conv` | 256, 3, 2 | 256×40×40 | PAN 下行 · P3 → P4 分辨率 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="45" /> | 18 | `Concat` | `[-1, 13]` | 768×40×40 | cat head P4（512）+ 下行 P3（256） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="46" /> | 19 | `C3k2` | 512, **True** ×2 | 512×40×40 | **P4/16-medium 检测输入** |
| <SrcLink file="v26:cfg_models_26\yolo26.yaml" :line="48" /> | 20 | `Conv` | 512, 3, 2 | 512×20×20 | PAN 下行 · P4 → P5 分辨率 |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="49" /> | 21 | `Concat` | `[-1, 10]` | 1536×20×20 | cat backbone P5（1024）+ 下行 P4（512） |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="50" /> | 22 | `C3k2` | 1024, **True**, **0.5**, **True** ×1 | 1024×20×20 | **P5/32-large 检测输入** · e=0.5 · cross_attn=True |
| <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="52" /> | 23 | `Detect` | `[16, 19, 22], nc` | — | Detect(P3, P4, P5)——**end2end=True 双分支收尾** |

**Head 关键节奏**：

- **标准 FPN 上行（#11-#16）+ PAN 下行（#17-#22）**——**v11~v14 通用骨架**，v13 的 HyperACE + FullPAD_Tunnel 骨架与 v14 的 DeformableA2C2f 全部撤出；
- **5 个融合块（#13/#16/#19/#22）都是 C3k2**——**回归 v11~v12 的 C3k2 融合范式**；
- **#22 是唯一「cross_attn=True + e=0.5」的 C3k2**：这是 v26 yaml 里唯一一处启用 `C3k2.attn`（Cross-Stage Attention）的位置——**P5-large 前用注意力补偿 PAN 下行的信息损失**；
- **#22 也是唯一 `n=1` 的 C3k2**：其余融合块都是 `n=2`——**最后一层只做 1 次 C3k2 迭代，节省计算**；
- **Detect 输入三处**：#16（P3, stride 8）、#19（P4, stride 16）、#22（P5, stride 32）——**与 v11~v14 完全一致**；
- **`end2end: True` + `reg_max: 1`**（yaml L9/L10）——**Detect 头内部走双分支：one2many + one2one**，NMS 在推理时消失。

### 1.3 Detect 双分支内部结构（end2end=True）

回看 [Detect.__init__](file:///f:/src/agent-yolo/source/yolo26/snapshot/nn/modules/head.py)（head.py L38-L154）：

```python
def __init__(self, ch=(256, 512, 1024), nc=80, ch_cls=1, ch_obj=1, ch_fg=32, reg_max=16, end2end=True):
    self.end2end = end2end
    self.reg_max = reg_max  # 1 → DFL 退化 Identity
    self.bbox_reg = (4 + reg_max) * 3  # reg_max=1 → 15 × 3 = 45

    # one2many 分支：常规 CV2/CV3 卷积（topk=10 TAL 分配器）
    self.cv2 = nn.ModuleList(BevmConv(ch[i], ch_fg, 3) for i in range(3))
    self.cv3 = nn.ModuleList(BevmConv(ch[i], (self.no - self.nk) * self.nl, 1) for i in range(3))

    if end2end:
        # one2one 分支：copy.deepcopy 复制一份（topk=1 TAL，速度优先）
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.bbox_reg * 2  # 双分支时通道加倍 → 90
```

**yaml 层的 Detect(#23) 内部实际结构**：

| 内部模块 | 尺寸 | one2many | one2one | 说明 |
|---------|------|---------|---------|------|
| `cv2` / `one2one_cv2` | 3×3 Conv | 3 个 | 3 个 | 特征精炼（`copy.deepcopy`） |
| `cv3` / `one2one_cv3` | 1×1 Conv | 3 个 | 3 个 | 分类+回归融合输出 |
| `dfl` | `F.linear` | 退化 | 退化 | **reg_max=1 → Identity** |

**要点**：

- **`copy.deepcopy` 是关键**——**one2one 分支完全独立于 one2many**，不共享权重；
- **reg_max=1 让 bbox 回归头从「4 方向 × 16 bins」缩到「4 方向 × 1 bin」**——**DFL 退化为 Identity，等价于直接输出 4 值**；
- **推理时走 one2one 分支**（`forward_one2one`）——**NMS-free，因为 topk=1 TAL 已经保证每 anchor 只有一个正样本**；
- **训练时同时跑两分支**（one2many 用 topk=10 TAL 分配密集监督，one2one 用 topk=1 TAL 学 NMS-free 决策）。

## 2. 十一份 yaml 变体结构对比

v26 首发十一份 yaml 变体，覆盖六个任务方向——**它们不是「同一模型的不同参数」，而是「不同 head 与 yaml 结构差异」**。

| 变体 | Detect 头 | Proto | Depth | Pose (RLE+Flow) | 目标任务 |
|------|-----------|-------|-------|-----------------|---------|
| `yolo26.yaml` | ✅ end2end | ❌ | ❌ | ❌ | Detection |
| `yolo26-cls.yaml` | `Classify` | ❌ | ❌ | ❌ | Classification (ImageNet) |
| `yolo26-seg.yaml` | `Segment26` | `Proto26` | ❌ | ❌ | Instance Segmentation |
| `yolo26-obb.yaml` | `OBB26` | ❌ | ❌ | ❌ | Oriented Detection (DOTAv1) |
| `yolo26-pose.yaml` | `Pose26` | ❌ | ❌ | ✅ RLE+RealNVP | Keypoint Pose (COCO) |
| `yolo26-depth.yaml` | `Depth` | ❌ | ✅ | ❌ | Depth Estimation (NYU) |
| `yolo26-segsemantic.yaml` | `SemanticSegment` | ❌ | ❌ | ❌ | Semantic Segmentation (Cityscapes) |
| `yolo26-ed.yaml` | `YOLOEDetect` | ❌ | ❌ | ❌ | Open-Vocabulary (MobileCLIP2) |
| `yolo26-ed-seg.yaml` | `YOLOESegment26` | `Proto26` | ❌ | ❌ | Open-Vocabulary Segmentation |
| `yolo26-ed-obb.yaml` | (ED OBB) | ❌ | ❌ | ❌ | Open-Vocabulary OBB |
| `yolo26-tvp.yaml` | (TVP) | — | ❌ | ❌ | Text-Vision Prompt |

**十一份变体的共同点**：
- 都基于 ultralytics 8.3.x 的 tasks.py / data.py / loss.py
- 都是 v26 版（`yolo26*.yaml` 命名）
- 都用 `end2end: True` + `reg_max: 1`（除分类）
- 档位都是 n/s/m/l/x 五档

**十一份变体的差异点**：
- `yolo26.yaml` 官方全量版：Detect(end2end=True) 双分支——**参数最少、速度最快、COCO mAP 最高**；
- `yolo26-cls.yaml`：骨干不变，去掉所有 FPN/PAN，直接用最后层做全局池化 + Classify 头——**ImageNet 71.4 → 79.9 top1**；
- `yolo26-seg.yaml`：Detect → Segment26，头部加 `Proto26`（继承 v10 Proto）——**实例分割 56.5 box / 47.0 mask**；
- `yolo26-obb.yaml`：Detect → OBB26，回归从 4 值 → 10 值（含 rotation + 4×2 顶点）——**DOTAv1 56.7 mAP**；
- `yolo26-pose.yaml`：Detect → Pose26，输出 keypoints，损失里加 RLE + RealNVP flow——**COCO 71.6 mAP**；
- `yolo26-depth.yaml`：Detect → Depth，输出 per-pixel depth，SILog + 4 尺度梯度 L1——**NYU Δ1=0.933**；
- `yolo26-segsemantic.yaml`：Detect → SemanticSegment，CE + Dice + Cityscapes 类权重——**Cityscapes 83.6 mIoU**；
- `yolo26-ed.yaml`：Detect → YOLOEDetect，多文本 prompt，融合 mobileclip2:b——**开放词汇检测**；
- `yolo26-ed-seg.yaml`：Segment26 → YOLOESegment26，开放词汇分割；
- `yolo26-ed-obb.yaml`：OBB26 → ED OBB；
- `yolo26-tvp.yaml`：TVP 变体，用于 Text-Vision Prompt 学习。

> **yaml 变体详细对比**：见 [variants 源码页](/versions/v26/source/yolo26-variants)。

## 3. 与 v8/v11/v12/v13/v14 结构对比

| 维度 | v8 | v11 | v12 | v13 | v14 | **v26** | 差异 |
|------|----|-----|-----|-----|-----|---------|------|
| 总层数 | 23 | 22 | 23 | 32 | 25 | **24** | 与 v11/v12 一致 |
| Backbone 层数 | 9 | 9 | 10 | 9 | 12 | **11** | v26 = v11 + C2PSA |
| Head 层数 | 14 | 13 | 13 | 23 | 13 | **14** | 加 1 层 Detect |
| 骨干入口 | Conv | Conv | Conv | Conv | CircularConv | **Conv** | 回到 Conv |
| 骨干浅层 | C3/C2f | C3k2 | C3k2 | DSC3k2 | C3k2 | **C3k2** | 回归 C3k2 |
| 骨干深段 | C3k2 ×4 | C3k2 ×2 | C2PSA ×2 | C2PSA ×2 | DeformableA2C2f | **C3k2 + SPPF + C2PSA** | v26 用 SPPF + C2PSA |
| Head 融合块 | Bottleneck-CSP | C3k2 | A2C2f | HyperACE | DeformableA2C2f | **C3k2** | 回归 C3k2 |
| 骨干新锚点 | — | — | — | ViewEmb+DomainAdapt | 3 处锚点 | **0** | v26 完全回归经典 |
| Detect 分支 | 单 | 单 | 单 | 单 | 单 | **双（end2end）** | v26 首次 E2E |
| DFL | reg_max=16 | reg_max=1 | reg_max=1 | reg_max=1 | reg_max=1 | **reg_max=1** | 完全一致（Identity） |
| NMS | 需要 | 需要 | 需要 | 需要 | 需要 | **不需要**（one2one） | v26 独有 |
| 任务方向 | 3 | 4 | 5 | 5 | 5 | **6** | 加 Depth + Semantic |
| scales 档数 | 5 | 5 | 4 | 4 | 5 | **5** | 与 v14 一致 |
| 许可证 | AGPL-3.0 | AGPL-3.0 | AGPL-3.0 | Apache-2.0 | AGPL-3.0 | **AGPL-3.0** | 与官方一致 |

**核心洞察**：v26 是**「v11 骨干的官方 2026 快照」**——**去掉 v14 的多域鲁棒锚点，回归 Ultralytics 官方谱系，用 Detect(end2end=True) 把 NMS 干掉，把 head 复杂度让给 E2E 双分支机制**。**这是「结构保守 + 训练激进」的组合**。

## 4. 五档 scales 详解（官方实测）

| 档位 | depth | width | max_channels | layers | Params | GFLOPs @640 |
|------|-------|-------|-------------|--------|--------|-------------|
| n | 0.50 | 0.25 | 1024 | 260 | 2,572,280 | 6.1 |
| s | 0.50 | 0.50 | 1024 | 260 | 10,009,784 | 22.8 |
| m | 0.50 | 1.00 | 512 | 280 | 21,896,248 | 75.4 |
| l | 1.00 | 1.00 | 512 | 392 | 26,299,704 | 93.8 |
| x | 1.00 | 1.50 | 512 | 392 | 58,993,368 | 209.5 |

- **v26 有 5 档**——与 v11/v14 一致；
- **n 档 2.57M 参数、6.1 GFLOPs**——**比 v14-n 的 5.0M/12.4G 更轻一半**（因为无 DomainAdaptiveLayer/ViewEmbedding）；
- **s 档 10.0M 参数、22.8 GFLOPs**——**与 v14-s 的 9.2M/21.5G 相近**，v26-s 略重但结构更简单；
- **m 档 21.9M 参数、75.4 GFLOPs**——**m 档从 s 档跳跃到 2× 参数**（因为 width 从 0.50 升到 1.00）；
- **l 档 26.3M 参数、93.8 GFLOPs**——**l 档用 depth=1.0**（不缩深度，只缩通道 max_channels=512）；
- **x 档 59.0M 参数、209.5 GFLOPs**——**x 档用 width=1.50**，是 n 档的 23 倍参数。

## 5. 与 v14 参数量对比（同档）

| 版本 | 档 | Params | FLOPs @640 | 结构亮点 |
|------|---|--------|-----------|---------|
| v14 Omni | n | 5.0M | 12.4G | CircularConv + ViewEmb + DomainAdapt ×2 |
| **v26** | **n** | **2.57M** | **6.1G** | Conv + C3k2 + SPPF + C2PSA + Detect(end2end) |
| v14 Omni | s | 9.2M | 21.5G | 3 处锚点 + 5 处 DeformableA2C2f |
| **v26** | **s** | **10.0M** | **22.8G** | 经典骨干 + C3k2 融合 + Detect(end2end) |
| v14 Omni | m | 17M | 39.1G | m 档补回 |
| **v26** | **m** | **21.9M** | **75.4G** | width=1.0 满通道 |
| v14 Omni | l | 35M | 78.2G | depth=1.0 |
| **v26** | **l** | **26.3M** | **93.8G** | depth=1.0 + max_ch=512 |
| v14 Omni | x | 55M | 118G | width=1.5 |
| **v26** | **x** | **59.0M** | **209.5G** | width=1.5 |

**要点**：
- **v26-n 比 v14-n 轻一半**（2.57M vs 5.0M）——**因为无 3 处锚点**；
- **v26-s 比 v14-s 重 8%**（10.0M vs 9.2M）——**因为 v26-s 用 C3k2 with Bottleneck**；
- **v26-l 比 v14-l 轻 25%**（26.3M vs 35M）——**因为 v26 l 档 max_channels=512 抑制了 P5 通道爆炸**；
- **v26-x 参数与 v14-x 相当**（59M vs 55M），但 FLOPs 高 78%（209.5 vs 118）——**因为 v26 x 档 width=1.5 时 C3k2 Bottleneck 通道加倍**。

> **详细性能数据**：见 [paper.md §5 关键指标](/versions/v26/paper#关键指标)。

## 6. Detect 双分支与 NMS 消亡

v26 最大结构变化是 `end2end=True`——**Detect 头内部分成 one2many + one2one 两个分支**。

| 分支 | topk | 用途 | 推理时使用 | NMS |
|------|------|------|-----------|-----|
| one2many | 10 | 训练密集监督 | ❌ | 需要 |
| **one2one** | **1** | **推理 NMS-free** | **✅** | **不需要** |

**yaml 层看 Detect(#23) 是一个模块**，但内部有 **6 个 Conv**（one2many cv2×3 + cv3×3 + one2one cv2×3 + cv3×3）——**这才是「260 layers, 2.57M params」的来源**。

**要点**：
- **推理时只跑 one2one 分支**——**one2many 分支被剪掉**（推理部署时 `head.one2many_mode = False`）；
- **训练时两个分支各走一遍 loss**：`E2ELoss = 0.8 × loss_o2m + 0.2 × loss_o2o`（衰减到 0.1/0.9）；
- **topk=1 的 one2one 分配器天然保证每 anchor 一个正样本**——**NMS 是「后验冗余」，可以完全去除**。

> **E2E 双分支详细推导**：见 [concepts.md §1 E2E 双分支检测](/versions/v26/concepts#1-e2e双分支检测端到端消除nms)。

## 7. Fast Mode / 推理开关

v26 **没有 v14 那种「YOLOV14_FAST 环境变量」**——**Ultralytics 官方采用更简化的部署策略**：

| 开关 | 训练模式 | 推理模式 | 说明 |
|------|---------|---------|------|
| `end2end=True` | one2many + one2one | **one2one only** | Detect.head 内部自动切换 |
| `reg_max=1` | Identity（DFL 退化） | Identity | 无需切换 |
| `one2many_mode` | True | False | 推理时剪掉 one2many |
| `flow_model`（Pose） | 启用 | 训练后蒸馏或保留 | RealNVP flow |

**要点**：
- **v26 的「Fast Mode」= 直接走 one2one 分支**——**推理时不加载 one2many 权重**，参数量从 ~10M 缩到 ~5M；
- **无环境变量开关**——**部署策略写在 `Detect.__init__` 里**，`if self.end2end:` 分支控制；
- **`flow_model`（Pose RLE）保留权重**——**RealNVP 流模型推理时仍在**，因为 RLE 需要 sigma 采样。

> **读者路标**：Deploy 策略在 [concepts.md §1/§5](/versions/v26/concepts#1-e2e双分支检测端到端消除nms) 有对应实现分支，本节是索引。

## 8. 参考

- [v26 架构总览](/versions/v26/architecture)
- [v26 概念详解](/versions/v26/concepts)
- [v26 损失全景](/versions/v26/losses)
- [v26 论文-代码偏差](/versions/v26/painpoints)
- [v14 层速查](/versions/v14/layers)
- 官方 YAML：[yolo26.yaml L20-L52](file:///f:/src/agent-yolo/source/yolo26/snapshot/cfg_models_26/yolo26.yaml#L20-L52)
- Detect 实现：<SrcLink file="v26:nn/modules/head.py" :line="38" :end="154" />
