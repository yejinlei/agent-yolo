# YOLO 演化事件

每一次演化都是一个 `from → to` 的边，记录引入/淘汰了什么概念、层、损失、技巧，以及为什么。

## v1 → v2：从"无 anchor + 2 box"到"anchor + BN + 捷径"

| 引入 | 淘汰 |
|------|------|
| Anchor 网格 | `sqrt=1` 加速 |
| BatchNorm | 前 100k 随机 best_index |
| Passthrough / Route / Upsample 层 | — |
| `random-translate` 数据增强 | — |

- VOC2007 mAP@0.5：**63.4 → 69.4**
- FPS：**35 → 160**
- 关键理由：v1 的 grid cell 只有 2 box，无先验宽高比，定位粗糙。

## v2 → v3：多尺度 FPN + 9 anchors

- 首次一次推理出 3 种分辨率 (13×13 / 26×26 / 52×52)
- 9 anchors 通过 K-Means 从 VOC 聚类
- COCO mAP@0.5: **57.9**，51 FPS
- 关键理由：多尺度解决小目标召回问题。

## v3 → v4：技巧大合集

- Mosaic 数据增强、SPP、CIoU、DPN、Mish、WAM
- COCO mAP@0.5: **44.8** (v3 是 50.4)
- 关键理由：不是新架构，是把已知最好的技巧集齐。

## v4 → v5：换栈 PyTorch

- 从 darknet C 切到 PyTorch
- 工程化优先：pip install、开箱即用
- 引入 Focus 层 (space-to-depth)
- 精度略有回退，但易用性大幅提升

## v5 → v6 / v7：分叉

- **v6 (Megvii)**：Reparameterization + EP2P 两阶段训练
- **v7 (WongKinYiu)**：Model Scaling + EVA Attention + EPE loss

## v5 → v8：回到主线，全新范式

- 切换 **Anchor-free**
- **Decoupled Head**（分类回归分离）
- **C2f** 结构（CSP 增强）
- 引入 Varifocal + DFL 损失
- COCO mAP@0.5:0.95: **44.9**（v5x）

## v8 → v9：PGT + GELAN

- Program Guided Training 提升小目标
- GELAN 架构增强特征表达
- COCO mAP@0.5:0.95: **50.5**

## v9 → v10：NMS-free

- 用 **NWD** (Normalized Wasserstein Distance) + Assigner 替代 NMS
- FreqLoss 提升小样本
- COCO mAP@0.5:0.95: **53.8**

## v8 → v11：EMA + C3k2

- Efficient Model Architecture
- C3k2 块（融合卷积核）
- COCO mAP@0.5:0.95: **53.4**

## v11 → v12：Global Attention

- Global Attention + Spatial Attention
- 参数量减少、精度提升
- COCO mAP@0.5:0.95: **53.6** (v12) vs **53.4** (v11)

## v12 → v13：HyperACE 超图融合 + DS 深度可分离

- **骨干/颈全 DS 化**：所有 3×3 Conv 换成 DSConv（DWConv + PWConv），骨干主块由 C3k2 换成 DSC3k2（yaml 里 7 次）；A2C2f 沿用 v12 但 `mlp_ratio` 从 1.2 提到 1.5
- **Head 首层加 HyperACE**：在骨干出口处对 P3/P4/P5 做**超图融合**——`AdaHyperedgeGen` 把初始邻接矩阵 P_0 通过可学习增量 ΔP 更新（`num_heads=8`），`AdaHGConv` 做 vertex→edge→vertex 的两段映射
- **FullPAD_Tunnel 门控 ×7**：每个上采样拼接点前后各插一个 `FullPAD_Tunnel`，用 `nn.Parameter(0.0)` 门控"直通 vs 融合"，训练初期等价直通、后期学出融合
- **DownsampleConv（AvgPool 2×2 + Conv 1×1）** 取代 3×3 s2 卷积；**l/x 档被 tasks.py 强制走 Identity**（等价退化成 v8 的 C2f 下采样）
- **档位收敛到 4 档**（n/s/l/x，砍掉 m）；cfg 目录回到 `models/v13/`（带 v 前缀，与 v12 一致，纠正 v11 的"去 v"破例）
- **许可证回 Apache-2.0**：iMoonLab 仓库改用 Apache-2.0，摆脱 v12 的 AGPL-3.0 传染顾虑
- COCO mAP@0.5:0.95（论文 TABLE I）：v13-N/S/L/X = **41.6 / 48.0 / 53.4 / 54.8**，与 v12 的 L/X 差距收敛到 <1
- 关键理由：DS 化降参数与 FLOPs；超图融合引入"跨尺度非两两相关"，弥补 v12 AAttn 只作用于单尺度的不足

## v13 → v14（学术线）：YOLO-Omni 多相机感知 + GRL 域对抗

**发布**：zhangcbb/yolov14 @ `0b8610d`，AGPL-3.0 fork，arXiv 2608.04720v3，2026-08-14。作者自述"We name our detector **YOLO-Omni**"。

| 引入 | 淘汰 |
|------|------|
| CircularConv（环状卷积，处理全景/360° 特征卷绕） | 传统 3×3 卷积边界 |
| ViewEmbedding（6 类视角 embedding 注入 P2 特征） | 视角无关的输入假设 |
| DomainAdaptiveLayer（视角/域自适应门控） | 单一域训练假设 |
| DeformableA2C2f（在 A2C2f 上叠加 deformable） | 静态注意力 |
| DomainAdversarialLoss + GRL（域对抗训练） | 纯监督学习 |
| CrossViewConsistencyLoss（跨视角一致性） | 单视角监督 |
| 4 基准：Game / Fisheye / Drone / Panorama | 只跑 COCO |
| cfg 目录独立到 `cfg_models_v14/` | 与主线共用 `models/` |

- **骨干**：`CircularConv → ViewEmbedding → DomainAdaptiveLayer → DeformableA2C2f` 四段式，每段都注入"视角/域"信息；**A2C2f 骨架沿用 v13（mlp_ratio 从 1.2 → 2.0）**，C3k2 浅层沿用 v11/v12/v13 家族
- **头**：Detect + v14DetectionLoss（三分量 + GRL 域对抗 + 4 条 schedule）
- **任务**：Detect / Segment / Pose / OBB（4 任务，比 v13 少）
- **档位**：n/s/m/l/x 五档（恢复 v11/v12 的 m 档）
- **许可证**：AGPL-3.0 fork（回到 v10~v12 的 AGPL 阵营）
- COCO mAP@0.5:0.95：s 档 49.1（T4/FP16 344 FPS），比 v13 s 档 48.0 高 1.1；四基准（Game/Fisheye/Drone/Panorama）均 >43 AP
- 关键理由：**「统一多域鲁棒检测」而非 COCO SOTA 竞赛**——通过视角 embedding + 域对抗让模型在游戏、鱼眼、无人机、全景四类非常规图像上都能保持高精度

## v13 → v26（官方线）：End2End 双分支 + 6 任务统一

**发布**：Ultralytics 官方 @ `2e241d0`，AGPL-3.0 + Enterprise 双许可，arXiv 2606.03748，2026-06 发布。

| 引入 | 淘汰 |
|------|------|
| End2End 双分支头（`copy.deepcopy(cv2/cv3)`，DFL → `nn.Identity()`） | 传统 DFL 分布回归 |
| E2ELoss（one2many + one2one + `decay` 退火 0.8→0.1） | 单一 one2many 分配 |
| `_grouped_topk`（`end2end=True` 无 NMS 推断） | NMS 后处理 |
| SAVPE（Slope-Aware Vertical Point Encoding） | 传统坐标编码 |
| Proto26（v26 分割原型） | v8~v13 的 Proto |
| RealNVP 流模型（用于 YOLOE 文本对齐） | 静态文本-图像对齐 |
| BNContrastiveHead（对比学习头） | 单一分类头 |
| Depth + SemanticSeg 任务 | 只有 Detect/Segment/Pose/OBB |
| YOLOE（open-vocabulary）+ World 模式 | 固定类别数 |
| mobileclip2:b 文本编码器 | 无文本编码器 |
| cfg 目录独立到 `cfg_models_26/` | 与主线共用 `models/` |

- **骨干**：Conv-C3k2×4-SPPF-C2PSA（回归 v8/v10/v11 家族，不用 v12/v13 的 Attention 块）
- **头**：`Detect(end2end=True, reg_max=1)` — `copy.deepcopy` 双分支
- **损失**：`E2ELoss`（one2many + one2one + decay 退火）
- **9 种模式**：Detect / Segment / Pose / OBB / Depth / SemanticSeg / CLS / RTDETR / YOLOE / World
- **CFG**：`reg_max: 1`（DFL 退化）、`agnostic_nms: False`、`tal_topk: 10`
- **许可证**：AGPL-3.0 + Enterprise 双许可（首次引入商业许可分流）
- COCO mAP@0.5:0.95：v26 X 档 **57.5**（T4/FP16 660 FPS），比 v13 X 档 54.8 高 2.7
- 关键理由：**「端到端统一 6 任务 + 无 NMS」**——通过双分支 + decay 退火 + `_grouped_topk` 消除后处理，同时把分割、姿态、深度、语义分割、开放词汇（YOLOE）都纳入同一套代码路径

---

**主线 + 双线路叙事**：从"如何更快"（v1→v2→v3）→"如何更准"（v3→v4→v8）→"如何更高效"（v9→v10→v11→v12）→"如何更细粒度地融合与压缩"（v13）→ **2026 双线路分叉**：

- **学术线 v14**（YOLO-Omni）：走向「多域鲁棒」——通过视角 embedding + 域对抗 + 环状卷积，把检测器从"单一域 COCO SOTA 竞赛"推向"多相机、多视角、多域统一"；
- **官方线 v26**（Ultralytics End2End）：走向「端到端 6 任务统一」——通过双分支 + decay 退火 + `_grouped_topk`，把 9 种模式（含开放词汇 YOLOE、深度、语义分割）统一在同一套代码路径，消除 NMS 与 DFL 后处理。

两条线路都从 v13 分叉，共享 DSConv + C3k2 的骨干遗产，但在「头设计」和「任务广度」上走向完全不同的技术方向。

> **注意**：主线 v26 使用 `NormalizingFlow` 家族中的 `RealNVP` 变体（RealNVP = Real-valued Non-volume preserving Flow）作为其**独立组件**，与本节的"RealNVP 主线"是两个含义完全不同的东西，不要混淆。详见 [v26 RealNVP 组件说明](/versions/v26/index)。

---

## 衍生分支演化事件

主线之外三条并行研究路径的演化，按"从哪个主线版本 fork 出去"排列。

### v8 → YOLO-World（2024-01，Tencent AI Lab，CVPR'24）

| 引入 | 淘汰 |
|------|------|
| 文本 prompt 分支（Phrase Grounding / Open-Vocabulary Detection） | 闭集固定 80 类 |
| **PHF**（Prompt-Handling Module） | — |
| **BNContrastiveHead**（对比学习头） | — |
| **WorldDetect** 检测头（同时兼容开闭集） | Detect 只做闭集 |

- **动机**：让 YOLOv8 从固定 80 类扩展为开放词表——用户直接输入自然语言类别描述
- **精度**：CO-COCO 20.6 mAP（v8 是 30.2 但只 80 类；CO-COCO 是 6000+ 类开放集）
- **工程代价**：backbone 每层额外注入一次文本特征，参数量增加约 2.5M
- **基座**：YOLOv8n / s / m / l / x 全系列

### YOLO-World → YOLOE（2025-03，清华 THU-MIG，ICCV'25）

| 引入 | 淘汰 |
|------|------|
| **Visual Prompt**（框选/点击/涂鸦） | — |
| **Prompt-Free** 模式（无需任何 prompt 的零样本检测） | 必须给文本 prompt |
| **ImagePoolingAttn（IAP）** 统一三种 prompt 通道 | PHF 只处理文本 |
| **YOLOEDetect / YOLOESegment**（检测 + 分割双头） | 仅检测 |

- **动机**：三种 prompt 输入方式统一到一个模型里，同时打通检测与分割
- **精度**：COCO-Prompt 53.8 AP；Prompt-Free 模式 60.9 AP（v8 63.7 AP）
- **基座**：v8 / v11 / v26 三基座，每个基座独立发布权重
- **重要点**：不是 v8 的"补丁"，是独立的 detector 家族

### v12 → YOLO-Master（2025-12，Tencent Youtu + SMU，CVPR'26）

| 引入 | 淘汰 |
|------|------|
| **RepRTA**（Reparameterizable Token-Aware Attention） | 普通卷积/注意力 |
| **ES-MoE**（Explicit Sparse MoE） + `ExpertRouter` + `SoftmaxTopkGate` + `TopKScheduler` | 稠密参数 |
| **DynamicRoutingLayer**（可替换 Conv 的 MoE 层） | 静态 Conv |
| **WorldALFF** neck | 固定 concat |
| **SAVPE / LRPC**（稀疏位置编码） | ViT 全位置编码 |

- **动机**：把"稀疏大参数 + 训练推理一致"塞进 YOLOv12 家族，对标 Transformer-based detector
- **精度**：**42.4 mAP @ 1.62 ms**（T4 单卡）——同时打赢速度与精度两条基线
- **基座**：YOLOv12n / s（仅小尺寸）
- **重要点**：**RepRTA ≠ RealNVP**——RealNVP 是主线 v26 里 Normalizing Flow 的独立组件，两者不共享代码与语义
- **对比锚点**：与 [v13 学术线](/versions/v13/index) 同期，但路线完全不同（v13 改 backbone/neck，YOLO-Master 改参数结构）

### 三条分支的横向对照

| 维度 | YOLO-World | YOLOE | YOLO-Master |
|------|-----------|-------|-------------|
| fork 基座 | v8 | v8 / v11 / v26 | v12 |
| 核心新增 | PHF | IAP + YOLOE Detect/Segment | ES-MoE + RepRTA |
| 能力扩展 | 开放词表 | 开放词表 + 视觉 prompt + Prompt-Free + Seg | 大参数量 + 稀疏推理 |
| 会议 | CVPR'24 | ICCV'25 | CVPR'26 |
| arXiv | 2401.17270 | 2503.07465 | 2512.23273 |
| 与主线关系 | 完全独立 | 完全独立 | 完全独立 |

### 与主线演化的耦合点

- **v8 → v11 → v12 → v26** 主线连续演化中，每一版都被至少一个衍生分支复用为基座
- 三个衍生分支**不进入** `docs/versions/`（无编号），而是归入 `docs/derivatives/`
- 主线编号序列与衍生家族**并行不取代**：v26 之后是否会出现 v27，与 YOLO-Master 是否升级为新主线是两个独立判断

