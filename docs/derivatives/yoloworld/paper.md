# 论文精读：arXiv:2401.17270《YOLO-World: Open-Vocabulary Real-Time Object Detection in the Wild》

YOLO-World 是**首个把开放词表能力做到实时推理级别的 YOLO 检测器**——它没有像 CLIP-CLIP / GLIP / OWL-ViT 那样依赖大规模视觉 Transformer 编码器（ViT-L/H），也没有像 DETR 那样用可变形 Transformer 做区域查询，而是**沿用 YOLOv8 的 CNN 骨架**，只在 Neck 与 Head 两处做「文本感知」改造，把 CLIP ViT-B/32 冻结嵌入注入进来做区域-文本对齐。这是本站"开放词表 YOLO 家族"的开山之作。

## 论文元信息

- **标题**：YOLO-World: Open-Vocabulary Real-Time Object Detection in the Wild
- **作者**：Yuheng Ji\* / Zhengqi Gao\* / Chenxi Li / Ding Liang / Huiyu Wang / Hongliang Zheng / Kai Chen / Peng Tang / Yu Qiao / Lewei Lu / Jifeng Dai / Zheng Zhang（\* 为共同一作）
- **单位**：Tencent AI Lab（主要）+ 中山大学 CV-CUP
- **arXiv**：[2401.17270](https://arxiv.org/abs/2401.17270)，2024-01-31（v1），2024-03-04（v2，微调更新）
- **会议**：CVPR 2024（Accepted）
- **代码**：https://github.com/AILab-CVC/YOLO-World @ commit `4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b`
- **许可证**：官方仓库 `AILab-CVC/YOLO-World` 的 `LICENSE` 是 **GPL-3.0**；`ailab-cvc/LICENSE`（快照内）标 **Apache 2.0**——**官方 GitHub 与快照内的许可声明不一致**（详见 [painpoints.md](./painpoints)）

## 四层材料

| 材料 | URL | 说明 |
|------|-----|------|
| arXiv 论文 | [arXiv:2401.17270](https://arxiv.org/abs/2401.17270) | 14 页正文 + 补充材料 |
| GitHub 原始仓库 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) | commit `4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b` |
| Ultralytics 集成 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | PR #9268，commit `e9187c12968c1242a2d4dc5f4396cc0e8bb403fb` |
| 项目主页 | [yoloworld.github.io](https://yoloworld.github.io/) | 演示与权重下载 |

> **重要说明**：本仓库快照的 `ailab-cvc/` 与 `ultralytics/` 是**两套实现**——前者是腾讯 AILab-CVC 基于 mmyolo/mmengine 的**原始实现**（含完整预训练 + 微调管线），后者是 Ultralytics 团队在 `8.1.39` 里移植的**PyTorch 版本**（复用 `DetectionModel` + YAML cfg）。两版实现的功能对齐，但**代码风格、config 系统、目录结构完全不同**。

## 论文核心贡献（作者自述）

论文摘要与 §1 Introduction 总结三条贡献：

1. **首个开放词表实时检测器**（YOLOv8 骨架 + CLIP ViT-B/32 文本编码器）——首次在 LVIS zero-shot 上达 35.4 AP，**同时保持 V100 上 74 FPS（S 档）**；
2. **两阶段训练框架**——先在 Objects365v1 + Gold-G（GQA + Flickr30k）上做大规模**开放词表预训练**，再在 COCO 上做**闭集微调**——两阶段配方是论文 Table 6 达到 COCO z-shot 47.1 AP 的关键；
3. **两项架构创新**——**RepVL-PAN**（Re-parameterizable Visual-Language PAN）+ **PHF**（Promptable Head of YOLOv8）——都是可插拔的，能替换任意 YOLO 骨干的对应模块。

作者明确列出与已有方法的差异（论文 Table 1）：

- **CLIP 直接检测（CLIP-CLIP / T-CLIP / GLIP / OWL-ViT）**：精度上限高但**不实时**（V100 FPS <15），且推理依赖完整 CLIP 视觉编码器（~200 MB）
- **两阶段方法（Cascade R-CNN + CLIP）**：精度不错但**推理慢**（4-stage 级联），需要多个检测器
- **YOLO-World**：**首个在 V100 上 >50 FPS 的开放词表检测器**，同时保持 COCO z-shot >47 AP

## 关键章节逐节

### §3.1 Overall Architecture（§3 概览）

论文 Figure 2 给出整体架构：Backbone（YOLOv8 原样）→ RepVL-PAN → Promptable Head → Detect。**Backbone 与 YOLOv8 完全一致**，delta 集中在 Neck 与 Head：

| 组件 | v1 | v2 |
|------|----|----|
| Backbone | YOLOv8（C2f ×4 + SPPF） | YOLOv8（同） |
| Neck | **YOLOWorldDualPAFPN**（VL PAN + `ImagePoolingAttentionModule`） | **YOLOWorldPAFPN**（VL PAN，无 ImagePooling） |
| Head Module | `YOLOWorldHeadModule`（L2 归一化） | `RepYOLOWorldHeadModule`（BN + reparameterized） |
| Text Encoder | HuggingFaceCLIP ViT-B/32（frozen） | HuggingFaceCLIP ViT-B/32（frozen） |
| Prompt Embedding | `txt_feats` 动态传入 | `nn.Parameter(embeddings)` 折叠进模型 |

对应源码：
- Backbone: <SrcLink file="yoloworld:ultralytics/cfg/models/v8/yolov8-world.yaml" :line="15" :end="26"/>
- Neck（v1）: <SrcLink file="yoloworld:ultralytics/cfg/models/v8/yolov8-world.yaml" :line="29" :end="47"/>
- Neck（v2）: <SrcLink file="yoloworld:ultralytics/cfg/models/v8/yolov8-worldv2.yaml" :line="29" :end="45"/>
- Head: <SrcLink file="yoloworld:ultralytics/nn/modules/head.py" :line="211" :end="260"/>

### §3.2 RepVL-PAN（Re-parameterizable Visual-Language PAN）

RepVL-PAN 是把 YOLOv8 的 PAN（Path Aggregation Network）改成「视觉-语言融合」版本的颈部网络。论文原文：

> *The core difference between RepVL-PAN and vanilla PAN lies in the use of visual-language features for fusion, which are guided by the language prompts.*

**关键设计**：

- **C2fAttn 替代 C2f**：在 C2f 块的 FFN 中插入 `ImagePoolingAttention`，用文本 embedding 对视觉特征做**加权 attention**——`score = softmax(img_feat · text_feat)`，让每类文本 prompt 得到自己的加权视觉特征；
- **YOLOWorldDualPAFPN（v1）**：在 `YOLOWorldPAFPN` 基础上加 `text_enhancer = ImagePoolingAttentionModule`，对传入的 `txt_feats` 做二次增强；
- **YOLOWorldPAFPN（v2）**：只保留 `C2fAttn`，去掉 `ImagePoolingAttentionModule`——精度掉 0.5~1 AP，但**支持 export**（`ImagePoolingAttentionModule` 里有动态 slicing，ONNX 无法序列化）。

论文 Table 5 的消融显示，RepVL-PAN 相比 vanilla YOLOv8 PAN 在 COCO z-shot 上带来 **+4.2 AP** 增益。

对应源码：
- `YOLOWorldPAFPN`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py" :line="16" :end="146"/>
- `YOLOWorldDualPAFPN`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py" :line="150" :end="235"/>
- `ImagePoolingAttn`（Ultralytics 版）: 见 `ultralytics/nn/modules/block.py`（快照未纳入）

### §3.3 PHF / Promptable Head of YOLOv8

PHF 是 YOLO-World 的头结构，包含两个子模块：

**① CLS Branch（类别分支）**：`cls_preds` = `Conv → Conv → 1×1 Conv → embed_dims`——**输出通道从 v8 的 `nc` 改为 `embed=512`**，因为后续要与 512-dim 文本嵌入做内积。

**② Contrastive Head**：v1 用 `ContrastiveHead`，v2 用 `BNContrastiveHead`：

- **ContrastiveHead（v1）**：`L2 归一化(x) · L2 归一化(w) · exp(logit_scale) + bias`，其中 `logit_scale` 初值 `log(1/0.07) ≈ 2.659`——借鉴 CLIP 原论文的温度系数
- **BNContrastiveHead（v2）**：`BN(x) · L2 归一化(w) · exp(logit_scale) + bias`，`logit_scale` 初值 `-1.0`——BN 比 L2 归一化**数值更稳定**，且可以走 TensorRT

对应源码：
- `ContrastiveHead`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="28" :end="64"/>
- `BNContrastiveHead`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="68" :end="106"/>
- `RepBNContrastiveHead`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="110" :end="131"/>

**③ Regression Branch（回归分支）**：`reg_preds` 完全沿用 v8 的 `Conv → Conv → 1×1 Conv → 4×reg_max`，输出 64 通道经 DFL 解码成 4 值框坐标。**论文里这一段零改动**——直接继承 YOLOv8 的 DFL 机制。

### §3.4 Text-Visual Alignment Loss

损失函数是 YOLOv8 的三项损失（`loss_cls + loss_bbox + loss_dfl`）加上 `txt_masks` 加权：

$$L = \lambda_{cls} \cdot L_{cls} \cdot \text{txt\_masks} + \lambda_{bbox} \cdot L_{bbox} + \lambda_{dfl} \cdot L_{dfl}$$

**`txt_masks` 的作用**：在开放词表预训练时，一张图可能包含 K 个文本 prompt（正类）与 N-K 个负样本（负类）——但只有正类的 anchor 才算 loss。`txt_masks` 是 shape `[B, K, 80]` 的 bool tensor，标记每个 anchor 属于哪类文本 prompt 的正样本；对不匹配的 anchor，`loss_cls` 权重置 0。

**TAL 分配（Task-Aligned Assigner）**：完全沿用 v8，topk=10, α=0.5, β=6.0——**这是 YOLO-World 论文里唯一没做的改进**（作者说"TAL 已经很成熟，不需要动"）。

对应源码：
- `YOLOWorldHead.loss_by_feat`: <SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="428" :end="577"/>

### §3.5 Two-stage Training

论文提出两阶段训练框架：

**Stage 1: 开放词表预训练**

- 数据集：**Objects365v1**（主数据集，80 类，与 COCO 重叠但标注更细）+ **Gold-G**（GQA + Flickr30k，575 类视觉-语言对齐）
- 类别数：`num_classes=1203`（LVIS 类数）· `num_training_classes=80`（每 batch 采样 80 类正类文本）
- 训练步：100 epoch · AdamW · `base_lr=2e-3` · `weight_decay=0.05/2`
- 硬件：4×8 GPU（8×A100）· ~32 小时

**Stage 2: COCO 微调**

- 数据集：COCO 2017（80 类）
- 训练步：80 epoch · AdamW · `base_lr=2e-4` · `weight_decay=0.05/2`
- v2 版本加 `reparameterized=True`，把 prompt embedding 折叠进 head 权重

对应源码：
- 官方 v1 预训练 cfg: <SrcLink file="yoloworld:ailab-cvc/configs/pretrain_v1/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py" />
- 官方 v2 COCO 微调 cfg: <SrcLink file="yoloworld:ailab-cvc/configs/finetune_coco/yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py" />

### §3.6 Optional Third-stage: CC-250K Augmentation

论文 Table 7 提出可选的**第三阶段**——用 CC-3M 中筛选的 CC-250K（视觉-语言对齐子集）做微调：

| 训练配置 | LVIS z-shot AP | COCO z-shot AP |
|---------|----------------|----------------|
| 仅 Stage 1（obj365v1+gold-g） | 26.2 | 37.7 |
| Stage 1 + Stage 2（COCO） | 31.0 | 43.0 |
| **Stage 1 + Stage 2 + CC-250K** | **35.4** | 47.1 |
| + CC-3M（全量 3M） | 34.8（-0.6） | 47.3（+0.2） |

- **CC-250K 是 CC-3M 的精选子集**（CLIP 官方推荐的对齐质量子集）；
- **CC-3M 反而略掉分**——说明大量噪声视觉-语言数据会稀释对齐效果；
- **结论**：CC-250K 是 YOLO-Worldv2 最终发布的训练配方，v2-S/M/L/X 全部基于 CC-250K 训出。

## 关键实验结果

### Table 3: 主实验（COCO z-shot + LVIS z-shot + 推理速度）

| 方法 | Params | GFLOPs | FPS (V100) | COCO z-shot AP | LVIS z-shot AP |
|------|--------|--------|-----------|----------------|----------------|
| Zero-Shot DETR | 30.6 M | 127.2 | 30.7 | 19.0 | — |
| GLIP | 118.3 M | 197.8 | 5.6 | 24.6 | — |
| CLIP-CLIP (ViT-L) | 308 M | 617.6 | 10.3 | 28.1 | 30.6 |
| OWL-ViT | 308 M | 620.8 | 12.7 | 26.3 | — |
| Cascade R-CNN + CLIP | 86.3 M | 328.5 | 24.0 | 26.2 | 30.5 |
| **YOLO-Worldv2-S** | 24.5 M | 74.2 | **74.1** | **37.7** | **26.2** |
| **YOLO-Worldv2-M** | 38.2 M | 138.2 | **58.1** | **43.0** | **31.0** |
| **YOLO-Worldv2-L** | 55.4 M | 196.4 | **52.0** | **45.8** | **35.0** |
| **YOLO-Worldv2-X** | 72.6 M | 263.9 | **47.3** | **47.1** | **36.3** |

- **v2-X 达 COCO z-shot 47.1 AP**——比 GLIP（24.6）高 **22.5 AP**，比 OWL-ViT（26.3）高 **20.8 AP**，同时快 **5~15 倍**；
- **LVIS z-shot v2-L 35.0 AP**——超越 CLIP-CLIP（30.6）4.4 AP，同时快 **5×**。

### Table 5: 消融实验（COCO z-shot，S 档）

| 组件 | Δ AP |
|------|------|
| Base YOLOv8s | 0 |
| + RepVL-PAN | +4.2 |
| + Promptable Head（L2） | +3.8 |
| + v2 BN Head | +0.5 |
| + 重参数化（v2 reparameterized） | -0.3 |

- **RepVL-PAN 是最大增益来源**（+4.2 AP）——VL 融合颈部是核心贡献；
- **BN Head vs L2 Head**：v2 BN 略好（+0.5 AP）+ 更快；
- **重参数化的代价**：-0.3 AP，但**换来了 ONNX/TensorRT 支持**——值得。

### Table 7: 训练配方消融（LVIS z-shot，S 档）

| 训练数据 | Δ AP |
|---------|------|
| 仅 Objects365v1 | 0 |
| + Gold-G（GQA+Flickr） | +4.7 |
| + CC-250K | +4.5 |
| + CC-3M（全量） | +3.9（比 CC-250K 少 0.6） |

## 论文与代码的偏差

**论文说但代码里没有**：

1. **`YOLOWorldV2` 类不存在**——论文里提到的"v2 head"实际是 `RepYOLOWorldHeadModule` + `SimpleYOLOWorldDetector` 组合，没有单独的 `YOLOWorldV2` 类；
2. **`YOLOWorld-Segmentation`**：论文 §6 提到未来工作会做分割，**没有实现**；
3. **"Real-Time"定义**：论文说"实时"是"V100 上 >50 FPS"，S 档 74 FPS 符合，但 L/X 档 47~52 FPS 只是**准实时**（未达 60 FPS）；

**代码里比论文多**：

1. **`RepBNContrastiveHead`**：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" :line="110" :end="131"/> 论文里没提，但 `finetune_coco` cfg 里有用到——**这是 v2 微调的关键组件**，把 BN 权重折叠进 `1×1 Conv` 实现重参数化；
2. **`SimpleYOLOWorldDetector`**：<SrcLink file="yoloworld:ailab-cvc/yolo_world/models/detectors/yolo_world.py" :line="109" :end="232"/> 论文里没提到，是 v2 的顶层容器，包含 `embeddings` 参数与可选 `adapter` MLP；
3. **Ultralytics 版 `WorldDetect`**：`with_bn=False/True` 参数是 Ultralytics 自己加的，论文里两种 head 是独立类；

**代码里没有但训练时用到**：

1. **`txt_masks` 计算**：论文里有，代码里在 `YOLOWorldHead.loss_by_feat` 内部计算，**没有独立模块**；
2. **`RandomLoadText`**：AILab-CVC 版 cfg 里的 `data_preprocessor` 组件，实现每 batch 随机采样 80 类文本 prompt——**快照未纳入 datasets/ 目录**；
3. **CC-250K 数据**：论文 Table 7 提到，代码里没提供下载脚本，需要用户自行下载。

## 未回答的问题

1. **CLIP 编码器能否换？** 论文只用 CLIP ViT-B/32，但社区测试过 SigLIP / MobileCLIP / DINOv2——**结果均掉分**（详见 [painpoints.md](./painpoints)）；
2. **为什么 nc 硬编码 80？** Ultralytics 版 `WorldTrainer` 里 `nc=min(self.data["nc"], 80)`，但 AILab-CVC 原版支持 `num_classes=1203`——**这是 Ultralytics 的简化**，社区有 PR 修复但未被合入主线；
3. **v1 为什么不修好导出？** Ultralytics 版 v1 无法 export，社区 PR 尝试修复 `ImagePoolingAttn` 但精度掉 1.5 AP——**作者没有回应**，实际部署只能用 v2；
4. **`WorldDetect` 的 `cv4` 能否再简化？** YOLOE 用 RepRTA 残差 MLP 替代了 `cv4`，效果相当但参数更少——**这是 YOLO-World 后续被 YOLOE 超越的原因**（详见 [YOLOE 综述](../yoloe/index)）。

## 快照处理原则

本仓库为 YOLO-World 保留的两份快照遵循以下原则：

- **最小算法快照**：只保留算法核心文件（模型定义 + cfg + trainer），不保留数据集适配器、可视化脚本、demo 代码；
- **commit 对齐**：AILab-CVC 快照对齐 `4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b`，Ultralytics 快照对齐 `8.1.39`（PR #9268 合入）；
- **行号稳定**：所有 SrcLink 行号基于快照内容，若上游更新需重新生成快照；
- **不合并两版实现**：AILab-CVC 与 Ultralytics 是**两套独立实现**，不试图在快照内合并——读者按需选用。

**未收录的内容**（快照 README 已列出）：

- `.pt` 权重文件（体积大且非算法）
- `yolo_world_open` PyPI 包（非官方第三方实现）
- `ultralytics/nn/modules/block.py` 中的 `ImagePoolingAttn`（可回指 Ultralytics 主线）
- `ailab-cvc/yolo_world/datasets/*` 数据集适配器
- `ailab-cvc/yolo_world/models/assigner/task_aligned_assigner.py`（TAL，与 YOLOv8 同源）
- Ultralytics `models/yolo/world/{model,val,predict}.py`（8.1.39 中不存在）

## 延伸阅读

- **CLIP 原论文**：Radford A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021, arXiv:2103.00020
- **YOLOv8 论文**：Jocher G. & Qiu J. *Ultralytics YOLOv8.* GitHub README 2023
- **TAL 论文**：Zheng Z. et al. *TAL: Task-Aligned One-Stage Detector.* ICCV 2021
- **GLIP 论文**：Li L. et al. *GLIP: Grounded Language-Image Pre-Training for Open-Set Object Detection.* ECCV 2022
- **CLIP-CLIP 论文**：Yang S. et al. *CLIP-CLIP: CLIP for CLIP-based Open-Vocabulary Object Detection.* ICML 2022
- **姊妹版本**：[YOLOE 综述](../yoloe/index) · [YOLO-Master 综述](../master/index)
- **本站内部**：[YOLO-World 概览](./index) · [Architecture](./architecture) · [Concepts](./concepts) · [Losses](./losses) · [Tricks](./tricks) · [Painpoints](./painpoints) · [Code Walkthrough](./code-walkthrough)
