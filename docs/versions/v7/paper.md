# YOLOv7 论文精读

> **YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**
> arXiv:[2207.02696](https://arxiv.org/abs/2207.02696)，2022-07-06 提交，中央研究院（Chien-Yao Wang、Alexey Bochkovskiy、Hong-Yuan Mark Liao）。
> 快照取自 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) main 分支 @ `a207844`。本页按论文行文精读，文末设「论文 vs 快照」对照。

## 1. Motivation：trainable bag-of-freebies

论文标题即论点：**把新纪录归功于一组"可训练的免费午餐"**——训练期加料、推理期零开销的方法，而不是更大的模型或更贵的推理算子。

- **bag-of-freebies** 的定义要对照 v4 的语境：只增加训练成本、不增加推理成本的技巧（数据增强、标签分配、损失设计、训练策略）；
- v7 的贡献全部落在这个范畴：E-ELAN 与 compound scaling 让网络**训练得动更大**，planned re-param 让多分支**推理时消失**，双头 coarse-to-fine 标签分配与 EMA 都只在训练期生效；
- 与 v6 的"为部署而生"对照：**v7 押注精度侧**（test-dev 51.4 AP 起步），用"推理零开销"约束守住实时性（V100 161 FPS）。

一句话：v4 回答"哪些 trick 有效"，v5 回答"怎么工程化"，v6 回答"怎么在生产硬件上又快又准"，**v7 回答"怎么把训练期的每一分投入都变成推理期的免费精度"**。

## 2. Network Design：E-ELAN 与 compound scaling（§3）

### 2.1 E-ELAN（§3.1）

- ELAN 出处引用 [1] = anonymous《Designing Network Design Strategies》2022，核心是"控制最短最长梯度路径、稳定梯度传播"；
- **E-ELAN 的 expand/shuffle/merge**：用 cardinality（分组数）+ group conv 只**增加梯度传播路径的多样性**、**不改变梯度路径本身**——论文原话是不破坏原有 gradient path，因此大模型训练仍能收敛；
- 对应快照：**没有独立的 E-ELAN 类**，落地形态是 yaml 里 Conv + 多路 Concat 块——骨干 4 路 `[[-1,-3,-5,-6],1,Concat,[1]]`<SrcLink file="v7:cfg/training/yolov7.yaml" :line="21" :end="28" />，头部 6 路 `[[-1,-2,-3,-4,-5,-6],1,Concat,[1]]`<SrcLink file="v7:cfg/training/yolov7.yaml" :line="82" :end="89" />。

### 2.2 concat 型模型的 compound scaling（§3.2）

- v4/v5 的缩放是 depth + width 独立缩放；**concat 型结构**（输出靠多路 Concat 拼通道）下，论文主张：**computational block 只缩 depth，transition layer 按与 computational block 相同的变化量缩 width**；
- 直觉：Concat 要求各路通道对齐，transition 层宽度若独立缩放会破坏拼接比例；同变化量缩放保持结构等比；
- 消融 Table 3：base **51.7 →** compound scaling **52.9 AP**（+1.2）；
- 对应快照：`depth_multiple: 1.0` / `width_multiple: 1.0`<SrcLink file="v7:cfg/training/yolov7.yaml" :line="3" :end="4" />——yolov7 基线配置本身就是 1.0 不缩放，缩放体现在 X/W6/E6/D6/E6E 家族配置里。

## 3. Planned Re-parameterized Convolution（§4.1）

- **RepConv [13] = RepVGG**（Ding, CVPR 2021）：训练态 3×3 + 1×1 + identity 三分支，推理态融合单卷积；
- 论文的核心贡献不是"再用一次 RepConv"，而是**规划**（planned）——逐层分析该不该有 identity 分支，得到 **RepConvN 规则**：**带 residual 连接或 concat 连接的卷积层不能有 identity 分支**（Fig.4）；
- 理由：ResNet 的残差相加、DenseNet 的拼接本身就是梯度高速公路，再叠 identity 分支会**破坏这类结构的梯度多样性**，训练收益反而受损；
- 对应快照：**没有 RepConvN 这个类**，规则由条件 identity 实现——`rbr_identity = (nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None)`<SrcLink file="v7:models/common.py" :line="463" :end="582" />；yolov7.yaml 三处 RepConv 都是 128→256 / 256→512 / 512→1024（c2≠c1）<SrcLink file="v7:cfg/training/yolov7.yaml" :line="135" :end="137" />，天然无 identity = RepConvN 行为；
- 消融 Table 4/5（论文内，本页不引数字）：对照不同位置加/不加 identity 分支的方案，验证"哪些层该用 RepConv、哪些层该用 RepConvN、哪些层不该重参数化"。

## 4. Label Assignment：lead-head-guided coarse-to-fine（§4.2）

论文观察：辅助头（auxiliary head）的标签分配**不该与主头（lead head）各算各的**，由此设计三件套：

1. **lead head guided label assigner**：用 lead head 的预测 + GT 生成一套软标签，**同时**供 lead 与 aux 两个 head 使用；
2. **coarse-to-fine**：约束严的 **fine label 给 lead head**，放宽约束的 **coarse label 给 auxiliary head**——aux 头保 recall，lead 头保精度；
3. **partial auxiliary head**：aux head 只接在网络部分层（E-ELAN 配置下），且 aux 分支**不含隐式知识**。

对应快照（`utils/loss.py`）：

- `ComputeLossAuxOTA`（train_aux.py 用）<SrcLink file="v7:utils/loss.py" :line="1176" :end="1697" />：`build_targets`（lead）走 `find_3_positive`（候选格偏置 **g=0.5**，fine）<SrcLink file="v7:utils/loss.py" :line="1646" :end="1654" />；`build_targets2`（aux）走 `find_5_positive`（**g=1.0**，候选格更宽 = coarse）<SrcLink file="v7:utils/loss.py" :line="1593" :end="1601" />；
- 两套目标由同一批预测 + GT 经相同 cost 匹配生成（lead-head guided 的代码体现）；aux 损失统一 ×0.25；
- 消融 Table 6/7/8：base **55.6 →** coarse-to-fine **55.9 →** partial aux **56.3 → 56.8 AP**（逐步加码）。

## 5. Training Freebies：三条推理免费的午餐（§4.3）

1. **conv-BN 融合**：推理前把 BN 折进卷积权重/偏置，训练态不省、推理态免费；
2. **YOLOR 隐式知识并进卷积**：ImplicitA（加性）/ ImplicitM（乘性）的隐式向量在推理时合并进 conv 的 bias / weight——代码落点 `IDetect.fuse()`<SrcLink file="v7:models/yolo.py" :line="178" :end="190" />，`torch.matmul` 把 ia 并进 bias、逐元素乘把 im 并进 weight/bias；隐式知识类本身见 <SrcLink file="v7:models/common.py" :line="433" :end="461" />；
3. **EMA（mean teacher [75]）**：最终推理模型用 EMA 权重。

三条的共同点：训练时引入额外表达力或稳定性，推理图里完全消失。

## 6. 实验结果

### 6.1 主指标（Table 2，V100 测速，AP 为 test-dev，从零训练只用 COCO）

| 模型 | 参数量 | FPS（V100） | AP<sup>test-dev</sup> | 备注 |
|------|--------|-------------|----------------------|------|
| YOLOv7 | 36.9M | 161 | **51.4** | 104.7 GFLOPs，val 51.2 |
| YOLOv7-X | 71.3M | 114 | 53.1 | |
| YOLOv7-W6 | 70.4M | 84 | 54.9 | 1280 输入 |
| YOLOv7-E6 | 97.2M | 56 | 56.0 | |
| YOLOv7-D6 | 154.7M | 44 | 56.6 | |
| YOLOv7-E6E | 151.7M | 36 | **56.8** | 集成模型 |
| YOLOv7-tiny-SiLU | 6.2M | 286 | 38.7 | |

### 6.2 baseline 对比（§5.2）

与 YOLOv4-tiny / YOLOX-tiny 等轻量级对比中，原文有"比 YOLOv4 少 75% 参数"一句——**与论文自身 Table 1 的参数量数字对不上**，疑为原文笔误，引用时请以 Table 1 为准（见 §7 漂移盘点）。

### 6.3 消融结构（Table 3~8）

论文消融逐项标价、逐项加码：

- **Table 3**：compound scaling 51.7 → 52.9（§2.2）；
- **Table 4/5**：planned re-param 逐位置方案对照（哪些层该/不该有 identity 分支）；
- **Table 6/7/8**：标签分配三件套 55.6 → 55.9 → 56.3 → 56.8（§4.2）。

### 6.4 与同期工作的对比（读表注意事项）

- 2022 年年中的两部作品路线相反：**v6 anchor-free + TAL + VFL**（美团，T4 口径），**v7 anchor-based + SimOTA 式双头分配 + CIoU/BCE**（中央研究院，V100 口径）——"anchor 是否过时"在这一年是**分叉**而非收敛；
- **FPS 跨报告不可比**：v6 全部数字在 T4 + TensorRT fp16、v7 在 V100；AP 口径也有 test-dev / val 之分；
- 可比的是方法论：v6 把预算花在部署工具链，v7 把预算花在训练期技巧——两条路线后来都证明了自身价值。

## 7. 论文 vs 快照：实现漂移盘点

| 维度 | 论文（2022-07） | 当前快照 main @ a207844 |
|------|----------------|------------------------|
| RepConvN | 命名一个"无 identity 分支"的 RepConvN | **无此类**；RepConv 用 `c2 == c1 and s == 1` 条件 identity 实现同一规则<SrcLink file="v7:models/common.py" :line="486" :end="486" /> |
| decoder 约束 | 提到"按到目标中心距离限制 objectness 上界" | IDetect decode 是标准 YOLOv3 式（`xy=(σ·2-0.5+grid)·stride`、`wh=(σ·2)²·anchor_grid`），**无此约束**<SrcLink file="v7:models/yolo.py" :line="134" :end="135" /> |
| §5.2 参数对比 | "比 YOLOv4 少 75% 参数" | 与 Table 1 数字不符，疑原文笔误 |
| E-ELAN | expand/shuffle/merge 的 cardinality + group conv | 无独立 E-ELAN 类，以 yaml Conv + 多路 Concat 块形态落地 |

**未核实项（谨慎措辞，本页不下断言）**：训练 schedule（epochs/batch/优化器）在论文附录、未与快照逐一核对；coarse-to-fine 的公式级细节与代码对应关系未逐行验证；RepCSP/RCSP/RepRCSP 在网络里的具体换位关系未完整核实。

**结论**：v7 的漂移集中在"论文命名 vs 代码形态"（RepConvN）与"论文叙述 vs 代码实现"（decoder 约束）两处，规则本身在代码里都成立、只是换了写法。读论文建动机，读代码定事实，两边互相印证。

## 8. 论文风格：学术界的精度宣言

- 与 v6"每个组件都标价"的技术报告范式不同，v7 是**会议论文式**叙事：动机（bag-of-freebies）→ 方法（E-ELAN / scaling / re-param / 标签分配）→ 消融逐项加码 → 全家族 SOTA 表；
- 亮点是**消融的累进式呈现**（55.6 → 55.9 → 56.3 → 56.8），每个组件的边际收益一目了然；
- 学术界 + v4 作者回归的背景，让"anchor-based 是否还有余量"这个问题在 2022 年得到了一个高分答案。

## 延伸阅读

- [损失函数：CIoU + BCE + 双头标签分配详解](/versions/v7/losses)
- [反向传播推导](/versions/v7/loss-gradient)
- [理论 Concept：trainable bag-of-freebies](/versions/v7/concepts)
- [Architecture 逐层图解](/versions/v7/architecture)
- [源码走读：从配置到损失](/versions/v7/code-walkthrough)
- [v6 论文精读（对照工业技术报告叙事）](/versions/v6/paper)
- [系列家族对照](/global/family) · [术语表](/glossary)
- 论文原文：[arXiv:2207.02696](https://arxiv.org/abs/2207.02696) · [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
