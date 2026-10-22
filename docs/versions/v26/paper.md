# 发布精读：Ultralytics YOLO26 arXiv:2606.03748

v26 是本站十四代之后**「学术线 + 官方线」双线路标的第一个官方线版本**——由 **Ultralytics（Glenn Jocher & Jing Qiu）** 在官方仓库 [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics) 发布，同期第三方学术线是 [v14 zhangcbb/yolov14](/versions/v14/index)（YOLO-Omni 多域鲁棒检测）。发布材料四层：arXiv 论文正文 + GitHub 仓库 + README 中的 "Models / Usage" 两大节详解 + Kaggle / Colab / Binder / Ultralytics Platform 五个官方托管渠道。

**关键定位**：README 首行 Banner 已直接指向 `yolo/yolo27`（<SrcLink file="v26:README.md" :line="125" :end="125" />）——**「YOLO27 — Coming Soon」**是官方公开预告，所以 v26 是「14 代官方主线里**最后一代有论文 + 完整公开权重的版本**」；v27 已进 R&D，官方口径明说「Models are not yet available, and no launch date is set」。v26 的官方定位不是单一 COCO SOTA 竞赛，而是**「六任务官方全家桶」**——Detect / Segmentation / Semantic Segmentation / Depth / Pose / Classification / OBB + YOLOE 开放词汇，全部在同一份 backbone 上。

## 四层材料

| 材料 | URL | 快照 |
|------|-----|------|
| arXiv 论文 | [arXiv:2606.03748](https://arxiv.org/abs/2606.03748) | 2026-06 首发 |
| GitHub 仓库 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | commit `2e241d0`（2026-06 快照） |
| README 精读 | 仓库根 `README.md` | 301 行 · 六任务性能表全公开 |
| 官方权重 | [ultralytics/assets/releases](https://github.com/ultralytics/assets/releases) | 30 份 .pt（n/s/m/l/x × 6 任务）全部公开 |
| Ultralytics Docs | [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26) | 官方文档站 |

> **注**：v26 与 v14 完全不同渠道——**v26 是 ultralytics 官方主线正统续作**，v14 是第三方 fork；两者同期（v26 2026-06，v14 2026-08）但走不同路线：v26 官方全家桶（六任务 + YOLOE），v14 学术多域鲁棒（Detect 单任务打四基准）。

## 论文元信息

- **标题**：Ultralytics YOLO26
- **作者**：Glenn Jocher & Jing Qiu（Ultralytics 官方）
- **arXiv**：2606.03748，Computer Vision and Pattern Recognition (cs.CV)，2026-06
- **代码**：`https://github.com/ultralytics/ultralytics`
- **许可证**：**AGPL-3.0 + Ultralytics Enterprise License 双许可**（<SrcLink file="v26:README.md" :line="280" :end="281" />）——官方首次出现「开源 + 商用」双轨，AGPL-3.0 用于开源分发（需遵守 copyleft），商用闭源需另购 Enterprise License

## 论文核心贡献（作者自述）

论文把 v26 的贡献总结为**四个正交方向**，每条对应一个模块 / 机制 + 一份官方权重家族：

1. **End-to-End Detection (E2E)**——`Detect` 头在 `end2end=True` 时复制出一份 `one2one_cv2/one2one_cv3`（`copy.deepcopy`），得到 one2many + one2one 双分支；`reg_max: 1` 让 DFL 退化为 `nn.Identity()`；`E2EDetectLoss` = 两个 `v8DetectionLoss`（topk=10 + topk=1）直接相加；`E2ELoss` 用 `decay` 函数让 one2many 权重退火到 `final_o2m`。对应 <SrcLink file="v26:nn/modules/head.py" :line="38" :end="154" /> 与 <SrcLink file="v26:utils/loss.py" :line="1263" :end="1315" />。
2. **Depth Estimation (Depth26)**——单目深度估计新任务；`Depth` head 用 `proj → interpolate + refine → head Conv → ConvTranspose2d → Conv → Conv2d(1ch)` 上采样，`self.head[-1].bias.data.fill_(0.182)` 初始化到约 1.2 米；`DepthLoss26` 用 SILog（`silog_lambda=1.0` scale-invariant）+ 4 尺度梯度 L1（`_grad_l1`）；`cal_a/cal_b` log-affine 校准 buffer。对应 <SrcLink file="v26:nn/modules/head.py" :line="785" :end="869" /> 与 <SrcLink file="v26:utils/loss.py" :line="1169" :end="1260" />。
3. **Pose RLE + RealNVP 流模型**——`Pose26` + `PoseLoss26` 新增 RLE（Rectified Log Error），`flow_model` 是 `RealNVP` 流模型做 `log_prob(error)` 计算概率密度；`PoseTaskAlignedAssigner` 特化姿态任务。对应 <SrcLink file="v26:utils/loss.py" :line="810" :end="932" /> 与 <SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2080" /> 与 <SrcLink file="v26:utils/tal.py" :line="396" :end="420" />。
4. **Semantic Segmentation (Semantic26)**——`SemanticSegment(P3, P4)` 双尺度语义分割头；`bake_argmax` 用于 TRT/Hailo 导出优化；`SemanticSegmentationLoss` 用 CE（`ignore_index=255, reduction="sum"`）+ BCE（nc=1 时）+ Cityscapes 类权重。对应 <SrcLink file="v26:nn/modules/head.py" :line="1926" :end="1991" /> 与 <SrcLink file="v26:utils/loss.py" :line="1389" :end="1442" />。

配套模块：**Classify 头**（ImageNet 分类）、**YOLOEDetect / YOLOESegment26**（开放词汇检测，`text_model: mobileclip2:b`）、**Proto26**（新掩码头，替代 v11 的 Proto）、**SAVPE**（Space-Adaptive Variational Prior Estimation，为 Semantic/Depth 提供位置编码）。

## 六任务 + 双许可家族

README "Models" 章节明确列出**六任务 × 五档 = 30 份官方权重**（<SrcLink file="v26:README.md" :line="139" :end="280" />）：

| 任务 | 数据集 | 数据集说明 | 官方权重数 |
|------|--------|-----------|-----------|
| Detection | COCO 2017 | 80 类 | 5（n/s/m/l/x） |
| Segmentation | COCO-Seg 2017 | 80 类 | 5 |
| Semantic Segmentation | Cityscapes | 19 类 | 5 |
| Depth Estimation | NYU Depth V2（Eigen test split） | 单目深度 | 5 |
| Pose Estimation | COCO-Pose | 17 关键点 | 5 |
| Classification | ImageNet | 1000 类 | 5 |
| OBB | DOTAv1 | 旋转框 | 5 |
| YOLOE（开放词汇） | 多数据集混合 | 文本 prompt | 5 |

**关键观察**：v26 是官方线**首次**在同一 backbone 上打包 Detect/Seg/Sem/Depth/Pose/Classify/OBB 全部七个子任务——之前 v8~v14 官方线最多 4 任务（Detect/Seg/Pose/OBB），Semantic/Depth/Classify/YOLOE 都是新加的。

## 与 v8 / v11 / v12 / v13 / v14 的关键差异（作者口径）

| 维度 | YOLOv8 | YOLOv11 | YOLOv12 | YOLOv13 | YOLOv14 (Omni) | **YOLO26** |
|------|--------|---------|---------|---------|---------------|------------|
| 骨干 | Conv + C2f + SPPF | Conv + C3k2 + SPPF | Conv + C3k2 + **A2C2f** | C3k2 + A2C2f + DSC3k2 + HyperACE | C3k2 + A2C2f + DeformableA2C2f | **Conv + C3k2 + SPPF + C2PSA（回归 v8 风格）** |
| 检测头 | Detect 单头 | Detect 单头 | Detect 单头 | Detect 单头 | Detect 单头 | **Detect + `end2end=True` 双分支** |
| DFL | reg_max=16 | reg_max=16 | reg_max=16 | reg_max=16 | reg_max=16 | **reg_max=1（DFL 退化为 Identity）** |
| 任务覆盖 | Detect/Seg/Pose/OBB | Detect/Seg/Pose/OBB | Detect/Seg/Pose/OBB | Detect/Seg/Pose/OBB | Detect（四域） | **Detect/Seg/Sem/Depth/Pose/Classify/OBB/YOLOE** |
| 结构文件 | v8n/s/m/l/x.yaml 五份 | yolo11.yaml 单份 | yolov12.yaml 单份 | yolov13.yaml 单份 | 6 份 yaml 变体 | **11 份 yaml 变体** |
| 论文 | 无 | 无 | arXiv:2502.12524 | arXiv:2506.17733v2 | arXiv:2608.04720v3 | **arXiv:2606.03748** |
| 许可证 | AGPL-3.0 | AGPL-3.0 | AGPL-3.0 | Apache-2.0 | AGPL-3.0 | **AGPL-3.0 + Enterprise License（首次双许可）** |
| 作者 | Glenn Jocher & Jing Qiu | Glenn Jocher & Jing Qiu | UB SUNY + UCAS | iMoonLab | zhangcbb | **Ultralytics 官方** |

## 论文正文关键章节（arXiv:2606.03748）

### III-A Overall Architecture（§3.1）

描述骨干-颈-头整体结构：Backbone 12 层（`Conv-C3k2-Conv-C3k2-Conv-C3k2-Conv-C3k2-SPPF-C2PSA`）+ Head 10 层（`Upsample×2+C3k2×4+Detect`）；Detect 输出三尺度 P3/P4/P5。对应 <SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="20" :end="52" />。

**关键设计**：**A2C2f 骨干被彻底移除**——`yolo26.yaml` 用的是 SPPF + C2PSA 组合（v8/v10/v11 风格），这是官方线 v8 以来**第一次回归经典骨干**。同时 `AAttn` 与 `A2C2f` 类仍在 block.py 保留（<SrcLink file="v26:nn/modules/block.py" :line="1650" /> 与 <SrcLink file="v26:nn/modules/block.py" :line="1804" />），供自定义 yaml 使用。

### III-B End-to-End Detection（§3.2）

E2E 双分支是 v26 的头号 trick。`Detect.__init__` 在 `end2end=True` 时（<SrcLink file="v26:nn/modules/head.py" :line="120" :end="140" />）：

- 复制 `cv2 → one2one_cv2`（box 分支）
- 复制 `cv3 → one2one_cv3`（cls 分支）
- 保留原始 `cv2/cv3` 作为 one2many 分支

**训练时**：one2many 走 topk=10 TAL，one2one 走 topk=1 TAL（每网格至多分配 1 个正样本），两个 loss 相加（<SrcLink file="v26:utils/loss.py" :line="1263" :end="1280" />）。

**推理时**：可以只保留 one2one 分支，`nms=False` 直接输出——这是「端到端无 NMS」的部署路径，精度代价 0.7~0.8 mAP 换 30~40% 推理时间（README "Speed" 说明栏）。

`E2ELoss`（<SrcLink file="v26:utils/loss.py" :line="1283" :end="1315" />）用 decay 函数：
```python
o2m = max(1 - x/epochs, 0) * (o2m_copy - final_o2m) + final_o2m
```
让 one2many 权重从初值线性退火到 `final_o2m`，最后阶段只剩 one2one——**保证最后阶段只优化 one2one 分支的收敛**。

### III-C Depth Estimation（§3.3）

单目深度估计是官方线**首次引入**的新任务。`Depth` head（<SrcLink file="v26:nn/modules/head.py" :line="785" :end="869" />）：

- `proj`：三尺度特征融合（P3/P4/P5）
- `interpolate + refine`：插值 + 精修
- `head Conv → ConvTranspose2d → Conv → Conv2d(1ch)`：从低分辨率恢复到输入分辨率
- `self.head[-1].bias.data.fill_(0.182)`：**初始化到约 1.2 米**（exp(0.182) ≈ 1.2）
- `cal_a/cal_b` log-affine 校准 buffer：**推理时对齐预测深度到真值尺度**

`DepthLoss26`（<SrcLink file="v26:utils/loss.py" :line="1169" :end="1260" />）：

- **SILog 项**（`silog_lambda=1.0` scale-invariant log loss，`silog_lambda=0.0` 时是 log-RMSE）：尺度不变
- **4 尺度梯度 L1**（`_grad_l1` 函数）：多尺度梯度监督，鼓励边缘对齐
- **`valid_f` mask + `keep` 逐图过滤**：处理无效深度值

**NYU Depth V2 数字**：n 档 Δ1=0.882 / x 档 0.933——都是含 multi-scale + 水平翻转 TTA + log-least-squares 对齐的口径。

### III-D Pose Estimation with RLE + RealNVP（§3.4）

v26 在 v8PoseLoss 基础上新增 **RLE（Rectified Log Error）+ 流模型**（<SrcLink file="v26:utils/loss.py" :line="810" :end="932" />）：

- `PoseLoss26` 继承 `v8PoseLoss`，接入 `PoseTaskAlignedAssigner`
- `flow_model` 是可选的 `RealNVP` 流模型（<SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2080" />），对误差分布做 `log_prob`
- 关键行 <SrcLink file="v26:utils/loss.py" :line="892" :end="932" />：`error.clamp(-100, 100)` 后再算 `flow_model.log_prob(error)`——**抑制极值影响 log_prob 稳定性**

**RealNVP 流模型**（<SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2080" />）：Normalizing Flow 的 Real NVP 变体，用于姿态估计的密集键点误差建模——**这是 v26 首次在 YOLO 系列里引入 Normalizing Flow**。

### III-E Semantic Segmentation（§3.5）

`SemanticSegment` head（<SrcLink file="v26:nn/modules/head.py" :line="1926" :end="1991" />）：

- `P3 + P4` 双尺度输入（不是 P3/P4/P5）
- `bake_argmax` 模式：训练时输出 logits，导出 TRT/Hailo 时把 argmax 融进图
- 19 类 Cityscapes 数据集（`nc: 19`）

`SemanticSegmentationLoss`（<SrcLink file="v26:utils/loss.py" :line="1389" :end="1442" />）：

- **CE**：`CrossEntropyLoss(ignore_index=255, reduction="sum")`——`ignore_index=255` 是 Cityscapes 的 invalid class 标记
- **BCE**（nc=1 时）：`BCEWithLogitsLoss(reduction="sum")`——二分类退化路径
- **CITYSCAPES_WEIGHT** 类权重：处理 19 类的严重类别不平衡

**Cityscapes 数字**：n 档 mIoU 78.3 / x 档 83.6。

### III-F YOLOE 开放词汇（§3.6）

`YOLOEDetect`（<SrcLink file="v26:nn/modules/head.py" :line="1091" :end="1460" />）与 `YOLOESegment26`（<SrcLink file="v26:nn/modules/head.py" :line="1461" :end="1800" />）：

- `text_model: mobileclip2:b`：CLIP 系小模型做文本 embedding
- 支持零样本检测 / 分割（用户输入文字 prompt 直接检测）
- 与 Detect / Segment 头同源，只是在 yaml 里改 `YOLOEDetect` / `YOLOESegment26`

### IV-A Experimental Setup（§4.1）

- **Datasets**：COCO 2017（118K train / 5K val）+ COCO-Seg + Cityscapes + NYU Depth V2 + COCO-Pose + ImageNet + DOTAv1；
- **Implementation**：PyTorch ≥ 1.8，Python ≥ 3.8，300 epoch，batch=64，640×640，mosaic=1.0；
- **E2E**：`end2end=True`（v26 默认），`E2ELoss` 权重退火；
- **延迟**：T4 + TensorRT10 FP16（CPU 用 ONNX 测）。

### IV-B Main Results（TABLE I-VI 六任务主表）

**Detect COCO（TABLE I）**：

| 型号 | mAP50-95 | mAP50-95(e2e) | CPU ONNX (ms) | T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
|------|---------|--------------|---------------|--------------------|------------|-----------|
| YOLO26n | 40.9 | 40.1 | 38.9 | **1.7** | 2.4 | 5.5 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 | 20.9 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 | 68.4 |
| YOLO26l | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 | 86.8 |
| YOLO26x | **57.5** | 56.9 | 525.8 | 11.8 | 55.7 | 194.4 |

**Segment COCO（TABLE II）**：n 档 39.6 box / 33.9 mask；x 档 56.5 box / 47.0 mask。

**Semantic Cityscapes（TABLE III）**：n 档 78.3 mIoU；x 档 83.6 mIoU。

**Depth NYU（TABLE IV）**：n 档 Δ1=0.882 / abs_rel=0.109；x 档 Δ1=0.933 / abs_rel=0.080。

**Pose COCO（TABLE V）**：n 档 57.2 mAP；x 档 71.6 mAP。

**Classification ImageNet（TABLE VI）**：n 档 71.4 top-1；x 档 79.9 top-1。

**OBB DOTAv1（TABLE VII）**：n 档 52.4 mAP；x 档 56.7 mAP。

## 论文与代码的偏差（本站发现）

v26 论文与 ultralytics@`2e241d0` 快照的对齐度极高（都是官方仓库），但仍有几处值得注意：

1. **A2C2f 类保留但 yaml 不用**：<SrcLink file="v26:nn/modules/block.py" :line="1804" /> 里 `A2C2f` 类完整实现（与 v12/v13 一致），但 `yolo26.yaml` 完全没引用——**推测是向后兼容（自定义 yaml 可以用）**，但官方推荐路径是 SPPF + C2PSA；
2. **AAttn 类保留但 yaml 不用**：<SrcLink file="v26:nn/modules/block.py" :line="1650" /> 里 `AAttn` 完整实现，yaml 同样不引用——**v26 官方口径：注意力骨干已不再是官方推荐，但仍在库里供自定义**；
3. **`reg_max: 1` 在 yaml 里是硬编码**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="10" /> `reg_max: 1`——**DFL 退化为 Identity**，但 block.py 里的 DFL 类仍在——**推理时会跳过 DFL 前向**，节省 ~0.3 ms；
4. **`end2end=True` 是 v26 默认**：<SrcLink file="v26:cfg_models_26/yolo26.yaml" :line="9" /> `end2end: True`——**默认开启双分支**；不想要 E2E 需手动关；
5. **Depth `cal_a/cal_b` 初始化为 0**：buffer 初始为 0，推理时通过 log-affine 校准——**校准流程不在训练主循环里**，需要在推理脚本里独立做；
6. **`nc: 1` 是 Depth yaml 的 parser 占位**：<SrcLink file="v26:cfg_models_26/yolo26-depth.yaml" /> 里 `nc: 1`——**Depth 是回归任务不是分类任务**，`nc=1` 只是让 parser 不报错，实际深度值走 `head Conv(1ch)`；
7. **YOLOE 依赖外部 CLIP 权重**：<SrcLink file="v26:cfg_models_26/yoloe-26.yaml" /> `text_model: mobileclip2:b`——**首次推理需自动下载 mobileclip2 权重**（约 50 MB）；
8. **官方权重全部公开**：30 份 .pt 全部可通过 `ultralytics/assets` releases 下载，Kaggle/Colab/Binder 一键运行——**这是 v14（IN PROGRESS）与 v26 最实质的差距**。

## 论文引用的相关方法

- **End-to-End 检测**：FCOS、CenterNet、TOOD；v10 的双头独立 loss；
- **深度估计**：DPT、MiDaS、Monodepth2、AD-Metric、Eigen 深度评估协议；
- **语义分割**：Cityscapes 基准、CE + Dice 组合、Categorical Focal Loss；
- **姿态估计**：OpenPose、HRNet、Real NVP Normalizing Flow；
- **开放词汇检测**：GLIP、OWL-VIT、CLIP 系模型；
- **YOLO 家族**：v8/v10/v11/v12/v13/v14 都在 Related Work 引用。

## 论文的贡献与快照事实的匹配度

- **强匹配**：Detect/Segment/OBB/Pose/Classify 五任务的类定义、E2E 双分支实现、Depth/Semantic 新任务全部完整实现、六任务性能表数字全部对齐 README；
- **弱匹配**：`A2C2f`/`AAttn` 类保留但 yaml 不用（是设计选择还是遗留？README 未说明）；`RealNVP` 的具体 flow 层数与耦合方式论文未细化；
- **不匹配**：无——v26 是官方仓库，README + 论文 + 代码三方同源，无 v14 那种「README 声称 vs 权重未发布」的偏差。

## 论文未回答但工程界关心的问题

1. **A2C2f 未来是否彻底移除？**：v26 官方口径是「经典骨干回归」，但类保留——**v27 会移除还是重新启用？** 官方未透露；
2. **E2E 双分支训练稳定性**：`E2ELoss` 的 `final_o2m` 默认值论文未说明，实际收敛曲线未公开；
3. **Depth 的校准流程**：`cal_a/cal_b` 需要在推理时独立做 log-affine 对齐——**具体对齐算法（median vs log-least-squares）在哪个脚本里？** README 未指向具体文件；
4. **RLE + RealNVP 的超参**：`error.clamp(-100, 100)` 的上下界是手调的、RealNVP 的 flow 层数与耦合方式未公开；
5. **Semantic 的 `nc=1` 二分类退化路径**：什么时候用 CE，什么时候用 BCE？代码里是 `if nc == 1` 判断，但论文未给出典型场景；
6. **YOLOE 的文本模型选择**：`mobileclip2:b` 是官方推荐，但 `text_model` 字段是否支持 `ViT-L/14@336` 等大模型？README 未说明；
7. **Enterprise License 的商业边界**：AGPL-3.0 + Enterprise 双许可，具体「商用需申请」的门槛标准未公开——**内部 SaaS 用是否需申请？** 建议联系 Ultralytics 官方；
8. **v27 发布时间**：README Banner 说 Coming Soon，但**具体发布时间未定**。

## 本站快照的处理原则

- **以 README + 论文 + 快照代码为准**：六任务数字来自 README "Models" 章节；结构描述以 `yolo26.yaml` + `head.py` + `loss.py` 为准；
- **区分官方线 vs 第三方学术线**：本站 v26 全部基于 ultralytics@`2e241d0`（官方）；同期第三方 v14 zhangcbb/yolov14（学术）见 [v14 概览](/versions/v14/index)；
- **不推测未披露内容**：v27 发布时间、Enterprise License 商业边界、RealNVP 具体耦合方式等论文未披露的部分，本站只记录"论文未提及"；
- **区分作者口径与代码事实**：A2C2f/AAttn 类保留但 yaml 不用的现象在 [痛点页](/versions/v26/painpoints) 中显式提醒。

## 延伸阅读

- [arXiv:2606.03748（v26 论文）](https://arxiv.org/abs/2606.03748)
- [GitHub 仓库 ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（commit `2e241d0`）
- [Ultralytics Docs YOLO26](https://docs.ultralytics.com/models/yolo26)
- [arXiv:2608.04720v3（v14 论文）](https://arxiv.org/abs/2608.04720) — 同期学术线
- [v13 发布精读](/versions/v13/paper) — 上一代官方线（若存在）
- [v14 发布精读](/versions/v14/paper) — 同期第三方学术线
- [Architecture 逐层图解](/versions/v26/architecture) — yaml 与骨干-颈-头逐层图解
- [理论 Concept](/versions/v26/concepts) — E2E / Depth SILog / Pose RLE / Semantic CE+Dice 机制解析
- [痛点追踪](/versions/v26/painpoints) — README 与代码偏差 8 项
