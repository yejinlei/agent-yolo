# YOLOv13 (2025-06)

> **v1~v12 读者路标**：YOL 系列**首个把「高阶超图（Hypergraph）」作为骨干与颈部主通路**的版本——YOLOv13 的骨架是「**A2C2f + DSC3k2 沿用 v12 骨干 + 引入 HyperACE + FullPAD 三条全管线连接**」的组合。首发载体有**两条并行线**：
>
> - **学术线**（本站主线，16 页全部内容以此为准）：`iMoonLab/yolov13`，commit `73289949533efac82bb5f72ec19b746618656bd2`，2025-11-18 更新；对应 arXiv [2506.17733v2 "YOLOv13: Enhancing YOLO for High-Resolution Multi-Object Detection through Hypergraph-based Adaptive Correlation Enhancement"](https://arxiv.org/abs/2506.17733)。作者 Wei Yang（iMoonLab）。许可证 Apache-2.0。仓库根附 `paper-yolov13.pdf`。
> - **Ultralytics 官方线**：2025-11 起在 ultralytics 仓库主线出现 `YOLO13` 命名（无独立论文、无独立仓库、docs 404）。本站**仅在 family/timeline 中标注存在**，不建子页面、不做深入分析。
>
> 快照目录：`source/yolov13/snapshot/`（iMoonLab/yolov13，`.git` 已清除，只留 `README.md` / `paper-yolov13.pdf` / `ultralytics/` 主干）。

## 相对 v12 的三个「大改」

| 维度 | v12 | **v13** |
|------|-----|---------|
| 骨干深段注意力 | A2C2f 内嵌 AAttn（FlashAttention 硬依赖） | **A2C2f 沿用** + **新增 HyperACE 主通路**（超图消息传递 + 高阶关联） |
| 骨干融合块 | C3k2（C3k/Bottleneck 内瓶） | **C3k2 保留浅层** + **新增 DSC3k2**（DSConv 内瓶）+ **新增 DSC3k** |
| Neck 结构 | FPN+PAN 三段融合 | **FPN+PAN 保留** + **FullPAD 三条全管线 tunnel**（骨干↔颈部、颈部内部、颈部↔头） |
| 头 Detect | `legacy=False`（DW-separable cv3） | **未动**（`head.py` 里没有任何 v13 专属头） |
| 损失 | v8DetectionLoss（三分量总装） | **未动**（`loss.py` 里 grep "v13|HyperACE|FullPAD" 零匹配） |

## 只想带走三件事

1. **v13 的新代码集中在 block.py 的 540 行**：L1375 DSBottleneck / L1414 DSC3k / L1472 DSC3k2 / L1549 AdaHyperedgeGen / L1626 AdaHGConv / L1677 AdaHGComputation / L1721 C3AH / L1764 FuseModule / L1804 HyperACE / L1858 DownsampleConv / L1891 FullPAD_Tunnel。**loss.py、tal.py、head.py 零改动**——v13 的「新」是纯结构改动，损失语义与 v8 完全一致。
2. **HyperACE 是唯一真正跑超图的模块**：`fuse → cv1(1×1) → chunk(3)` 三路 → 两条 **C3AH**（AdaHGComputation = flatten → AdaHGConv = 生成自适应超边 + 两阶段消息传递 + 残差）与 `L` 个 **DSC3k**（DSConv 深度可分离）并行 → concat → cv2(1×1)。整张 yaml 里只出现 1 次 HyperACE（#23），出现在 head 首个 block，把骨干 P3/P4/P5 三路拉进来做**一次性的多尺度超图融合**。
3. **FullPAD 的三条全管线连接就是 7 个 FullPAD_Tunnel**：yaml 里 7 次 `[-1, k]` 形式的 `FullPAD_Tunnel`，`out = x[0] + gate * x[1]`（gate 初始化为 0.0 标量参数）——**是「学习门控」而不是「固定融合」**。论文消融 TABLE III：`000→47.1 / 100→47.8 / 010→47.6 / 001→47.7 / 111→48.0`（YOLOv13-S），三个子组件单独消融增益 0.5~0.9 AP。

## 主要结果（arXiv 2506.17733 TABLE I，COCO 主表）

| 型号 | FLOPs (G) | Params (M) | AP50:95 | AP50 | AP75 | Latency (ms, T4/FP16) |
|------|----------|-----------|---------|------|------|----------------------|
| YOLOv13-N | 6.4  | 2.5  | **41.6** | 58.7 | 44.8 | **1.28** |
| YOLOv13-S | 20.8 | 9.0  | **48.0** | 65.2 | 52.0 | 2.98 |
| YOLOv13-L | 88.4 | 27.6 | **53.4** | 70.9 | 58.1 | 8.63  |
| YOLOv13-X | 199.2 | 64.0 | **54.8** | 72.0 | 59.8 | 14.67 |
| YOLOv12-S | 21.4 | 9.3  | 47.1   | 64.2 | 51.0 | 2.82  |
| YOLOv12-L | 88.9 | 26.4 | 53.0   | 70.0 | 57.9 | 7.10  |
| YOLOv12-X | 199.0 | 59.1 | 54.4   | 71.1 | 59.3 | 12.46 |

**结论**：v13-S 相对 v12-S 少 0.6 GFLOPs、少 0.3 M 参数、AP 高 0.9；v13-L 在同 FLOPs 下 AP 高 0.4、延迟高 1.53 ms；v13-X 在同 FLOPs 下 AP 高 0.4、参数多 4.9 M、延迟高 2.21 ms。**"更多高阶信息，代价是延迟换回一部分"**——不是纯白拿。

## 训练配方（iMoonLab/yolov13 README L60-80 + 论文 §IV-A2）

- **600 epoch, batch=256, 640×640, lr=0.01 SGD, warmup 3 epoch, linear decay**
- 增强：Mosaic + Mixup + copy_paste
- 硬件：N/S 档 4 或 8 × RTX 4090；L/X 档 4 或 8 × A800
- 延迟测法：T4 + TensorRT FP16

## 页面地图

| 主题 | 页面 |
|------|------|
| 论文精读（arXiv 2506.17733v2） | [paper](/versions/v13/paper) |
| Architecture 逐层图解 | [architecture](/versions/v13/architecture) |
| 理论 Concept（HyperACE / FullPAD / DS） | [concepts](/versions/v13/concepts) |
| 架构与层速查 | [layers](/versions/v13/layers) |
| 损失函数 | [losses](/versions/v13/losses) |
| 反向传播推导 | [loss-gradient](/versions/v13/loss-gradient) |
| 训练技巧 | [tricks](/versions/v13/tricks) |
| 痛点追踪 | [painpoints](/versions/v13/painpoints) |
| 源码走读 | [code-walkthrough](/versions/v13/code-walkthrough) |

## 源码浏览（iMoonLab/yolov13@`73289949`）

- [nn/modules/block.py](/versions/v13/source/block-py) —— v13 新增 11 类，540 行
- [nn/modules/conv.py](/versions/v13/source/conv-py) —— DSConv（v13 新增，L84）
- [nn/modules/head.py](/versions/v13/source/head-py) —— 无 v13 改动
- [utils/loss.py](/versions/v13/source/loss-py) —— 无 v13 改动
- [utils/tal.py](/versions/v13/source/tal-py) —— 无 v13 改动
- [cfg/models/v13/yolov13.yaml](/versions/v13/source/yolov13-yaml) —— 50 行

## 与 v12 的对应关系速查

| 想从 v12 理解某个概念 | v13 里的对应物 |
|--------------------|--------------|
| A2C2f 内嵌 AAttn（骨干 P4/P5） | **沿用**，仍是骨干 #6/#8 |
| C3k2（骨干浅层 / P5-large 收尾） | **保留**（浅层 #2/#4），并新增 **DSC3k2**（#3/#5/#10/#13/#16/#19/#21）替换骨干中段 |
| C2f / C3k2 内瓶 | 内瓶换成 **DSBottleneck**（DSConv k1=3 + DSConv k2=5, d2=1） |
| FPN+PAN 三段融合 | 保留（#10/#13/#16/#19），额外**加 7 条 FullPAD_Tunnel** 做跨尺度/跨头/跨骨干的软融合 |
| 骨干出口 SPPF / C2PSA | **都去掉了**（和 v12 一致，v13 也没有 SPPF） |
| Detect cv3 DW-separable | **未动**（`head.py` 没有 v13 版本，仍是 v8 的 Detect） |
| v8DetectionLoss 三分量总装 | **未动**（`loss.py` 零改动） |

## 参考

- Yang W. *YOLOv13: Enhancing YOLO for High-Resolution Multi-Object Detection through Hypergraph-based Adaptive Correlation Enhancement.* arXiv:2506.17733v2, 2025-06-21（v2 更新 2025-11）. https://arxiv.org/abs/2506.17733
- iMoonLab/yolov13. https://github.com/iMoonLab/yolov13  @ commit `73289949533efac82bb5f72ec19b746618656bd2`
- Ultralytics. https://github.com/ultralytics/ultralytics （YOLO13 出现在 ultralytics 主线，无独立论文/仓库/docs）
