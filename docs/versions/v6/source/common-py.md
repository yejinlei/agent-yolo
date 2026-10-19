# yolov6/layers/common.py

YOLOv6 的积木盒：**RepVGG 重参数化家族 + SPP 简化版 + BiFusion 融合**。核心是 `switch_to_deploy()`——训练态三分支结构在部署前融合为单卷积。

> 快照路径：`source/yolov6/snapshot/yolov6/layers/common.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v6:yolov6/layers/common.py" />

**阅读路线**：

- 基础单元：[L26 ConvModule](#L26)（act_type 可选 ReLU/SiLU/HSiLU）、[L67 ConvBNSiLU](#L67)
- SPP 家族：[L115 SimSPPF](#L115)（简化 SPPF，少一次激活）、[L171 CSPSPPF](#L171)（CSP 版，骨干尾部用）
- 重参数化核心：[L197-319 RepVGGBlock](#L197-L319)——训练三分支 rbr_dense/rbr_1x1/rbr_identity，部署融合 [L302-319 switch_to_deploy](#L302-L319)、矩阵融合 [L278-300 _fuse_bn_tensor](#L278-L300)
- QA 系列（v6 自身 v4.0）：[L322 QARepVGGBlock](#L322)、[L396 QARepVGGBlockV2](#L396)——训练态对齐推理态，量化更友好
- 组合块：[L569 RepBlock](#L569)、[L590 BottleRep](#L590)（alpha 加权残差）、[L634 BepC3](#L634)（CSPStackRep）、[L653 MBLABlock](#L653)
- 融合与工厂：[L695 BiFusion](#L695)（Neck 三路融合）、[L721 get_block](#L721)（按 training_mode 返回块类型）

## 与 v5 models/common.py 的对比

| 维度 | v5 common.py | v6 common.py |
|------|--------------|--------------|
| 定位 | Focus/C3/SPP 等通用积木 | Rep 重参数化家族 + SPPF 简化版 |
| 训练/推理结构 | 完全一致 | 训练多分支、推理单卷积 |
| 部署融合 | 无（靠 torchscript 追踪） | `switch_to_deploy()` 显式融合 BN |
| CSP 实现 | C3（3 conv） | BepC3/CSPStackRep（Rep 块堆栈） |
| 激活 | SiLU 统一 | Rep 块 ReLU + 可选 SiLU/HSiLU |
