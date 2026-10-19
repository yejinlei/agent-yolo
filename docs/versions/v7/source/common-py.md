# models/common.py

YOLOv7 的积木仓库：**E-ELAN 风格 CSP 块 + 计划性重参数化 RepConv + YOLOR 隐式知识**。论文里的 RepConvN 在快照中没有独立类，而是用 RepConv 的条件 identity 分支实现同一规则。

> 快照路径：`source/yolov7/snapshot/models/common.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v7:models/common.py" />

**阅读路线**：

- 基础单元：[L99-111 Conv](#L99-L111)（Conv+BN+SiLU，[L110 fuseforward](#L110) 为融合 BN 后的前向）
- 感受野与 CSP：[L195 SPP](#L195)、[L209 Bottleneck](#L209)、[L262-280 SPPCSPC](#L262-L280)（骨干尾部扩感受野：CSP 拆分 + 5/9/13 三路 MaxPool，yolov7.yaml 的 #51 层）
- CSP 家族：[L307 BottleneckCSPA](#L307)、[L323 BottleneckCSPB](#L323)、[L340 BottleneckCSPC](#L340)（A/B/C 三种堆叠深度，派生出 Res/Ghost/Rep 各系变体）
- YOLOR 隐式知识：[L433-443 ImplicitA](#L433-L443)（加性向量）、[L446-456 ImplicitM](#L446-L456)（乘性向量）——§4.3 freebie 的原料
- 重参数化核心 [L463-582 RepConv](#L463-L582)：训练三分支 rbr_dense/rbr_1x1/rbr_identity；[L486 条件 identity](#L486)——`c2 == c1 and s == 1` 才建 BN 分支，否则为 None，这正是论文 RepConvN"残差/Concat 处禁 identity"规则的代码实现；[L498-507 forward](#L498-L507) 三分支相加再激活；[L524-552 _fuse_bn_tensor](#L524-L552) 内含 [L536-544 id_tensor 技巧](#L536-L544)（identity 分支等价中心置 1 的 3×3 单位阵卷积）；[L584-643 fuse_repvgg_block](#L584-L643) 三分支相加为单卷积
- Rep 组合块：[L646 RepBottleneck](#L646) → [L678 起 RepRes 系列](#L678)（RepRes/RepResCSP/RepResX/RepResXCSP）
- v5 遗产：[L796 Focus](#L796)、[L808 SPPF](#L808)（tiny 等变体配置仍引用）
- 融合工具与 OREPA：[L1032 transI_fusebn](#L1032)（conv-BN 融合函数）、[L1038 ConvBN](#L1038)、[L1072 OREPA_3x3_RepConv](#L1072)、[L1224 RepConv_OREPA](#L1224)（在线重参进阶版）
- Swin 系列（实验性骨干）：[L1367 WindowAttention](#L1367) 起，[L1472 SwinTransformerLayer](#L1472)、[L1602 STCSPA](#L1602) 等，共 82 个类/函数

## 与 v5/v6 common.py 的对比

| 维度 | v5 common.py | v6 common.py | v7 common.py |
|------|--------------|--------------|--------------|
| 主打积木 | Conv/Bottleneck/C3 | RepVGGBlock/QARepVGG | RepConv（计划性重参）+ CSP 块 |
| 重参数化 | 无 | 骨干/Neck 全 Rep 化 | 有选择地用 RepConv，且规则禁 identity |
| identity 分支 | — | 训练态固定三分支 | 条件化：`c2==c1 and s==1` 才有 |
| 隐式知识 | 无 | 无 | ImplicitA/ImplicitM（YOLOR） |
| CSP 形态 | C3（3 conv） | BepC3/CSPStackRep | BottleneckCSPA/B/C、SPPCSPC |
| 感受野模块 | SPP | SimSPPF/CSPSPPF | SPPCSPC（+ SPPF 备用） |
| 部署融合入口 | 靠 torchscript 追踪 | `switch_to_deploy()` | `fuse_repvgg_block()` + `Conv.fuseforward` |

相关的头与损失见 [models/yolo.py](/versions/v7/source/yolo-py) 与 [utils/loss.py](/versions/v7/source/loss-py)。
