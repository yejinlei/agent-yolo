# ultralytics/nn/modules/block.py

YOLOv8 的积木仓库：**DFL 期望值积分 + SPPF 串行池化 + C2f 梯度分流**。v7 的 common.py 有 82 个类/函数（RepConv、隐式知识、Swin 系）；v8 的 block.py 只导出 17 个——不是变少，而是拆家：Conv 住 conv.py、头住 head.py，本文件只留「块」。

> 快照路径：`source/yolov8/snapshot/ultralytics/nn/modules/block.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v8:ultralytics/nn/modules/block.py" />

**阅读路线**：

- [L18-36 DFL](#L18-L36)：[L27-30 固定权重卷积](#L27-L30) 预置 0..15 序列且 `requires_grad_(False)`——卷积即期望值积分；[L32-35 forward](#L32-L35) softmax 后把 64 维分布回归折成 4 个 ltrb 距离
- [L120-135 SPPF](#L120-L135)：[L128 单个 MaxPool](#L128) 串行池化 3 次、[L135 四路 Concat](#L135)——等效 SPP(5,9,13) 但只建一个池化层；v5 遗产在 v8 转正为骨干收尾（yolov8.yaml 层 9）
- [L169-189 C2f](#L169-L189)：[L176 cv2 入通道 (2+n)*c](#L176) 暗示输出路数；[L179-183 forward](#L179-L183) chunk 成两半后 `y.extend` 逐级过 Bottleneck、共 n+2 路全拼接；[L185-189 forward_split](#L185-L189) 是给 ONNX 导出备的 split 版
- [L272-284 Bottleneck](#L272-L284)：3×3 + 3×3 双卷积，[L280 add](#L280) 由 shortcut 与 c1==c2 共同决定——yaml 骨干 C2f 带 True、head 默认 False 的落点
- 对照组：[L103-117 SPP](#L103-L117)（并行三池化）、[L152-166 C2](#L152-L166)、[L192-205 C3](#L192-L205)——v8 配置均不引用，留作兼容与积木库存货
- RT-DETR 支援：[L54 HGStem](#L54)、[L82 HGBlock](#L82)（PPHGNetV2）、[L218 RepC3](#L218)——单库多任务在积木层的证据

## 与 v7 models/common.py 的对比

| 维度 | v7 common.py | v8 block.py |
|------|--------------|-------------|
| 体量 | 82 个类/函数（含 Swin/OREPA） | 17 个导出，Conv/头已外迁 |
| 主打积木 | ELAN Concat 堆叠 + SPPCSPC | C2f（split → n×Bottleneck → 全拼接） |
| 重参数化 | RepConv 三分支 + 条件 identity | 无（RepConv 仅 RT-DETR neck 使用） |
| 隐式知识 | ImplicitA/ImplicitM | 无——靠结构化梯度分流替代外挂向量 |
| 感受野 | SPPCSPC（CSP 拆分 + 三池化） | SPPF（串行 3 次 MaxPool） |
| 梯度通路 | E-ELAN expand/shuffle/merge | C2f n+2 路 concat，ELAN 思想的轻量落地 |

Conv 基础单元在 conv.py L26、Detect 的 cv2/cv3 双分支在 head.py L37-39（均站外，SrcLink 回退 GitHub）；建网入口见 [tasks.py 源码页](/versions/v8/source/tasks-py)，结构配置见 [yolov8.yaml 源码页](/versions/v8/source/yolov8-yaml)。
