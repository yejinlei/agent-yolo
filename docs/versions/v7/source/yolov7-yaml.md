# cfg/training/yolov7.yaml

YOLOv7 标准模型的结构配置：**140 行、anchor-based、ELAN 堆叠骨干 + PANet 双向融合头**，末行 IDetect 接 3 组 anchors。这是 v7 主打的 36.9M/104.7G/51.4 AP 模型的全部结构描述。

> 快照路径：`source/yolov7/snapshot/cfg/training/yolov7.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v7:cfg/training/yolov7.yaml" />

**阅读路线**：

- 超参：[L1-4 nc=80 + depth_multiple=1.0 + width_multiple=1.0](#L1-L4)——全尺寸基准版，缩放变体见同目录 `yolov7x/w6/e6/d6/e6e/tiny` 等 yaml
- Anchors：[L6-10 3 组锚点](#L6-L10)（P3/8、P4/16、P5/32）——**v7 是 anchor-based**，与 v6 的 anchor-free 配置形成 2022 年的路线分叉
- 骨干 [L13-71](#L13-L71)：Conv stem 与 ELAN 堆叠 [L15-28](#L15-L28)（[L27 四路 Concat](#L27) 汇聚 -1/-3/-5/-6 四个深度的特征）；[L30-34 MP 下采样](#L30-L34) 出 P3/8 → [L42-48](#L42-L48) 出 P4/16 → [L56-62](#L56-L62) 出 P5/32 → [L70](#L70) 收尾
- Head [L74-140](#L74-L140)：[L75 SPPCSPC](#L75)（#51，骨干尾部扩感受野）；上采样融合 [L77-89](#L77-L89)（[L79 route P4](#L79)、[L88 六路 Concat](#L88)）与 [L91-103](#L91-L103)（[L93 route P3](#L93)）；MP 下行 [L105-118](#L105-L118)（[L109 三路 Concat](#L109)）与 [L120-133](#L120-L133)（[L124](#L124)）——经典 PANet 双向
- 三处 RepConv：[L135-137](#L135-L137)（128→256 / 256→512 / 512→1024）——c2≠c1，RepConv 的 identity 分支自动为 None，即论文 RepConvN"带 Concat 连接的卷积禁 identity"规则
- 检测头：[L139 IDetect](#L139)（`[[102,103,104], 1, IDetect, [nc, anchors]]`，三尺度各一个 RepConv 输出接入）
- 同级目录：`cfg/deploy/` 放部署用 yaml（含 `yolov7-tiny-silu.yaml`）；`cfg/baseline/` 放 yolor、yolov3、yolov4-csp 等对照配置

## 与 v5 yolov5s.yaml / v6 yolov6s.py 的对比

| 维度 | v5 models/yolov5s.yaml | v6 configs/yolov6s.py | v7 cfg/training/yolov7.yaml |
|------|------------------------|------------------------|------------------------------|
| 形式 | 标准 yaml | 可执行 Python dict | 标准 yaml |
| Anchors | 3 组（anchor-based） | 无（anchor-free） | 3 组（anchor-based） |
| 骨干积木 | Focus + C3 + SPP | EfficientRep（RepVGG） | Conv + ELAN Concat 堆叠 + MP |
| Neck | SPP + PANet | RepBiFPANNeck | SPPCSPC + PANet + 六路 Concat |
| 重参数化 | 无 | 全 Rep 化 | 计划性 3 处 RepConv |
| 检测头 | Detect | EffiDeHead | IDetect（带 ia/im 隐式知识） |
| 缩放 | depth/width_multiple | 同名参数 + num_repeats | depth/width_multiple（论文 §3.2 compound scaling） |

头的实现见 [models/yolo.py](/versions/v7/source/yolo-py)，积木定义见 [models/common.py](/versions/v7/source/common-py)。
