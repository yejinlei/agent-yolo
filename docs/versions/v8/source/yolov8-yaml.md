# ultralytics/models/v8/yolov8.yaml

YOLOv8 的结构配置：**46 行、anchor-free、C2f 骨干 + FPN/PAN 双向融合 + 三尺度 Detect**。对比 v7 的 140 行（anchors 三组 + ELAN 多路 Concat），行数近乎减半——anchor-free 砍掉 anchors 段，ELAN 的显式拼接收进 C2f 模块内部。

> 快照路径：`source/yolov8/snapshot/ultralytics/models/v8/yolov8.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v8:ultralytics/models/v8/yolov8.yaml" />

**阅读路线**：

- [L5 nc=80](#L5)；[L6-12 scales 复合缩放](#L6-L12)：`[depth, width, max_channels]` 五档——n/s 共享 1024 通道上限、m 768、l/x 512，max_channels 截断大模型通道增长；[L8 n 档注释](#L8) 自带官方摘要：225 层 / 3,157,200 参数 / 8.9 GFLOPs
- 骨干 [L15-26](#L15-L26)：层 0-9——Conv s2 ×2 → [L19 C2f(128, True)](#L19) → P3/8 → [L21 C2f(256, True)](#L21)（yaml 重复 6）→ P4/16 → [L23 C2f(512, True)](#L23) → P5/32 → [L25 C2f(1024, True)](#L25) → [L26 SPPF(1024, 5)](#L26) 收尾；`True` 即开 shortcut，全 yaml 仅骨干四处
- Head [L29-46](#L29-L46)：
  - FPN 上行：[L30 Upsample](#L30) + [L31 Concat 骨干 P4](#L31) → [L32 C2f(512)](#L32)；[L34-36 下到 P3/8](#L34-L36) 出小目标特征
  - PAN 下行：[L38 Conv s2](#L38) + [L39 Concat](#L39) → [L40 C2f(512)@18](#L40)；[L42-44 下到 P5/32](#L42-L44)——head 的 C2f 不传 True，shortcut 默认 False
  - [L46 Detect 三尺度收尾](#L46)：from 15/18/21、参数仅 nc——yaml 里没有任何 anchor 字段，anchor-free 的结构证据
- 职责分离：训练超参（增益 box 7.5 / cls 0.5 / dfl 1.5、mosaic 等）全部在 cfg/default.yaml，本文件只管结构

## 与 v7 cfg/training/yolov7.yaml 的对比

| 维度 | v7 yolov7.yaml | v8 yolov8.yaml |
|------|----------------|----------------|
| 行数 | 140 行（anchors + backbone + head） | 46 行（scales + backbone + head） |
| anchor | P3/8、P4/16、P5/32 三组写死 | 无 anchor 字段 |
| 骨干 | Conv + ELAN 多路 Concat + MP 下采样 | Conv + C2f（梯度分流在模块内） |
| 感受野 | SPPCSPC（层 51） | SPPF（层 9） |
| Neck | PANet + 六路 Concat + 3×RepConv | Upsample/Concat + C2f（无 Rep） |
| 检测头 | IDetect，参数 [nc, anchors] | Detect，参数 [nc]（解耦头、无 objectness） |
| 缩放 | depth/width_multiple 两参数 | scales 五档含 max_channels |
| 变体方式 | w6/e6/d6/tiny 各写独立 yaml | 同一 yaml，scale=n/s/m/l/x 一词切换 |

逐层解读见 [layers](/versions/v8/layers)，积木实现见 [block.py 源码页](/versions/v8/source/common-py)，建网函数见 [tasks.py 源码页](/versions/v8/source/tasks-py)，损失见 [loss.py 源码页](/versions/v8/source/loss-py)。
