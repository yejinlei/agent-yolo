# configs/yolov6s.py

YOLOv6s 的**Python dict 配置**——v6 抛弃了 v5 的 yaml，配置本身是可执行的 Python（`model` / `solver` / `data_aug` 三段式），import 即校验、可写注释逻辑。

> 快照路径：`source/yolov6/snapshot/configs/yolov6s.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v6:configs/yolov6s.py" />

**阅读路线**：

- model 段：[L2-39](#L2-L39)——`depth_multiple=0.33` / `width_multiple=0.50`（L5-6）；backbone [L7-13](#L7-L13)（EfficientRep，`num_repeats=[1,6,12,18,6]`、[L11 fuse_P2=True](#L11)、[L12 cspsppf=True](#L12)）；neck [L14-18](#L14-L18)（RepBiFPANNeck）；head [L19-38](#L19-L38)（EffiDeHead，stride 8/16/32、[L30 atss_warmup_epoch=0](#L30)、[L31 iou_type='giou'](#L31)、[L32-33 use_dfl=False / reg_max=0](#L32-L33)——蒸馏续训才设 True/16）
- solver 段：[L41-51](#L41-L51)——SGD + Cosine，lr0=0.01、momentum=0.937、weight_decay=0.0005
- data_aug 段：[L53-65](#L53-L65)——HSV/translate/scale/fliplr，[L63 mosaic=1.0](#L63)、[L64 mixup=0.0](#L64)（默认关）

## 与 v5 yaml 的对比

| 维度 | v5 models/yolov5s.yaml | v6 configs/yolov6s.py |
|------|------------------------|------------------------|
| 形式 | 数据格式，语言中立 | 可执行 Python，语言绑定 |
| 校验时机 | yaml.load + parse_model 运行时 | import 时即报语法错 |
| 训练超参 | hyp yaml 单独一份 | solver + data_aug 同文件三段 |
| 结构描述 | from/to 层列表 | 三个组件 dict（backbone/neck/head） |
| 缩放 | depth/width_multiple | 同名参数 + num_repeats 列表 |

同目录还有 `yolov6{n,t,s,m,l,x}.py`、`*6.py`（P6 四尺度）、`lite/`（端侧）、`mbla/`、`qarepvgg/`、`repopt/`（RepOptimizer/QAT）等变体。
