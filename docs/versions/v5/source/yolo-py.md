# models/yolo.py

YOLOv5 模型定义：**Detect 检测头 + Model 构建器 + parse_model yaml 解析器**。本站所有 `v5:models/yolo.py:行号` 引用均指向本页。

> 快照路径：`source/yolov5/snapshot/models/yolo.py`（commit `f5b8f7d`，tag `v5.0`）。点击左侧行号可复制/分享行锚点；如 `#L49-L56` 可高亮多行区间。

<SourceViewer file="v5:models/yolo.py" />

**阅读路线**：

- Detect 头：[L24-38 构造（anchors/stride 初始化）](#L24-L38) → [L40-58 forward（训练/推理分支）](#L40-L58) → [L53-55 解码公式（xy/wh）](#L53-L55) → [L60-63 _make_grid](#L60-L63)
- Model 构建：[L66-85 __init__（yaml 加载）](#L66-L85) → [L89-98 stride 计算与 anchors 归一化](#L89-L98) → [L146-154 _initialize_biases（obj/cls bias 初始化）](#L146-L154)
- parse_model：[L201-205 yaml 解析入口](#L201-L205) → [L217-226 模块实例化（Conv/C3/SPP/Focus 白名单）](#L217-L226) → [L242-252 nn.Sequential 组装](#L242-L252)

## 与 v4 的关键差异

- v4 `yolo.c` 是 C 手写 forward/backward；v5 全部由 PyTorch autograd 接管
- v4 的 anchors 写在 cfg 里固定；v5 在 Model init 时通过 `m.anchors /= m.stride.view(-1, 1, 1)` 归一化到 grid 单位
- v5 新增 `_initialize_biases`：obj bias 初始化为 `log(8 / (640/s)²)`，假设每张 640 图平均 8 个物体
