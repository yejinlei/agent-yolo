# ultralytics/nn/tasks.py

YOLOv8 的建模器：**BaseModel 基类 + 四类任务模型 + parse_model 建网函数**。v7 把 Model 与五种检测头塞在 models/yolo.py 一个文件里；v8 则头归 head.py、积木归 modules/，tasks.py 只管「YAML → nn.Sequential」的建网流水线，一套代码同时服务 detect/segment/pose/classify——单库多任务的工程支点。

> 快照路径：`source/yolov8/snapshot/ultralytics/nn/tasks.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v8:ultralytics/nn/tasks.py" />

**阅读路线**：

- 基类 [L22-170 BaseModel](#L22-L170)：[L42-64 _forward_once](#L42-L64) 按 `m.f` 取 from 层、`self.save` 决定缓存——FPN/PAN 的跳连全靠这两行；[L91-114 fuse](#L91-L114) Conv+BN 融合入口
- 检测建模 [L173-246 DetectionModel](#L173-L246)：[L185 调 parse_model 建网](#L185)；[L189-197 构建 strides](#L189-L197)——256 像素假输入一次前向反推 Detect 的 stride 8/16/32，并顺手调 bias_init
- 建网核心 [L462-535 parse_model](#L462-L535)：
  - [L467-475 scales 复合缩放](#L467-L475)：从 yaml 取 depth/width/max_channels 三参数，`yolov8n.yaml` 由文件名尾字母选中 n 档
  - [L494-495 模块白名单](#L494-L495)：Conv/C2f/SPPF 等自动推通道，[L498 make_divisible](#L498) 一行同时完成 width 缩放与 max_channels 截断
  - [L501-503 C2f 重复数 n 注入 args](#L501-L503) 后置 n=1——depth 缩放只在此处生效（n 档 0.33×6 → 2 层）
  - [L517-520 Detect 分支](#L517-L520)：把三个 from 层的通道数附给头；[L524 逐层实例化](#L524)
- YAML 与权重侧：[L538 yaml_model_load](#L538)（`yolov8n` → `yolov8.yaml + scale=n`）、[L556 guess_model_scale](#L556)、[L574 guess_model_task](#L574)、[L360 torch_safe_load](#L360)（检测到 v5 权重直接报不向前兼容）

## 与 v7 models/yolo.py 的对比

| 维度 | v7 models/yolo.py | v8 tasks.py |
|------|-------------------|-------------|
| 模型类 | Model 一个（检测专用） | BaseModel + Detection/Segmentation/Pose/Classification 四类 |
| 检测头 | IDetect 等五头同文件 | 头在 nn/modules/head.py，建模与头解耦 |
| 缩放参数 | depth/width_multiple 两参数 | + max_channels 三参数复合缩放 |
| anchor | yaml anchors 经 parse_model 传入头 | 无 anchor 概念，头只收通道列表 |
| 任务猜测 | 无 | guess_model_task 按头类型/文件名判断任务线 |
| 兼容层 | 无 | v5 权重加载报错 + 缺依赖自动安装 |

Detect 头（head.py L21-71）与 Conv（conv.py L26）未设站内页，SrcLink 会回退 GitHub 快照外链；结构配置见 [yolov8.yaml 源码页](/versions/v8/source/yolov8-yaml)，积木见 [block.py 源码页](/versions/v8/source/common-py)，损失模块见 [loss.py 源码页](/versions/v8/source/loss-py)。
