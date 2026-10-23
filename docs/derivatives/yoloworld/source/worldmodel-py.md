# ultralytics/nn/tasks_worldmodel.py

YOLO-World v1/v2 的建模器外壳：**`WorldModel(DetectionModel)` 只 92 行、只在 v8 `DetectionModel` 上追加两件事**——`txt_feats` 缓冲（CLIP 编码后 L2 归一化的 `[B, nc, 512]` 张量）+ `predict` 里对三种「文本感知」模块（`C2fAttn` / `ImagePoolingAttn` / `WorldDetect`）做分发。原仓库该段位于 `ultralytics/nn/tasks.py:558-646`，快照里独立成文方便锚点定位。

> 快照路径：`source/yoloworld/snapshot/ultralytics/nn/tasks_worldmodel.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/nn/tasks_worldmodel.py" />

**阅读路线**：

- [L5 WorldModel](#L5)：继承 `DetectionModel`——v8 主线的建网（`parse_model`）、`fuse`、`strides` 全部沿用
- [L8-12 __init__](#L8-L12)：`self.txt_feats = torch.randn(1, nc or 80, 512)` 是**占位符**（推理前必须 `set_classes` 覆盖）；`self.clip_model = None` 预留缓存 CLIP 的位——`cache_clip_model=True` 时不重复加载
- [L14-33 set_classes](#L14-L33)：`txt_feats` 缓冲的核心 API——`clip.tokenize(text)` 分批编码 → L2 归一化（`/norm(p=2, dim=-1, keepdim=True)`）→ `reshape(-1, len(text), 512)`；末行 `self.model[-1].nc = len(text)` 同步检测头——**这就是 open-vocabulary 切换类别的唯一入口**
- [L35-76 predict](#L35-L76)：v1/v2 的整套 forward——先设备/形状对齐（`repeat(len(x), 1, 1)`）→ 循环 `self.model`：
  - [L56 m.f != -1](#L56)：按 from 索引取上游层输出（同 v8 `_forward_once`）
  - [L60-61 C2fAttn](#L60-L61)：接收 `txt_feats` 做**图像↔文本双向 attention**，更新图像特征
  - [L62-63 WorldDetect](#L62-L63)：接收 `ori_txt_feats`（未更新版本）做对比打分
  - [L64-65 ImagePoolingAttn](#L64-L65)：**唯一修改 `txt_feats` 的地方**——v2 cfg 里此模块被删，等价于分支不执行
  - [L71-75 embed](#L71-L75)：中间层可采样池化，用于 re-identification / retrieval
- [L78-91 loss](#L78-L91)：`self.criterion(preds, batch)`——`preds` 来自 `self.forward(batch["img"], txt_feats=batch["txt_feats"])`，`batch["txt_feats"]` 由 [train.py `preprocess_batch`](/derivatives/yoloworld/source/train-py) 注入

## 与 v8 主线 DetectionModel 的对比

| 维度 | v8 DetectionModel | WorldModel |
|------|-------------------|------------|
| 父类 | BaseModel | DetectionModel |
| 缓冲 | 无 | `txt_feats`（占位 `[1, nc, 512]`）+ `clip_model` |
| 类别切换 | 权重决定（`nc` 固定） | `set_classes(text_list)` 动态 |
| forward 签名 | `forward(x, profile, visualize, augment, embed)` | 增加 `txt_feats`、`embed` 参数 |
| 特殊模块 | 无 | `C2fAttn` / `ImagePoolingAttn` / `WorldDetect` |
| 数据字典 | `{img, cls, bboxes}` | 追加 `{texts, txt_feats}`（`preprocess_batch` 注入） |

`WorldModel` 本身只承担「文本接入点」的角色，具体的 `C2fAttn` attention 计算、`WorldDetect` 的对比头结构分别在 [head.py 的 WorldDetect](/derivatives/yoloworld/source/head-py) 与 AILab-CVC 侧的 [yolo_world_head.py](/derivatives/yoloworld/source/head-ailab-py) 实现。文本注入的动机与 `ImagePoolingAttn` 的取舍见 [架构页](../architecture)，`txt_feats` 的语义与训练时的 `txt_masks` 加权见 [concepts 页](../concepts) 与 [losses 页](../losses)。
