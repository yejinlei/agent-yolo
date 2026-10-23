# ultralytics/models/yolo/world/train_world.py

YOLO-World 的**从 scratch 训练器**——`WorldTrainerFromScratch(WorldTrainer)` 108 行、在 [微调 WorldTrainer](/derivatives/yoloworld/source/train-py) 基础上只覆盖 3 件事：`build_dataset`（把 Objects365 + GQA + Flickr30k 多个源拼接成一个 concat dataset）、`get_dataset`（解析自定义 `train`/`val` 二级 dict 数据格式）、`final_eval`（默认走 LVIS mini-val split）。docstring 里给的示例是论文里的 10M 图像混合训练配方——`Objects365.yaml` 作 yolo_data、`flickr30k` + `GQA` 作 grounding_data、`lvis.yaml` 作 val。

> 快照路径：`source/yoloworld/snapshot/ultralytics/models/yolo/world/train_world.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/models/yolo/world/train_world.py" />

**阅读路线**：

- [L8-37 class + docstring 示例](#L8-L37)：完整用法——`YOLOWorld("yolov8s-worldv2.yaml").train(data=data, trainer=WorldTrainerFromScratch)`，`data` 是二级 dict，`train.yolo_data` 是 yolo 格式路径列表、`train.grounding_data` 是 `{img_path, json_file}` 字典列表、`val.yolo_data` 是单条 yolo 数据集
- [L39-43 __init__](#L39-L43)：与父类同——只归一化 overrides
- [L45-64 build_dataset](#L45-L64)：训练侧的核心——[L56-61 列表推导](#L56-L61) 对 `img_path` 每个元素二选一：字符串走 `build_yolo_dataset(multi_modal=True)`（同 [train.py L77-79](/derivatives/yoloworld/source/train-py#L77-L79)）、字典走 `build_grounding(img_path, json_file)`（处理 COCO 格式标注）；[L62 concat](#L62) 用 `YOLOConcatDataset` 拼接，单源直接返回原始 dataset
- [L66-97 get_dataset](#L66-L97)：解析用户传入的自定义数据格式——[L74-75 assert train/val](#L74-L75) 强制二级结构；[L76 check_det_dataset 逐个校验](#L76)；[L77-83 LVIS 特殊处理](#L77-L83)：val 必须单数据集、含 `lvis` 时改用 `minival` split；[L83-92 双模式循环](#L83-L92)：`yolo_data` 走 train/val 字段、`grounding_data` 直接 append；[L93-95 强制 nc/names](#L93-L95)：以 val 数据集为准——**这是把 10M 训练源统一到同一 label 空间的唯一方式**
- [L99-101 plot_training_labels](#L99-L101)：直接 pass，禁止标签可视化——多源数据的标签分布无法用一张直方图画清楚
- [L103-108 final_eval](#L103-L108)：把 `args.data` 强制设成 `val.yolo_data[0]`，split 根据数据集名判断（含 `lvis` 用 `minival`，否则用 `val`）——**从 scratch 训练结束时只在 LVIS mini-val 上报分**

## 与微调 WorldTrainer 的对比

| 维度 | WorldTrainer（微调） | WorldTrainerFromScratch |
|------|---------------------|------------------------|
| 数据集输入 | 标准 yolo `data.yaml`（`path: train/val`） | 自定义二级 dict（`train.yolo_data` + `train.grounding_data` + `val.yolo_data`） |
| 多源混合 | ❌ | ✅（`YOLOConcatDataset` 拼 yolo + grounding） |
| CLIP 编码 | 训练时动态编码 texts（`preprocess_batch`） | 同（继承父类） |
| `nc` 来源 | `data.yaml` 的 `nc` 字段 | val 数据集的 `nc`（`final_data["nc"] = data["val"][0]["nc"]`） |
| 最终评估 | 用户 `data.yaml` 的 val split | 强制 LVIS mini-val / val |
| 典型数据 | COCO8、自定义闭集 | Objects365 + GQA + Flickr30k + LVIS |
| 适用场景 | 领域适配 | 论文复现 / 预训练 |

`WorldTrainerFromScratch` 与微调的差别不在模型结构，而在**数据配方**——从 scratch 训练时视觉 backbone 与 `txt_feats`（v2 是 `embeddings`）从零学习，配合 Objects365 的 365 类做长尾覆盖，再叠加 GQA/Flickr30k 的 grounding 数据补齐开放词汇标注。真正的预训练 loss（`txt_masks` 加权对比损失）在 [head-ailab-py L428-577](/derivatives/yoloworld/source/head-ailab-py#L428-L577) 实现。多源数据与 LVIS mini-val 的评估约定见 [concepts 页](../concepts)，微调 vs 从 scratch 的取舍脉络见 [痛点页](../painpoints)。
