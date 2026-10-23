# ultralytics/models/yolo/world/train.py

YOLO-World 的**微调训练器**——`WorldTrainer(DetectionTrainer)` 只 91 行、只覆盖 4 个方法：`on_pretrain_routine_end`（冻结 CLIP text encoder）、`get_model`（`nc` 硬编码到 80）、`build_dataset`（`multi_modal=True` 打开文本分支）、`preprocess_batch`（把 `texts` 用 CLIP 编码后塞回 `batch["txt_feats"]`）。这个训练器是**微调（fine-tune）**模式，输入是已标好闭集标签的数据集（如 COCO8、自定义 80 类）；从 scratch 训练走另一个 [train_world.py](/derivatives/yoloworld/source/train-world-py)。

> 快照路径：`source/yoloworld/snapshot/ultralytics/models/yolo/world/train.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/models/yolo/world/train.py" />

**阅读路线**：

- [L18-28 on_pretrain_routine_end](#L18-L28)：训练回调——[L20-23 RANK 判断](#L20-L23) 只在主进程把 EMA 模型的类别同步到 `names`（`set_classes(names, cache_clip_model=False)`）；[L25-27 clip.load + 冻结参数](#L25-L27)：`ViT-B/32` 加载后 `p.requires_grad_(False)` 全冻结——**CLIP text encoder 在微调模式下只作为「文本编码器」而非可训练参数**
- [L31 WorldTrainer](#L31)：继承 `yolo.detect.DetectionTrainer`——所有优化器、EMA、调度器、AMP 等 v8 主线机制全沿用
- [L45-49 __init__](#L45-L49)：只做参数归一化，`overrides is None` 时置空 dict
- [L51-65 get_model](#L51-L65)：[L55-60 WorldModel 构造](#L55-L60)——`nc=min(self.data["nc"], 80)` 硬编码到 80，注释写明「这是图像里不同文本样本的最大数量，不是实际 nc」；[L61-62 权重加载](#L61-L62) 支持预训练 `.pt`；[L63 注册回调](#L63) 把 `on_pretrain_routine_end` 挂到训练循环
- [L67-79 build_dataset](#L67-L79)：与 v8 唯一差异是 [L78 `multi_modal=mode == "train"`](#L78)——训练时才把 `texts` 字段传入 dataset，val 时不传（val 阶段类别由 `on_pretrain_routine_end` 一次性同步）
- [L81-90 preprocess_batch](#L81-L90)：CLIP 编码核心——[L86 `itertools.chain(*batch["texts"])`](#L86) 把 `[B, [texts]]` 展平成一维 → `clip.tokenize` → `text_model.encode_text` → L2 归一化（`/ norm(p=2)`）→ [L90 reshape](#L90) 回 `[B, K, 512]`（K = 每张图的文本数，通常 `num_classes`）

## 与 v8 DetectionTrainer 的对比

| 维度 | v8 DetectionTrainer | WorldTrainer |
|------|--------------------|--------------|
| 数据集 | `build_yolo_dataset` | 同 + `multi_modal=True` |
| batch 字段 | `{img, cls, bboxes}` | 追加 `{texts, txt_feats}` |
| CLIP 依赖 | 无 | `clip.load("ViT-B/32")` 冻结加载 |
| `nc` | 用户配置 | 硬编码 `min(nc, 80)` |
| EMA 类别 | 无 | `set_classes(names)` 同步 EMA |
| 训练循环 | 直接 loss | 同（loss 内部走 `WorldModel.loss`） |

CLIP 冻结意味着微调阶段**文本编码不会更新**——只有 vision backbone、`C2fAttn`、`WorldDetect` 会学习。想训 `embeddings`（v2 的 `SimpleYOLOWorldDetector` 那种可训练 prompt），得走 AILab-CVC 仓库的 [YOLOWorldDetector](/derivatives/yoloworld/source/detector-py) 或本仓库的 [WorldTrainerFromScratch](/derivatives/yoloworld/source/train-world-py)。`preprocess_batch` 的 `txt_feats` 缓冲与 [WorldModel.predict](/derivatives/yoloworld/source/worldmodel-py) 的 `txt_feats` 参数是同名同形张量，训练/推理两侧共用。CLIP 冻结的动机与从 scratch 训练的对比见 [concepts 页](../concepts)，损失计算见 [losses 页](../losses)。
