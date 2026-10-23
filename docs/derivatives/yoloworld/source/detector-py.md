# ailab-cvc/yolo_world/models/detectors/yolo_world.py

AILab-CVC 原始仓库（Apache 2.0）的**顶层 detector**——232 行、含两个 `@MODELS.register_module()` 类：`YOLOWorldDetector`（v1，L12）与 `SimpleYOLOWorldDetector`（v2，L109）。两者都继承 `mmyolo.models.detectors.YOLODetector`，但把 v8 检测头换成了**双通道**设计：一个走 `backbone.forward_image` 出图像特征，一个走 `backbone.forward_text` 出文本特征；文本特征在训练时随 `batch_data_samples` 传入、推理时通过 `reparameterize(texts)` 预编码后缓存。

> 快照路径：`source/yoloworld/snapshot/ailab-cvc/yolo_world/models/detectors/yolo_world.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ailab-cvc/yolo_world/models/detectors/yolo_world.py" />

**阅读路线**：

- [L12 YOLOWorldDetector](#L12)：**v1 顶层**——需要运行时 CLIP（走 backbone 的 `forward_text`）
- [L14-23 __init__](#L14-L23)：接收 `mm_neck`（是否多模态颈部）、`num_train_classes`、`num_test_classes`
- [L25-33 loss](#L25-L33)：`bbox_head.num_classes` 训练时设为 `num_train_classes`；`extract_feat` 返回 `(img_feats, txt_feats, txt_masks)` 三元组，`txt_masks` 是文本掩码——**v1 loss 的核心签名**
- [L35-56 predict](#L35-L56)：`bbox_head.num_classes = txt_feats[0].shape[0]`——推理时类别数由传入的文本数决定（open-vocabulary 入口）
- [L58-61 reparameterize(texts)](#L58-L61)：**推理前的文本预编码 API**——`self.backbone.forward_text(texts)` 一次性编码，缓存到 `self.text_feats`；之后 `extract_feat` 若发现 `self.text_feats` 存在则跳过 text forward
- [L75-105 extract_feat](#L75-L105)：数据来源判定——[L80-91 五分支判断](#L80-L91)：`None` 走缓存、`dict` 走 `texts`、`list + SampleList` 走 `data_sample.texts`、`hasattr(self, 'text_feats')` 走缓存、否则报错；[L94-99 图像/文本 forward 分流](#L94-L99)：有缓存只跑 `forward_image`，无缓存跑 `backbone(batch_inputs, texts)` 同时返回双特征；[L100-104 neck](#L100-L104)：`mm_neck=True` 时 neck 接收 `(img_feats, txt_feats)`
- [L109 SimpleYOLOWorldDetector](#L109)：**v2 顶层**——用可训练 `embeddings` 替代运行时 CLIP
- [L111-131 __init__](#L111-L131)：新增参数 `prompt_dim=512`（embed 维度）、`num_prompts=80`、`embedding_path`（预加载 numpy 权重）、`reparameterized=False`、`freeze_prompt=False`、`use_mlp_adapter=False`
- [L133-155 embeddings + adapter 分支](#L133-L155)：非 reparameterize 模式下构造 `self.embeddings = nn.Parameter(L2_normalize(randn))`；`freeze_prompt` 控制 `requires_grad`；`use_mlp_adapter=True` 时增加一个 `[Linear, ReLU, Linear]` MLP adapter
- [L157-168 loss](#L157-L168)：`reparameterized=True` 时 head 只吃 `img_feats`（文本权重已折入 conv），否则吃 `(img_feats, txt_feats)` 双通道
- [L170-194 predict](#L170-L194)：同 loss 分支逻辑
- [L196-209 _forward](#L196-L209)：`forward` 供 DDP / `torch.save` 用，也走两条路径
- [L211-232 extract_feat](#L211-L232)：v2 极简版——只跑图像 `self.backbone(batch_inputs, None)`；非 reparameterize 时把 `self.embeddings[None]` 广播为 `[1, K, D]`，可选过 adapter + 残差 + L2 归一化，最后 `repeat(img_feats[0].shape[0], 1, 1)` 广播到 batch 维；neck 阶段同 v1

## v1 vs v2 顶层对比

| 维度 | v1 YOLOWorldDetector (L12) | v2 SimpleYOLOWorldDetector (L109) |
|------|---------------------------|----------------------------------|
| 文本来源 | 运行时 `backbone.forward_text(texts)`（CLIP 编码） | 可训练 `embeddings` 参数或预加载 `.npy` |
| `reparameterize` API | ✅（L58） | 通过 `reparameterized=True` 走另一路径 |
| `txt_masks` | ✅ 训练时必需 | ❌（loss 不再需要 mask） |
| MLP adapter | ❌ | ✅ 可选 |
| 冻结 prompt | 不适用（用冻结 CLIP） | `freeze_prompt=True` 可关梯度 |
| 训练时头签名 | `loss(img_feats, txt_feats, txt_masks, batch)` | `reparameterized=False`：`loss(img_feats, txt_feats, batch)`；`=True`：`loss(img_feats, batch)` |
| 对应 cfg | `yolov8-world.yaml` | `yolov8-worldv2.yaml` |
| 部署友好度 | ❌（依赖 CLIP 运行时） | ✅（`reparameterized=True` 时可与 vision 融合） |

两顶层 detector 都是纯「调度器」——真正的 `bbox_head`（对比头 + 分配 + loss）在 [yolo_world_head.py](/derivatives/yoloworld/source/head-ailab-py)，颈部（多模态 PAFPN）在 [yolo_world_pafpn.py](/derivatives/yoloworld/source/pafpn-py)。`txt_feats` 的双来源设计与 `reparameterize` 缓存机制见 [架构页](../architecture)，v1 运行时 CLIP 与 v2 可训练 embeddings 的取舍见 [痛点页](../painpoints)。
