# ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py

AILab-CVC 仓库的**核心 head 实现**——749 行、6 个 `@MODELS.register_module()` 类，是理解 YOLO-World 数学的必读页。三个对比头（`ContrastiveHead` L28、`BNContrastiveHead` L68、`RepBNContrastiveHead` L110）+ 两个 head module（`YOLOWorldHeadModule` L135、`RepYOLOWorldHeadModule` L293）+ 顶层 `YOLOWorldHead` L348。`loss_by_feat` L428-L577 是**训练时的对比学习损失**，包含 AILab-CVC 独门的两处机制：`txt_masks` 三重加权（L521-529）+ `world_size` 多卡补偿（L571-577）。

> 快照路径：`source/yoloworld/snapshot/ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py" />

**阅读路线**：

### 对比头三兄弟

- [L28 ContrastiveHead](#L28)：v1 头——**L2 归一化**
  - [L42-44 参数](#L42-L44)：`bias=0`、`logit_scale = log(1/0.07) ≈ 2.659`（CLIP 论文里的 τ=0.07 温度）
  - [L46-64 forward](#L46-L64)：`F.normalize(x, dim=1, p=2)` + `F.normalize(w, dim=-1, p=2)` 双侧 L2 → `einsum('bchw,bkc->bkhw', x, w)` → `x * logit_scale.exp() + bias`
- [L68 BNContrastiveHead](#L68)：v2 头——**BN 归一化**
  - [L82 norm = build_norm_layer(norm_cfg)](#L82)：BN 层，`embed_dims` 通道
  - [L85 logit_scale = -1.0](#L85)：注释「-1.0 is more stable」——比 v1 更保守
  - [L88-106 forward](#L88-L106)：`x = self.norm(x)`（BN，不再 L2）+ `w = F.normalize(w, dim=-1, p=2)`（文本仍 L2）→ 同 `einsum` → `x * logit_scale.exp() + bias`
- [L110 RepBNContrastiveHead](#L110)：v2 reparameterize 后的头——**只有 BN + 1×1 Conv**
  - [L124-125 norm + Conv2d(embed_dims, num_guide_embeds, 1)](#L124-L125)：把对比头的权重折叠成 1×1 卷积，无 `logit_scale`
  - [L127-131 forward(x)](#L127-L131)：`norm → conv`，签名只吃图像特征——文本权重已固化进 conv

### Head Module

- [L135 YOLOWorldHeadModule(YOLOv8HeadModule)](#L135)：v1/v2 通用 head module
  - [L142-153 __init__](#L142-L153)：新增 `embed_dims`、`use_bn_head`、`use_einsum`、`freeze_all`
  - [L155-165 init_weights](#L155-L165)：cls_pred 最后一层 bias 归零，对比头 bias 用 `log(5/nc/(640/stride)²)` 初始化——对应「0.01 目标 × 80 类 × 640/stride 采样点」的类别先验
  - [L167-229 _init_layers](#L167-L229)：三个尺度各自的 `cls_preds`（两层 3×3 + 1×1 到 `embed_dims`）、`reg_preds`（两层 3×3 + 1×1 到 `4*reg_max`）、`cls_contrasts`（BNContrastiveHead 或 ContrastiveHead 由 `use_bn_head` 决定）
  - [L241-244 train](#L241-L244)：`freeze_all=True` 时每次 `train()` 都强制冻结（供从 scratch 训练预热阶段）
  - [L246-254 forward](#L246-L254)：把 `txt_feats`/`txt_masks` 按尺度数复制，`multi_apply` 并行跑三尺度
  - [L256-289 forward_single](#L256-L289)：[L262-263 cls_embed → cls_logit](#L262-L263)；[L265-271 txt_masks 处理](#L265-L271)：训练时 `cls_logit = cls_logit * txt_masks` + 把 `txt_masks==0` 位置置 `-10e6`（防止未激活类在 softmax 里被采样）；[L273-283 bbox](#L273-L283)：DFL 分布 → `matmul(proj)`
- [L293 RepYOLOWorldHeadModule](#L293)：v2 reparameterize 版——**只吃 img_feats**
  - [L300-305 __init__](#L300-L305)：强制 `use_bn_head=True`、`use_einsum=False`
  - [L308-314 cls_contrasts 全部换成 RepBNContrastiveHead](#L308-L314)
  - [L316-339 forward_single](#L316-L339)：无 `txt_masks` 参数、无 mask 置 `-10e6`
  - [L341-344 forward](#L341-L344)：签名只收 `img_feats`

### 顶层 Head

- [L348 YOLOWorldHead(YOLOv8Head)](#L348)：v1/v2 共用顶层，只多一个 `world_size=-1` 参数
- [L351-353 __init__](#L351-L353)：`world_size=-1` 表示自动从 `get_dist_info()` 取
- [L357-368 loss](#L357-L368)：`outs = self(img_feats, txt_feats, txt_masks)` → `loss_by_feat(outs, bboxes_labels, img_metas)`
- [L370-394 loss_and_predict](#L370-L394)：给 `DDP` / one-stage 训练用
- [L396-400 forward](#L396-L400)：转调 `self.head_module(img_feats, txt_feats, txt_masks)`
- [L401-417 predict](#L401-L417)：调用 `predict_by_feat` 后处理
- [L428-577 loss_by_feat](#L428-L577)：**训练 loss 核心**
  - [L461-479 特征图尺寸跟踪](#L461-L479)：`featmap_sizes_train` 变则重建 `flatten_priors_train` + `stride_tensor`
  - [L481-485 GT 预处理](#L481-L485)：`gt_instances_preprocess` 拼出 `[B, max_k, 5]`（label + xyxy）
  - [L488-508 预测展开](#L488-L508)：三尺度 flatten → `bbox_coder.decode`
  - [L510-517 assigner](#L510-L517)：DFL assigner 输出 `assigned_bboxes` / `assigned_scores` / `fg_mask_pre_prior`
  - [L519 assigned_scores_sum](#L519)：`clamp(min=1)` 防除零
  - [L521-529 ★ txt_masks 加权](#L521-L529)：把 `batch_text_masks` 展成 `[B, n, K]` 权重矩阵 → `(loss_cls * cls_weight).sum(dim=-1).sum()`——**只有被激活的类别才计入 loss**，未激活类被完全 mask 掉；无 `txt_masks` 时退化为普通 sum
  - [L530 loss_cls / assigned_scores_sum](#L530)：除以 assigned_scores_sum 得到「每正样本平均 loss」
  - [L532-534 rescale bbox](#L532-L534)：`/stride_tensor` 统一到输入尺度
  - [L536-570 bbox + DFL loss](#L536-L570)：`mask_select` 取正样本 → `loss_bbox(pred, target, weight)`、`loss_dfl(dist, ltrb, avg_factor=assigned_scores_sum)`
  - [L571-577 ★ world_size 补偿](#L571-L577)：`get_dist_info()` 取多卡数——DDP 下梯度已被平均，loss 需乘回 `world_size` 才与单机对齐

## 对比头三兄弟

| 维度 | ContrastiveHead (L28) | BNContrastiveHead (L68) | RepBNContrastiveHead (L110) |
|------|-----------------------|-------------------------|-----------------------------|
| 归一化 | L2（`F.normalize`） | BN（`self.norm`） | BN（`self.norm`） |
| `logit_scale` | `log(1/0.07) ≈ 2.659` | `-1.0`（更保守） | 无 |
| `bias` | ✅ | ✅ | 无（折进 conv bias） |
| forward 签名 | `forward(x, w)` | `forward(x, w)` | `forward(x)` |
| 输出维度 | 与输入 `w` 相同（=`nc`） | 与输入 `w` 相同（=`nc`） | `num_guide_embeds`（= `nc`） |
| 可导出 ONNX | ❌（`einsum` + 动态 `w`） | ⚠️（`einsum` + BN） | ✅（纯 `BN + Conv`） |
| 用于 v1 cfg | ✅（`with_bn=False`） | ❌ | ❌ |
| 用于 v2 cfg | ❌ | ✅（`with_bn=True`，训练） | ✅（`reparameterized=True`，推理） |
| 温度学习 | ✅ `logit_scale` 可学习 | ✅ | 已折叠（不再学习） |

`txt_masks` 加权的数学含义与 `world_size` 补偿的动机见 [losses 页](../losses)；BN vs L2 归一化的工程取舍见 [layers 页](../layers) 与 [痛点页](../painpoints)；三尺度结构与 assigner 见 [架构页](../architecture)。
