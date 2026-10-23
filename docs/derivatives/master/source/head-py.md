# ultralytics/nn/modules/head.py

YOLO 检测头（Detect）与所有任务头集合：**~1900 行**，包含 Detect / Segment / OBB / Pose / Classify / YOLOE / RTDETR / v10Detect 等 16 个头类。Master 直接继承，未做任何修改——Detect 是 Master 训练/推理的最终出口。

> 快照路径：`source/yolomaster/snapshot/ultralytics/nn/modules/head.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="master:ultralytics/nn/modules/head.py" />

**阅读路线**：

- [L1-L22 文件头](#L1-L22)：`NOT_MACOS14` / `dist2bbox` / `make_anchors` / `fuse_conv_and_bn` 等依赖注入
- [L23-L34 __all__](#L23-L34)：暴露的头类列表
- **Detect**（核心）[L37-L272](#L37)：
  - [L37 __init__ 签名](#L37)：`nc=80 / reg_max=16 / end2end=False / ch=()`
  - [L78-L88 类属性](#L78-L88)：
    - `dynamic = False`：force grid reconstruction
    - `export = False`：导出模式标志
    - `max_det = 300`：最大检测数
    - `legacy = False`：**关键**——v3/v5/v8/v9/v11 兼容性标志
    - `xyxy = False`：输出格式
  - [L89-L120 __init__ 主体](#L89-L120)：
    - [L104-L107 cv2](#L104-L107)：box 分支——两个 3×3 Conv + 一个 1×1 Conv
    - [L108-L119 cv3](#L108-L119)：**双分支**——
      - `legacy=True` 走普通 Conv（v3/v5/v8/v9/v11 老权重）
      - `legacy=False` 走 DW-separable（DWConv + Conv）——**Master 走这个分支**
    - [L120 dfl](#L120)：DFL(reg_max)，reg_max=16 时用，否则 Identity
  - [L122-L125 end2end 分支](#L122-L125)：one2one_cv2 / one2one_cv3 是 cv2/cv3 的 deepcopy
  - [L126-L144 属性](#L126-L144)：one2many / one2one / end2end getter
  - [L146-L155 forward_head](#L146-L155)：拼接 box + cls 预测
  - [L157-L177 forward](#L157-L177)：
    - 训练态：返回 preds（含 one2one 或 end2end）
    - 推理态：走 _inference → postprocess
    - [L164-L170 P2 gradient bridge](#L164-L170)：`p2_o2o_gradient_alpha` 可选
  - [L179-L200 _inference](#L179-L200)：`make_anchors` + `dist2bbox` 解码
  - [L202-L214 bias_init](#L202-L214)：初始化 box/cls 偏置
  - [L241-L266 get_topk_index](#L241-L266)：NMS-free 的 Top-K 选择——`torch.gather` 而非高级索引（ONNX 兼容）
  - [L268-L270 fuse](#L268-L270)：融合时删掉 one2many 头
- **Segment** [L273-L376](#L273)：Detect + mask 预测
- **OBB / Pose / Classify / YOLOE / v10Detect** [L436-L1823](#L436)：其他任务头

## 与 v12 Detect 的对比

| 维度 | v12 Detect | Master Detect |
|------|------------|---------------|
| 行数 | ~230 | ~230 |
| cv3 结构 | 双分支（legacy/现代） | **完全一致** |
| end2end | 支持 | 完全一致 |
| P2 gradient bridge | 无 | 完全一致（v12 引入） |
| NMS-free Top-K | 支持 | 完全一致 |

**Master 的关键差异不在 Detect 本身，而在 `legacy` 标志**：

- `parse_model` 遇到 ES_MOE 时把 `legacy` 置为 False（[code-walkthrough.md §2.1](../code-walkthrough#2-建网-parse_model-的-es_moe-独立分支)）
- 于是 Detect 走 DW-separable cv3 分支——**比 v12 默认（legacy=True）更快**
- 这也是为什么 Master 推理速度可以比 v12 更快（除了参数少之外）

**end2end 与 cv3 双分支的关系**：

- `end2end=False`（默认，训练态）：用 cv2/cv3 走 one2many，v8DetectionLoss 处理
- `end2end=True`（导出态）：one2one_cv2/one2one_cv3 是拷贝，走 NMS-free Top-K
- Master yaml 里 Detect 参数只有 `[nc]`，`end2end=False` 默认——**训练态走 one2many**

**为什么 Master 不用 v13 的 FullPAD / HyperACE**：Master 基于 v12 fork，v13 的 head 改进没有回合并——**Master 与 v13 是两个正交的分支**（[痛点 6 §Master 未合并 v13](../painpoints#6-master-未合并-v13-pain-not-merged-v13)）。

MoE 主模块见 [modules.py 源码页](./moe-modules-py)，损失集成见 [losses 页](../losses)，架构总览见 [architecture 页](../architecture)，训练见 [train.py 源码页](./train-py)。

## 常见问题

**Q：cv3 的双分支是什么？**
A：`legacy=True` 走普通 Conv（v3/v5/v8/v9/v11 老权重），`legacy=False` 走 DW-separable Conv（DWConv + Conv）——**Master 走这个分支**，因为 parse_model 遇到 ES_MOE 会置 `legacy=False`。

**Q：end2end=True 时 Detect 怎么工作？**
A：`end2end=True` 时 cv2/cv3 有 deepcopy 副本 `one2one_cv2/one2one_cv3`，训练时用 `one2many` 头，推理时用 `one2one` 头 + NMS-free Top-K（[get_topk_index](#L241)）。Master yaml 里 `end2end=False`，所以走 v8DetectionLoss。

**Q：reg_max=16 是什么意思？**
A：DFL（Distribution Focal Loss）的分布维度——reg_max=16 意味着每个方向有 16 个 bin，用来预测 box 的位置分布。reg_max=1 时无 DFL，直接输出 box。

**Q：P2 gradient bridge 是什么？**
A：[L164-L170 p2_o2o_gradient_alpha](#L164-L170)——end2end 训练时，`x_detach + alpha * (x - x_detach)` 让 one2one 头也接受梯度，避免 one2many 与 one2one 头分离。默认 alpha=0 不启用。

**Q：为什么 Master 不用 v13 的 Detect 改进？**
A：Master 基于 v12 fork，v13 的 Detect 改进（如 HyperACE、FullPAD）没有回合并。Master 与 v13 是**两个正交分支**。
