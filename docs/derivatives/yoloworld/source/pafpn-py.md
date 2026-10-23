# ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py

AILab-CVC 的**多模态颈部**——235 行、两个 `@MODELS.register_module()` 类：`YOLOWorldPAFPN`（L16，v2 用）与 `YOLOWorldDualPAFPN`（L150，v1 用）。都继承 `mmyolo.models.necks.yolov8_pafpn.YOLOv8PAFPN`，把 v8 PAFPN 的 `CSPLayerWithTwoConv` 替换成带 `guide_channels` + `embed_channels` + `num_heads` 的 attention 版本。v1 额外多一个 `text_enhancer`（`ImagePoolingAttentionModule`），在 top-down 完成后**反向更新** `txt_feats`——这是 v1 与 v2 的核心分叉点，也是 v1 无法导出 ONNX 的元凶。

> 快照路径：`source/yoloworld/snapshot/ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py" />

**阅读路线**：

- [L16 YOLOWorldPAFPN](#L16)：v2 多模态颈部——纯「图像↔文本 attention」，无反向更新
- [L20-48 __init__](#L20-L48)：新增 `guide_channels`（文本向量维度，通常 256 或 512）、`embed_channels`（每个 attention block 的 embed 维度列表）、`num_heads`（多头数列表）、`block_cfg`（默认 `CSPLayerWithTwoConv`）
- [L50-76 build_top_down_layer](#L50-L76)：构造 top-down 阶段的 attention block——`in_channels = (in[idx-1] + in[idx])`（拼接后通道）、`out_channels = out[idx-1]`、并注入 `guide_channels` + `embed_channels` + `num_heads`（都经 `widen_factor` / `deepen_factor` 缩放）
- [L78-104 build_bottom_up_layer](#L78-L104)：bottom-up 同结构，通道索引改为 `idx` / `idx+1`
- [L106-146 forward(img_feats, txt_feats)](#L106-L146)：
  - [L112-114 reduce layers](#L112-L114)：把 backbone 三尺度特征降到 `out_channels`
  - [L116-129 top-down](#L116-L129)：`upsample → concat → attention_block(top_down_inputs, txt_feats)`——**每层都传入 `txt_feats`** 做双向 attention
  - [L131-139 bottom-up](#L131-L139)：`downsample → concat → attention_block(..., txt_feats)`
  - [L141-146 out_layers](#L141-L146)：三尺度各过一个 1×1 Conv 得到最终 head 输入
- [L150 YOLOWorldDualPAFPN(YOLOWorldPAFPN)](#L150)：**v1 多模态颈部**——增加 `text_enhancer`
- [L152-172 __init__](#L152-L172)：新增 `text_enhancder` 参数（拼写沿用官方），默认 `{type='ImagePoolingAttentionModule', embed_channels=256, num_heads=8, pool_size=3}`
- [L187-194 text_enhancer 构建](#L187-L194)：把 `image_channels` / `text_channels` / `num_feats` 补齐后 `MODELS.build`——**这就是 v1 cfg L38 `ImagePoolingAttn` 的原始位置**
- [L196-235 forward](#L196-L235)：
  - [L199-217 reduce + top-down](#L199-L217)：与 v2 一致
  - [L219 ★ txt_feats = self.text_enhancer(txt_feats, inner_outs)](#L219)：**唯一的反向更新**——从 top-down 后的三尺度图像特征聚合出新的 `txt_feats`
  - [L220-228 bottom-up](#L220-L228)：用**更新后**的 `txt_feats` 做 attention
  - [L230-235 out_layers](#L230-L235)：同 v2

## v1 vs v2 颈部对比

| 维度 | v2 YOLOWorldPAFPN (L16) | v1 YOLOWorldDualPAFPN (L150) |
|------|------------------------|-----------------------------|
| 父类 | YOLOv8PAFPN | YOLOWorldPAFPN |
| `text_enhancer` | ❌ | ✅（`ImagePoolingAttentionModule`） |
| 图像↔文本 attention | ✅（每层 attention block） | ✅（同） |
| 文本→图像反向更新 | ❌（`txt_feats` 单向只读） | ✅（L219，top-down 后重新计算） |
| top-down 后处理 | 无 | `text_enhancer(txt_feats, inner_outs)` |
| bottom-up 使用的 txt_feats | 原始输入 | 经 enhancer 更新后 |
| `pool_size` | N/A | 默认 3（三尺度池化） |
| `num_heads` | 列表（每层可不同） | 列表 + enhancer 独立 `num_heads=8` |
| cfg 里对应模块 | `C2fAttn` 4 处 | `C2fAttn` 4 处 + `ImagePoolingAttn` 1 处 |
| 可导出 ONNX | ✅ | ❌（enhancer 内含 Python `for` + 切片） |
| 对应 cfg | `yolov8-worldv2.yaml` | `yolov8-world.yaml` |

`text_enhancer` 是 v1 精度上限的关键——它让文本表征能反向吸收图像侧的三尺度信息，形成**双向交互**；但同时也锁死了部署路径。v2 用「纯 BN + 单向 attention」把精度换一点点回到可部署。多模态 attention 的数学结构见 [架构页](../architecture)，`ImagePoolingAttentionModule` 的完整实现细节（本快照未包含，属 `necks/` 子目录另一文件）与 `C2fAttn` 的关系见 [layers 页](../layers)，v1 → v2 的取舍脉络见 [痛点页](../painpoints)。
