# ultralytics/nn/modules/head.py

Ultralytics 侧的 **`WorldDetect(Detect)` 检测头**——继承 v8 `Detect`，只做三件事：把 cv3 的最终卷积从 `nn.Conv2d(c3, nc, 1)` 换成 `nn.Conv2d(c3, embed=512, 1)`；把 `cv4` 从不存在改成对比头（`BNContrastiveHead` 或 `ContrastiveHead`，由 cfg 第三参 `with_bn` 决定）；`forward` 里在 `cv2` 输出后 `torch.cat` 拼接 cv4 的对比 logit。原仓库该段位于 `head.py:211-260`，快照把它与 v8 其余检测头（`Segment`、`Classify`、`YOLOELiteHead` 等）并列。

> 快照路径：`source/yoloworld/snapshot/ultralytics/nn/modules/head.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="yoloworld:ultralytics/nn/modules/head.py" />

**阅读路线**：

- [L211 WorldDetect](#L211)：唯一的新头类
- [L212 __init__(nc, embed, with_bn, ch)](#L212)：`nc` 是真实类别数、`embed` 是 cv3 输出维度（v1/v2 cfg 都设 512）、`with_bn` 由 cfg 第三参控制（v1=False / v2=True）、`ch` 是三尺度输入通道
- [L215 c3 = max(ch[0], min(self.nc, 100))](#L215)：cv3 隐藏通道——`min(nc, 100)` 让 20 类的微调数据集与 80 类主数据集共享同一宽度；`max(ch[0], ...)` 保证不小于输入
- [L216 cv3 ModuleList](#L216)：三个尺度各自的 `[Conv(x,c3,3) → Conv(c3,c3,3) → Conv2d(c3,embed=512,1)]`——**输出维度不再等于 `nc`**，而是固定 `embed=512`（对齐 CLIP 的 ViT-B/32 文本向量维度）
- [L217 cv4 = BNContrastiveHead(embed) if with_bn else ContrastiveHead()](#L217)：v1 走 L2 归一化对比头、v2 走 BN 归一化——两侧实现在 [AILab-CVC head-ailab-py](/derivatives/yoloworld/source/head-ailab-py) 有完整版本
- [L219 forward(x, text)](#L219)：`x[i] = cat((cv2[i](x[i]), cv4[i](cv3[i](x[i]), text)), 1)`——cv2 出 `reg_max*4=64` 维回归、cv4 出 `nc` 维对比 logit，通道维拼接
- [L228 cat across scales](#L228)：推理路径——三尺度 `[B, 64+nc, H, W]` → view 后沿最后维拼成 `[B, 64+nc, n_total]`
- [L233-237 export 分支](#L233-L237)：TF 系（saved_model/pb/tflite/edgetpu/tfjs）用切片避免 FlexSplitV；其余用 `split((reg_max*4, nc), 1)`
- [L244-246 tflite/edgetpu 预计算 norm](#L244-L246)：把 `strides / (stride[0] * grid_size)` 预乘进 box，提高数值稳定性
- [L248 dbox](#L248)：`decode_bboxes(dfl(box), anchors) * strides`——与 v8 Detect 完全一致
- [L253-260 bias_init](#L253-L260)：只重置 box head 的 bias 为 1.0；class head 的 bias 初始化在 AILab-CVC 的 `YOLOWorldHeadModule.init_weights` 里做（见 [head-ailab-py L155-165](/derivatives/yoloworld/source/head-ailab-py#L155-L165)）

## 与 v8 Detect 的对比

| 维度 | v8 Detect | WorldDetect |
|------|-----------|-------------|
| 父类 | `nn.Module` | `Detect`（继承） |
| cv3 输出通道 | `nc` | `embed=512` |
| cv4 | 无 | 对比头（BNContrastiveHead 或 ContrastiveHead） |
| forward 签名 | `forward(x)` | `forward(x, text)` |
| 类别维度来源 | 权重固定 | `txt_feats` 决定（cfg 里 `embed=512` 是文本向量维度） |
| 是否可 reparameterize | N/A | `with_bn=True` 时可重参数化为 1×1 Conv |
| cfg 里 head 行 | `Detect [nc, ...]` | `WorldDetect [nc, 512, with_bn]` |

v1 cfg（[yolov8-world.yaml L48](/derivatives/yoloworld/source/yolov8-world-yaml#L48)）写 `WorldDetect [nc, 512, False]`，v2 cfg（[yolov8-worldv2.yaml L46](/derivatives/yoloworld/source/yolov8-worldv2-yaml#L46)）写 `WorldDetect [nc, 512, True]`——**只有第三参不同**，触发完全不同的对比头实现。真正的对比头数学（`F.normalize` + `einsum` + `logit_scale`）在 [head-ailab-py L28-107](/derivatives/yoloworld/source/head-ailab-py#L28-L107) 有完整版；Ultralytics 侧是简写导入。文本感知检测头的架构原理见 [架构页](../architecture)，BN 与 L2 归一化的取舍见 [layers 页](../layers) 与 [痛点页](../painpoints)。
