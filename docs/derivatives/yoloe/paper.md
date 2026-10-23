# 论文精读：YOLOE: Real-Time Seeing Anything（arXiv:2503.07465）

**"zero inference and transferring overhead"**——YOLOE 是本站在**开放词表 YOLO 家族里第一条把『可重参数化』作为设计哲学的路线**。文本、视觉、无 prompt 三种输入走同一条检测头，训练结束后一条 `fuse()` 折叠所有开放词表痕迹，得到与闭集 YOLOv8 / YOLO11 / YOLO26 **完全同构**的推理图。

> **重要纠偏（本站口径）**：RepRTA 是 **Re-parameterizable Region-Text Alignment**，一个 `Residual(SwiGLUFFN(embed, embed))` 残差 MLP。**它不是 Normalizing Flow，也不是 RealNVP**——论文正文与代码里都没有 Normalizing Flow；社区贴子里常把 RepRTA 混淆为 RealNVP（可能是把 MobileCLIP 或 YOLO-World Qwen2 的记忆错乱了），这是需要纠正的常见误传。

## 四层材料

| 材料 | URL / 位置 | 快照 / 版本 |
|------|-----------|------------|
| arXiv 论文 | [arXiv:2503.07465](https://arxiv.org/abs/2503.07465) | v1 2025-03-10；ICCV 2025 收录 |
| 论文复现仓库 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) | commit `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`（AGPL-3.0） |
| Ultralytics 集成 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) PR #19775 | tag `v8.3.99` @ `2fbf7015b9e89294b59e37d78e9b5266d94d7139`（YOLOE 首发） |
| Ultralytics YOLOE-26 | 同上 | tag `v8.4.0` @ `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`（YOLOE-26 系列） |
| 本站快照 | `source/yoloe/snapshot/` | Ultralytics 8.3.99 主 + THU-MIG `README.md` + `LICENSE` |
| Hugging Face 权重 | [jameslahm/yoloe](https://huggingface.co/jameslahm/yoloe) | 6 档 v8/11 × {T/V/PF} × {seg/coco-pe/coco} |

> **本站口径**：论文数字以 THU-MIG/yoloe README 表格为准；结构描述以 Ultralytics 8.3.99 集成代码为准；YOLOE-26 系列的骨干描述以 Ultralytics 8.4.0 为准。**论文与代码的偏差在文末单独一节列出**。

## 论文元信息

- **标题**：YOLOE: Real-Time Seeing Anything
- **作者**：Ao Wang\* / Lihao Liu\* / Hui Chen / Zijia Lin / Jungong Han / Guiguang Ding（\* 等贡献）
- **单位**：**清华大学 THU-MIG**（Mobile Intelligence Group）
- **arXiv**：[2503.07465](https://arxiv.org/abs/2503.07465)，Computer Vision and Pattern Recognition (cs.CV)，2025-03-10
- **发表**：**ICCV 2025**
- **代码**：`https://github.com/THU-MIG/yoloe` @ `40cd606c`
- **许可证**：**AGPL-3.0**（copyleft，商用受限）
- **致谢引用**：Ultralytics、YOLO-World、MobileCLIP、lvis-api、CLIP、GenerateU——**YOLOE 是一个「站在六个肩膀上」的项目**，尤其是 YOLO-World 和 MobileCLIP。

## 核心贡献（作者自述）

论文摘要里把贡献总结为**三条正交方法**，每条对应一个可插拔模块，最后统一在 `YOLOEDetect` 头里汇合：

1. **RepRTA（Re-parameterizable Region-Text Alignment）**——文本对齐策略。用**可重参数化的轻量辅助网络**精炼预训练文本嵌入，做视觉-文本对齐；训练后与 `BN` + `1×1 Conv` **一起折叠**，**推理与迁移零开销**。对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="588" :end="619"/>。
2. **SAVPE（Semantic-Activated Visual Prompt Encoder）**——视觉提示编码器。用**解耦的 semantic branch + activation branch** 做空间感知的视觉提示嵌入，**只用最少复杂度**换准确率。对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="348" :end="400"/>。
3. **LRPC（Lazy Region-Prompt Contrast）**——无 prompt 策略。用**内置大词汇表 + 专门 embedding** 识别所有物体，**避开昂贵的语言模型依赖**。对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="403" :end="435"/>。

配套贡献（论文未作为"三大贡献"列出但代码里明确存在）：

- **BNContrastiveHead**——`1×1 Conv → BN → (cls_feat·txt_feat)·logit_scale + bias` 的可学习温度对比头；`fuse()` 前是"可对比结构"，`fuse()` 后是普通 `1×1 Conv`；对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="462" :end="509"/>；
- **YOLOEDetect / YOLOESegment**——闭集 Detect 头的可重参数化变体；`cv3` 输出通道从 `nc` 改为 `embed=512`，额外挂 `reprta` + `savpe` + `cv4`；对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="438" :end="654"/>；
- **六 Trainer 类**——三阶段流水线（Text → Visual → Prompt-Free）+ seg 版；对应 <SrcLink file="yoloe:ultralytics/models/yolo/yoloe/train.py" :line="20" :end="355"/>。

## 关键概念澄清（本站口径）

### RepRTA ≠ RealNVP（重要纠偏）

社区贴子里常把 RepRTA 与 RealNVP 混淆。**这是错误**：

| 维度 | RepRTA（YOLOE） | RealNVP（Dinh et al. 2016） |
|------|----------------|----------------------------|
| 全称 | Re-parameterizable Region-Text Alignment | Real-valued Non-linear Invertible Neural Networks |
| 归属 | YOLOE 论文新增 | 生成模型 Normalizing Flow 家族 |
| 结构 | `Residual(SwiGLUFFN(embed, embed))` | Affine Coupling Layer × N + log-det Jacobian |
| 目的 | 精炼文本嵌入，做视觉-文本对齐 | 密度估计 / 无样本生成 |
| 是否可逆 | **不要求可逆** | **要求可逆**（用于 Jacobian） |
| 计算量 | 一次前向 | 需要正逆两次前向 |
| 在 YOLOE 里的角色 | 文本 prompt 编码器 | **不存在** |

**为什么会被误传？** 三个可能来源：

1. **MobileCLIP 是苹果 2023 年发的、基于 DistilBERT 的轻量 CLIP**——名字里都没有 "RealNVP"，但两者发音近似；
2. **YOLO-World v2 用 SigLIP**——SigLIP 是 Google 2023 年的 CLIP 替代，也不是 Normalizing Flow；
3. **RealNVP 是"Real-V 系列"名字**——"YOLOE 的文本分支叫 RepRTA，缩写里也有个 R"——缩写撞车导致记忆串味。

**结论**：论文正文 §3.2 与代码 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="464"/> 里明确写的是 `self.reprta = Residual(SwiGLUFFN(embed, embed))`，**没有 Normalizing Flow，也没有 Affine Coupling**——任何写"YOLOE 用 RealNVP 做文本编码"的资料都是错的。

### 可重参数化（Re-parameterization）的含义

"Re-parameterizable" 是 YOLOE 的核心设计词，与 MobileNet 的 Conv-BN-Fuse、RepVGG 的 branch-fuse 是同一族设计模式：

- **训练期**：`cv3[-1]`（普通 `1×1 Conv`）→ `BNContrastiveHead`（BN + logit_scale + bias）→ `reprta`（SwiGLU 残差）——**三条支路分开学**，梯度不互相干扰；
- **推理期**：`fuse()` 把 `BN` 折叠进 `1×1 Conv` 的权重和偏置、把 `reprta` 直接删掉（`self.reprta = nn.Identity()`）、把 `txt_feats` 通过外积融合进 `1×1 Conv` 的权重——**结果就是一个普通的 `1×1 Conv(nc_in, nc_out, 1)`**，ONNX / TensorRT / CoreML 路径与闭集 YOLO 完全一致；
- **迁移期**：因为折叠后与闭集 YOLOv8 完全同构，`Linear probing` / `Full tuning` 走 Ultralytics 标准 COCO finetune 管线，**不需要任何迁移代码**——这是"zero transferring overhead"的承诺。

对应代码 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="469" :end="513"/>：

```python
def fuse(self, txt_feats):
    if self.is_fused: return
    assert not self.training
    txt_feats = txt_feats.to(torch.float32).squeeze(0)
    for cls_head, bn_head in zip(self.cv3, self.cv4):
        conv = cls_head[-1]
        logit_scale = bn_head.logit_scale
        bias = bn_head.bias
        norm = bn_head.norm
        t = txt_feats * logit_scale.exp()
        conv = fuse_conv_and_bn(conv, norm)
        w = conv.weight.data.squeeze(-1).squeeze(-1)
        b = conv.bias.data
        w = t @ w
        b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
        b2 = torch.ones_like(b1) * bias
        conv = nn.Conv2d(conv.in_channels, w.shape[0], kernel_size=1).requires_grad_(False)
        conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
        conv.bias.data.copy_(b1 + b2)
        cls_head[-1] = conv
        bn_head.fuse()
    del self.reprta
    self.reprta = nn.Identity()
    self.is_fused = True
```

**折叠后的关键公式**：`W_new = txt_feats @ W_1x1`——**文本嵌入被"烧"进权重矩阵的每一行**，推理时不需要再传文本；`nc_out` 由 `txt_feats` 的行数（词表大小）决定。

### Text-Prompt / Visual-Prompt / Prompt-Free 三种模式

论文里三种 prompt 模式共用同一份 `YOLOEDetect` 头，只在**推理入口**上分岔：

| 模式 | 训练阶段 | 输入 | 编码器 | 推理路径 | 权重后缀 |
|------|---------|------|--------|---------|---------|
| **Text-Prompt** | 阶段 ①（30 epoch） | CLIP tokenizer 编码的类目文本 | RepRTA + BNContrastiveHead | `YOLOEDetect.forward(x, cls_pe)` | `.pt` |
| **Visual-Prompt** | 阶段 ②（2 epoch） | SAM 2.1 生成的 mask | SAVPE | `YOLOEDetect.forward(x, cls_pe)` 但 `cls_pe = savpe(x, vp)` | `-vp.pt` |
| **Prompt-Free** | 阶段 ③（1 epoch） | 内置 1000 词 vocabulary | LRPCHead + `set_vocab` | `YOLOEDetect.forward_lrpc(x)` | `-pf.pt` |

**入口分岔代码** <SrcLink file="yoloe:ultralytics/nn/head.py" :line="564" :end="574"/>：

```python
def forward(self, x, cls_pe, return_mask=False):
    if hasattr(self, "lrpc"):  # prompt-free inference
        return self.forward_lrpc(x, return_mask)
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
    ...
```

`hasattr(self, "lrpc")` 是**运行期判断**——不是通过构造参数控制，而是**通过 `set_vocab` 挂载一个 `lrpc` 属性来切换**。这个设计有点非主流，但在 Ultralytics 的 Trainer 管线里很省事。

## 论文正文关键章节

### III-A Overall Architecture（§3.1）

论文 Fig 1 展示六阶段流水线：Backbone（沿用 YOLOv8 / YOLO11 / YOLO26 零改动）→ Neck（PAFPN 零改动）→ `cv2`（位置分支，零改动）→ `cv3`（类别分支，输出 `embed=512`）→ `BNContrastiveHead`（`cls_pe` 融合）→ `DFL + Decode + NMS`。

**关键结构性改变**：

- `cv3[i]` 的最后卷积输出从 `nc`（闭集类数）改为 `embed=512`（可学习 prompt embedding 维度）；
- 新增 `cv4[i]` 做对比融合：`cls_feat = bn_head(cv3_feat) * logit_scale + bias`，然后 `logits = (cls_feat @ cls_pe.T)`；
- 新增 `reprta`、`savpe` 两个文本/视觉 prompt 编码器；
- `fuse()` 在训练结束或加载权重时调用，把三条支路折叠为普通 `1×1 Conv`。

对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="443" :end="466"/>。

### III-B RepRTA 详解（§3.2）

RepRTA 是文本对齐的核心，由三部分组成：

1. **MobileCLIP 文本编码器**（外部预训练）——把文本 prompt 编码为 512-dim 向量；`text_model=mobileclip:blt`（Ultralytics 8.3.99 首发）或 `mobileclip2:b`（8.4.0）；
2. **RepRTA 残差 MLP**——`Residual(SwiGLUFFN(512, 512))`，把 MobileCLIP 的原始嵌入映射到"YOLOE 空间"；
3. **BNContrastiveHead**——用 `cls_feat` 与 `txt_feat` 做内积，乘以 `logit_scale.exp()` 得到 logit，再加 `bias`。

**零初始化保证训练稳定** <SrcLink file="yoloe:ultralytics/nn/head.py" :line="612" :end="615"/>：

```python
nn.init.zeros_(self.m.w3.bias)
nn.init.zeros_(self.m.w3.weight)  # l 档需改为 nn.init.constant_(self.m.w3.weight, 1e-6)
```

**训练初期 RepRTA 退化为 Identity**（`x + 0 = x`）——MobileCLIP 的原始嵌入直接进入 BNContrastiveHead，先稳住对比学习，再让 MLP 慢慢学习文本空间的微调。**这是 RepRTA 训练不稳定性的重要缓解**，也是本站[训练技巧](./tricks)里反复强调的"必须用 `nn.init.zeros_` 初始化"的原因。

**为什么用 SwiGLU？** SwiGLU（SiLU-Gated Linear Unit）是 LLM 时代的主流 FFN 激活（PaLM / OPT / GPT-NeoX）——`hidden = F.silu(x1) * x2` 相当于用一半的隐藏维度做 gate，另一半做 value，比传统 ReLU FFN 参数效率高。**RepRTA 用 SwiGLU 是 LLM 时代的做法，符合"文本对齐 = 语言模型子问题"的直觉**。

### III-C SAVPE 详解（§3.3）

SAVPE 是视觉提示编码器，把 SAM 2.1 生成的 mask 转换成视觉 prompt embedding。核心是**解耦 semantic branch 与 activation branch**：

```python
class SAVPE(nn.Module):
    def __init__(self, ch, c=512, e=512):
        self.cv1 = nn.Sequential(Conv(ch[0], c, 3), Conv(c, c, 3))   # semantic branch
        self.cv4 = nn.ModuleList(Conv(ch[i], c, 1) for i in range(len(ch)))
        self.cv5 = nn.ModuleList(Conv(ch[i], c, 1) for i in range(len(ch)))
        self.cv6 = nn.ModuleList(Conv(ch[i], e, 1) for i in range(len(ch)))
```

**对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="348" :end="400"/>**。

`forward(x, vp)` 里 `vp` 是 `(B, N, H, W)` 的归一化 mask：

- **semantic branch** 从 P3 特征图算出每个位置的类别语义：`sem = cv1(x[0])`；
- **activation branch** 对 P3/P4/P5 三个尺度的特征做 `Q`，用 `vp`（视觉 prompt）做 `K`，softmax 归一化后加权求和，得到空间感知的视觉 prompt 嵌入。

**为什么要解耦？** 论文 §3.3 说："Visual prompts are often noisy or imprecise, so a decoupled design allows the network to attend to semantic context when the prompt is uncertain."——**activation branch 负责"在哪里"，semantic branch 负责"是什么"，两条支路独立可训练**。

### III-D LRPC 详解（§3.4）

LRPC 是无 prompt 模式的核心，用内置 vocabulary 替代文本编码：

```python
class LRPCHead(nn.Module):
    def __init__(self, e, vocab, topk=500, t=0.05):
        self.vocab = vocab  # (V, e) 词表
        self.topk = topk
        self.t = t          # 阈值
```

**对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="403" :end="435"/>**。

`forward(cls_feat, loc_feat, conf, max_det)` 里的"Lazy"体现在**阈值剪枝**：先用低阈值过一遍 `cls_feat @ vocab.T`，只保留 `score > t` 的 top-K 词，再对剩下的做精细推理。**论文用 "Lazy Region-Prompt Contrast" 命名，"Lazy" 就是指这个阈值剪枝机制**。

**vocabulary 从哪来？** `set_vocab(names)` 从内置的 COCO 80 + LVIS 常见类里选 1000 词（实际词表在 `ultralytics/models/yolo/yoloe/vocab.py` 里，未在本快照保留）。**1000 词是工程折中**——太少召回不够，太多推理变慢；论文 §3.4 有消融：500 / 1000 / 2000 词的 AP 与 FPS 权衡。

### III-E BNContrastiveHead 与可重参数化（§3.5）

BNContrastiveHead 是把"对比学习"落到"1×1 卷积"的关键桥：

```python
class BNContrastiveHead(nn.Module):
    def __init__(self, dim, max_neg_cat=1000):
        self.norm = nn.BatchNorm2d(dim, affine=True)  # affine=True 才能 fuse
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
    def forward(self, x, txt_feats):
        x = self.norm(x) * self.logit_scale.exp() + self.bias
        return txt_feats @ x  # 内积
```

**对应 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="447" :end="462"/>**。

`affine=True` 是关键——如果 `affine=False`，BN 就没有 `weight` / `bias` 参数，无法与后续 `1×1 Conv` 折叠；`logit_scale` 是**可学习的温度**（初始 `log(1/0.07) ≈ 2.66`），类似 CLIP 的 `logit_scale`。

**为什么用 BN 而不是 LayerNorm？** BN 有 `weight` / `bias` / `running_mean` / `running_var` 四个参数，**这四个参数都可以与 `1×1 Conv` 代数折叠**（`fuse_conv_and_bn`）；LN 不行，因为它在 batch 维上聚合，无法与空间卷积代数折叠。**这是 YOLOE 唯一用 BN 的地方**，其他 YOLO 系列都是 LN 或 LayerNorm，这是**可重参数化设计对架构的强约束**。

### III-F 三阶段训练流水线（§4.1）

论文的三阶段训练流程（`train.py` 里对应不同的 Trainer 类）：

```
Stage 1 (30 epoch): Text prompt
  - 数据：Objects365v1 + GQA + Flickr30k（Objects365v1 用 box，GQA+Flickr30k 用 grounding）
  - 冻结：骨干 + Neck（0）
  - 可训：cv2, cv3, cv4, reprta, proto, cv5
  - 目的：学 RepRTA + BNContrastiveHead 的基础对齐

Stage 2 (2 epoch): Visual prompt
  - 数据：Objects365v1（用 SAM 2.1 生成 mask）
  - 冻结：骨干 + Neck + RepRTA + BNContrastiveHead + cv2/cv3
  - 可训：SAVPE（30M 参数）
  - 目的：只训视觉 prompt 编码器，不动其他参数
  - 特殊：需要 seg2det 转换，训完再 get_vp_segm 补回分割头

Stage 3 (1 epoch): Prompt-free
  - 数据：LVIS 单类版本（`generate_lvis_sc.py` 生成）
  - 冻结：骨干 + Neck + RepRTA + BNContrastiveHead
  - 可训：LRPC vocabulary embedding
  - 目的：让模型学会从内置 vocabulary 里选词
```

**对应 Ultralytics 6 Trainer 类**（<SrcLink file="yoloe:ultralytics/models/yolo/yoloe/train.py" :line="20" :end="355"/>）：

| Trainer | 场景 | 权重后缀 |
|---------|------|---------|
| `YOLOETrainer` | 阶段 1 全参微调 | `.pt` |
| `YOLOEPETrainer` | Linear probing（只训 `cv3[-1]`） | `-coco-pe.pt` |
| `YOLOETrainerFromScratch` | 阶段 1 从零训练 | `.pt` |
| `YOLOEPEFreeTrainer` | 阶段 3 Prompt-Free vocabulary 训练 | `-pf.pt` |
| `YOLOEVPTrainer` | 阶段 2 视觉 prompt 训练（SAVPE） | `-vp.pt` |
| seg 版本（`YOLOEPEFreeTrainer` 等） | 分支配 seg 头 | `-seg-*` |

**为什么三阶段？** 论文 §4.1 说："For visual prompts, because only SAVPE is trained, we can adopt the detection pipeline with less training time"——**分阶段训练的核心是让每一阶段的可训参数尽量少，收敛步数尽量少**——这是 YOLOE 训练成本比 YOLO-Worldv2 低 26.7% 的关键。

## 实验结果详解

### LVIS Zero-shot Detection（TABLE II）

**YOLOE-v8-S 在 LVIS `minival` 上**：

| 指标 | YOLO-Worldv2-S | YOLOE-v8-S (T) | YOLOE-v8-S (V) | Δ vs YOLO-World |
|------|----------------|-----------------|-----------------|----------------|
| Params | 24.5 M | 12 M | 13 M | -51% |
| Time（8×RTX4090） | 32.6 h | 12.0 h | — | -26.7% |
| FPS（T4 + TensorRT） | 218.0 | 305.8 | 64.3 | +1.4× |
| $AP$（Fixed） | 24.4 | **27.9** | 26.2 | +3.5 |
| $AP_r$（region/class） | 21.7 | **22.3** | 21.3 | +0.6 |
| $AP_c$（conjunction） | 26.5 | **27.8** | 27.7 | +1.3 |
| $AP_f$（phrase） | 25.9 | **29.0** | 25.7 | +3.1 |

**口径提醒**：LVIS `minival` 用的是 **Fixed AP** 而非 Standard AP，`AP_f` 的 29.0 是短语识别的固定集合指标，不是开放域短语检索。

**Text 优于 Visual 的原因**：Visual prompt 阶段只训 SAVPE 的 30M 参数（骨干 + RepRTA 都冻结），**收敛步数远小于 Text 阶段**——这是设计取舍，不是缺陷。

### LVIS Zero-shot Segmentation（TABLE III）

**Standard AP<sup>m</sup>**（LVIS `val` 集）：

| 型号 | $AP^m$ (T/V) | $AP_r^m$ (T/V) | $AP_c^m$ (T/V) | $AP_f^m$ (T/V) |
|------|--------------|-----------------|-----------------|-----------------|
| YOLOE-v8-S | 17.7 / 16.8 | 15.5 / 13.5 | 16.3 / 16.7 | 20.3 / 18.2 |
| YOLOE-v8-M | 20.8 / 20.3 | 17.2 / 17.0 | 19.2 / 20.1 | 24.2 / 22.0 |
| YOLOE-v8-L | 23.5 / 22.0 | 21.9 / 16.5 | 21.6 / 22.1 | 26.4 / 24.3 |
| YOLOE-11-S | 17.6 / 17.1 | 16.1 / 14.4 | 15.6 / 16.8 | 20.5 / 18.6 |
| YOLOE-11-M | 21.1 / 21.0 | 17.2 / 18.3 | 19.6 / 20.6 | 24.4 / 22.6 |
| YOLOE-11-L | 22.6 / 22.5 | 19.3 / 20.5 | 20.9 / 21.7 | 26.0 / 24.1 |

**YOLOE-v8-L 在 $AP_r^m$ 上 Visual 反而略低于 Text**（16.5 vs 21.9）——**是 SAVPE 视觉提示在 L 档上的固有短板**，可能与 L 档语义容量过大、视觉 prompt 覆盖不足有关。

### COCO Downstream Transfer（TABLE V）

**关键点：折叠后与闭集 YOLOv8 / YOLO11 完全同构**：

| 训练模式 | 型号 | Epochs | $AP^b$ | vs 闭集 YOLOv8-L |
|---------|------|--------|--------|------------------|
| Linear probing | YOLOE-v8-S | 10 | 35.6 | — |
| Linear probing | YOLOE-v8-M | 10 | 42.2 | — |
| Linear probing | YOLOE-v8-L | 10 | **45.4** | **+0.6** |
| Linear probing | YOLOE-11-L | 10 | 45.1 | — |
| Full tuning | YOLOE-v8-S | 160 | 45.0 | — |
| Full tuning | YOLOE-v8-M | 80 | 50.4 | — |
| Full tuning | **YOLOE-v8-L** | 80 | **53.0** | **+0.6** |
| Full tuning | YOLOE-11-M | 80 | 51.3 | — |
| Full tuning | YOLOE-11-L | 80 | 52.6 | — |

**"Nearly 4× less training time"**——YOLOE-v8-L Full tuning 80 epoch vs 闭集 YOLOv8-L 通常 300+ epoch，训练时长缩短 4× 而精度还高 0.6 AP，**是 YOLOE "预训练 → 迁移" 范式的强卖点**。

**为什么迁移能"少 4× 训练时间"？** 三个原因叠加：

1. **RepRTA 已经学会"通用视觉-文本对齐"**——在 Objects365v1 + GQA + Flickr30k 上训过 30 epoch，视觉特征已经与语义空间对齐；COCO 迁移时不用从零学"什么是狗、什么是车"；
2. **折叠后与闭集同构，无迁移代价**——`fuse()` 已经把 RepRTA + BN + `1×1 Conv` 折叠成一个 `1×1 Conv`，COCO 迁移就是普通 Detect 头训练；
3. **`Linear probing` 只训最后 `1×1 Conv`**——从 12M 参数降到 < 1M 参数，收敛极快，10 epoch 就能达到 45.4 AP。

**迁移与从零训练的对比**（YOLOE-v8-L）：

| 方式 | Epochs | 训练时长 | $AP^b$ | 备注 |
|------|--------|---------|--------|------|
| YOLOE-v8-L 预训练 → LP | 10 | ~1 h | 45.4 | 只训 `cv3[-1]` |
| YOLOE-v8-L 预训练 → FT | 80 | ~8 h | 53.0 | 全参微调 |
| 闭集 YOLOv8-L 从零训 | 300 | ~30 h | 52.4 | Ultralytics 官方配方 |
| **YOLOE-v8-L 全 FT** | **80** | **~8 h** | **53.0** | **+0.6 AP、-73% 时长** |

### YOLOE-v8 与 YOLOE-11 的横比

| 指标 | YOLOE-v8-S | YOLOE-11-S | YOLOE-v8-L | YOLOE-11-L |
|------|-----------|-----------|-----------|-----------|
| Params | 12 M | 10 M | 45 M | 26 M |
| LVIS T $AP$ | 27.9 | 27.5 | 35.9 | 35.2 |
| LVIS V $AP$ | 26.2 | 26.3 | 34.2 | 33.7 |
| LVIS PF $AP$ | 21.0 | 20.6 | 27.2 | 26.3 |
| COCO LP $AP^b$ | 35.6 | 37.0 | 45.4 | 45.1 |
| COCO FT $AP^b$ | 45.0 | 46.2 | 53.0 | 52.6 |
| T4 TRT FPS（Text） | 305.8 | 301.2 | 102.5 | 130.5 |
| Time（8×4090） | 12.0 h | 13.0 h | 22.5 h | 23.5 h |

**发现**：YOLOE-11 系列的**参数比 v8 少**（L 档 26 M vs 45 M），但精度接近、FPS 更快——**这是因为 YOLO11 骨干用了 C3k2 替代 C2f，参数效率更高**；v8 系列精度略高（+0.7 AP 左右），可能与 YOLO11 训练配方差异有关。

### YOLOE-26 系列补充（Ultralytics 8.4.0）

YOLOE-26 系列在 Ultralytics 8.4.0（2026-01-14）首发，**论文未讨论**，只有 Ultralytics README 里的数字：

| 型号 | Params | $AP^b$（COCO） | 骨干 | 备注 |
|------|--------|----------------|------|------|
| YOLOE-26-n | ~4 M | — | C3k2 + C2PSA | 8.4.0 首发，速度优先 |
| YOLOE-26-s | ~9 M | — | C3k2 + C2PSA | — |
| YOLOE-26-m | ~20 M | — | C3k2 + C2PSA | — |
| YOLOE-26-l | ~35 M | — | C3k2 + C2PSA + reg_max=1 | 骨干升级到 C2PSA |
| YOLOE-26-x | ~45 M | — | C3k2 + C2PSA + reg_max=1 | — |

**关键差异**：YOLOE-26 骨干从 `C2f`（v8）升级为 `C3k2 + C2PSA`（YOLO26 骨干），**回归头从 DFL 换成 `reg_max=1`**——这是 Ultralytics 在 YOLO26 上的整体架构升级，YOLOE 头保持不变。见 [YOLO26 概览](/versions/v26/index)。

### Prompt-Free（TABLE IV）

| 型号 | Params | $AP$ | $AP_r$ | $AP_c$ | $AP_f$ | FPS（T4 PyTorch） |
|------|--------|------|--------|--------|--------|-------------------|
| YOLOE-v8-S | 13M | **21.0** | 19.1 | 21.3 | 21.0 | 95.8 |
| YOLOE-v8-M | 29M | 24.7 | 22.2 | 24.5 | 25.3 | 45.9 |
| YOLOE-v8-L | 47M | **27.2** | 23.5 | 27.0 | 28.0 | 25.3 |
| YOLOE-11-S | 11M | 20.6 | 18.4 | 20.2 | 21.3 | 93.0 |
| YOLOE-11-M | 24M | 25.5 | 21.6 | 25.5 | 26.1 | 42.5 |
| YOLOE-11-L | 29M | 26.3 | 22.7 | 25.8 | 27.5 | 34.9 |

**Prompt-Free 精度略低于 Text-Prompt**：YOLOE-v8-S PF 21.0 vs Text 27.9，差 6.9 AP——**vocabulary 只有 1000 词，覆盖度受限**；但**Prompt-Free 95.8 FPS 是三种模式里最快的**，因为不依赖文本编码器。

## 论文与代码的偏差（本站发现）

YOLOE 论文（THU-MIG/yoloe @ `40cd606c`）与 Ultralytics 8.3.99 集成的对齐度非常高，但仍有几处值得注意：

1. **`text_model` 版本号不一致**：THU-MIG 仓库 README L123-124 用 `mobileclip_blt.pt`（MobileCLIP-B LT），Ultralytics 8.3.99 首发也是 `mobileclip:blt`；但 **Ultralytics 8.4.0 起升级为 `mobileclip2:b`**——论文里的数字是基于 MobileCLIP v1 训的，8.4.0 权重是重训的；

2. **骨干版本分岔**：论文只报 YOLOE-v8 与 YOLOE-11 两系；**YOLOE-26 系列在论文发布 10 个月后（2026-01-14）由 Ultralytics 8.4.0 首发**——论文未讨论，YOLOE-26 的骨干是 `C3k2 + C2PSA + reg_max=1`，与论文 v8/v11 的骨干完全不同；

3. **视觉 prompt 训练流程的"seg2det 再补回"**：THU-MIG README L268-275 明确写出 "First, obtain the detection model → train the SAVPE module → use get_vp_segm.py to add the segmentation head"——**这是训练管线的一个非直觉约定**，Ultralytics 8.3.99 集成代码里没有把这套流程封装成 `YoloEVPTrainer` 一步到位，需要用户手动跑三段脚本；

4. **`reprta` 初始化的档间差异**：<SrcLink file="yoloe:ultralytics/nn/head.py" :line="613" :end="615"/> 里注释写："For models with l scale, please change the initialization to `nn.init.constant_(self.m.w3.weight, 1e-6)`"——**L 档需要手改初始化常数**，这是论文没讨论的工程细节；

5. **Prompt-free vocabulary 的具体词表未公开**：THU-MIG README 只说"built-in 1000 词 vocabulary"，但**具体是哪 1000 词未开源**——`ultralytics/models/yolo/yoloe/vocab.py` 在 8.3.99 快照里也未保留，**Prompt-Free 的复现依赖 8.4.0 起官方发的 `-pf.pt` 权重**；

6. **`fuse()` 的数值稳定性**：`txt_feats.to(torch.float32)` 强制转 fp32 是因为 `fuse_conv_and_bn` 的 `running_var` 在 fp16 下会溢出；**论文没提这个细节**，但工程上很关键，训练精度用 fp16 时 `fuse()` 前必须转 fp32；

7. **`YOLOESegment` 的 `return_mask` 逻辑**：<SrcLink file="yoloe:ultralytics/nn/head.py" :line="635" :end="654"/> 里 `return_mask` 是 `YOLOEDetect.forward` 的第三个参数，`YOLOESegment.forward` 没暴露这个参数——**Seg 版想拿到 mask 索引需要走 `forward_lrpc` 的间接路径**，这是 API 上的一个不一致；

8. **Ultralytics 的 `YOLOE` 入口在 `models/yolo/model.py`**：<SrcLink file="yoloe:ultralytics/models/yolo/model.py" :line="738" :end="880"/>——**不是独立的 `models/yolo/yoloe/model.py`**，与 v10 / v11 的顶层入口组织方式不同，可能是为了代码复用的考虑。

## 论文未回答但工程界关心的问题

1. **RepRTA 训练的稳定性边界**：`w3.weight` 零初始化 + SwiGLU 的非线性组合，**如果 warmup 步数不足会不会发散？** 论文没讨论，但 <SrcLink file="yoloe:ultralytics/nn/head.py" :line="613" :end="615"/> 的注释暗示 L 档必须用 `1e-6` 而非 `0`——**这是一个"经验常数"，不是理论保证**；

2. **vocabulary 1000 词是怎么选的**：论文 §3.4 有消融但没给选词算法——**是 COCO 80 + LVIS 高频 920，还是别的策略？** 复现时需要自己选；

3. **MobileCLIP 替换成 SigLIP 会怎样**：MobileCLIP 是"CLIP 的 10× 小版本"，SigLIP 是"CLIP 的精度替代品"——**换成 SigLIP 会不会破坏 RepRTA 的可重参数化？** 论文没讨论；

4. **AGPL-3.0 商用的具体边界**：AGPL-3.0 的 §13 规定了网络服务也算"分发"，**SaaS 部署 YOLOE 模型是否触发 AGPL 传染性？** 论文与仓库都没讨论，本站[痛点页](./painpoints)有专门一节讨论；

5. **CLIP tokenizer 77 token 上限**：CLIP 的 tokenizer 硬编码 77 token，**prompt 超过 77 token 会被截断**——对长描述类应用（"small brown dog on red couch near window"）影响显著；论文没讨论；

6. **Visual Prompt 的 SAM 2.1 依赖**：SAVPE 训练需要 SAM 2.1 生成的 mask，**SAM 2.1 本身也是 AGPL-3.0**（Meta 的许可），**两个 AGPL 组合的商用边界？** 论文没讨论；

7. **CVE-2026-111865 与 `autobackend.py`**：Ultralytics 主库的 `autobackend.py` 有 eval 注入漏洞，**影响至 YOLOE 首发 commit `40cd606c`**——Ultralytics 8.4.0 已修，但用 YOLOE 8.3.99 的用户需注意升级；

8. **YOLOE-26 系列的骨干差异**：YOLOE-26 用 `C3k2 + C2PSA + reg_max=1`，**回归头从 DFL 换成 reg_max=1 的 C2PSA 分支**——这会影响 `fuse()` 的折叠路径吗？**论文没讨论，需要看 Ultralytics 8.4.0 的 head.py**。

## YOLOE 与 YOLO-World 的学术血脉对比

YOLOE 是**开放词表 YOLO 家族的第三条路线**，与 YOLO-World 是**平行并列**关系：

| 维度 | YOLO-World v1/v2 | **YOLOE** | 关键差异 |
|------|------------------|-----------|---------|
| 作者 | CV-CUP / 深圳大学 / Tencent | **清华 THU-MIG** | 团队不同 |
| 论文 | arXiv:2401.17270 / 2411.13503 | **arXiv:2503.07465（ICCV 2025）** | YOLOE 晚 2 年 |
| 基座 | Ultralytics YOLOv8 | **Ultralytics YOLOv8 / v11 / v26** | YOLOE 三基座 |
| 文本对齐 | Qwen2-0.5B / SigLIP | **RepRTA + MobileCLIP-B(LT)** | YOLO-World 用 LLM，YOLOE 用轻量 CLIP |
| 视觉对齐 | 无 | **SAVPE** | YOLOE 独有 |
| 无 prompt | 无 | **LRPC + 1000 词 vocab** | YOLOE 独有 |
| 与闭集同构 | 否（推理时保留 LLM） | **是（fuse 后完全同构）** | YOLOE 的核心卖点 |
| 推理参数 | 12 M + Qwen2 250 M | **12 M** | YOLOE 推理期小 20× |
| 许可证 | Apache-2.0 | **AGPL-3.0** | YOLOE 商用受限 |
| 分发 | 独立 repo + ultralytics 集成 | **Ultralytics 8.3.99 起原生集成** | YOLOE 官方集成更早 |

**学术脉络**：YOLO-World 是**"用大模型做文本对齐"**的路线（Qwen2 做 zero-shot grounding），YOLOE 是**"用轻量 CLIP + 可重参数化"**的路线——**两条路线都在追求 open-vocabulary YOLO，但工程取舍完全相反**：

- YOLO-World 追求**推理精度上限**，愿意接受 250M Qwen2 的推理开销；
- YOLOE 追求**推理零开销**，接受 MobileCLIP 的精度上限，靠 RepRTA 折叠换同构；
- **两种路线正交**，见 [YOLO-World 综述](/derivatives/yoloworld/index)。

## 关键数字速查

- **论文 arXiv 首发**：2025-03-10（arXiv:2503.07465）
- **ICCV 2025 收录**：2025-10 前后
- **Ultralytics 集成首发**：2025-03-30（8.3.99，PR #19775）
- **YOLOE-26 系列首发**：2026-01-14（8.4.0，YOLO26 Models Release）
- **LVIS zero-shot 精度**：YOLOE-v8-S 27.9 AP（Text）/ 26.2 AP（Visual）/ 21.0 AP（Prompt-Free）
- **LVIS vs YOLO-Worldv2**：+3.5 AP，-26.7% 训练时长，+1.4× 推理速度，-51% 参数
- **COCO finetune**：YOLOE-v8-L Linear probing 45.4 AP（vs 闭集 44.8，+0.6）；Full tuning 53.0 AP（+0.6）
- **推理速度**：YOLOE-v8-S 305.8 FPS（T4 + TensorRT）
- **训练时长**：YOLOE-v8-S 12.0 h（8×RTX4090，Text 阶段 30 epoch）
- **参数量**：S 12M / M 27M / L 45M；YOLOE-26 补齐 N/S/M/L/X 五档
- **RepRTA 隐藏维度**：512（`embed=512`）
- **vocabulary 大小**：1000 词（Prompt-Free）
- **CLIP tokenizer 上限**：77 token
- **许可证**：AGPL-3.0
- **已知 CVE**：CVE-2026-111865（`autobackend.py` eval 注入）

## 深入阅读

- [YOLOE 概览（含电梯演讲 + 快速上手）](./index)
- [Architecture 逐层图解（RepRTA + SAVPE + LRPC 三分支详解）](./architecture)
- [理论 Concept（RepRTA 残差、SAVPE 解耦、LRPC 对比、BN 折叠）](./concepts)
- [架构与层速查（v8 / 26 双 yaml + 各档 scales）](./layers)
- [损失函数（TVPDetectLoss + Prompt 分支）](./losses)
- [训练技巧（分阶段：Text → Visual → Prompt-Free）](./tricks)
- [痛点追踪（RepRTA 稳定性 / CLIP 77 token / SAM 2.1 标注 / AGPL / CVE）](./painpoints)
- [源码走读（head.py + tasks.py + train.py 逐文件精读）](./code-walkthrough)
- [YOLO-World 综述](/derivatives/yoloworld/index) · [YOLO-Master 综述](/derivatives/master/index)
- [YOLOv8 概览](/versions/v8/index) · [YOLO11 概览](/versions/v11/index) · [YOLO26 概览](/versions/v26/index)

## 参考

- Wang A., Liu L., Chen H., Lin Z., Han J., Ding G. *YOLOE: Real-Time Seeing Anything.* arXiv:2503.07465, ICCV 2025, 2025-03-10. https://arxiv.org/abs/2503.07465
- THU-MIG/yoloe. https://github.com/THU-MIG/yoloe @ commit `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`
- Ultralytics. https://github.com/ultralytics/ultralytics · tag `v8.3.99` @ `2fbf7015b9e89294b59e37d78e9b5266d94d7139` · tag `v8.4.0` @ `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`
- DIN, X. et al. *MobileCLIP: Fast, Low-resource and Transferable CLIP Models.* ICCV 2023. https://github.com/apple/ml-mobileclip
- Dinh, L., Sohl-Dickstein, J., Bengio, S. *Density estimation using Real NVP.* ICLR 2017. https://arxiv.org/abs/1605.08803
- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. https://arxiv.org/abs/2103.00020
- Xu, H. et al. *YOLO-World: Open-Vocabulary Real-Time Detection.* ICCV 2025. https://arxiv.org/abs/2401.17270
