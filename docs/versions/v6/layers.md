# YOLOv6 架构与层速查

> 以 yolov6s（`EfficientRep + RepBiFPANNeck + EffiDeHead`）为例；"使用次数"按 `configs/yolov6s.py` 缩放后（`training_mode='repvgg'`）统计。

## 1. 模块清单（按 common.py + models 顺序）

| 模块 | 文件 | 功能 | 使用次数 |
|------|------|------|----------|
| `ConvModule` | common.py L24 | Conv + BN + 可选激活（基础单元） | 被各封装复用 |
| `ConvBNReLU` | common.py L55 | Conv + BN + ReLU | 颈部 ~14 |
| `ConvBNSiLU` | common.py L65 | Conv + BN + SiLU | 头部 9（3 层 × 3） |
| `SimSPPF` / `SimCSPSPPF` | common.py L113 / L159 | P5 感受野扩展（cspsppf=True 用后者） | 1 |
| `Transpose` | common.py L181 | 转置卷积上采样 | 2（BiFusion 内） |
| `RepVGGBlock` | common.py L197 | 三分支重参数化块 | ~35（骨干 19 + 颈 16） |
| `QARepVGGBlock` | common.py L322 | 量化友好 rep 块（qarepvgg 模式） | 0（默认模式不用） |
| `RepBlock` | common.py L569 | n 个 rep 块的阶段容器 | 8（骨干 4 + 颈 4） |
| `BottleRep` | common.py L590 | α 加权残差双 rep 块 | BepC3 内部 |
| `BepC3` | common.py L634 | CSP 结构（CSP 骨干/颈用） | 0（yolov6s 不用） |
| `MBLABlock` | common.py L653 | 多分支层聚合（大模型变体） | 0（yolov6s 不用） |
| `BiFusion` | common.py L695 | 三路特征融合 | 2 |
| `get_block` | common.py L721 | 按 training_mode 选基本块 | 1（build_network 调用） |
| `EfficientRep` | efficientrep.py L7 | 骨干 | 1 |
| `RepBiFPANNeck` | reppan.py L132 | 双向 BiFusion 颈 | 1 |
| `Detect`（EffiDeHead） | effidehead.py L10 | 解耦检测头 | 1（内部 3 层 × 5 子层） |
| `build_effidehead_layer` | effidehead.py L142 | 构造头的 15 层子模块 | 1 |
| `build_network` | yolo.py L55 | 总装：缩放 + 实例化三部件 | 1 |

## 2. ConvModule — 基础单元

```python
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation_type, padding=None, groups=1, bias=False):
        self.conv = nn.Conv2d(..., padding=kernel_size // 2, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation_table.get(activation_type)  # None 则不激活
```

- **padding = k // 2**（same padding），默认 `bias=False`（BN 吸收）
- `activation_type=None` 时输出 `bn(conv(x))`——RepVGGBlock 分支内部正是这种"无激活 conv"
- `ConvBNReLU` / `ConvBNSiLU` / `ConvBN` / `ConvBNHS` 都是它的薄封装（'relu' / 'silu' / None / 'hardswish'）

<SrcLink file="v6:yolov6/layers/common.py" :line="24" :end="92" />

## 3. RepVGGBlock — 核心重参数化块

```python
class RepVGGBlock(nn.Module):
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):        # 部署态
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.nonlinearity(self.se(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))  # 训练态
```

- 训练态三分支：`rbr_dense`（3×3+BN）、`rbr_1x1`（1×1+BN）、`rbr_identity`（仅 `in==out and stride==1` 时存在）
- **assert k==3, p==1**：重参数化数学只对 3×3/1×1 成立
- `switch_to_deploy()`：`get_equivalent_kernel_bias()` 折算三分支 → 建 `rbr_reparam`（带 bias 的 3×3）→ 删除旧分支
- 激活是 **ReLU**（`nonlinearity = nn.ReLU()`），与 v5 全网 SiLU 不同

<SrcLink file="v6:yolov6/layers/common.py" :line="197" :end="319" />

## 4. RepBlock 与 BottleRep

```python
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1)))
        if block == BottleRep:
            self.conv1 = BottleRep(..., weight=True)
            n = n // 2                      # 每个 BottleRep 含 2 个 rep 块
            self.block = nn.Sequential(*(BottleRep(...) for _ in range(n - 1)))
```

```python
class BottleRep(nn.Module):
    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs
```

- `alpha` 是可学习标量（`weight=True`），残差加权 $outputs + \alpha x$；`in != out` 时无 shortcut
- yolov6s 的骨干/颈 RepBlock 都是 `block=RepVGGBlock`（不减半）；BepC3 内部才用 BottleRep

<SrcLink file="v6:yolov6/layers/common.py" :line="569" :end="608" />

## 5. BepC3 — CSPStackRep（CSP 变体）

```python
class BepC3(nn.Module):  # CSPStackRep Block
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        c_ = int(out_channels * e)          # hidden = 一半
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        self.m = RepBlock(c_, c_, n=n, block=BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```

- 结构与 v5 C3 同构（1×1 → 瓶颈栈 → 1×1，旁路 concat），瓶颈换成 BottleRep
- `MBLABlock`（多分支层聚合）是更深的 CSP 变体：分支数按 `n_list` 动态生成，内含 BottleRep3
- 二者用于 `CSPBepBackbone` / 大模型（M/L）变体；yolov6s 的 EfficientRep 路线不用

<SrcLink file="v6:yolov6/layers/common.py" :line="634" :end="692" />

## 6. SimSPPF / CSPSPPF — P5 感受野扩展

```python
class SPPFModule(nn.Module):
    def forward(self, x):
        x = self.cv1(x)                     # 1×1 降维到 c_ = in/2
        y1 = self.m(x)                      # MaxPool k=5, s=1, same
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))   # 4 路 concat
```

- 串行 3 次 MaxPool(k=5) 等效大核（5/9/13）金字塔，但只算 3 次小池化
- `SimSPPF` = ReLU 版（配 RepVGGBlock）；`SPPF` = SiLU 版（配 ConvBNSiLU）
- `CSPSPPFModule` 在此之上加 CSP 旁路（`cv1~cv7`，旁路 `cv2` 直连 + `cv7` 汇合），`cspsppf=True` 时使用（yolov6s 默认）

<SrcLink file="v6:yolov6/layers/common.py" :line="95" :end="178" />

## 7. Transpose 与 BiFusion — 颈部融合件

```python
class BiFusion(nn.Module):
    def forward(self, x):
        x0 = self.upsample(x[0])                        # Transpose（ConvTranspose2d）
        x1 = self.cv1(x[1])                             # 1×1 对齐同层
        x2 = self.downsample(self.cv2(x[2]))            # 1×1 + 3×3 s2 压下层
        return self.cv3(torch.cat((x0, x1, x2), dim=1)) # 1×1 收拢 3·out → out
```

- `Transpose` 是 `nn.ConvTranspose2d`（kernel 2, stride 2），比 nearest upsample 更硬件友好
- `cv1/cv2/cv3` 均为 ConvBNReLU；`in_channels=[c0, c1]` 描述前两路输入，第三路固定为 `out_channels`（下采样前后同宽）

<SrcLink file="v6:yolov6/layers/common.py" :line="181" :end="194" />
<SrcLink file="v6:yolov6/layers/common.py" :line="695" :end="718" />

## 8. EfficientRep — 骨干

```python
self.stem = block(3, channels_list[0], k=3, s=2)                 # 3→32 @320
self.ERBlock_2 = nn.Sequential(block(32, 64, s=2), RepBlock(64, 64, n=2))
self.ERBlock_3 = nn.Sequential(block(64, 128, s=2), RepBlock(128, 128, n=4))
self.ERBlock_4 = nn.Sequential(block(128, 256, s=2), RepBlock(256, 256, n=6))
self.ERBlock_5 = nn.Sequential(block(256, 512, s=2), RepBlock(512, 512, n=2),
                               SimCSPSPPF(512, 512, k=5))
```

- `forward` 在 `fuse_P2=True` 时返回 **4 路** `(x3, x2, x1, x0)`（P2~P5），否则 3 路
- `channel_merge_layer` 选择逻辑：`block==ConvBNSiLU` 时用 SiLU 版 SPPF/CSPSPPF，否则 ReLU 版
- `EfficientRep6`（P6 五输出）/ `CSPBepBackbone`（BepC3 版）是同文件的兄弟实现

<SrcLink file="v6:yolov6/models/efficientrep.py" :line="28" :end="118" />

## 9. RepBiFPANNeck — 颈部

```python
def forward(self, input):
    (x3, x2, x1, x0) = input                      # fuse_P2 四输入
    fpn_out0 = self.reduce_layer0(x0)             # 512→128 @20
    f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
    f_out0 = self.Rep_p4(f_concat_layer0)         # 128 @40
    fpn_out1 = self.reduce_layer1(f_out0)         # 128→64 @40
    f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
    pan_out2 = self.Rep_p3(f_concat_layer1)       # 64 @80 → 头 P3
    down_feat1 = self.downsample2(pan_out2)
    pan_out1 = self.Rep_n3(torch.cat([down_feat1, fpn_out1], 1))   # 128 @40 → 头 P4
    down_feat0 = self.downsample1(pan_out1)
    pan_out0 = self.Rep_n4(torch.cat([down_feat0, fpn_out0], 1))   # 256 @20 → 头 P5
    return [pan_out2, pan_out1, pan_out0]
```

- FPN 下行（top-down）用 BiFusion，PAN 上行（bottom-up）用 plain 3×3 s2 + concat
- 通道取自合并通道表（yolov6s 缩放后）`[32,64,128,256,512, 128,64,64,128,128,256]`，索引 5~10 属于 neck
- `RepPANNeck`（concat 双路版）、`RepBiFPANNeck6`、`CSPRepPANNeck`、`Lite_EffiNeck` 为同文件兄弟实现

<SrcLink file="v6:yolov6/models/reppan.py" :line="215" :end="237" />

## 10. Detect（EffiDeHead）— 检测头

```python
class Detect(nn.Module):
    def __init__(self, num_classes=80, num_layers=3, head_layers=None, use_dfl=True, reg_max=16):
        self.no = num_classes + 5
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)  # DFL 期望
        # 每层 5 子层：stems / cls_convs / reg_convs / cls_preds / reg_preds
```

每检测层结构（`build_effidehead_layer`，`chx=[6,8,10]` 取通道）：

| 子层 | 类型 | 通道（yolov6s） | 输出 |
|------|------|------------------|------|
| stem | ConvBNSiLU 1×1 | c → c | c×H×W |
| cls_conv | ConvBNSiLU 3×3 | c → c | c×H×W |
| reg_conv | ConvBNSiLU 3×3 | c → c | c×H×W |
| cls_pred | nn.Conv2d 1×1 | c → 80 | 80×H×W |
| reg_pred | nn.Conv2d 1×1 | c → 4（DFL 时 68） | 4×H×W |

- `initialize_biases`：cls 先验 `prior_prob=1e-2`，reg bias=1；`proj_conv` 权重固定为 `linspace(0, reg_max, reg_max+1)`，不参与训练
- 训练 `forward` 返回 `(x, cls_score_list, reg_distri_list)`；推理 `forward` 内做 DFL 期望、`generate_anchors(mode='af')`、`dist2bbox`，最后 `cat([box, ones, cls])` 输出 `(bs, 8400, 85)`
- yolov6s 默认 `use_dfl=False, reg_max=0`：`reg_pred` 只出 4 维；蒸馏续训才开 `reg_max=16`

<SrcLink file="v6:yolov6/models/effidehead.py" :line="10" :end="47" />
<SrcLink file="v6:yolov6/models/effidehead.py" :line="142" :end="293" />

## 11. get_block 与 training_mode

```python
def get_block(mode):
    if mode == 'repvgg':      return RepVGGBlock
    elif mode == 'qarepvgg':  return QARepVGGBlock
    elif mode == 'qarepvggv2':return QARepVGGBlockV2
    elif mode == 'hyper_search': return LinearAddBlock
    elif mode == 'repopt':    return RealVGGBlock
    elif mode == 'conv_relu': return ConvBNReLU
    elif mode == 'conv_silu': return ConvBNSiLU
```

- `build_network` 里 `block = get_block(config.training_mode)`，之后骨干/颈所有 rep 块都用它实例化
- `training_mode` 不在 `configs/yolov6s.py` 里，`tools/train.py` 缺省补 `'repvgg'`
- `build_network` 还按 `fuse_ab` / `distill_ns` 选择不同的检测头实现（effidehead_fuseab / effidehead_distill_ns），默认走 `effidehead.Detect`

<SrcLink file="v6:yolov6/layers/common.py" :line="721" :end="737" />
<SrcLink file="v6:yolov6/models/yolo.py" :line="55" :end="133" />

## 12. 输出张量布局

**训练**：`forward` 返回三元组

| 张量 | 形状 | 说明 |
|------|------|------|
| feats | 3 个：(bs,64,80,80) (bs,128,40,40) (bs,256,20,20) | 头部输入 |
| cls_score_list | (bs, 8400, 80) | sigmoid 后的分类分数 |
| reg_distri_list | (bs, 8400, 4) | ltrb 距离（DFL 时 68 = 4×17 分布） |

**推理**：单张量 `(bs, 8400, 85)`

- `8400 = 80² + 40² + 20²`（strides 8/16/32）
- 85 = 4 box（xywh，原图坐标）+ 1 占位（恒 1）+ 80 类 sigmoid 分数

## 13. 模型缩放参数

| 参数 | 配置字段 | yolov6s 值 | 作用 |
|------|-----------|-----------|------|
| `depth_multiple` | model | 0.33 | 重复次数：`max(round(n×0.33),1)`（n=1 不缩放） |
| `width_multiple` | model | 0.50 | 通道：`make_divisible(c×0.5, 8)` |
| `num_repeats` | backbone / neck | [1,6,12,18,6] / [12,12,12,12] | 缩放后 [1,2,4,6,2] / [4,4,4,4] |
| `out_channels` | backbone / neck | [64,128,256,512,1024] / [256,128,128,256,256,512] | 缩放后 [32,64,128,256,512] / [128,64,64,128,128,256] |
| `fuse_P2` | backbone | True | 骨干四输出 |
| `cspsppf` | backbone | True | ERBlock_5 用 SimCSPSPPF |
| `use_dfl` / `reg_max` | head | False / 0 | 蒸馏续训设 True / 16 |
| `strides` | head | [8, 16, 32] | 检测层步长 |
| `atss_warmup_epoch` | head | 0 | >0 时先 ATSS 后 TAL |

<SrcLink file="v6:configs/yolov6s.py" :line="2" :end="39" />

## 延伸阅读

- [Architecture 逐层图解](/versions/v6/architecture)
- [理论 Concept](/versions/v6/concepts)
- [损失函数详解](/versions/v6/losses)
- [v5 架构与层速查（对比）](/versions/v5/layers)
