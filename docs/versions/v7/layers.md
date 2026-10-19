# YOLOv7 架构与层速查

> 以 `cfg/training/yolov7.yaml`（E-ELAN 骨干 + PAN 头 + IDetect，anchor-based）为例；"使用次数"按该 yaml 逐行统计。

## 1. 模块清单（按 common.py + yolo.py 顺序）

| 模块 | 文件 | 功能 | 使用次数 |
|------|------|------|----------|
| `MP` | <SrcLink file="v7:models/common.py" :line="30" /> | MaxPool(k=2, s=2) 下采样，配双分支 Conv 组成过渡层 | 5（骨干 3 + 头 2） |
| `SP` | <SrcLink file="v7:models/common.py" :line="39" /> | 大核 MaxPool 空间池化（备用件） | 0 |
| `Concat` | <SrcLink file="v7:models/common.py" :line="56" /> | 沿通道维拼接，ELAN / FPN / PAN 的汇合件 | 15（4 路 ×4 + 6 路 ×4 + 二路 ×5 + 三路 ×2） |
| `Conv` | <SrcLink file="v7:models/common.py" :line="99" /> | Conv + BN + SiLU 基础单元 | 79（骨干 41 + 头 38） |
| `SPP` | <SrcLink file="v7:models/common.py" :line="195" /> | 串行大核池化金字塔（v4 式） | 0 |
| `Bottleneck` | <SrcLink file="v7:models/common.py" :line="209" /> | 1×1 + 3×3 残差瓶颈（Darknet 式） | 0（yolov7.yaml 不用） |
| `SPPCSPC` | <SrcLink file="v7:models/common.py" :line="262" /> | CSP 版 SPP：5/9/13 三池化 + 旁路 concat | 1（# 51） |
| `BottleneckCSPA/B/C` | <SrcLink file="v7:models/common.py" :line="307" /> | 三种 CSP 瓶颈堆叠（L307/L323/L340） | 0（yolov7.yaml 不用） |
| `ImplicitA` | <SrcLink file="v7:models/common.py" :line="433" /> | 加性隐式知识 `implicit + x`（均值 0 起步） | 每个头内置 |
| `ImplicitM` | <SrcLink file="v7:models/common.py" :line="446" /> | 乘性隐式知识 `implicit * x`（均值 1 起步） | 每个头内置 |
| `RepConv` | <SrcLink file="v7:models/common.py" :line="463" /> | 三分支重参数化卷积（3×3 + 1×1 + 条件 identity） | 3（# 102-104） |
| `RepBottleneck` | <SrcLink file="v7:models/common.py" :line="646" /> | cv2 换成 RepConv 的 Bottleneck | 0（yolov7.yaml 不用） |
| `Focus` | <SrcLink file="v7:models/common.py" :line="796" /> | 空间信息折叠进通道（v5 stem） | 0 |
| `SPPF` | <SrcLink file="v7:models/common.py" :line="808" /> | 串行小池化等效大核（v5 头部入口） | 0 |
| `Detect` | <SrcLink file="v7:models/yolo.py" :line="23" /> | 原版 v3 式 anchor 检测头 | 0 |
| `IDetect` | <SrcLink file="v7:models/yolo.py" :line="97" /> | 带 ImplicitA/M 的默认检测头 | 1（# 105） |
| `IAuxDetect` | <SrcLink file="v7:models/yolo.py" :line="311" /> | lead + aux 双头（train_aux.py 用） | 0（yolov7.yaml 不用） |
| `Model` | <SrcLink file="v7:models/yolo.py" :line="508" /> | 总装类：yaml → 网络 + stride 校准 | 1 |
| `parse_model` | <SrcLink file="v7:models/yolo.py" :line="736" /> | yaml 解析器：缩放 + 逐层实例化 | 1 |

## 2. yolov7.yaml 逐块速查

### 2.1 全局参数与 anchors

| 配置项 | 值 | 说明 | 位置 |
|--------|----|------|------|
| `nc` | 80 | 类别数 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="2" /> |
| `depth_multiple` | 1.0 | 深度缩放，基准即全深 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="3" /> |
| `width_multiple` | 1.0 | 宽度缩放，基准即全宽 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="4" /> |
| anchors P3/8 | `[12,16, 19,36, 40,28]` | 80×80 层 3 组先验框 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="8" /> |
| anchors P4/16 | `[36,75, 76,55, 72,146]` | 40×40 层 3 组先验框 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="9" /> |
| anchors P5/32 | `[142,110, 192,243, 459,401]` | 20×20 层 3 组先验框 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="10" /> |

**3 组 anchors × 3 个尺度 = anchor-based**：推理输出每格 3 个预测（对照 v6 anchor-free 每格 1 个）。

### 2.2 backbone 逐块（# 0 – # 50）

| 层索引 | 块 | 结构要点 | 位置 |
|--------|----|----------|------|
| # 0 | stem | Conv 32, 3×3 s1 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="15" /> |
| # 1-2 | P1/2 | Conv 64 s2 + Conv 64 s1 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="17" :end="18" /> |
| # 3-11 | P2/4 ELAN | Conv 128 s2 → 两支 1×1 降维 → 4×3×3 堆叠 → 4 路 Concat `[[-1,-3,-5,-6],1,Concat]` → 1×1 收拢 256 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="20" :end="28" /> |
| # 12-16 | MP 过渡 | MP + 双路 1×1 + 3×3 s2 + 2 路 Concat，出 128 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="30" :end="34" /> |
| # 17-24 | P3/8 ELAN | 同 P2 骨架（128 分支 → 512） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="35" :end="42" /> |
| # 25-29 | MP 过渡 | 同上，出 256 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="44" :end="48" /> |
| # 30-37 | P4/16 ELAN | 同骨架（256 分支 → 1024） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="49" :end="56" /> |
| # 38-42 | MP 过渡 | 同上，出 512 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="58" :end="62" /> |
| # 43-50 | P5/32 ELAN | 同骨架（256 分支 → 1024） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="63" :end="70" /> |

骨干规律：每个 ELAN 块 = 1×1 入口两支 → 4 个 3×3 级联 → **4 路 Concat `[[-1,-3,-5,-6],1,Concat]`**（取后 4 个不同深度的特征）→ 1×1 收拢；每级之间插一个 MP 过渡层下采样（Concat 在 <SrcLink file="v7:cfg/training/yolov7.yaml" :line="27" /> 等 4 处）。

### 2.3 head 逐块（# 51 – # 105）

| 层索引 | 块 | 结构要点 | 位置 |
|--------|----|----------|------|
| # 51 | SPPCSPC | 5/9/13 池化金字塔 + CSP 旁路，出 512 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="75" /> |
| # 52-55 | FPN1 | 1×1 降维 + Upsample + route #37（骨干 P4）+ 2 路 Concat | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="77" :end="80" /> |
| # 56-63 | P4 ELAN | **6 路 Concat `[[-1,-2,-3,-4,-5,-6],1,Concat]`** 块，出 256 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="82" :end="89" /> |
| # 64-67 | FPN2 | 1×1 + Upsample + route #24（骨干 P3）+ 2 路 Concat | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="91" :end="94" /> |
| # 68-75 | P3 ELAN | 6 路 Concat ELAN，出 128（P3 检测输入 #75） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="96" :end="103" /> |
| # 76-80 | PAN 下行 1 | MP + 双 1×1 + 3×3 s2 + `Concat[-1,-3,63]`（接回 #63） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="105" :end="109" /> |
| # 81-88 | P4 ELAN | 6 路 Concat ELAN，出 256（P4 检测输入 #88） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="111" :end="118" /> |
| # 89-93 | PAN 下行 2 | MP + 双 1×1 + 3×3 s2 + `Concat[-1,-3,51]`（接回 #51） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="120" :end="124" /> |
| # 94-101 | P5 ELAN | 6 路 Concat ELAN，出 512（#101） | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="126" :end="133" /> |
| # 102-104 | RepConv ×3 | `[75→256]` `[88→512]` `[101→1024]`，3×3 s1 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="135" :end="137" /> |
| # 105 | IDetect | `[[102,103,104], 1, IDetect, [nc, anchors]]`，P3/P4/P5 三输出 | <SrcLink file="v7:cfg/training/yolov7.yaml" :line="139" /> |

头部是标准 PAN：SPPCSPC 入口 → FPN 自上而下（Upsample + Concat）→ 自下而上（MP + Concat 接回骨干或 SPPCSPC 输出）；与骨干 ELAN 的 4 路 Concat 不同，头部 ELAN 用 **6 路 Concat** `[[-1,-2,-3,-4,-5,-6],1,Concat]`（L88/L102/L117/L132 共 4 处）。三个检测输入 #75/#88/#101 先各过一个 RepConv（# 102-104）再进 IDetect。

## 3. Conv 与 MP — 基础单元与过渡层

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))        # conv-BN 融合后的部署路径
```

```python
class MP(nn.Module):
    def __init__(self, k=2):
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)                     # 裸 MaxPool(k=2, s=2)
```

- **Conv** = Conv + BN + SiLU，`autopad` 保证 same padding，`bias=False` 由 BN 吸收；`fuseforward` 对应论文 §4.3 的 conv-BN 融合 freebie
- **MP 类本体只是 MaxPool**；yaml 里的"MP 块"是组合：`[-1,1,MP,[]]` 后接两支 1×1（分别作用于 `-1` 与 `-3`）+ 3×3 s2 + 2 路 Concat——下采样与通道过渡一次完成
- `SP`（L39-45）是 k×k 大池化备用件，yolov7.yaml 不用

<SrcLink file="v7:models/common.py" :line="99" :end="111" />
<SrcLink file="v7:models/common.py" :line="30" :end="45" />

## 4. Bottleneck 与 RepBottleneck — 瓶颈双胞胎

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        c_ = int(c2 * e)                     # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

```python
class RepBottleneck(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)   # 硬编码，忽略传参
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)                 # 3×3 支换成 RepConv
```

- Bottleneck：1×1 降维 + 3×3 处理 +（`c1 == c2` 时）残差相加
- **RepBottleneck 有个 quirk**：`super().__init__` 把 `shortcut=True, g=1, e=0.5` 写死，签名里的 shortcut/g/e 实际被忽略；随后仅把 cv2 换成 RepConv——配合 RepBottleneckCSP 系列（<SrcLink file="v7:models/common.py" :line="654" /> 起）给其他配置用
- 两者都不在 yolov7.yaml 里：E-ELAN 骨干直接堆 Conv，不用瓶颈

<SrcLink file="v7:models/common.py" :line="209" :end="219" />
<SrcLink file="v7:models/common.py" :line="646" :end="651" />

## 5. SPPCSPC — 头部入口感受野扩展

```python
class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        c_ = int(2 * c2 * e)                 # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)        # CSP 旁路
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)    # 收 4 路池化
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)    # 汇合主路 + 旁路

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
```

- 双路 CSP：主路 cv1 → cv3(3×3) → cv4 之后并联 5/9/13 三个 MaxPool(s=1) + 原特征共 4 路 concat → cv5/cv6；旁路 cv2 直连；cv7 汇合
- 隐藏通道 `c_ = 2·c2·e`（比 SPPF 的 `c1 // 2` 更宽）
- yolov7.yaml 只用一次（# 51，输入 #50 的 1024 → 512），位于头部入口
- 与 v5 SPPF 的差异：SPPF 串行 3 次 k=5 池化等效 5/9/13；SPPCSPC 真开三条池化支路 + CSP 旁路

<SrcLink file="v7:models/common.py" :line="262" :end="280" />

## 6. RepConv — 重参数化卷积

```python
class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        assert k == 3
        assert autopad(k, p) == 1
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                                           nn.BatchNorm2d(num_features=c2))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                                         nn.BatchNorm2d(num_features=c2))

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))      # 部署态
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
```

- 训练态三分支：3×3+BN、1×1+BN、identity(BN)；`assert k == 3`（重参数化数学只对 3×3/1×1 成立）
- **条件 identity 就是论文 RepConvN 规则的落地**：`c2 == c1 and s == 1` 才建 identity 分支——否则为 None，天然满足"带 residual/concat 的卷积不带 identity 分支"。yolov7.yaml 三处 RepConv（# 102-104）都是升维（128→256 / 256→512 / 512→1024），自动无 identity（快照里没有名为 RepConvN 的类）
- 折叠：`get_equivalent_kernel_bias` → `_fuse_bn_tensor`（L524-552）；identity 分支用 **id_tensor 技巧**（单位阵塞进 3×3 中心，L536-544）等价成 1×1 conv
- `repvgg_convert` 导出 kernel/bias；`fuse_conv_bn`（L561-582）处理普通 conv-BN；`RepConv_OREPA`（<SrcLink file="v7:models/common.py" :line="1224" />）是 OREPA 训练加速变体，yaml 不用

<SrcLink file="v7:models/common.py" :line="463" :end="582" />
<SrcLink file="v7:models/common.py" :line="524" :end="552" />

## 7. ImplicitA 与 ImplicitM — YOLOR 隐式知识

```python
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x             # 加性，均值 0 起步

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x             # 乘性，均值 1 起步
```

- YOLOR 隐式知识：可学习的 (1, C, 1, 1) 张量广播加/乘到特征上，起点分别近似恒等（加 0 / 乘 1）
- 训练期是普通参数；推理期被 `IDetect.fuse()` / `IAuxDetect.fuse()` 折进输出 conv 的 bias/weight（加性走 matmul、乘性逐元素乘）——论文 §4.3 "推理免费"的代码落点
- 在检测头里 `ia` 接特征侧（通道 = 输入通道数），`im` 接输出侧（`no*na` 通道）

<SrcLink file="v7:models/common.py" :line="433" :end="443" />
<SrcLink file="v7:models/common.py" :line="446" :end="456" />

## 8. IDetect — 默认检测头

```python
class IDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        self.no = nc + 5                                    # 每 anchor 输出数
        self.nl = len(anchors)                              # 检测层数
        self.na = len(anchors[0]) // 2                      # 每层 anchor 数
        self.m  = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)   # 特征侧隐式加
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)  # 输出侧隐式乘

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))              # 隐式加 → 1×1 输出卷积
            x[i] = self.im[i](x[i])                         # 隐式乘
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:                           # 推理解码
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]              # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)
```

- 每层 1×1 输出卷积出 `na×(nc+5) = 255` 通道，view+permute 成 (bs, 3, ny, nx, 85)
- **解码是标准 YOLOv3 式 anchor 解码**：xy = (2σ−0.5+grid)×stride、wh = (2σ)²×anchor_grid——对照 v6 anchor-free 的 ltrb/DFL 路线
- 训练返回 3 路特征图列表；推理返回 `(torch.cat(z, 1), x)` 元组
- `fuse()`（L178-190）：ImplicitA 折进 conv bias（`bias += W·ia`，matmul），ImplicitM 折进 bias/weight（乘）；`fuseforward`（L140-176）是融合后的推理路径

<SrcLink file="v7:models/yolo.py" :line="97" :end="208" />
<SrcLink file="v7:models/yolo.py" :line="178" :end="190" />

## 9. IAuxDetect — 辅助双头

```python
class IAuxDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        self.m  = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])   # lead 头
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])   # aux 头
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])       # 只配 lead
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))        # lead：隐式知识 + 输出卷积
            x[i] = self.im[i](x[i])
            x[i + self.nl] = self.m2[i](x[i + self.nl])   # aux：纯 1×1，无隐式知识
            ...
        return x if self.training else (torch.cat(z, 1), x[:self.nl])
```

- 输入 6 路特征（3 lead + 3 aux，由 train_aux.py 的网络配置提供）；**aux 头 `m2` 是纯 1×1 卷积、没有 ia/im**——partial auxiliary head（辅助分支不含隐式知识）的代码体现
- 推理只输出 lead（`z` 只收集 lead 路，返回 `x[:self.nl]`）；aux 只参与训练
- 配 `ComputeLossAuxOTA`：lead 用 `find_3_positive`（g=0.5，fine）、aux 用 `find_5_positive`（g=1.0，coarse）、aux 损失统一 ×0.25（详见[损失函数详解](/versions/v7/losses)）
- `fuse()`（L401-413）只折 lead 的隐式知识——aux 没得折

<SrcLink file="v7:models/yolo.py" :line="311" :end="430" />
<SrcLink file="v7:models/yolo.py" :line="401" :end="413" />

## 10. Focus 与 SPPF — v5 遗产

```python
class Focus(nn.Module):
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        c_ = c1 // 2
        ...
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
```

- Focus：隔行采样把空间折叠进通道（v5 的 P1 stem）；SPPF：串行 3 次 k=5 池化等效 5/9/13 大核（v5 的 P5 头部入口）
- 二者都是 v5 遗产，yolov7.yaml 均不用（stem 用 Conv s2、头部入口用 SPPCSPC），保留在 common.py 服务 baseline 配置

<SrcLink file="v7:models/common.py" :line="796" :end="821" />

## 11. parse_model — yaml 装配器

```python
def parse_model(d, ch):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)                                   # 输出通道 = anchors × (类 + 5)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        n = max(round(n * gd), 1) if n > 1 else n         # depth gain
        if m in [Conv, RepConv, SPPCSPC, Bottleneck, ...]:
            c1, c2 = ch[f], args[0]
            if c2 != no:                                 # 检测头输出通道不缩
                c2 = make_divisible(c2 * gw, 8)          # width gain
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([ch[x] for x in f])              # 注入各输入层通道表
    return nn.Sequential(*layers), sorted(save)
```

- 每层四元组 `[from, number, module, args]`；`from` 支持负索引与显式层号——yaml 里 route #37/#24/#63/#51 就是显式层号
- depth gain 只对 `n > 1` 生效（yolov7.yaml 多数 n=1，不缩）；width gain 走 `make_divisible(×, 8)`，且 `c2 != no` 才缩
- Detect 家族把各输入层通道表 append 进 args；`save` 记录被跳连引用的层号；`eval` 机制让 `nc`/`anchors` 直接进 args

<SrcLink file="v7:models/yolo.py" :line="736" :end="813" />

## 12. 输出张量布局

**训练**（IDetect，yolov7.yaml 默认）：返回 3 张量的列表

| 张量 | 形状 | 说明 |
|------|------|------|
| x[0] | (bs, 3, 80, 80, 85) | P3/8，每格 3 anchors × 85 |
| x[1] | (bs, 3, 40, 40, 85) | P4/16 |
| x[2] | (bs, 3, 20, 20, 85) | P5/32 |

IAuxDetect（train_aux.py）返回 6 张量：前 3 个 lead 同上，后 3 个 aux 同形状（纯 1×1 输出）。

**推理**：`(torch.cat(z, 1), x)` 的第一分量是 **(bs, 25200, 85)**

- `25200 = 3 anchors × (80² + 40² + 20²)`——anchor-based，每格 3 个预测
- 85 = 4 box（xywh，原图坐标）+ 1 objectness + 80 类
- **对照 v6**：anchor-free 单头输出 (bs, 8400, 85)（8400 = 80²+40²+20²）；v7 的 25200 = 3 × 8400，多出的 3 倍正是每格 3 组先验框

## 13. 模型缩放参数

| 参数 | 配置字段 | yolov7 值 | 作用 |
|------|-----------|-----------|------|
| `nc` | nc | 80 | 类别数；IDetect 的 `no = nc + 5 = 85`（每 anchor） |
| `depth_multiple` | depth_multiple | 1.0 | 重复数 `max(round(n×gd), 1)`（n=1 不缩） |
| `width_multiple` | width_multiple | 1.0 | 通道 `make_divisible(c×gw, 8)` |
| `anchors` | anchors | 3 层 × 3 组 | `na = 3` → 推理 25200 = 3×8400 |
| `no`（parse_model） | 派生 | 255 | `no = na×(nc+5)`，1×1 输出卷积通道数 |

- 缩放公式均在 `parse_model`：depth <SrcLink file="v7:models/yolo.py" :line="751" />、width <SrcLink file="v7:models/yolo.py" :line="765" />、no <SrcLink file="v7:models/yolo.py" :line="740" />
- v7 论文的 concat 型 compound scaling：computational block 只缩 depth、transition 层按同样变化量缩 width——基准配置（本表）两个 multiple 都是 1.0，缩放留给 yolov7-x 等变体配置

<SrcLink file="v7:cfg/training/yolov7.yaml" :line="2" :end="10" />

## 延伸阅读

- [Architecture 逐层图解](/versions/v7/architecture)
- [理论 Concept](/versions/v7/concepts)
- [损失函数详解](/versions/v7/losses)
- [v6 架构与层速查（对比）](/versions/v6/layers)
