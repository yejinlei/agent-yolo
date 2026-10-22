# ultralytics/nn/modules/conv.py

YOLOv13 的卷积模块仓库：**369 行文件里，v13 的净增量只有一个新类——`DSConv`（L84-101）**，同时把 `"DSConv"` 加进 `__all__`（L25）。DSConv 是 v13 骨干「轻量化改造」的基础积木：DSBottleneck / DSC3k / DSC3k2（block.py L1375/L1414/L1472）与 HyperACE 里的低阶分支（L1842）全都建立在它上面。其它 v8-v12 保留的 `Conv / Conv2 / DWConv / DWConvTranspose2d / ConvTranspose / Focus / GhostConv / RepConv / ChannelAttention / SpatialAttention / CBAM / Concat / Index` 一字未动。

> 快照路径：`source/yolov13/snapshot/ultralytics/nn/modules/conv.py`（369 行，commit `73289949`，2025-11-18）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：**v13 论文（arXiv 2506.17733）里「real-time」两个字的直接落点就在 DSConv 上**——用 depthwise + pointwise 两步卷积替代普通卷积，参数量与计算量都大幅下降（在 `groups=c_in` 下，DWConv 部分只有 `k²·c_in` 参数、1×1 部分 `c_in·c_out` 参数，总和远小于 `k²·c_in·c_out`）。HyperACE 的高阶分支用 AdaHGConv，低阶分支用 DSConv 拼装，两条通路各取所需。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="1" :end="26" /> | 文件头 + 导入 + `__all__` | **v13 新增 `"DSConv"` 到 `__all__`（L25）** |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="29" :end="35" /> | `autopad` | 与 v8-v12 一字不差 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="38" :end="56" /> | **Conv** | v8 基类：Conv2d + BN + SiLU；v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="59" :end="82" /> | **Conv2** | 简化 RepConv；v13 保留但 yaml 不引用 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="84" :end="101" /> | **DSConv** | **v13 新增**：DWConv(k) + PWConv(1×1) + BN + SiLU |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="103" :end="118" /> | LightConv | HGNet 引入，v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="121" :end="126" /> | DWConv | `g=math.gcd(c1, c2)` 深度可分离；v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="129" :end="134" /> | DWConvTranspose2d | v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="137" :end="155" /> | ConvTranspose | 上采样，v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="158" :end="174" /> | Focus | v5 时代遗留，v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="177" :end="190" /> | GhostConv | GhostNet，v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="193" :end="294" /> | RepConv | RT-DETR 用，v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="297" :end="339" /> | ChannelAttention / SpatialAttention / CBAM | v13 保留 |
| <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="342" :end="369" /> | Concat / Index | 基础算子，v13 保留 |

<SourceViewer file="v13:ultralytics/nn/modules/conv.py" />

## DSConv（L84-101）：v13 唯一的新增

```python
class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))
```

**结构四步**：

1. **`dw`**：`nn.Conv2d(c_in, c_in, k, s, p, dilation=d, groups=c_in)`——**groups=c_in** 意味着每通道一个独立的 `k×k` 卷积核（真正的 depthwise）；
2. **`pw`**：`nn.Conv2d(c_in, c_out, 1, 1)`——**pointwise 1×1**，负责通道混合；
3. **`bn`**：`nn.BatchNorm2d(c_out)`——BN 在 pw 之后（与 v12 Detect 里 `DWConv(x, x, 3) + Conv(x, c3, 1)` 的顺序一致，但整个 1×1 里也包含 BN）；
4. **`act = SiLU()`**：与整个 YOLO 家族的默认激活保持一致。

**与 `Conv` / `DWConv` 的关系**：

- 传统 `Conv`（L38-56）：`Conv2d + BN + SiLU`，一个卷积做「空间 + 通道」双维融合——参数 `k²·c_in·c_out`；
- `DWConv`（L121-126）：`g=math.gcd(c1, c2)` 的深度卷积——当 `c1 == c2` 时等价于 `groups=c1`，是 DSConv 的 dw 部分；但 DWConv 本身没有 pw 部分；
- **DSConv = DWConv(c_in, c_in, k) + Conv(c_in, c_out, 1) + BN + SiLU**——把两步组合成一个可复用模块。

**DSConv 参数量**：`k²·c_in + c_in·c_out + 2·c_out`（BN）；对比传统 `Conv(c_in, c_out, k)`：`k²·c_in·c_out + 2·c_out`。**当 c_in ≈ c_out 且 k=3 时，DSConv 参数约为 Conv 的 1/c_out**——这就是「轻量化」的量化含义。

**关键设计点**：

- **`bias=False` 默认**：与整个 YOLO 家族的无偏置风格一致（BN 之后不需要 bias）；
- **`p=None` 时自动 `p = (d * (k - 1)) // 2`**：标准的 same-padding，支持 dilation；
- **`s` 支持 >1**：DSConv 可以直接做下采样（v13 yaml L16/L18 的 `DSConv [512/1024, 3, 2]` 就是 s=2）；
- **`forward` 顺序：dw → pw → bn → act**——注意 BN 在 1×1 之后，不是分别放在 dw 和 pw 后各一份（v13 选择「合并 BN」的更紧凑写法）。

## DSConv 在 v13 里的引用点

DSConv 在 v13 里有两类消费者：

1. **yaml 直接引用**（<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="16" /> 与 L18）：
   - 骨干 layer 5：`DSConv [512, 3, 2]`——P3→P4 下采样（v12 这里是 `Conv [512, 3, 2]`）；
   - 骨干 layer 7：`DSConv [1024, 3, 2]`——P4→P5 下采样（v12 这里是 `Conv [1024, 3, 2]`）；
   - 这两处替换是 v13 骨干「轻量化」的第一手证据：把普通 3×3 卷积换成 DSConv 3×3 卷积，参数量下降一个数量级。

2. **block.py 间接引用**：
   - <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1405" :end="1406" /> DSBottleneck 内部两个 DSConv；
   - <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1523" :end="1547" /> DSC3k2 里 `dsc3k=False` 分支的 DSBottleneck 走 DSConv；
   - <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1842" /> HyperACE 里 `dsc3k=False` 分支用 DSBottleneck（间接走 DSConv）；
   - <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1458" :end="1470" /> DSC3k 内部的 DSBottleneck 也走 DSConv。

## 与 v12 conv.py 的对比

| 维度 | v12 conv.py | **v13 conv.py** |
|------|-------------|------------------|
| 文件行数 | 367 | **369**（+2） |
| 新增模块 | — | **DSConv（L84-101）** |
| `__all__` 差异 | 无 DSConv | **新增 "DSConv"（L25）** |
| 其他模块 | Conv/Conv2/DWConv/DWConvTranspose2d/ConvTranspose/Focus/GhostConv/RepConv/ChannelAttention/SpatialAttention/CBAM/Concat/Index | **完全一致** |
| Conv 定义 | 标准 Conv2d + BN + SiLU | **一字未改** |
| DWConv 定义 | `g=math.gcd(c1, c2)` | **一字未改** |

**给读者的建议**：v13 的 conv.py 是「v12 + DSConv」——读的时候把 DSConv 当作 v13 新增块来看即可，其它模块按 v8/v11/v12 的既有理解走即可。DSConv 里的 `groups=c_in` 是关键，别把它误读成 `groups=math.gcd(c1, c2)`（那是 DWConv 的写法）；DSConv 强制 dw 的输入输出通道相同，靠 pw 部分完成通道变换。

## 与其他文件的关系

- v13 骨干结构：<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="9" :end="19" /> 骨干 layer 5/7 直接用 DSConv 做下采样
- DSConv 消费者：<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1375" :end="1547" /> DSBottleneck / DSC3k / DSC3k2 三类模块；<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1804" :end="1856" /> HyperACE 的 DSC3k/DSBottleneck 分支
- 检测头 cv3 分类分支：<SrcLink file="v13:ultralytics/nn/modules/head.py" :line="46" :end="57" /> Detect.cv3 用 DWConv + Conv（不是 DSConv，但结构相似——两段 DW-separable + 1×1）
- block.py 里其它 v13 新增块见 [block-py 源码页](/versions/v13/source/block-py)；yaml 里 DSConv 的使用位置见 [yolov13-yaml 源码页](/versions/v13/source/yolov13-yaml)；检测头见 [head-py 源码页](/versions/v13/source/head-py)；损失与分配器见 [loss-py 源码页](/versions/v13/source/loss-py) 与 [tal-py 源码页](/versions/v13/source/tal-py)。
