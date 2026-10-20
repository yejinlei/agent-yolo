# models/common.py

这是 YOLOv9 快照 `models/common.py` 的源码浏览页：**GELAN 积木仓库 + PGI 可学习先验 + CBNet 通道级连接**。1200 行，与 v7 的 models/common.py 同源（v7 有 82 个类/函数），v9 往里添了三组新积木——[RepNCSPELAN4](#repncspelan4l580-598gelan-核心块)（GELAN 核心块）、ImplicitA/ImplicitM（PGI 先验）、CBLinear/CBFuse（可逆辅助分支的通道级连接）。

> 快照路径：`source/yolov9/snapshot/models/common.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v9:models/common.py" />

**阅读路线**：

- 降采样家族：[L60-67 AConv](#L60-L67)（avg_pool 2×1 + conv3s2）、[L70-83 ADown](#L70-L83)（avg_pool → chunk 两半 → 一半 conv3s2 / 一半 maxpool3s2+conv1 → concat）——v8 Conv-s2 路线的替代品，yolov9-c.yaml 的 P2→P5 全走这里
- 重参数化：[L86-180 RepConvN](#L86)——v7 RepConv 的 N 版，[L100 bn 恒为 None](#L100) 的细节见下方卡片
- [L373 RepNCSP](#L373)：CSP 结构里套 RepNBottleneck，GELAN 两级串联的内层单元
- [L554-558 Silence](#L554-L558)：恒等层——yolov9-c.yaml 的第 0 层，辅助分支从"原始输入"抽头的锚点
- GELAN 家族：[L563-577 SPPELAN](#L563-L577)（骨干收尾的 SPP-ELAN）、[L580-598 RepNCSPELAN4](#L580-L598)（核心块，下方逐行注释）
- PGI 先验：[L605-624 ImplicitA/ImplicitM](#L605-L624)——v7 YOLOR 隐式知识原样复用
- CBNet 通道级连接：[L631-650 CBLinear/CBFuse](#L631-L650)——可逆辅助分支的抽头与回注
- 对照：[L655 DetectMultiBackend](#L655) 等推理后端代码占了大半篇幅，属 v5-v7 遗产，本页不展开

## 关键片段逐行走读

> 已读过 [v7 common.py](/versions/v7/source/common-py)/[v8 block.py](/versions/v8/source/common-py)？v7 读者只需关注三组新积木；v8 读者注意目录名回来了（v8 的积木在 nn/modules/block.py，v9 回到 models/common.py）——布局洗牌是 [v9 痛点页](/versions/v9/painpoints) 记录的延续性事实。

### RepNCSPELAN4（L580-598）：GELAN 核心块

<SrcSnippet file="v9:models/common.py" :start="580" :end="598" hl="582-593" title="GELAN 核心块：RepNCSPELAN4" />

::: details 逐行注释（L580–L598）
- **L582**：`__init__(c1, c2, c3, c4, c5=1)` —— c1/c2 进出通道，c3 中间通道，c4 串联支路通道，c5 是传给 RepNCSP 的重复数（yaml 里恒为 1）。
- **L584**：`self.c = c3//2` —— 每一半的通道数，forward_split 用。
- **L585**：`cv1 = Conv(c1, c3, 1, 1)` —— 1×1 把输入升到 c3 通道，随后对半劈开。
- **L586**：`cv2 = Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))` —— 第一级串联支路：半宽 RepNCSP + 3×3 Conv。
- **L587**：`cv3 = Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))` —— 第二级串联支路，**输入是 cv2 的输出**——两级级联是 GELAN 与 C2f 的分水岭：C2f 是 n 个 Bottleneck 逐级并联全拼接（n+2 路），GELAN 只有两条串联支路（4 路 concat）。
- **L588**：`cv4 = Conv(c3+(2*c4), c2, 1, 1)` —— 收束层：c3（劈开的两半）+ 2×c4（两条支路）= 4 路 concat 后压回 c2。
- **L590-593**：`forward` —— `self.cv1(x).chunk(2, 1)` 劈成两半入列表，`m(y[-1])` 逐级喂给 cv2、cv3（`y[-1]` 取上一级输出，即串联），最后 4 路 concat 过 cv4。
- **L595-598**：`forward_split` —— 用 `split((self.c, self.c), 1)` 替代 chunk 的等价实现，ONNX/TensorRT 导出更友好；调用时机由 fuse/导出逻辑决定。
:::

### SPPELAN（L563-577）：SPP-ELAN 收尾

<SrcSnippet file="v9:models/common.py" :start="563" :end="577" hl="569-577" title="SPPELAN：SPP-ELAN 收尾块" />

::: details 逐行注释（L563–L577）
- **L565**：`__init__(c1, c2, c3)` —— yaml 层 10 传 `[512, 256]`，c3=256 是池化支路通道。
- **L568**：`cv1 = Conv(c1, c3, 1, 1)` —— 先压到 c3。
- **L569-571**：`cv2/cv3/cv4 = SP(5)` —— 三个 5×5 MaxPool **串行**堆叠（SP 类即单次池化），等效 SPP(5,9,13) 但只注册一个池化核——v8 SPPF 的同款思路。
- **L572**：`cv5 = Conv(4*c3, c2, 1, 1)` —— 原始 + 3 级池化共 4 路收束。
- **L574-577**：`forward` —— `y = [self.cv1(x)]` 起头，`y.extend(m(y[-1]) for m in [...])` 串行追加池化输出，cat 后过 cv5。
:::

### AConv（L60-67）与 ADown（L70-83）：平均池化降采样家族

<SrcSnippet file="v9:models/common.py" :start="60" :end="83" hl="66,78-83" title="AConv / ADown：平均池化降采样" />

::: details 逐行注释（L60–L83）
- **L63**：`cv1 = Conv(c1, c2, 3, 2, 1)` —— AConv 的主体，3×3 步幅 2。
- **L66**：`avg_pool2d(x, 2, 1, 0, False, True)` —— 2×2 窗口、步幅 1 的平均池化（ceil_mode=True），先把特征"糊"一遍再卷——下采样前的低通滤波。
- **L73**：`self.c = c2 // 2` —— ADown 把输出通道对半分给两条支路。
- **L74-75**：`cv1 = Conv(c1//2, self.c, 3, 2, 1)`、`cv2 = Conv(c1//2, self.c, 1, 1, 0)` —— 两条支路的卷积核。
- **L78**：先 avg_pool 全局糊化（步幅 1，不减分辨率）。
- **L79**：`x.chunk(2, 1)` —— 通道对半劈开。
- **L80**：一半走 cv1（conv3s2，提取+降采样）。
- **L81-82**：另一半 maxpool3s2（硬降采样保留极值）再 cv1×1 调通道。
- **L83**：`cat((x1, x2), 1)` —— 两条异构支路拼回 c2。对比 v8 的 `Conv(c1, c2, 3, 2)` 单路：v9 用"均值/极值两路"换更稳的下采样特征。
:::

### ImplicitA / ImplicitM（L605-624）：PGI 的可学习先验

<SrcSnippet file="v9:models/common.py" :start="605" :end="624" hl="609-613,620-624" title="PGI 可学习先验：ImplicitA / ImplicitM" />

::: details 逐行注释（L605–L624）
- **L609**：`self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))` —— 加性先验是逐通道的单像素张量，**自由参数**：不依附任何固定函数。
- **L610**：`normal_(self.implicit, std=.02)` —— 零均值小方差初始化，起点近似恒等。
- **L612-613**：`forward` 返回 `self.implicit + x` —— 对特征做逐通道平移。
- **L620**：乘性先验初始化为全 1。
- **L621**：`normal_(self.implicit, mean=1., std=.02)` —— 起点近似恒等缩放。
- **L623-624**：`forward` 返回 `self.implicit * x` —— 逐通道缩放。
- PGI 视角：这两个模块插在辅助分支/检测头前，训练期给"可编程梯度"提供自由度；与 v7 的 YOLOR 隐式知识（[v7 common.py L433-456](/versions/v7/source/common-py#L433-L456)）逐行同源。
:::

### CBLinear / CBFuse（L631-650）：通道级连接

<SrcSnippet file="v9:models/common.py" :start="631" :end="650" hl="635-639,646-650" title="CBLinear / CBFuse：通道级抽头与回注" />

::: details 逐行注释（L631–L650）
- **L635**：`conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)` —— 一个卷积输出 **sum(c2s) 个通道**，为多级拆分做准备。
- **L637-639**：`forward` —— `self.conv(x).split(self.c2s, dim=1)` 按 c2s 列表（如 `[256, 512, 512]`）拆成多路返回。yolov9-c.yaml 里层 25 输出三路，供三个不同尺度的 CBFuse 各取所需——同一份特征按需"批发"。
- **L642-644**：`CBFuse(idx)` 无可学习参数，idx 记录从输入元组取第几路。
- **L647**：`target_size = xs[-1].shape[2:]` —— 以**最后一路**（最深/最小尺度）为目标尺寸。
- **L648**：前面的路全部 `F.interpolate(..., mode='nearest')` 对齐尺寸。
- **L649**：`torch.sum(torch.stack(res + xs[-1:]), dim=0)` —— 逐元素相加回注主干。多级可逆分支的"汇流点"就这么两行。
:::

### RepConvN（L86-180）：重参数化的 N 版

<SrcSnippet file="v9:models/common.py" :start="86" :end="111" hl="100-102,108-111" title="RepConvN：训练态与部署态" />

::: details 逐行注释（L86–L111 与融合段 L134-180）
- **L87-89**：docstring 自述基于 RepVGG——v7 痛点 2 的"论文命名漂移"在此闭环：v7 论文写 RepConvN、代码叫 RepConv；v9 快照类名就叫 RepConvN。
- **L94**：`assert k == 3 and p == 1` —— 只支持 3×3 s1。
- **L100**：`self.bn = None` —— 与 v7 RepConv 的关键差异：v7 在 `c2 == c1 and s == 1` 时**条件创建** BN identity 分支，v9 的 `__init__` 不再创建，训练态默认只有 3×3 + 1×1 双卷积（L101-102）。
- **L104-106**：`forward_fuse` —— 部署态前向，走融合后的单卷积。
- **L108-111**：`forward` —— `id_out = 0 if self.bn is None else self.bn(x)`，然后 `act(conv1 + conv2 + id_out)`。identity 分支的机制**在场**（BN 在则生效），只是默认关闭。
- **L113-117**：`get_equivalent_kernel_bias` —— 三分支核相加：3×3 + pad 后的 1×1 + identity。
- **L145-151**：`_fuse_bn_tensor` 的 identity 分支用 id_tensor 技巧——BN 等价于"中心置 1 的 3×3 单位阵卷积 + BN"。
- **L161-180**：`fuse_convs` —— 算出等价核后新建单 Conv2d，删掉 conv1/conv2。与 v7 [fuse_repvgg_block](/versions/v7/source/common-py#L584-L643) 同型。
:::

## 与 v7/v8 的对比

| 维度 | v7 common.py | v8 block.py | v9 common.py |
|------|--------------|-------------|--------------|
| 体量 | 82 个类/函数 | 17 个导出（积木拆家） | 1200 行，v7 底子 + 三组新积木 |
| 主打积木 | SPPCSPC + ELAN Concat | C2f（n+2 路并联拼接） | RepNCSPELAN4（两级串联 4 路） |
| 重参数化 | RepConv 三分支 + 条件 identity | 无（仅 RT-DETR neck 用） | RepConvN 双卷积，identity 默认关 |
| 隐式知识/先验 | ImplicitA/ImplicitM | 无 | ImplicitA/ImplicitM 原样复用（PGI 先验） |
| 感受野 | SPPCSPC（CSP + 三池化） | SPPF | SPPELAN（SPPF 同型，ELAN 命名） |
| 下采样 | Conv s2 / MaxPool | Conv s2 | AConv / ADown（均值+极值两路） |
| 通道级连接 | 无 | 无 | CBLinear/CBFuse（可逆辅助分支专用） |

建网入口见 [models/yolo.py 源码页](/versions/v9/source/yolo-py)（DualDDetect 与 parse_model），双头损失见 [loss_tal_dual.py 源码页](/versions/v9/source/loss-tal-dual-py)，结构配置见 [yolov9-c.yaml 源码页](/versions/v9/source/yolov9-c-yaml)；GELAN/PGI 的叙事版见 [YOLOv9 痛点页](/versions/v9/painpoints)。
