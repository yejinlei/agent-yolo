# ultralytics/nn/modules/block.py

YOLOv13 的模块仓库：**1916 行文件里，v13 全部新增积木集中在 L1375-1916（共 542 行）**——11 个新类分三组：**DSConv 家族**（DSBottleneck L1375 / DSC3k L1414 / DSC3k2 L1472，把 C2f/C3 的 Bottleneck 换成 depthwise-separable）、**Hypergraph 家族**（AdaHyperedgeGen L1549 / AdaHGConv L1626 / AdaHGComputation L1677 / C3AH L1721 / FuseModule L1764 / HyperACE L1804，v13 招牌的 Adaptive Hypergraph Convolution）、**融合家族**（DownsampleConv L1858 / FullPAD_Tunnel L1891，FullPAD 范式用的门控残差与下采样）。A2C2f（L1297 附近，v12 引入）仍在文件里被 v13 骨干 layer 6/8 引用，`Conv / DWConv / Bottleneck / C2f / C3k2` 等 v8-v12 遗留模块一字未动。

> 快照路径：`source/yolov13/snapshot/ultralytics/nn/modules/block.py`（1916 行，commit `73289949`，2025-11-18，iMoonLab/yolov13，AGPL-3.0）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：**v13 是一次「双骨干结构改造 + 注意力范式升级」的双重跃迁**——一侧把 C2f 家族的骨干积木（Bottleneck/C3k2）换成 depthwise-separable 变体（DSBottleneck/DSC3k2）以降低实时开销，另一侧把 v12 的 Area-Attention 升级为 v13 的 Adaptive Hypergraph Convolution（AdaHGComputation 是 v12 AAttn 的替代方案）。**HyperACE（L1804）是 v13 head 的第一个 block**（yaml L22），它把三尺度骨干特征（P3/P4/P5）通过 FuseModule 融合、再用两个 C3AH 分支 + 若干 DSC3k 分支一起送进最终 cv2 卷积，是 v13 「Advanced Hypergraph-Based Attention for Real-Time Object Detection」（arXiv 2506.17733）的核心。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1" :end="50" /> | 文件头 + 导入 | 依赖 conv.py 的 Conv、math、torch、F |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="172" :end="185" /> | SPPF | v8 骨架，**v13 yaml 不引用** |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="220" :end="360" /> | Bottleneck / BottleneckCSP | 常规 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="360" :end="480" /> | C2f | DSC3k2 的父类（继承 C2f） |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="560" :end="620" /> | C3 / C3Ghost | DSC3k 继承 C3 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="725" :end="735" /> | C3k2 | v11 引入，v13 不用（换成 DSC3k2） |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1160" :end="1164" /> | FlashAttention 导入 | 保留 v12 的 `try/except + assert False` 硬依赖 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1166" :end="1246" /> | AAttn | v12 引入，v13 保留但 yaml 不直接引用 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1249" :end="1297" /> | ABlock | v12 引入 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1297" :end="1373" /> | A2C2f | **v13 yaml 骨干 layer 6/8 使用（a2=True）** |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1375" :end="1411" /> | **DSBottleneck** | v13 新增：DSConv(k1=3) + DSConv(k2=5, d=d2) + 可选 residual |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1414" :end="1470" /> | **DSC3k** | v13 新增：C3 骨架 + DSBottleneck 序列 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1472" :end="1547" /> | **DSC3k2** | v13 新增：C2f 骨架 + DSBottleneck 或 DSC3k |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1549" :end="1624" /> | **AdaHyperedgeGen** | v13 招牌：动态超边原型 + 参与矩阵 A 生成 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1626" :end="1675" /> | **AdaHGConv** | v13 招牌：AdaHyperedgeGen + 两阶段消息传递 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1677" :end="1719" /> | **AdaHGComputation** | v13 招牌：4D ↔ 序列包装器，接 CNN 骨干 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1721" :end="1762" /> | **C3AH** | v13 招牌：CSP 骨架 + AdaHGComputation 主通路 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1764" :end="1802" /> | **FuseModule** | v13 招牌：三尺度 downsample/upsample/concat |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1804" :end="1856" /> | **HyperACE** | v13 招牌：Fuse + 双 C3AH + 双 DSC3k 并联 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1858" :end="1889" /> | **DownsampleConv** | v13 新增：AvgPool2d(2) + 可选 1×1 通道翻倍 |
| <SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1891" :end="1916" /> | **FullPAD_Tunnel** | v13 新增：门控残差 `x[0] + gate * x[1]` |

<SourceViewer file="v13:ultralytics/nn/modules/block.py" />

## DSConv 家族：把 Bottleneck 换成 DSConv

### DSBottleneck（L1375-1411）

```python
def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
    super().__init__()
    c_ = int(c2 * e)
    self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
    self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
    self.add = shortcut and c1 == c2

def forward(self, x):
    y = self.cv2(self.cv1(x))
    return x + y if self.add else y
```

- 结构与 C2f 里的 `Bottleneck` 完全对称——两个 DSConv 串联 + 可选 residual；
- **默认 k1=3、k2=5、d2=1**：第二层是 5×5 深度可分离卷积，比经典 C2f Bottleneck 的 3×3 感受野更大，但成本只增加一点（因为 groups=c_in）；
- `shortcut and c1 == c2`——只有通道数不变时才加 residual，与 Bottleneck 一致；
- DSConv 本体在 <SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="84" :end="101" />，见 [conv-py](/versions/v13/source/conv-py) 页。

### DSC3k（L1414-1470）：C3 骨架 + DSBottleneck

```python
def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
    super().__init__(c1, c2, n, shortcut, g, e)
    c_ = int(c2 * e)
    self.m = nn.Sequential(
        *(DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))
    )
```

- 继承自 v11 时代的 `C3`——保留 C3 的「split 后 concat 再 cv3」骨架；
- 内部 Bottleneck 换成 DSBottleneck，`e=1.0` 意味着内层不做通道扩展（外层已扩展过一次）；
- **v13 yaml 不直接使用 DSC3k**，只在 HyperACE 内部通过 `dsc3k=True` 分支间接引用（L1842）。

### DSC3k2（L1472-1547）：C2f 骨架 + DSBottleneck / DSC3k

```python
def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
    super().__init__(c1, c2, n, shortcut, g, e)
    if dsc3k:
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2)
            for _ in range(n)
        )
    else:
        self.m = nn.ModuleList(
            DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
            for _ in range(n)
        )
```

- 继承自 v11 引入的 C2f 家族 `C3k2` 的父类 `C2f`——完全复用 C2f 的拼接式结构；
- **参数名从 v11 的 `c3k` 改为 `dsc3k`**——命名语义更清晰（DS=Bottleneck 家族），但**默认 `k2=7`**（比 v11 C3k2 的 3 更大，符合 v13 「更大感受野、更低计算」的取向）；
- **v13 yaml 骨干 layer 2/4**（`[256/512, False, 0.25]`）与 **head layer 17/21/26/30** 都用 DSC3k2，`dsc3k=False` 骨干走 DSBottleneck、`dsc3k=True` head 走 DSC3k。

## Hypergraph 家族：v13 招牌的四件套 + 一个包装器

### AdaHyperedgeGen（L1549-1624）：动态超边与参与矩阵

```python
def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
    self.head_dim = node_dim // num_heads
    self.context = context
    self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
    nn.init.xavier_uniform_(self.prototype_base)
    if context in ("mean", "max"):
        self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
    elif context == "both":
        self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)

def forward(self, X):
    B, N, D = X.shape
    if self.context == "mean":
        context_cat = X.mean(dim=1)
    elif self.context == "max":
        context_cat, _ = X.max(dim=1)
    else:
        avg_context = X.mean(dim=1); max_context, _ = X.max(dim=1)
        context_cat = torch.cat([avg_context, max_context], dim=-1)
    prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
    prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets
    X_proj = self.pre_head_proj(X)
    X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
    proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
    logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
    logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)
    logits = self.dropout(logits)
    return F.softmax(logits, dim=1)
```

**这是 v13 论文的核心机制**——把静态超图（v12 时代 AAttn 里的 attention 矩阵）换成**「动态超边原型 + 上下文调制 + 多头相似度」**：

1. **`prototype_base` 是可学习参数**（`num_hyperedges × node_dim`）——一组固定的超边原型；
2. **`context_net`**：从 X 的全局上下文（mean、max、或拼接）生成偏移量，加到原型上得到 per-sample 的 `prototypes`——**每条超边都被输入调制**；
3. **多头相似度**：把 X 与 prototypes 都切分成 `num_heads` 份，`bmm / sqrt(head_dim)` 得到相似度，多头的结果在 head 维上 `mean` 再 softmax；
4. **`softmax(dim=1)`**：沿 N（vertex）方向归一化——**每条超边（列）对它的参与顶点做 softmax**，得到稠密的参与矩阵 A（形状 `(B, N, M)`）；
5. **`dropout`** 加在 logits 上（softmax 之前），符合标准 transformer 惯例。

**与 v12 AAttn 的关键区别**：v12 AAttn 是 q/k/v 自注意（q、k 都从 X 里来）；v13 AdaHyperedgeGen 是**「X ↔ 动态原型」的异源注意**——原型是超图的边，X 是顶点，`context_net` 让边随输入变化。

### AdaHGConv（L1626-1675）：两阶段消息传递

```python
def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
    self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
    self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
    self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

def forward(self, X):
    A = self.edge_generator(X)
    He = torch.bmm(A.transpose(1, 2), X)   # (B, M, D) 顶点 → 超边
    He = self.edge_proj(He)
    X_new = torch.bmm(A, He)               # (B, N, D) 超边 → 顶点
    X_new = self.node_proj(X_new)
    return X_new + X
```

- 经典的 hypergraph convolution：`He = A^T X` 把顶点聚合到超边（vertex-to-edge）、`X_new = A He` 再把超边消息分发回顶点（edge-to-vertex）；
- `edge_proj` / `node_proj` 都是 `Linear + GELU` 的两层小 MLP——**没有 BN**（Transformer 风格）；
- **末端 residual `X_new + X`**——保证训练稳定；
- **激活是 GELU 而非 SiLU**——与 v12 ABlock 的 `Conv` 激活选择不同，但整个 v13 的 hypergraph 家族都统一用 GELU。

### AdaHGComputation（L1677-1719）：4D ↔ 序列包装器

```python
def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
    self.hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads, dropout, context)

def forward(self, x):
    B, C, H, W = x.shape
    tokens = x.flatten(2).transpose(1, 2)   # (B, C, H, W) → (B, H*W, C)
    tokens = self.hgnn(tokens)
    x_out = tokens.transpose(1, 2).view(B, C, H, W)
    return x_out
```

- 让 AdaHGConv 能直接嵌入 CNN 骨干——**没有位置编码**（依赖残差与全局上下文即可），**没有 padding**（保持 H×W 网格）；
- 与 v12 AAttn 一样都是「把 H×W 摊平成 N=H·W 个 token」的做法——但两者复杂度不同：AAttn 是 O(N²·D)，AdaHGConv 是 O(N·M·D)（M 是超边数，远小于 N）。

### C3AH（L1721-1762）：CSP 骨架 + AdaHGComputation

```python
def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
    c_ = int(c2 * e)
    assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
    num_heads = c_ // 16
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c1, c_, 1, 1)
    self.m = AdaHGComputation(embed_dim=c_, num_hyperedges=num_hyperedges,
                              num_heads=num_heads, dropout=0.1, context=context)
    self.cv3 = Conv(2 * c_, c2, 1)

def forward(self, x):
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

- **完全对标 v8 的 C2f 骨架**：cv1 走主通路、cv2 走 shortcut、cv3 融合；
- **关键约束 `assert c_ % 16 == 0`**——因为 `num_heads = c_ // 16` 是硬编码（与 v12 A2C2f 的 `c_ % 32 == 0` 类似但门槛更低）；
- **`e=1.0` 默认**：不像 C3/C2f 用 `e=0.5`，HyperACE 里 C3AH 走的是 `c_ = c2 * e2 = c2 * 1 = c2`；
- **主通路是 AdaHGComputation 而不是 Bottleneck**——这就是 v13 「用超图替代注意力」的具体落地。

### FuseModule（L1764-1802）：三尺度对齐

```python
def __init__(self, c_in, channel_adjust):
    self.downsample = nn.AvgPool2d(kernel_size=2)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    if channel_adjust:
        self.conv_out = Conv(4 * c_in, c_in, 1)
    else:
        self.conv_out = Conv(3 * c_in, c_in, 1)

def forward(self, x):
    x1_ds = self.downsample(x[0])
    x3_up = self.upsample(x[2])
    x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
    out = self.conv_out(x_cat)
    return out
```

- **输入是三个 feature map 的 list**（HyperACE 从骨干 layer 4/6/8 = P3/P4/P5 拿）；
- **x[0] 下采样**（P3 64×64 → 32×32）、**x[1] 不动**（P4 32×32）、**x[2] 上采样**（P5 16×16 → 32×32）——**统一到中间尺度**；
- `channel_adjust=True` 时 `Conv(4*c_in, c_in, 1)`——多输入一通道（可能来自 v13 里 FuseModule 被 DownsampleConv 复用），False 时 `Conv(3*c_in, c_in, 1)`；
- v13 yaml L22 HyperACE 参数第 8 位 `True` 传给 `channel_adjust=True`。

## HyperACE（L1804-1856）：v13 head 的核心入口

```python
def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False,
             e1=0.5, e2=1, context="both", channel_adjust=True):
    self.c = int(c2 * e1)
    self.cv1 = Conv(c1, 3 * self.c, 1, 1)
    self.cv2 = Conv((4 + n) * self.c, c2, 1)
    self.m = nn.ModuleList(
        DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
        for _ in range(n)
    )
    self.fuse = FuseModule(c1, channel_adjust)
    self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
    self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

def forward(self, X):
    x = self.fuse(X)
    y = list(self.cv1(x).chunk(3, 1))
    out1 = self.branch1(y[1])
    out2 = self.branch2(y[1])
    y.extend(m(y[-1]) for m in self.m)
    y[1] = out1
    y.append(out2)
    return self.cv2(torch.cat(y, 1))
```

**结构解读**：

1. **`fuse`**：先把三尺度特征合并到中间尺度（P4 尺度）——这是 v13 head 的第一个动作；
2. **`cv1(x).chunk(3, 1)`**：把融合特征切成 3 份 y0/y1/y2（每份 `self.c` 通道）；
3. **两个 C3AH 分支**都吃 y[1]——**并联的两路超图注意力**，得到 out1/out2；
4. **`self.m`** 是从 y0 出发的 DSC3k/DSBottleneck 链（低阶卷积分支）；
5. **`y[1] = out1`**：覆盖第二份；**`y.append(out2)`**：把 out2 追加到 y 尾部；
6. **`cv2(cat(y, 1))`**：最终拼接 `(4 + n) * c` 通道——**y 的最终长度是 `[y0, out1, y2, ...m 输出..., out2]`**，共 `4 + n` 项。

**HyperACE 参数缩放特例**（tasks.py <SrcLink file="v13:ultralytics/nn/tasks.py" :line="1075" :end="1088" />）：

```python
elif m is HyperACE:
    legacy = False
    c1 = ch[f[1]]
    c2 = args[0]
    c2 = make_divisible(min(c2, max_channels) * width, 8)
    he = args[1]
    if scale in "n":
        he = int(args[1] * 0.5)     # n 档 M=4
    elif scale in "x":
        he = int(args[1] * 1.5)     # x 档 M=12
    args = [c1, c2, n, he, *args[2:]]
    n = 1
    if scale in "lx":                # L/X 档
        args.append(False)           # 追加 shortcut=False（HyperACE 第 6 位是 shortcut）
```

- **n 档**：yaml 里 `he=8` → `he=int(8*0.5)=4`；
- **x 档**：`he=8` → `he=int(8*1.5)=12`；
- **s/l 档**：`he` 保持 8；
- **l/x 档额外 `args.append(False)`**：`HyperACE.__init__` 的第 6 个位置参数是 `shortcut=False`（默认值）——这里显式再传一次保持行为一致（其实等于默认值，属于作者防御性代码）。

## DownsampleConv 与 FullPAD_Tunnel：FullPAD 范式的两个零件

### DownsampleConv（L1858-1889）

```python
def __init__(self, in_channels, channel_adjust=True):
    self.downsample = nn.AvgPool2d(kernel_size=2)
    if channel_adjust:
        self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
    else:
        self.channel_adjust = nn.Identity()

def forward(self, x):
    return self.channel_adjust(self.downsample(x))
```

- **AvgPool2d(2) 下采样**——非学习；
- `channel_adjust=True` 时接 1×1 卷积把通道翻倍（P3→P4、P4→P5 的经典模式）；`False` 时保持通道不变（用 Identity）；
- **v13 yaml L24** 只有一处引用：`[[9], 1, DownsampleConv, []]`——HyperACE 的输出被下采样供 FullPAD_Tunnel 用；
- **tasks.py L1089-1095 特例**：l/x 档 `args.append(False)` 且 `c2=c1`——l/x 档不翻倍通道（因为通道已经足够大）。

### FullPAD_Tunnel（L1891-1916）

```python
def __init__(self):
    super().__init__()
    self.gate = nn.Parameter(torch.tensor(0.0))

def forward(self, x):
    out = x[0] + self.gate * x[1]
    return out
```

- **单个标量参数 `gate`**（初始 0.0）——训练初期等于直通（只保留 x[0]），训练过程中 gate 会缓慢增大让 x[1] 参与；
- **语义**：`x[0]` 是原始特征（例如 Concat 出的 P4 主干），`x[1]` 是「增强特征」（例如来自 HyperACE 的输出）——**gate 是「增强信号」的全局强度系数**；
- **无 BN、无激活**：只是一个可学习的标量门控；
- v13 yaml 用了 **7 次 FullPAD_Tunnel**（L25/L26/L27/L32/L38/L43/L48）——每处都是「Concat 输出的主干特征 + HyperACE/DownsampleConv 通路的增强特征」。

## 与 v12 block.py 的对比

| 维度 | v12 block.py | **v13 block.py** |
|------|--------------|------------------|
| 文件行数 | 1354 | **1916**（+562） |
| 新增块位置 | L1160-1354 | **L1375-1916**（新增 542 行） |
| 招牌模块 | AAttn / ABlock / A2C2f | **AdaHyperedgeGen / AdaHGConv / AdaHGComputation / C3AH / FuseModule / HyperACE** |
| 骨干积木 | C3k2（Bottleneck）+ A2C2f | **DSC3k2（DSBottleneck）+ A2C2f** |
| 位置编码 | AAttn.pe（7×7 DW conv） | **AdaHyperedgeGen 无显式 PE**（依赖 context + residual） |
| 复杂度 | O(N²·D)（自注意） | **O(N·M·D)（超图，M<<N）** |
| 激活 | SiLU（Conv） | **GELU（超图分支）+ SiLU（卷积分支）** |
| 外部硬依赖 | flash-attn（assert False） | **仍保留**（L1160-1164，但 yaml 不引用 AAttn 相关类） |
| v12 遗留 | — | **AAttn / ABlock / A2C2f 保留但仅骨干 layer 6/8 使用 A2C2f** |
| head 起点 | A2C2f（P4/P5 融合） | **HyperACE**（三尺度融合） |
| 门控残差 | 无 | **FullPAD_Tunnel**（7 处使用） |
| 下采样 | Conv stride=2 | **DownsampleConv**（AvgPool + 可选 Conv）+ Conv stride=2 |

## 与其他文件的关系

- 骨干积木依赖：<SrcLink file="v13:ultralytics/nn/modules/conv.py" :line="84" :end="101" /> `DSConv`（DSBottleneck/DSC3k/DSC3k2 都基于它）
- 结构配置：<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="13" :end="19" /> 骨干用 DSC3k2 × 2 + DSConv × 2 + A2C2f × 2；<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="22" /> head 首块 HyperACE；<SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="31" :end="47" /> head 用 DSC3k2 × 4 + FullPAD_Tunnel × 7
- 建模入口：<SrcLink file="v13:ultralytics/nn/tasks.py" :line="1075" :end="1095" /> `parse_model` 里 HyperACE / DownsampleConv / FullPAD_Tunnel 三个类的缩放特例
- 检测头：[head.py](/versions/v13/source/head-py)（v13 未新增头类）
- 损失：<SrcLink file="v13:ultralytics/utils/loss.py" :line="157" :end="259" /> `v8DetectionLoss`（v13 完全沿用）
- 分配器：<SrcLink file="v13:ultralytics/utils/tal.py" :line="14" :end="258" /> `TaskAlignedAssigner`（v13 完全沿用）
- DSConv 拆解见 [conv-py 源码页](/versions/v13/source/conv-py)；yaml 结构与缩放见 [yolov13-yaml 源码页](/versions/v13/source/yolov13-yaml)；损失总装见 [loss-py 源码页](/versions/v13/source/loss-py)；分配器见 [tal-py 源码页](/versions/v13/source/tal-py)；检测头见 [head-py 源码页](/versions/v13/source/head-py)。
