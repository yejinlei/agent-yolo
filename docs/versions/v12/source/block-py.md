# ultralytics/nn/modules/block.py

YOLOv12 的模块仓库：**1354 行的文件里，v12 的核心新增积木全部集中在 L1160-1354**——`AAttn`（L1166，Area-attention）+ `ABlock`（L1249，AAttn + Conv-based FFN）+ `A2C2f`（L1300，R-ELAN 结构：C2f 骨架里把 Bottleneck 换成 ABlock）。C3k2（L725）/ C3k（L736）/ C2PSA（L1004 附近）作为 v11 遗留保留，**C3k2 在 v12 yaml 中仍被使用**（骨干 layer 2/4/20），而 C2PSA 在 v12 快照中**代码存在但 yaml 不引用**（与 v11 的 C3k2 状态相反）。

> 快照路径：`source/yolov12/snapshot/ultralytics/nn/modules/block.py`（1354 行）。点击左侧行号可复制/分享行锚点。

**这个文件在 v12 里的角色**：**v12 是 Attention 从 v11 的「尾端外挂 C2PSA」变成「骨干 P4/P5 主通路内嵌 A2C2f」的一次结构性转向**。三件套 AAttn / ABlock / A2C2f 全部是新代码（L1160-1354），且 AAttn 硬依赖 `flash_attn` 库（L1160-1163 的 `assert False` 兜底）。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1" :end="50" /> | 文件头 + 导入 | FlashAttention 在此处不 import，在 L1160 单独 try/except |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="60" :end="170" /> | Conv / DWConv 依赖（外部文件） | block.py 依赖 conv.py 中的 Conv / DWConv |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="172" :end="185" /> | SPPF | v8 骨架，**v12 yaml 不引用**（v8/v10/v11 都用） |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="220" :end="360" /> | Bottleneck / BottleneckCSP | 常规 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="360" :end="480" /> | C2f | 基类，被 C3k2/A2C2f 继承 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="560" :end="620" /> | C3 / C3Ghost | 常规 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="620" :end="720" /> | RepNCSPELAN4 / C2i | v7 遗留 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="725" :end="735" /> | **C3k2** | v11 引入，**v12 yaml 骨干 layer 2/4/20 使用** |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="736" :end="744" /> | **C3k** | C3 变体，A2C2f 的 `a2=False` 分支用它 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1000" :end="1060" /> | **C2PSA** | v11 引入；**v12 yaml 不引用** |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1160" :end="1164" /> | **FlashAttention 导入** | `try/except` + `assert False`——**硬依赖，缺失即崩溃** |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1166" :end="1246" /> | **AAttn** | Area-attention 主体 |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1249" :end="1297" /> | **ABlock** | AAttn + Conv-based FFN，`mlp_ratio=1.2` |
| <SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1300" :end="1354" /> | **A2C2f** | R-ELAN；C2f 骨架 + ABlock 或 C3k |

<SourceViewer file="v12:ultralytics/nn/modules/block.py" />

## FlashAttention 硬依赖（L1160-1164）

```python
try:
    from flash_attn.flash_attn_interface import flash_attn_func
except Exception:
    assert False, "import FlashAttention error! Please install FlashAttention first."
from timm.models.layers import trunc_normal_
```

**这是 v12 的最大工程约束**：`block.py` 文件顶部**不 import flash_attn**（避免污染 v11 时代的调用路径），而是在 L1160-1163 单独 `try/except` + `assert False`。也就是说：

- 只要 `import ultralytics` 就能加载 block.py 的所有基础模块（Conv/Bottleneck/C2f 等），但**只要实例化 AAttn/ABlock/A2C2f 就会走 `flash_attn_func`**——即使 CPU 上也不 import；
- **未安装 flash-attn 时 `assert False` 直接抛异常**——v12 训练/推理的强制前置条件，没有优雅回退；
- 只有 AAttn.forward 内部有 `if x.is_cuda:` 的判断：CUDA 走 `flash_attn_func`（L1218-1223），CPU 走手写 softmax（L1224-1234），但 **CPU 分支也需要 flash_attn 模块存在**（因为 import 已经执行过 `assert`）。

**给读者的建议**：v12 训练前 `pip install flash-attn --no-build-isolation` 是硬前提；`assert False` 兜底意味着没有 fallback，不能靠代码 patch 绕过。

## AAttn（L1166-1246）：Area-attention 主体

```python
class AAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)   # L1200
        self.proj = Conv(all_head_dim, dim, 1, act=False)       # L1201
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)  # L1202 7×7 DW PE
```

**结构三件套**：

1. **`qkv`**：一个 1×1 Conv 一次产出 q/k/v 三份（`all_head_dim * 3` 通道）——不是三个独立的 Conv，节省计算；
2. **`proj`**：1×1 Conv 输出回原通道 `dim`；
3. **`pe`**：**7×7 DW 卷积位置编码**——关键设计。`g=dim` 意味着分组数=通道数（等价于 DWConv），`stride=1, padding=3`，激活关闭。**注意 `pe` 的作用对象不是 q/k/v，而是 v**——见 forward。

### forward（L1205-1246）

- L1207 `B, C, H, W = x.shape; N = H * W`——输入是 `(B, C, H, W)` 的 4D 特征图；
- L1210 `qkv = self.qkv(x).flatten(2).transpose(1, 2)`——`(B, 3C, N)`，即每像素 3C 维；
- **L1211-1213 area > 1 时把 qkv 切分成 `area` 份**：`qkv.reshape(B * area, N // area, C * 3)`——每个 area 内的 attention 独立计算，`area=4` 意味着 64×64 特征图分成 4×16=256 个位置的子块 attention；
- L1214-1216 `q, k, v = qkv.view(B, N, num_heads, head_dim * 3).split([head_dim, head_dim, head_dim], dim=3)`——按最后一维切成 q/k/v 三份；
- **L1218-1223 CUDA 分支**：`flash_attn_func(q.contiguous().half(), k.contiguous().half(), v.contiguous().half()).to(q.dtype)`——注意**强制转 half**，FlashAttention 要求 fp16/bf16，算完再转回原 dtype；
- **L1224-1234 CPU 分支**：手写 softmax attention——标准 `(q @ k^T) * head_dim^-0.5` → `exp(x - max)` → normalize → `v @ attn^T`；
- **L1236-1239 area > 1 恢复形状**；
- **L1241-1244 关键**：`x = x + self.pe(v)` 然后 `x = self.proj(x)`——**位置编码 pe(v) 加到 x 上（不是加到 attention 输出上）**，这是 v12 的独有设计：v 走 7×7 DW conv 提取局部位置信息，然后与 x 相加再走 1×1 投影。
- L1245 return x。

**注意 pe 加在 x 上而不是加到 q/k/v 上**——这意味着「局部位置编码」是**输出侧的信息注入**，而不是常见的「输入侧的 positional embedding」。这个设计让 7×7 DW 卷积在 attention 之后再做，与 attention 的**全局感受野互补**——attention 管全局，pe 管局部，两者相加得到综合特征。

**推荐配置**：类文档写「recommend that `dim//num_heads` 是 32 或 64 的倍数」——A2C2f 里 `num_heads = c_ // 32` 是硬编码，正好让 `head_dim = 32` 落在推荐范围。

## ABlock（L1249-1297）：AAttn + Conv-based FFN

```python
class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
```

- **`mlp_ratio=1.2`**：FFN 展开比只有 1.2——非经典 Transformer 的 4.0。**1.2 是「轻量 FFN」**，把 MLP 的容量控制在最小可行范围，符合 v12 「real-time」的定位；
- **FFN 用 Conv 而非 Linear**：`Conv(dim, hidden, 1) → Conv(hidden, dim, 1)`——两次 1×1 卷积等价于 Linear，但保持「整个模块只用 Conv」的统一风格；
- **trunc_normal_ 初始化 `std=0.02`**：从 `timm.models.layers` 引入的截断正态初始化，是 ViT/Swin 家族的常见做法——`std=0.02` 比默认 Kaiming 小很多，配合 layer-scale（见 A2C2f）保证训练初期 residual 路径稳定；
- **forward 无 LayerNorm**：ViT 标准是 pre-norm（`x = x + attn(ln(x))`），v12 是 **post-residual**（`x = x + attn(x)`），与 YOLO 家族一贯的 BN+SiLU 风格一致。

## A2C2f（L1300-1354）：R-ELAN 结构

```python
class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # L1332
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."  # L1333

        num_heads = c_ // 32  # L1336 硬编码

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01  # L1341
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + (self.gamma * self.cv2(torch.cat(y, 1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.cv2(torch.cat(y, 1))
```

**五个关键设计点**：

1. **`assert c_ % 32 == 0`（L1333）**：通道必须能被 32 整除——因为 `num_heads = c_ // 32` 是硬编码。yaml 里的 512/1024 基准通道经 width 与 `max_channels` 缩放后乘 `e=0.5`：n 档 #6/#8 → c_=64/128，s 档 → 128/256，m/l/x 档 → 256/256（P5 被 max_channels=512 封顶），均为 32 的倍数。scale 缩放后 width=0.25 时 256 → 64，仍然满足；但用户自定义 yaml 时若把通道改成 128 就会踩这个断言（`c_=64`，`64%32=0` 通过；但改成 96 就 `96%32=0` 也通过；改成 112 就崩）；

2. **`num_heads = c_ // 32`（L1336）**：**硬编码**——head_dim 永远是 32。v12 五档实际取值：n 档 P4/P5 → 2/4，s 档 → 4/8，m/l/x 档 → 8/8（P5 的 c_ 被 max_channels=512 封顶为 256，**不会出现 heads=16**）。**这个公式不允许外部覆盖**，即使想调 num_heads 也只能改 A2C2f 类本身；

3. **`gamma = 0.01`（L1341-1342）**：**layer-scale 参数**，仅在 `a2 and residual` 都为真时启用。yaml 写的是 `A2C2f, [512, True, 4]`——**第三位是 area 不是 residual**，`residual` 由 `parse_model`（tasks.py L1034-1038）只对 **L/X 档**追加 `True`（并带 `mlp_ratio=1.2`）。因此 **L/X 档骨干 #6/#8 有 gamma（0.01 初始化），n/s/m 档为 None**；head 三处 a2=False 永远为 None；

4. **`a2` 决定走 ABlock 还是 C3k**（L1345）：`a2=True` 走 `ABlock(c_, num_heads, mlp_ratio, area)`（每 `n` 段 2 个 ABlock）；`a2=False` 走 `C3k(c_, c_, 2, shortcut, g)`（每 `n` 段 2 个 C3k Bottleneck）——**这是一个隐藏的双分支**：yaml 的 a2 参数实际上控制「是否真的启用注意力」；

5. **forward 的 residual 加法**（L1352-1354）：只有 `gamma is not None` 时才走 `x + gamma * cv2(...)`——即「有 gamma 就带 layer-scale 残差、无 gamma 就不加残差」。n/s/m 档骨干 A2C2f 的 residual=False，走「**无残差 + 无 gamma 的纯 C2f 骨架（ABlock 替换 Bottleneck）**」；**L/X 档骨干 #6/#8 由 parse_model 补 residual=True，走 `x + gamma * cv2(...)` 的 layer-scale 残差**；head 段三处 A2C2f（a2=False）永远没有 gamma。

**给读者的建议**：A2C2f 是 v12 的招牌模块，`residual` 默认 `False`——**L/X 档由 `parse_model` 自动补 `residual=True`**，n/s/m 档保持关闭；自定义 yaml 想启用 layer-scale，需要手动传 `[c2, True, area, True]`（第四个位置是 residual）。

## C3k2（L725-735）：v11 引入，v12 yaml 骨干与 head 都用

```python
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```

- v11 引入、v12 沿用；
- `c3k=False` 时内嵌 `Bottleneck`（等价于 C2f），`c3k=True` 时内嵌 `C3k`（大核变体）；
- v12 yaml 使用点：骨干 layer 2/4（`c3k=False, e=0.25`）、head layer 20（`c3k=True`）；
- 详见 [layers](/versions/v12/layers#c3k2)。

## C3k（L736-744）：C3 的 k 变体

```python
class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
```

- C3 的 k 参数化版本（kernel size 可配）；
- **A2C2f 的 `a2=False` 分支直接用 C3k**（`block.py` L1345）——所以 head 段的三个 A2C2f 实际执行的是 C3k 骨架；
- 详见 [layers](/versions/v12/layers#c3k)。

## C2PSA（L1004 附近）：v11 引入，v12 yaml 不引用

v11 引入的 Position-wise Self-Attention 模块，v11 yaml 用在 backbone 尾部（`C2PSA, [1024, True, 5]`）。**v12 保留代码但 yaml 不引用**——与 v11 的 C3k2 状态相反（v11 保留但 yaml 不用的模块在 v12 变成骨干主力，v11 骨干主力的 C2PSA 在 v12 变成遗留）。

## 各模块与 v12 yaml 的使用情况

| 模块 | 行段 | v12 yaml 使用点 | 状态 |
|------|------|----------------|------|
| Conv | conv.py | 骨干 layer 0/1/3/5/7、head 多处长 | ✅ 主用 |
| DWConv | conv.py | Detect.cv3 分类头 | ✅ 主用（v12 Detect legacy=False） |
| C2f | block.py | A2C2f 的父类（`class A2C2f(nn.Module)`——注意 A2C2f **不继承 C2f**，是独立类） | ❌ yaml 不直接使用 |
| C3k2 | block.py L725 | 骨干 layer 2/4/20 | ✅ 主用 |
| C3k | block.py L736 | A2C2f `a2=False` 分支（即 head layer 11/14/17 的实质） | ✅ 隐式使用 |
| AAttn | block.py L1166 | ABlock.attn | ✅ 隐式使用（仅 a2=True 时） |
| ABlock | block.py L1249 | A2C2f.m（a2=True 分支） | ✅ 隐式使用（仅 a2=True 时） |
| A2C2f | block.py L1300 | 骨干 layer 6/8、head layer 11/14/17 | ✅ 主用 |
| C2PSA | block.py L1004 | 无 | ❌ yaml 不引用（v11 遗留） |
| SPPF | block.py L172 | 无 | ❌ yaml 不引用（v12 去 SPPF） |

## 与 v8 / v10 / v11 block.py 的对比

| 维度 | v8.0.100 | v10（8.1.34 fork） | **v11（8.3.0）** | **v12** |
|------|----------|--------------------|------------------|---------|
| 文件行数 | ~630 | ~830 | ~1080 | **1354** |
| 新增积木 | C2f / SPPF | C2fCIB / PSA / SCDown | C3k2 / C2PSA | **AAttn / ABlock / A2C2f** |
| 外部硬依赖 | 无 | 无 | 无 | **flash-attn（assert False 兜底）** |
| FlashAttention 导入 | 无 | 无 | 无 | **L1160-1163 单独 try/except** |
| 位置编码 | 无 | 无 | 无（C2PSA 有相对位置） | **AAttn.pe 7×7 DW conv** |
| layer-scale | 无 | 无 | 无 | **A2C2f.gamma（n/s/m 关；L/X 档骨干开）** |
| num_heads 计算 | — | — | — | **c_ // 32 硬编码** |
| FFN 展开比 | — | — | — | **2.0（A2C2f 默认）；L/X 档 1.2（parse_model 传）** |
| v11 遗留状态 | — | — | — | **C3k2 使用 / C2PSA 保留不用** |

## 与其他文件的关系

- 骨干 yaml：<SrcLink file="v12:ultralytics/cfg/models/v12/yolov12.yaml" :line="15" :end="25" /> 层 6/8 A2C2f；<SrcLink file="v12:ultralytics/cfg/models/v12/yolov12.yaml" :line="28" :end="45" /> head 层 11/14/17 A2C2f、层 20 C3k2
- 检测头：[head.py L21](/versions/v12/source/head-py) `Detect` 使用 DWConv（conv.py 提供）
- 建模入口：<SrcLink file="v12:ultralytics/nn/tasks.py" :line="298" /> `DetectionModel`
- 分配器：<SrcLink file="v12:ultralytics/utils/tal.py" :line="14" /> `TaskAlignedAssigner`
- 损失：<SrcLink file="v12:ultralytics/utils/loss.py" :line="157" /> `v8DetectionLoss`

A2C2f / ABlock / AAttn 三件套的语义解释见本页正文；训练链路与推理链路的全景见 [code-walkthrough](/versions/v12/code-walkthrough)；结构配置见 [yolov12.yaml 源码页](/versions/v12/source/yolov12-yaml)；检测头 DW-separable 分支见 [head.py 源码页](/versions/v12/source/head-py)；损失总装见 [loss.py 源码页](/versions/v12/source/loss-py)；分配器见 [tal.py 源码页](/versions/v12/source/tal-py)。
