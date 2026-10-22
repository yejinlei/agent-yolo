# YOLOv13 训练技巧

> **v1~v12 读者路标**：v13 的官方配方**几乎逐字复制 v12**——README L176-186 与 [v12 tricks §1](/versions/v12/tricks#1-官方配方600-epoch--batch-256-trick-official-recipe) 只差三处：`epochs=600` 相同、`batch=256` 相同、`mixup` 与 `copy_paste` 的分档注释从「五档」缩到「四档」（v13 无 m 档）。真正属于 v13 的「训练技巧」不在配方层，而在**结构层**：论文消融 TABLE IV 给出的 `M`（AdaHyperedgeGen 头数）值域从 **2 → 16**、TABLE III 的 FullPAD 三档独立消融、**C3AH 的 `num_heads = c_ // 16` 硬约束**、**tasks.py 对 HyperACE 强制 `n=1`**。本页 8 节盘点这些训练侧事实。

## 全部条目速查

| 条目 | 位置 | 说明 | 推理是否零开销 |
|------|------|------|----------------|
| 官方配方 600 epoch / batch 256 | README L178-185 | 与 v12 逐键相同 | 是（纯配置） |
| **分档 mixup / copy_paste（四档）** | README L183-184 | n 0.0/0.1 · s 0.05/0.15 · l 0.15/0.5 · x 0.2/0.6（**无 m 档**） | 是（训练期） |
| 分档 scale 增强 | README L181 | n 档 0.5；s/l/x 全 0.9 | 是（训练期） |
| **AdaHyperedgeGen `num_heads`** | block.py L1598 | yaml 传 `8`（`[512, 8, True, True, 0.5, 1, "both"]`）；论文消融 2→16 单调收敛到 8 | 是（结构） |
| **C3AH `num_heads = c_ // 16`** | block.py L1734 | 隐藏通道必须是 16 的倍数；`assert c_ % 16 == 0`（L1733） | 否（建网期） |
| **HyperACE `n` 被强制为 1** | tasks.py L1075-1095 | yaml 写 `2`，`parse_model` 里 `n = 1`——**重复次数不可调** | 否（结构） |
| **DownsampleConv 在 l/x 档强制 Identity** | tasks.py L1080-1093 | L/X 档走 Identity，n/s 档走 AvgPool+Conv(1×1, 2×ch) | 否（结构） |
| **AdaHyperedgeGen `dropout=0.1`** | block.py L1611 | 超边生成的固定 dropout；`num_heads` 从 yaml 传入 | 是（训练期） |
| **FullPAD_Tunnel `gate = nn.Parameter(0.0)`** | block.py L1898 | 门控标量从 0 学起，`out = x[0] + gate * x[1]` | 是（推理期） |
| Detect `legacy` 由建网器赋值 | head.py · tasks.py | **与 v12 相同**——`legacy=False`，DW-separable cv3 | 否（结构） |
| 损失增益 7.5 / 0.5 / 1.5 | default.yaml | 与 v8/v11/v12 逐键相同 | 是（初始化） |
| `optimizer: auto` + `warmup_bias_lr: 0.0` | default.yaml · trainer.py | 与 v12 相同（auto 内部强制置 0） | 是（训练期） |
| TAL `tal_topk=10` 显式 α=0.5 β=6.0 | loss.py L176 | 与 v12 相同 | 是（训练期） |
| close_mosaic 10 | default.yaml L33 | 600 epoch 下相对占比 1.7% | 是（训练期） |

## 1. 官方配方：600 epoch / batch 256 (`trick-official-recipe`)

> **v1~v12 读者路标**：v12 的官方配方是「v8 之后第一次动刀」——600 epoch + 分档增强；**v13 完全沿用**，唯一变化是把「分档注释」从五档缩到四档（**没有 m 档**）。

**动机**：v13 的 HyperACE + FullPAD 让骨干与颈部之间的信息流更复杂，作者沿用 v12 的长训练周期把 AdaHyperedgeGen 的超边权重与 FullPAD 的门控参数训稳——两者都是**动态生成 / 从 0 学起**的参数，需要更长周期收敛。

**官方训练命令**（README L171-186，逐行照抄）：

```python
from ultralytics import YOLO

model = YOLO('yolov13n.yaml')

results = model.train(
  data='coco.yaml',
  epochs=600,
  batch=256,
  imgsz=640,
  scale=0.5,          # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,          # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,     # S:0.15; L:0.5; X:0.6
  device="0,1,2,3",
)
```

**与 v12 官方的逐键对比**：

| 键 | v12 官方（n 档） | v13 官方（n 档） | 差异 |
|----|-----------------|-----------------|------|
| epochs | 600 | **600** | 同 |
| batch | 256 | **256** | 同 |
| imgsz | 640 | **640** | 同 |
| scale | n=0.5 · s/m/l/x=0.9 | n=0.5 · s/l/x=0.9 | **无 m 档** |
| mosaic | 1.0 | **1.0** | 同 |
| mixup | n=0.0 / s=0.05 / m=l=0.15 / x=0.2 | n=0.0 / s=0.05 / l=0.15 / x=0.2 | **无 m 档** |
| copy_paste | n=0.1 / s=0.15 / m=0.4 / l=0.5 / x=0.6 | n=0.1 / s=0.15 / l=0.5 / x=0.6 | **m=0.4 消失** |

**推理开销**：零开销。全部是训练期配置。

## 2. 分档增强矩阵：无 m 档，l 档与 v12 m 档错开 (`trick-per-scale-aug`)

> **v1~v12 读者路标**：v12 是五档差异化（n/s/m/l/x），**v13 缩到四档**（n/s/l/x）——yaml 里根本没定义 `m` 档（`yolov13.yaml` L4-L7），README 分档注释也去掉了 m。

**四档矩阵**（README L181-184 的注释值整理）：

| 档位 | scale | mixup | copy_paste | 说明 |
|------|-------|-------|------------|------|
| n | 0.5 | 0.0 | 0.1 | 最小模型，几乎不用 MixUp |
| s | 0.9 | 0.05 | 0.15 | 弱 MixUp |
| l | 0.9 | 0.15 | 0.5 | MixUp 与 CopyPaste 都上调 |
| x | 0.9 | 0.2 | 0.6 | 最强增强 |

**与 v12 五档的差异**：

- **v12 m 档（mixup=0.15, copy_paste=0.4）** 在 v13 消失——**m 档消失后，v13 l 档的 CopyPaste 从 0.5 直接跳到 v12 m 档的 0.5**（v12 l 档的 0.5 保留），但 MixUp 也从 0.15 保留；
- **v13 l 档 ≈ v12 l 档**：CopyPaste 与 MixUp 完全一致，只有 `scale` 增强相同；
- **v13 x 档 ≈ v12 x 档**：完全一致。

**为什么没有 m 档**（本站推断，作者未在 README 或论文中明说）：

1. **v13 的 scales 结构里 `max_channels` 从 s 档的 1024 掉到 l 档的 512**——`yolov13.yaml` L6 `l: [1.00, 1.00, 512]`，L7 `x: [1.00, 1.50, 512]`；
2. **n 与 s 档** max_channels=1024（沿用 v12），但 l 与 x 档**上限被压到 512**——这是「DS 体系下宽度不再需要大」的设计选择；
3. 若插入一个 m 档，会卡在 n/s 与 l/x 的 `max_channels` 断层之间，配方不好调，作者索性去掉。

**推理开销**：零开销。

## 3. AdaHyperedgeGen `num_heads` 的论文消融（TABLE IV） (`trick-hyperace-heads`)

> **v1~v12 读者路标**：v8~v12 的注意力头数要么按 `c // 64`（v11 C2PSA）、要么按 `c // 32`（v12 A2C2f）**硬编码推导**，从未给出消融；**v13 是 YOLO 系列第一个在论文里给出注意力头数消融表的版本**——TABLE IV 扫 2/4/8/16，结论是**头数越大 AP 略高但差异极小（0.3 AP）**。

**代码事实**（<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1549" :end="1599" />）：

```python
class AdaHyperedgeGen(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        self.dim = dim
        self.num_heads = num_heads
        self.scale = self.dim ** (-0.5)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.context_net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
```

- **`num_heads=8`** 是 yaml 直接传入的（`yolov13.yaml` L22：`[-1, 8, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]`），**不是推导**；
- **多头相似度 = softmax(dim=1)**：`P = P_0 + context_net(...)`，`P` 形状 `(B, num_heads, S, S)`，多头独立 softmax 归一（每超边一个分布）；
- **`dropout=0.1` 是硬编码常量**，yaml 层无法覆盖。

**论文 TABLE IV 消融**（YOLOv13-S，M = num_heads）：

| M | AP50:95 | 相对 M=8 差 |
|---|---------|-----------|
| 2 | 47.8 | −0.2 |
| 4 | 47.9 | −0.1 |
| **8** | **48.0** | 基线 |
| 16 | 48.1 | +0.1 |

**四个要点**：

1. **收益单调但幅度极小**：从 2 → 16，AP 只涨 0.3——远小于 FullPAD 三组件的 0.5~0.9 消融增益；
2. **8 头是「甜点」**：8 → 16 只多 0.1 AP，但计算量翻倍（`P` 变成 2× 大小）——**默认 8 是精度与效率的平衡点**；
3. **`num_heads` 与 `dim` 独立**：`AdaHyperedgeGen` 里没有像 v12 A2C2f 那样的 `num_heads = c_ // 32` 推导，**yaml 直接写数**；
4. **`num_heads` 是 HyperACE 层的直接参数**：`yolov13.yaml` L22 的第 2 个数字就是它，改起来不用碰源码。

**推理开销**：**零**（`num_heads` 是结构常量），但**大 num_heads 增加超边矩阵 `P` 的存储**：`num_heads=16` 时 `P` 是 2× 大小。

## 4. C3AH 的通道约束：`c_ % 16 == 0` (`trick-c3ah-constraint`)

> **v1~v12 读者路标**：v12 的 A2C2f 断言 `c_ % 32 == 0`（[tricks §4](/versions/v12/tricks#4-a2c2f-的通道约束c_-32--0-trick-channel-constraint)）；**v13 的 C3AH 断言 `c_ % 16 == 0`**——**门槛减半**，因为 C3AH 内嵌的是超图消息传递而不是标准 Transformer 注意力，头数约束更松。

**代码事实**（<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1721" :end="1741" />）：

```python
class C3AH(C3):
    def __init__(self, c1, c2, n=1, e=0.5, g=1, num_heads=8, shortcut=True):
        super().__init__(c1, c2, n, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 16 == 0, "C3AH hidden channels must be multiple of 16"
        self.num_heads = num_heads  # 或从 dim // 16 推导
        self.m = nn.Sequential(*[C3AHBlock(c_, num_heads) for _ in range(n)])
```

- **`c_` 必须是 16 的倍数**——否则建网直接 `AssertionError`；
- **`num_heads` 是独立参数**（默认 8）——与 C3AH 通道无关，由 HyperACE 层传入（L1804-1838 里 `C3AH(c_, n=..., num_heads=num_heads)`）；
- **`C3AH` 继承 `C3`**——CSP 骨架完全沿用 v8 C3，只是把内部的 Bottleneck 换成 `C3AHBlock`（内含 `AdaHGComputation`）。

**v13 里 C3AH 的实际出现**（`yolov13.yaml` L22 的 HyperACE 内部）：

- **yaml 层看不到 C3AH**——它是 HyperACE 的**内部子块**，`yolov13.yaml` 里 HyperACE 出现 1 次（L22），HyperACE 内部构造 2 个 C3AH（`L=1` 表示 1 个 DSC3k 并联）；
- **`num_heads=8` 从 HyperACE 参数传入**：`yolov13.yaml` L22 `[512, 8, True, True, 0.5, 1, "both"]` 的第 2 位就是 num_heads；
- **`L=1` 是 DSC3k 的并联数**：`yolov13.yaml` L22 的第 6 位是 `1`，表示 HyperACE 里并联 1 个 DSC3k。

**具体影响**：

- **n/s 档**：`max_channels=1024`，`HyperACE` 的 `c_` 由 `512 × 0.5 = 256` 得出（yaml L22 是 512），`256 % 16 == 0` ✓；
- **l/x 档**：`max_channels=512`，同样得到 `c_=256` ✓；
- **自定义 `e` 容易崩**：想给 C3AH 设 `e=0.25`（浅层轻量），`512 × 0.25 = 128 % 16 == 0` ✓，但 `c2=256` 时 `256 × 0.25 = 64 % 16 == 0` ✓——**比 v12 的 32 断言宽松一倍**。

**推理开销**：零开销——但属于**建网期硬约束**，配置错直接崩。

## 5. HyperACE `n` 被 `parse_model` 强制为 1 (`trick-hyperace-n-1`)

> **v1~v12 读者路标**：v8 C2f / v11 C3k2 / v12 A2C2f 的 `n` 都是**从 yaml 传入**（yaml L22 `2` 就是 repeats=2）；**v13 的 HyperACE 在 `parse_model` 里被强制覆盖为 1**——**yaml 写了 `2` 也没用**。

**代码事实**（<SrcLink file="v13:ultralytics/nn/tasks.py" :line="1075" :end="1095" />）：

```python
if m is HyperACE:
    # args: [c2, num_heads, fuse, channel_adjust, e, L, mode]
    args.insert(2, 1)  # n = 1
    n = 1
    ...
```

yaml L22 `[-1, 8, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]` 的第二列是 `8`（repeats），但 `parse_model` 把它替换为 `1`——**HyperACE 只出现一次、重复一次**。

**为什么这样设计**：

1. **HyperACE 是骨干融合的核心模块**——yaml 里只在 head 第 1 块（L22）出现一次，**重复多次会浪费算力**；
2. **论文中 HyperACE 只有一个实例**（Fig. 2 骨架图）；
3. **`L=1` 是 DSC3k 并联数**，通过另一个参数控制内部并联，不是 repeats——**作者用「并行深度」代替「串行重复」**。

**具体影响**：

- **yaml 写 `n=2` 会被静默改成 `n=1`**——不报错，只是建出来的模型和你预期的不一样；
- **想改 HyperACE 的 repeats 必须改源码**（tasks.py L1080）——**这是 v13 唯一一个「yaml 层的数不生效」的特例**；
- **v12 的 A2C2f `n` 从 yaml 生效**（见 [v12 tricks §2.1](/versions/v12/tricks#2-分档-mixup--copy_paste-矩阵v12-的核心配方创新-trick-per-scale-aug)）——**v13 与 v12 在这点上行为不一致**。

**推理开销**：零开销——但属于**建网期行为陷阱**。

## 6. DownsampleConv 在 l/x 档强制 Identity (`trick-downsample-identity`)

> **v1~v12 读者路标**：v8~v12 的下采样是标准 `Conv(..., stride=2)`；**v13 引入 `DownsampleConv`**——`AvgPool2d(2) + Conv(1×1, 2×ch)`，**但在 l/x 档被 `parse_model` 强制改成 Identity**。

**代码事实**（<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1858" :end="1889" />）：

```python
class DownsampleConv(nn.Module):
    def __init__(self, c1, c2=None, use_avgpool=True):
        super().__init__()
        self.use_avgpool = use_avgpool
        if use_avgpool:
            self.pool = nn.AvgPool2d(2, stride=2)
            self.conv = nn.Conv2d(c1, c1, 1, 1, 0)   # 通道不变
        else:
            self.pool = None
            self.conv = None
        self.c1 = c1
        self.c2 = c2 if c2 is not None else c1
```

**`parse_model` 的档位特判**（<SrcLink file="v13:ultralytics/nn/tasks.py" :line="1080" :end="1093" />）：

```python
if m is DownsampleConv:
    if scale in "lx":
        args = [1]  # 传 use_avgpool=False → Identity
    else:
        args = [2]  # 传 use_avgpool=True
```

**实际五档（四档）行为**：

| 档位 | `use_avgpool` | `out` 计算 | 参数 |
|------|---------------|-----------|------|
| n / s | True | `AvgPool2d(2)(x) → Conv(1×1, c1)(x)` | `1 × c1 × c1` |
| l / x | **False** | **Identity（原样返回 x）** | 0 |

**为什么 l/x 档走 Identity**：

1. **l/x 档 `max_channels=512`**——下采样前后通道数不变（yaml L11 `[-1, 1, Conv, [128, 3, 2, 1, 2]]` 的 128 是硬编码），不需要通道加倍；
2. **AvgPool + Conv 会引入额外参数与延迟**——l/x 档本身算力就大，去掉这一层是「省参数」的工程选择；
3. **论文里也没说为什么**——这是 `parse_model` 里的**静默决策**，yaml 层看不出来。

**具体影响**：

- **n/s 与 l/x 的 DownsampleConv 权重结构不同**——n/s 有 `pool.conv.weight` 键，l/x 是纯 Identity（无参数）；
- **跨档迁移权重必须 `strict=False`**：n/s 权重里有 `model.23.pool.conv.weight`，l/x 没有；
- **l/x 档的 P3/P4 从骨干拉到 head 时不做下采样**——这有点违反 FPN 直觉（一般大模型需要更多下采样来聚合大物体），但作者选择「让骨干自己下采样」；
- **yaml L24 `[-1, 1, DownsampleConv, []]`** 空 args 是特例——`parse_model` 会自己补。

**推理开销**：l/x 档零开销（Identity）；n/s 档多一个 `AvgPool + 1×1 Conv`。

## 7. FullPAD_Tunnel `gate = 0` 从 0 学起 (`trick-fullpad-gate`)

> **v1~v12 读者路标**：v8~v12 的颈部融合都是**固定 Cat / Add**——没有可学习门控；**v13 是 YOLO 系列第一个引入「可学习门控融合」的版本**，且**门控初始化为 0**——这是 LayerScale 思想的再次应用（呼应 v12 的 A2C2f gamma=0.01）。

**代码事实**（<SrcLink file="v13:ultralytics/nn/modules/block.py" :line="1891" :end="1916" />）：

```python
class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2
        return x[0] + self.gate * x[1]
```

**三个要点**：

1. **`gate` 是标量参数**（0.0 初始化）——`torch.tensor(0.0)` 生成 0 维张量，`requires_grad=True`；
2. **`out = x[0] + gate * x[1]`**——**主通路是 `x[0]`，副通路被 gate 加权**；gate=0 时完全等价于 `identity(x[0])`；
3. **7 处 FullPAD_Tunnel 全部共享相同的构造**——每条连接的 gate 独立训练，不是共享参数（yaml L25/L26/L27/L32/L38/L43/L48 各一个 `FullPAD_Tunnel` 实例）。

**训练含义**：

- **初始等价于「不加信息」**——第一个 epoch，`out = x[0]`，训练损失只看主通路；gate 从梯度反推学到「加入 `x[1]` 的信息权重」；
- **反向传播路径**：`dL/dgate = (x[1]) * (dL/dout)`——`x[1]` 有信号时 gate 才会更新，**否则 gate 一直停在 0**（"死门"）；
- **论文 TABLE III 消融**：000→47.1 / 100→47.8 / 010→47.6 / 001→47.7 / 111→48.0——**说明所有 7 个 gate 都能被有效训出非 0 值**，没有死门；
- **训练末期可检查 `gate.detach().item()`** 判断哪条连接被激活——这是一个「免费的模型可解释性工具」，v8~v12 都没有。

**推理开销**：**零**——`gate * x[1]` 是标量乘张量，可被 TensorRT 常量折叠。

## 8. 与 v8 / v9 / v10 / v11 / v12 配方完整对比 (`trick-compare`)

> **v1~v12 读者路标**：把六代配方摆在一起，v13 的位置很清楚——**配方层几乎逐字复制 v12**，**结构层第一次做「可学习门控融合」**。

| 项目 | v8 | v9 | v10 | v11 | v12 | **v13** |
|------|----|----|-----|-----|-----|--------|
| 损失类 | v8DetectionLoss | v8 + 0.25 辅助 | v10DetectLoss | v8DetectionLoss | v8DetectionLoss | **v8DetectionLoss（与 v11/v12 相同）** |
| 损失权重 | 7.5/0.5/1.5 | 7.5/0.5/1.5 + 0.25 辅 | 6 项 × 1.0 | 7.5/0.5/1.5 | 7.5/0.5/1.5 | **7.5/0.5/1.5** |
| TAL topk | 10 | 10+10 | 10+1 | 13（类默认） | 10（显式） | **10（显式，与 v12 相同）** |
| TAL α / β | 0.5/6.0 | 0.5/6.0 | 0.5/6.0 | 1.0/6.0 | 0.5/6.0 | **0.5/6.0** |
| 优化器 | auto（→SGD） | auto | auto | auto | auto | **auto（→SGD）** |
| lr0 / lrf | 0.01/0.01 | 0.01/0.01 | 0.01/0.01 | 0.01/0.01 | 0.01/0.01 | **0.01/0.01** |
| warmup_bias_lr | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | **0.0** |
| epochs（官方） | 100/500 | — | 500 | 100 | 600 | **600** |
| batch（官方） | 16~64 | — | — | — | 256 | **256** |
| Mosaic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| MixUp | 0.0 | 0.0 | 0.0 | 0.0 | 分档 0.0~0.2 | **分档 0.0~0.2（四档）** |
| CopyPaste | 0.0 | 0.0 | 0.0 | 0.0 | 分档 0.1~0.6 | **分档 0.1~0.6（四档）** |
| scale 增强 | 0.5 | 0.5 | 0.5 | 0.5 | 分档 0.5/0.9 | **分档 0.5/0.9（四档）** |
| close_mosaic | 10 | 10 | 10 | 10 | 10 | **10** |
| AMP | True | True | True | True | True | **True** |
| **FlashAttention** | — | — | — | — | 硬依赖 | **硬依赖（沿用 v12）** |
| **timm** | — | — | — | — | 硬依赖 | **硬依赖** |
| **num_heads 消融** | — | — | — | — | — | **✓ TABLE IV（2/4/8/16）** |
| **可学习门控融合** | — | — | — | — | — | **✓ FullPAD_Tunnel（7 处）** |
| 训练入口类 | YOLO | YOLOv9 | YOLOv10 | YOLO | YOLO | **YOLO** |

**四句话总结**：

1. **v13 的训练配方几乎逐字复制 v12**——600 epoch / batch 256 / auto→SGD / 分档增强，唯一变化是**没有 m 档**；
2. **v13 的结构层第一次有可学习门控**——FullPAD_Tunnel 的 `gate = 0` 从 0 学起，是 LayerScale 思想的第二次应用（第一次是 v12 的 A2C2f gamma=0.01）；
3. **v13 是 YOLO 系列第一个给出注意力头数消融的版本**——TABLE IV 的 M=2/4/8/16 让 `num_heads` 的选择有实验依据，而 v8~v12 都是「拍脑袋」；
4. **v13 有两个 `parse_model` 特例**：HyperACE `n` 强制 1、DownsampleConv 在 l/x 档强制 Identity——**都是 yaml 层看不出来的静默行为**（见 [痛点页 §2/§3](/versions/v13/painpoints)）。

## 9. 训练命令与常见场景 (`trick-commands`)

> **v1~v12 读者路标**：v10 需要 `from ultralytics import YOLOv10`；v11/v12/v13 都回到 `YOLO`——**v13 没有 `YOLOv13` 独立类**，`YOLO("yolov13n.yaml")` 即可。

**官方复现配方**（600 epoch，需要多卡）：

```bash
yolo detect train \
  model=yolov13n.yaml \
  data=coco.yaml \
  epochs=600 batch=256 imgsz=640 \
  scale=0.5 mosaic=1.0 mixup=0.0 copy_paste=0.1 \
  device=0,1,2,3
```

**单卡小数据集微调**（加载官方权重，降 lr、缩 epoch）：

```bash
yolo detect train \
  model=yolov13n.pt \
  data=mydataset.yaml \
  epochs=50 batch=32 imgsz=640 \
  lr0=0.001 patience=15
```

- **官方权重来自 GitHub Release**：`yolov13{n/s/l/x}.pt`（README L155-158），**不是 ultralytics 的资产仓库**——`YOLO("yolov13n.pt")` 会去 `iMoonLab/yolov13` 的 release 下载；
- **微调时不要动 `copy_paste`**：默认 0.1 是 COCO 尺度上的值，小数据集上 CopyPaste 需要分割掩码，检测数据集通常没有——建议显式传 `copy_paste=0.0`；
- **AMP 默认开启**：`amp: True`；A2C2f 内部本来就强制 fp16（沿用 v12），关 AMP 也省不掉这层降精度。

**环境自检**（训练前）：

```python
import torch
from flash_attn.flash_attn_interface import flash_attn_func   # 失败即环境不全
from timm.models.layers import trunc_normal_
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

**推理开销**：以上命令全部是训练期配置；但**环境依赖（flash-attn）在推理期同样存在**（沿用 v12，见 [v12 painpoints §1](/versions/v12/painpoints#1-flashattention-硬依赖导入即崩-pain-flash-attn)）。

## 延伸阅读

- [理论 Concept](/versions/v13/concepts)（HyperACE / FullPAD / DS 三个概念的机制解析）
- [YOLOv13 损失函数](/versions/v13/losses)——v13 无新损失类，沿用 v8 三分量
- [YOLOv13 反向传播推导](/versions/v13/loss-gradient)——HyperACE 与 FullPAD 的梯度路径
- [YOLOv13 痛点追踪](/versions/v13/painpoints)——C3AH 硬断言、HyperACE n 被覆盖、DownsampleConv 静默 Identity
- [YOLOv12 训练技巧](/versions/v12/tricks)——上一代「分档增强首次出现」的对照
- [YOLOv11 训练技巧](/versions/v11/tricks)——训练配方回到 v8 的对照
- [YOLOv8 训练技巧](/versions/v8/tricks)——v13 配方主干的直接出处
- [术语表](/glossary)——Mosaic、MixUp、CopyPaste、EMA、cosine、close_mosaic、FlashAttention、LayerScale、FullPAD 等术语索引
