# YOLOv12 训练技巧

v12 的技巧页有一个**和 v11 完全相反的结论**：v11 是「训练配方一行没改」，v12 则是**v8 之后第一次在配方上动刀**——README 给出的官方训练命令把 epochs 拉到 **600**、batch 提到 **256**，并且 **mixup 与 copy_paste 按 n/s/m/l/x 五档逐档给出不同值**（n 档 mixup=0.0 / copy_paste=0.1，x 档放大到 0.2 / 0.6）。这是本站在 v8~v11 四代「逐键同源」之后第一次记录到**分档差异化配方**。除此之外还有三处工程侧改动：FlashAttention 成为**硬依赖**（`requirements.txt` 直接钉了一个 Linux + CUDA 11 + Python 3.11 的 whl）、A2C2f 的通道数被 `assert c_ % 32 == 0` **硬约束**、`gamma` layer-scale 在 yaml 层**默认不启用**。本页 9 节盘点这些训练侧事实。

## 全部条目速查

| 条目 | 代码位置 | 说明 | 推理是否零开销 |
|-------|----------|------|----------------|
| 官方配方 600 epoch / batch 256 | README L76-86 | 比 default.yaml 出厂（100/16）激进得多 | 是（纯配置） |
| **分档 mixup / copy_paste** | README L81-84 | n 0.0/0.1 · s 0.05/0.15 · m 0.15/0.4 · l 0.15/0.5 · x 0.2/0.6 | 是（训练期） |
| 分档 scale 增强 | README L81 | n 档 0.5；s/m/l/x 全 0.9 | 是（训练期） |
| **FlashAttention 硬依赖** | requirements.txt L3 · block.py L1159-1162 | 缺包直接 `assert False`；版本钉死 Linux+cu11+py311+torch2.2.2 | 否（推理也要，除非走非 CUDA 分支） |
| A2C2f 通道断言 | block.py L1333 | `assert c_ % 32 == 0`——width 缩放与 max_channels 必须让隐藏通道是 32 倍数 | 否（建网期） |
| `num_heads = c_ // 32` 硬编码 | block.py L1336 | head 数不随档位可调，只由隐藏通道推导 | 否（建网期） |
| gamma layer-scale 仅 L/X 档开启 | block.py L1342 · tasks.py L1034-1038 | `parse_model` 给 L/X 档补 `residual=True, mlp_ratio=1.2`；n/s/m 档 gamma=None、mlp_ratio=2.0 | 否（结构） |
| Detect `legacy=False`（DW-separable cv3） | head.py L32 · L46-57 · tasks.py L1058 | `parse_model` 因模型含 C3k2/A2C2f 把 `legacy=False` 写给 Detect | 否（结构） |
| 损失增益 7.5 / 0.5 / 1.5 | default.yaml L99-101 | 与 v8/v10/v11 逐键相同 | 是（初始化） |
| `optimizer: auto` + `warmup_bias_lr: 0.0` | default.yaml L26 · L98 · trainer.py L788 | auto 内部会强制把 warmup_bias_lr 置 0 | 是（训练期） |
| TAL `tal_topk=10` 显式 α=0.5 β=6.0 | loss.py L160 · L176 | 与 tal.py 类默认 topk=13 α=1.0 **不一致** | 是（训练期） |
| close_mosaic 10 | default.yaml L33 | 最后 10 epoch 关 Mosaic | 是（训练期） |

## 1. 官方配方：600 epoch / batch 256 (`trick-official-recipe`)

> **v1~v11 读者路标**：v7 的技巧是论文论证的 [freebies](/glossary#trainable-bag-of-freebies-可训练免费午餐)；v8 把它变成工程默认值（见 [v8 tricks](/versions/v8/tricks)）；v9 加了 0.25 辅助头权重（见 [v9 tricks](/versions/v9/tricks)）；v10 回退到 v8 配方（见 [v10 tricks](/versions/v10/tricks)）；v11 继续沿用 v8 配方、只把 TAL 默认超参从 topk=10 α=0.5 改成 topk=13 α=1.0（见 [v11 tricks §5](/versions/v11/tricks#5-tal-topk13-α10-β60唯一配方级改动-trick-tal-defaults)）；**v12 是四代之后第一次在配方矩阵上做分档差异化**。

**动机**：v12 的卖点是「注意力进骨干也能实时」——AAttn 的计算量比卷积大，需要更长的训练周期把注意力权重训稳；作者用 600 epoch 换精度，同时按档位**放大增强强度**：越大的模型容量越足，越能吃得下更强的 mixup / copy_paste。

**官方训练命令**（README L70-95，逐行照抄）：

```python
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

results = model.train(
  data='coco.yaml',
  epochs=600,
  batch=256,
  imgsz=640,
  scale=0.5,          # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,          # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,     # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)
```

> **注意**：README 只给了 n 档的完整命令 + 其它四档的注释值——**没有给出 s/m/l/x 的完整命令**，也没有说明这四档的 batch 与设备。这是 [痛点页 §4](/versions/v12/painpoints#4-训练配方只给了-n-档完整命令-pain-recipe-incomplete) 记录的可复核性问题。

**与 default.yaml 出厂的差距**（`ultralytics/cfg/default.yaml`）：

| 键 | default.yaml 出厂 | README 官方配方 | 位置 |
|----|------------------|----------------|------|
| epochs | 100 | **600** | L12 |
| batch | 16 | **256** | L15 |
| imgsz | 640 | 640 | L16 |
| scale | 0.5 | n=0.5 / s·m·l·x=0.9 | L110 |
| mosaic | 1.0 | 1.0 | L117 |
| mixup | 0.0 | n=0.0 / s=0.05 / m=l=0.15 / x=0.2 | L118 |
| copy_paste | **0.1** | n=0.1 / s=0.15 / m=0.4 / l=0.5 / x=0.6 | L119 |

<SrcLink file="v12:ultralytics/cfg/default.yaml" :line="12" :end="16" />
<SrcLink file="v12:ultralytics/cfg/default.yaml" :line="117" :end="119" />

**推理开销**：零开销。以上全部是训练期配置，导出的权重与这些超参无关。

## 2. 分档 mixup / copy_paste 矩阵：v12 的核心配方创新 (`trick-per-scale-aug`)

> **v1~v11 读者路标**：v8~v11 四代的增强配置都是「一套值走五档」——Mosaic 1.0 / MixUp 0.0 / CopyPaste 0.0（v11 口径），**档位之间只有 depth/width 不同，增强完全一致**。v12 第一次把增强强度与模型容量绑定。

**矩阵**（README L81-84 的注释值整理）：

| 档位 | scale | mixup | copy_paste | 说明 |
|------|-------|-------|------------|------|
| n | 0.5 | 0.0 | 0.1 | 最小模型，几乎不用 MixUp，只用最轻的 CopyPaste |
| s | 0.9 | 0.05 | 0.15 | 开始引入弱 MixUp |
| m | 0.9 | 0.15 | 0.4 | CopyPaste 大幅上调 |
| l | 0.9 | 0.15 | 0.5 | CopyPaste 继续上调 |
| x | 0.9 | 0.2 | 0.6 | 最强增强 |

**为什么这样设计**（作者未在论文或 README 明说，本站从容量与过拟合的关系推断）：

1. **小模型欠拟合、大模型过拟合**：n 档容量小（2.60M），强增强会让它连训练集都拟合不好，所以 mixup=0.0；x 档 59.22M，容易记住训练集，需要用 0.2 的 MixUp + 0.6 的 CopyPaste 压过拟合；
2. **scale 增强同理**：n 档保持 0.5 的默认缩放抖动，s~x 档统一提到 0.9——大模型能承受更大的尺度变化；
3. **CopyPaste 是 v12 唯一「全线开启」的增强**：从 n 的 0.1 一路加到 x 的 0.6，是五档里跨度最大的键（6×）——说明作者认为**前景粘贴**是对注意力骨干最有效的数据增强。

> **注意 CopyPaste 与 segment 的关系**：`default.yaml` L119 的注释写的是 `(float) segment copy-paste (probability)`——在 ultralytics 主线里该键主要用于分割任务；v12 的检测训练显式把它开到 0.1~0.6，属于**检测任务上的非典型用法**，本站按 README 原样记录，不做语义扩展。

**推理开销**：零开销。增强只在 dataloader 里发生。

## 3. FlashAttention：训练环境的硬门槛 (`trick-flash-attn`)

> **v1~v11 读者路标**：v8~v11 的训练环境只要 `torch + torchvision` 就能跑（CPU 也能慢跑）；**v12 第一次把第三方 CUDA 算子写进硬依赖**——`block.py` 在**模块导入阶段**就 `import flash_attn`，失败直接 `assert False`。

**代码事实**（<SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1159" :end="1163" />）：

```python
try:
    from flash_attn.flash_attn_interface import flash_attn_func
except Exception:
    assert False, "import FlashAttention error! Please install FlashAttention first."
from timm.models.layers import trunc_normal_
```

- `try/except` **不是**「有就用、没有就退化」——except 分支直接 `assert False`，**导入失败即进程崩溃**，连 `YOLO("yolov12n.yaml")` 都建不起来；
- 同一段还硬依赖 `timm.models.layers.trunc_normal_`（ABlock 的 ViT 风格初始化，见 §5）——`timm` 也是必装项；
- 因此 **v12 无法在纯 CPU 机器上建网**，也无法在没编好 flash-attn 的 GPU 机器上建网。

**requirements.txt 钉死的环境**（仓库 `requirements.txt` L1-L4）：

| 行 | 内容 | 含义 |
|----|------|------|
| L1 | `torch==2.2.2` | 固定 torch 小版本 |
| L2 | `torchvision==0.17.2` | 与 torch 2.2.2 配套 |
| L3 | `flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` | **预编译 whl**：CUDA 11 + torch 2.2 + C++11 ABI FALSE + Python 3.11 + **仅 Linux x86_64** |
| L4 | `timm==1.0.14` | `trunc_normal_` 的来源 |

**实践含义**：

- **Windows / macOS 用户装不上**：whl 名里 `linux_x86_64` 写死了平台，Windows 用户必须自己源码编译 flash-attn（需要匹配 CUDA 与 torch 版本，编译耗时长）；
- **Python 版本锁死 3.11**：`cp311` 是 Python 3.11 的 ABI tag，3.10/3.12 都得自己编；
- **`cxx11abiFALSE`**：对应老 ABI 编译选项，装错 ABI 会在运行时抛 `undefined symbol`；
- **训练与推理都吃这个依赖**：`AAttn.forward` 的 CUDA 分支（<SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1217" :end="1222" />）对 q/k/v 调 `.half()`——**即使你用 AMP=False 全 fp32 训练，注意力内部也被强制降到 fp16**。

**训练命令**（先装环境再训）：

```bash
# 1) 按 requirements.txt 装（Linux + CUDA 11 + Python 3.11）
pip install -r requirements.txt

# 2) 训练
yolo detect train model=yolov12n.yaml data=coco.yaml epochs=600 batch=256 imgsz=640
```

**推理开销**：**非零**——AAttn 在推理路径上同样调用 flash-attn（CUDA 分支），所以导出与部署也要处理该算子（见 [痛点页 §1](/versions/v12/painpoints#1-flashattention-硬依赖导入即崩-pain-flash-attn)）。

## 4. A2C2f 的通道约束：`c_ % 32 == 0` (`trick-channel-constraint`)

> **v1~v11 读者路标**：v8 的 C2f / v11 的 C3k2 对隐藏通道**没有任何整除断言**——`c_ = int(c2 * e)` 取整后直接用；v10 的 C2fCIB 也没有；**v12 的 A2C2f 第一次加了硬断言**。

**代码事实**（<SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1330" :end="1336" />）：

```python
c_ = int(c2 * e)  # hidden channels
assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
# num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
num_heads = c_ // 32
```

两条硬规则：

1. **`c_` 必须是 32 的倍数**——否则建网直接 `AssertionError: Dimension of ABlock be a multiple of 32.`；
2. **`num_heads` 完全由 `c_` 推导**——`c_ // 32`，没有独立旋钮。注释里那行被注掉的「`c_//64` 若 ≥2 则用 `c_//64`」说明作者试过更少的 head，最终选了固定 `//32`。

**实际五档的取值**（`e=0.5`，`c2` 先被 `max_channels` 与 width 缩放）：

| 档位 | yaml c2 | width | 缩放后 c2 | `c_ = c2×0.5` | `num_heads` | head_dim |
|------|---------|-------|-----------|---------------|-------------|----------|
| n | 512 | 0.25 | 128 | 64 | 2 | 32 |
| s | 512 | 0.50 | 256 | 128 | 4 | 32 |
| m | 512 | 1.00 | 512 | 256 | 8 | 32 |
| l | 512 | 1.00 | 512 | 256 | 8 | 32 |
| x | 512 | 1.50 | 768 | 384 | 12 | 32 |

- **`head_dim` 恒为 32**——因为 `num_heads = c_ // 32` 与 `head_dim = dim // num_heads` 互为逆运算，**任何档位下每头都是 32 维**；这也是 §1 断言存在的原因（`c_` 不是 32 倍数会出现非整数 head 数）；
- **x 档是唯一 `num_heads=12` 的配置**——`max_channels: 512` 在 x 档对 A2C2f 不封顶（512×1.5=768），所以 x 档注意力头数比 l 档多 50%；
- **c_ 的路径依赖**：`c2` 来自 yaml（512/1024/256），先经 `make_divisible(width × c2, 8)` 与 `max_channels` 封顶，再乘 `e=0.5`——**任一步改动都可能让断言失败**。

**自定义模型时的注意事项**：

- 若你想给 A2C2f 设 `e=0.25`（v11 C3k2 在浅层常用 0.25），必须保证 `c2×width×0.25` 是 32 的倍数；
- 若你想改 `max_channels`，注意 A2C2f 出现在 yaml L23/L25/L31/L35/L39 **五处**，改一处封顶值会影响这五处的 `c_`；
- **`area` 与 head 数无关**：yaml 里 `area=4`（P4）与 `area=1`（P5）只影响特征图切分块数，不影响 `num_heads`（见 [concepts §2](/versions/v12/concepts#2-aattnarea-attention-与-flashattention-硬依赖-concept-aattn)）。

**推理开销**：零开销——但属于**建网期硬约束**，配置错直接崩。

## 5. gamma layer-scale：只有 L/X 档开启 (`trick-gamma-off`)

> **v1~v11 读者路标**：v8 的 C2f 有 `shortcut` 但**没有 layer-scale**；v11 的 C3k2 也没有；**v12 的 A2C2f 第一次引入 `gamma` 参数（layer-scale）**——而且**不是全关**：`parse_model` 会给 L/X 档自动补上 `residual=True`，因此 **L/X 档 5 处 A2C2f 都有 gamma，n/s/m 档全部没有**。

**代码事实**（<SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1341" :end="1354" />）：

```python
init_values = 0.01  # or smaller
self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
...
def forward(self, x):
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    if self.gamma is not None:
        return x + (self.gamma * self.cv2(torch.cat(y, 1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return self.cv2(torch.cat(y, 1))
```

**关键在 `parse_model` 的档位特判**（<SrcLink file="v12:ultralytics/nn/tasks.py" :line="1034" :end="1038" />）：

```python
if m is A2C2f:
    legacy = False
    if scale in "lx":  # for L/X sizes
        args.append(True)    # residual=True
        args.append(1.2)     # mlp_ratio=1.2
```

yaml 里 A2C2f 只写 `[c2, a2, area]`，但 `parse_model` 会在 **L/X 档**把它补成 `[c1, c2, n, a2, area, True, 1.2]`，即 `residual=True, mlp_ratio=1.2`；n/s/m 档保持 5 个参数，`residual=False, mlp_ratio=2.0`（A2C2f 自己的默认值）。

**五档真实配置**：

| 档位 | `residual` | `gamma` | `mlp_ratio` | A2C2f 最外层残差 |
|------|-----------|---------|-------------|------------------|
| n / s / m | False | **None（不创建）** | **2.0**（A2C2f 默认） | 无 |
| l / x | **True** | **0.01 初始化的 `c2` 维向量** | **1.2** | **有（带 layer-scale）** |

三个要点：

1. **`gamma` 只在 `a2 and residual` 都真时创建**——n/s/m 档 `residual=False` → `gamma=None`，`forward` 走最后的 `return self.cv2(torch.cat(y, 1))`，**A2C2f 是纯前馈块**，残差只存在于 ABlock 内部；
2. **L/X 档走另一条路径**——`gamma` 是 `0.01 × ones(c2)` 的可学习向量，`forward` 返回 `x + gamma ⊙ cv2(cat(y))`，**A2C2f 变成带 layer-scale 的残差块**；
3. **`mlp_ratio` 也跟着档位变**——n/s/m 用 A2C2f 默认的 2.0，L/X 被 `parse_model` 显式传 1.2。**同一个模块在五档里其实有两种 FFN 宽度**（ABlock 自己的默认值是 1.2，只有被 A2C2f 显式传参时才会变）。

**与此配套的初始化**（<SrcLink file="v12:ultralytics/nn/modules/block.py" :line="1284" :end="1291" />）：

```python
self.apply(self._init_weights)

def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
```

- **ABlock 内所有 Conv2d 都用截断正态 `std=0.02`**——这是 ViT 的标准初始化，与 ultralytics 主线 Detect/Conv 的 `kaiming_uniform` 完全不同；
- **`nn.Conv2d` 是 `Conv` 类的基类**——`isinstance(m, nn.Conv2d)` 会命中 ABlock 内**所有** `Conv` 模块，包括 AAttn 的 `qkv` / `proj` / `pe`；
- **`m.bias` 全部置 0**。

**训练含义**：

- **L/X 档的残差是「作者替用户开的」**：yaml 层看不出来（`[512, True, 4]` 五档写法完全一样），只有读了 `parse_model` 的 L1036 才知道；
- **这解释了 L/X 档参数量的跳变**：yaml 注释里 l 档 26.45M / x 档 59.22M，其中包含每处 A2C2f 的 `c2` 维 gamma 向量（5 处 × 512/1024）——量级不大，但**权重键不同**；
- **n/s/m 与 l/x 的权重完全不能互换**：`gamma` 键是否存在决定了 state_dict 结构，跨档加载必须 `strict=False`。

**推理开销**：L/X 档多一个逐通道乘法（可忽略）；n/s/m 档零开销。

## 6. Detect `legacy=False`：DW-separable 分类头的训练含义 (`trick-dw-head`)

> **v1~v11 读者路标**：v8 的 Detect 头 `cv3` 是三段全 3×3 卷积（`Conv(x,c3,3) → Conv(c3,c3,3) → Conv2d(c3,nc,1)`）；v10 的 `v10Detect` 用 depthwise-separable 重写；**v11 的 `Detect` 已经是 DW-separable（无开关）**；**v12 给 `Detect` 补了 `legacy` 类属性（默认 `False`）**，由 `parse_model` 显式赋值 `m.legacy = legacy`——`legacy=True` 才切回 v8 全卷积，默认行为与 v11 相同，yaml 里仍只写 `Detect`。

**代码事实**（<SrcLink file="v12:ultralytics/nn/modules/head.py" :line="32" /> · <SrcLink file="v12:ultralytics/nn/modules/head.py" :line="46" :end="57" />）：

```python
legacy = False  # backward compatibility for v3/v5/v8/v9 models
...
self.cv3 = (
    nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
    if self.legacy
    else nn.ModuleList(
        nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1),
        )
        for x in ch
    )
)
```

**两条分支的结构差异**：

| 分支 | 结构 | 层数 | 感受野 | 参数（大致） |
|------|------|------|--------|------------|
| `legacy=True`（v8/v11） | `Conv(x,c3,3) → Conv(c3,c3,3) → Conv2d(c3,nc,1)` | 3 段 | 5×5 等效 | 较高（全卷积） |
| `legacy=False`（v12 默认） | `DWConv(x,x,3) → Conv(x,c3,1) → DWConv(c3,c3,3) → Conv(c3,c3,1) → Conv2d(c3,nc,1)` | **5 段** | 5×5 等效 | **显著更低**（DW 只做通道内卷积） |

- **`cv2` 回归分支没变**：仍是 `Conv(x,c2,3) → Conv(c2,c2,3) → Conv2d(c2,4·reg_max,1)`——**`legacy` 只影响分类分支 `cv3`**；
- **`legacy` 由建网器赋值**（<SrcLink file="v12:ultralytics/nn/tasks.py" :line="1053" :end="1058" />）：`parse_model` 里 `legacy = True` 是初值，遇到 `C3k2`（<SrcLink file="v12:ultralytics/nn/tasks.py" :line="1030" :end="1033" />）或 `A2C2f`（L1034-1035）就置 `False`，最后 `m.legacy = legacy` **写在 Detect 实例上**——v12 的 yaml 含 C3k2 与 A2C2f，所以 `legacy=False`；
- **`bias_init` 仍然生效**（<SrcLink file="v12:ultralytics/nn/modules/head.py" :line="133" :end="140" />）：`b[-1]` 在两条分支里都是最后一个 `nn.Conv2d(c3, nc, 1)`，所以 `log(5/nc/(640/s)²)` 的偏置初始化路径不变；
- **`yaml` 层完全看不出这个差异**：`yolov12.yaml` 最后一行只写 `Detect`，不写 `legacy`——**换头是 `parse_model` 的隐式赋值，不是配置行为**（见 [concepts §5](/versions/v12/concepts#5-detect-dw-separable-分支yaml-层面无反映的隐性差异-concept-detect-dw-separable)）。

**训练含义**：

- **参数更少、梯度更稀疏**：DWConv 的每个输出通道只依赖单个输入通道，反向传播时 `cv3` 的梯度通路比全卷积窄——前几轮的分类头收敛可能略慢于 v8 头；
- **不能靠改 yaml 切回 legacy**：`Detect` 不接收 `legacy` 参数（它是**类属性**），要改只能改 Python 源码或继承覆写；
- **官方权重是 `legacy=False`**：加载 `yolov12n.pt` 时 `cv3` 的结构必须与源码默认值一致，**改了默认值就加载不了官方权重**（`cv3.0.0.weight` 等键形状不匹配）。

**推理开销**：**降低**——DW-separable 分类头 FLOPs 与参数量都低于全卷积，且不引入额外算子（`DWConv` 由 `Conv` + `groups` 实现，导出后是标准 `Conv` 节点）。

## 7. 优化器 auto 与 `warmup_bias_lr: 0.0` (`trick-optimizer`)

> **v1~v11 读者路标**：v6 起是 SGD + cosine；v8~v11 的 `default.yaml` 都写 `optimizer: auto`（auto 在迭代数 >10000 时选 SGD，否则选 AdamW）；v11 的 `warmup_bias_lr` 是 `0.1`，**v12 改成 `0.0`**。

**代码事实**：

| 键 | v12 值 | 位置 | 与 v11 差异 |
|----|--------|------|------------|
| `optimizer` | `auto` | default.yaml L26 | 同 |
| `lr0` | 0.01 | L92 | 同 |
| `lrf` | 0.01 | L93 | 同（终点 lr = 1e-4） |
| `momentum` | 0.937 | L94 | 同 |
| `weight_decay` | 0.0005 | L95 | 同 |
| `warmup_epochs` | 3.0 | L96 | 同 |
| `warmup_momentum` | 0.8 | L97 | 同 |
| **`warmup_bias_lr`** | **0.0**（注释留 `0.1`） | L98 | **v11 是 0.1** |
| `cos_lr` | False | L32 | 同（`lrf` 仍在 cosine 路径上生效） |

**为什么改成 0.0**：`optimizer=auto` 在 `build_optimizer` 里有一段硬编码（<SrcLink file="v12:ultralytics/engine/trainer.py" :line="779" :end="788" />）：

```python
if name == "auto":
    ...
    name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
    self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
```

- **auto 模式内部会把 `warmup_bias_lr` 强制改成 0.0**——v12 的 `default.yaml` 把这行显式写成 `0.0` 并保留注释 `# 0.1`，**等于把运行时行为前置到配置文件**；
- **600 epoch × COCO 的迭代数远超 10000** → auto 会选 **SGD（lr=0.01, momentum=0.9）**——注意 `momentum` 从 `default.yaml` 的 0.937 被 auto **覆盖成 0.9**；
- **若用 AdamW 路径（迭代数 ≤10000）**：`lr0` 改为 `lr_fit = 0.002×5/(4+nc)`——COCO 80 类时约 1.19e-4。

**训练日志预期**（600 epochs，COCO80，batch 256，imgsz 640，v12-n）：

- **前 3 epochs（warmup）**：因为没有 bias lr 抬升，box/cls loss 下降比 v8 平缓——分类头的 `log(5/nc/(640/s)²)` 偏置从第一轮就是「每格 0.01 目标」的先验；
- **3~300 epochs（主下降）**：Mosaic 全开，注意力权重逐步从「近似恒等」训成「区域注意力」；
- **300~590 epochs（精调）**：mAP50-95 缓慢爬升；
- **590~600 epochs（close_mosaic）**：Mosaic 关闭，定位指标最后一跳。

**推理开销**：零开销。

## 8. close_mosaic 与数据增强出厂值 (`trick-close-mosaic`)

> **v1~v11 读者路标**：`close_mosaic` 从 v8 起引入，v8.1+ 出厂即 10；v9/v10/v11 沿用 10；**v12 也是 10**——600 epoch 下这 10 个 epoch 只占 1.7%（v8 的 100 epoch 下是 10%）。

**出厂增强值**（`ultralytics/cfg/default.yaml` L105-119）：

| 增强 | 概率 / 范围 | v12 位置 | 与 v11 差异 |
|------|------------|----------|------------|
| Mosaic | 1.0 | L117 | 同 |
| MixUp | 0.0（官方配方分档上调） | L118 | 同 |
| **CopyPaste** | **0.1** | L119 | **v11 是 0.0** |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | L105-107 | 同 |
| degrees | 0.0 | L108 | 同 |
| translate | 0.1 | L109 | 同 |
| scale | 0.5（官方配方 s~x 档上调到 0.9） | L110 | 同 |
| shear / perspective | 0.0 / 0.0 | L111-112 | 同 |
| flipud / fliplr | 0.0 / 0.5 | L114-115 | 同 |
| `copy_paste_mode` | `"flip"` | L121 | v11 L119 也是 flip |

<SrcLink file="v12:ultralytics/cfg/default.yaml" :line="105" :end="121" />

**两处真实的默认值改动**（相对 v11）：

1. **`copy_paste: 0.0 → 0.1`**——出场就开 10% 的前景粘贴，是 v8~v11 四代里第一次；
2. **`warmup_bias_lr: 0.1 → 0.0`**——见 §7。

**close_mosaic 的语义没变**：`epochs - close_mosaic` 之后把 mosaic 概率清零。**但 600 epoch 下的相对占比只有 1.7%**——这与 v8 的 10%（100 epoch）差别很大，说明作者的官方配方把 close_mosaic 当作「末尾去噪」而不是「长阶段微调」。

**推理开销**：零开销。

## 9. 与 v8 / v9 / v10 / v11 配方完整对比 (`trick-compare`)

> **v1~v11 读者路标**：把五代配方摆在一起，v12 的位置很清楚——**主干（损失 + 优化器 + 增强键）仍是 v8，唯一新增的是「分档差异化」与「FlashAttention 环境依赖」**。

| 项目 | v8 | v9 | v10 | v11 | **v12** |
|------|----|----|-----|-----|--------|
| 损失类 | v8DetectionLoss | v8DetectionLoss + 0.25 辅助 | v10DetectLoss | v8DetectionLoss | **v8DetectionLoss（与 v11 相同，`tal_topk` 参数沿用）** |
| 损失权重 | 7.5/0.5/1.5 | 7.5/0.5/1.5 × 1.0 + 0.25 辅 | 6 项 × 1.0 | 7.5/0.5/1.5 | **7.5/0.5/1.5** |
| TAL topk | 10 | 10 + 10 | 10 + 1 | 13（类默认） | **10（loss.py L176 显式）** |
| TAL α / β | 0.5 / 6.0 | 0.5 / 6.0 | 0.5 / 6.0 | 1.0 / 6.0 | **0.5 / 6.0（显式）** |
| 优化器 | auto（→SGD） | auto | auto | auto | **auto（→SGD）** |
| lr0 / lrf | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | **0.01 / 0.01** |
| momentum / wd | 0.937 / 5e-4 | 0.937 / 5e-4 | 0.937 / 5e-4 | 0.937 / 5e-4 | **0.937 / 5e-4**（auto 覆盖为 0.9） |
| warmup_epochs | 3.0 | 3.0 | 3.0 | 3.0 | **3.0** |
| warmup_bias_lr | 0.1 | 0.1 | 0.1 | 0.1 | **0.0**（v12 改动） |
| epochs（官方） | 100/500 | — | 500 | 100 | **600** |
| batch（官方） | 16~64 | — | — | — | **256** |
| Mosaic | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| MixUp | 0.0 | 0.0 | 0.0 | 0.0 | **分档 0.0~0.2** |
| CopyPaste | 0.0 | 0.0 | 0.0 | 0.0 | **分档 0.1~0.6** |
| scale 增强 | 0.5 | 0.5 | 0.5 | 0.5 | **分档 0.5 / 0.9** |
| close_mosaic | 10 | 10 | 10 | 10 | **10** |
| AMP | True | True | True | True | **True** |
| **FlashAttention** | 不需要 | 不需要 | 不需要 | 不需要 | **硬依赖（导入即崩）** |
| **timm** | 不需要 | 不需要 | 不需要 | 不需要 | **硬依赖（trunc_normal_）** |
| 训练入口类 | YOLO | YOLOv9 | YOLOv10 | YOLO | **YOLO（v8 通用管线）** |
| 训练日志损失名 | 3 项 | 6 项 | 6 项 | 3 项 | **3 项** |

**四句话总结**：

1. **v12 的损失与优化器主干仍是 v8**——7.5/0.5/1.5、auto→SGD、cosine、3 epoch warmup，一个都没变；
2. **v12 第一次做「分档增强」**——mixup / copy_paste / scale 三键按 n/s/m/l/x 给出不同值，这是 v8 之后配方层最大的变化；
3. **v12 第一次把训练环境变成硬门槛**——flash-attn + timm 缺失即 `assert False`，Windows 与 CPU 用户无法直接训练；
4. **v12 的 TAL 参数是「显式写死 v8 值」**——`loss.py L176` 显式传 `alpha=0.5, beta=6.0` 与 `topk=tal_topk(默认10)`，**与 `tal.py L28` 的类默认 `topk=13, alpha=1.0` 不一致**，容易误读（见 [losses §4](/versions/v12/losses#4-tal_topk-参数化v12-唯一改动)）。

## 10. 训练命令与常见场景 (`trick-commands`)

> **v1~v11 读者路标**：v10 需要 `from ultralytics import YOLOv10`；v11/v12 都回到 `YOLO`——**v12 没有 `YOLOv12` 独立类**，`YOLO("yolov12n.yaml")` 即可。

**官方复现配方**（600 epoch，需要多卡）：

```bash
yolo detect train \
  model=yolov12n.yaml \
  data=coco.yaml \
  epochs=600 batch=256 imgsz=640 \
  scale=0.5 mosaic=1.0 mixup=0.0 copy_paste=0.1 \
  device=0,1,2,3
```

**单卡小数据集微调**（加载官方权重，降 lr、缩 epoch）：

```bash
yolo detect train \
  model=yolov12n.pt \
  data=mydataset.yaml \
  epochs=50 batch=32 imgsz=640 \
  lr0=0.001 patience=15
```

- **官方权重来自 GitHub Release**：`yolov12{n/s/m/l/x}.pt`（README L56-60），**不是 ultralytics 的资产仓库**——`YOLO("yolov12n.pt")` 会去 `sunsmarterjie/yolov12` 的 release 下载；
- **微调时不要动 `copy_paste`**：默认 0.1 是 COCO 尺度上的值，小数据集上 CopyPaste 需要分割掩码，检测数据集通常没有——建议显式传 `copy_paste=0.0`；
- **AMP 默认开启**：`amp: True`（L35）；但 AAttn 内部**本来就强制 fp16**（见 §3），关 AMP 也省不掉这层降精度。

**环境自检**（训练前）：

```python
import torch
from flash_attn.flash_attn_interface import flash_attn_func   # 失败即环境不全
from timm.models.layers import trunc_normal_
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

**推理开销**：以上命令全部是训练期配置；但**环境依赖（flash-attn）在推理期同样存在**，部署时需单独处理（见 [painpoints §1](/versions/v12/painpoints#1-flashattention-硬依赖导入即崩-pain-flash-attn)）。

## 延伸阅读

- [理论 Concept](/versions/v12/concepts)（A2C2f / AAttn / ABlock / 去 SPPF / DW-separable 五个概念的机制解析）
- [YOLOv12 损失函数](/versions/v12/losses)——`tal_topk` 参数化与 v8 三分量总装
- [YOLOv12 反向传播推导](/versions/v12/loss-gradient)——A2C2f / AAttn / ABlock 的梯度路径
- [YOLOv12 痛点追踪](/versions/v12/painpoints)——FlashAttention 硬依赖、`num_heads` 硬编码、配方不完整
- [YOLOv11 训练技巧](/versions/v11/tricks)——上一代「训练配方一行没改」的对照
- [YOLOv8 训练技巧](/versions/v8/tricks)——v12 配方主干的直接出处
- [YOLOv10 训练技巧](/versions/v10/tricks)——唯一有「专属训练入口类」的对照
- [术语表](/glossary)——Mosaic、MixUp、CopyPaste、EMA、cosine、close_mosaic、FlashAttention 等术语索引
