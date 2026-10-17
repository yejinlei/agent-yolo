# YOLOv5 架构与层速查

## 1. 模块清单（按 common.py + yolo.py 顺序）

| 模块 | 文件 | 功能 | 使用次数 |
|------|------|------|----------|
| `Conv` | common.py L33 | Conv + BN + SiLU | 10+ |
| `Focus` | common.py L163 | 4路切片 → conv | 1 |
| `C3` | common.py L126 | CSP Bottleneck 3 conv | 9 |
| `SPP` | common.py L149 | 多尺度 maxpool | 1 |
| `Concat` | common.py L203 | torch.cat | 4 |
| `Detect` | yolo.py L24 | 检测头 | 1（内部3分支） |
| `Bottleneck` | common.py L94 | 残差 bottleneck | C3 内部 |
| `nn.Upsample` | yaml | 最近邻上采样 | 2 |

## 2. Conv — 基础单元

```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else ...
```

- **autopad**：`padding = k // 2`（same padding）
- **无 bias**：`bias=False`（BN 会吸收 bias）
- **默认激活 SiLU**：`act=True` 即 `nn.SiLU()`

## 3. C3 — 核心残差模块

```python
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```

- `e=0.5` 表示 hidden channel 是输入的一半
- `shortcut` 控制是否加残差（neck 中为 False，骨干中为 True）
- `n` 会被 `depth_multiple` 缩放

## 4. Detect — 检测头

```python
class Detect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        self.nc = nc
        self.no = nc + 5  # 85
        self.nl = len(anchors)  # 3
        self.na = len(anchors[0]) // 2  # 3
        self.grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.register_buffer('anchor_grid', ...)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
```

- `ch` = [256, 512, 1024]（三个头的输入通道）
- `m` 是 3 个 1×1 conv，输出通道 = `na × no` = 3 × 85 = 255

## 5. 推理时的 Detect forward

```python
def forward(self, x):
    z = []
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # 1×1 conv
        bs, _, ny, nx = x[i].shape
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            z.append(y.view(bs, -1, self.no))
    return x if self.training else (torch.cat(z, 1), x)
```

## 6. SPP — 空间金字塔

```python
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```

## 7. 输出张量布局（三个头同构）

每个头输出 shape：`[bs, 3, 85, ny, nx]`（训练）或 `[bs, 3, ny, nx, 85]`（推理，permute 后）。

- `bs` = batch size
- `3` = anchors per head
- `85` = 4 + 1 + 80
- `ny, nx` = grid 尺寸（80/40/20）

## 8. 模型缩放参数

| 参数 | yaml 字段 | 作用 |
|------|-----------|------|
| `depth_multiple` | yaml | C3 重复次数缩放 |
| `width_multiple` | yaml | 通道数缩放 |
| `anchors` | yaml | 锚点定义（3×3） |
| `nc` | yaml | 类别数 |

## 延伸阅读

- [Architecture 逐层图解](/versions/v5/architecture)
- [理论 Concept](/versions/v5/concepts)
- [源码走读](/versions/v5/code-walkthrough)
