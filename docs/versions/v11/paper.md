# 论文与发布精读

> **YOLO11 没有正式 arXiv 论文**——这是 YOLO 系列自 v2 起第一次「无论文发布」的官方版本。
>
> 全部权威材料三层：
> 1. **官方博客**：ultralytics.com/blog 上的《YOLO11: 100% Faster, 100% Smarter》（2024-09-30 前后）
> 2. **仓库 README**：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) tag `v8.3.0`（commit `6e43d1e1e5db72afbf686dee6745669bcb124b0a`，tag 对象 sha `a46b49a27a97d44c642882a405c8b57979e6d368`，664 文件）
> 3. **PyPI release**：`ultralytics 8.3.0`，PR #16539，作者 Glenn Jocher & Jing Qiu，tag 消息 `ultralytics 8.3.0 YOLO11 Models Release (#16539)`，日期 2024-09-30
>
> 本页按发布叙事精读：材料来源 → 官方主张 → C3k2 与 C2PSA 的设计动机 → 五档参数与性能表 → 与 v10 的结构性对比 → 批判性评述。快照可复核的每一句都标注 `<SrcLink>` 徽章；博客声明但快照不可复核的，一律注明「**官方声明，snapshot 内不可复核**」。

## 1. 材料来源：为什么这一页叫「论文与发布精读」

从 v2 起 YOLO 每一版都有论文：v3 是 arXiv:1910.06822，v4 是 arXiv:2004.10934，v5 没有独立论文但 UADY 团队有说明，v7 是 arXiv:2207.02696，v9 是 arXiv:2402.13616，v10 是 arXiv:2405.14447。**v11 是这条链上的第一次断裂**：官方博客《YOLO11: 100% Faster, 100% Smarter》替代了 arXiv 预印本。

这个断裂带来三个读法上的变化：

1. **没有方法学章节**：博客的写作是「产品发布」体，不是「方法」体——不给公式推导、不给消融实验、不给数据集上的分档消融，只给「更快、更聪明」的对比表与博客口径的推理延迟；
2. **没有实现细节的独立说明**：`C3k2` 的 `__init__(c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)` 六参含义 <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="723" />、`C2PSA` 的 CSP + Attention + FFN 拼装 <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="1002" :end="1041" />——这些细节**代码里有，博客里没写**，README 也只是一笔带过；
3. **指标口径不可复核**：博客表里的 mAP50-95 与延迟、与 v8 的横向对比数字，**snapshot 里没有实验脚本、没有训练日志、没有预训练权重对应的 eval 结果**——是「作者宣称」而非「社区可复现」。

本页的所有权威事实因此遵循两条边界：
- **快照内可定位的事实**（代码、行号、结构、参数）→ 附 `<SrcLink>` 徽章；
- **博客声明的事实**（精度、延迟、同档对比、"100% Faster, 100% Smarter"）→ 注明「**官方声明，snapshot 内不可复核**」。

> **v1~v10 读者路标**：v8 读者的记忆锚点是「以代码为论文」——见 [v8 论文精读](/versions/v8/paper)；v11 比 v8 更极端：**连 README 的「以代码为论文」都要靠 blog 补齐**，博客里没写的方法学就没有。

## 2. 官方主张：一句话概括博客叙事

博客首页的三个数字是 YOLO11 的整个营销叙事——**「100% Faster, 100% Smarter」**：

- **100% Faster**：博客口径「YOLO11 相对 YOLOv8 同精度延迟降低 1.8 倍」——这是博客数字，**snapshot 内不可复核**（v11 README 快照不给延迟列，也不给基准环境）；
- **100% Smarter**：博客口径「YOLO11 相对 YOLOv8 同档位 mAP50-95 提升 0.8~2.3 个百分点」——这是博客数字，见第 5 节精度表的横向对比；
- **架构简化**：博客首页提到「更简洁的架构」（更简洁的措辞是 "more efficient architecture"），对应代码里的两件事——**用 C3k2 换 C2f、用 C2PSA 换 SPPF+PSA**；博客里没有单独解释为什么这样设计，只有代码层面的替换事实 <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="723" /> <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="1002" :end="1041" />；
- **单头 + NMS 回归**：博客没有讨论 NMS，但代码里 Detect 头就是 v8 的原版 <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="68" />，损失也是 v8DetectionLoss <SrcLink file="v11:ultralytics/utils/loss.py" :line="157" />——**NMS-free 部署这条 v10 立起来的路径，在 v11 快照里完全没走**；
- **AGPL-3.0**：与 v8 / v10 一致，商用闭源仍受双轨约束（背景见 [v8 论文精读 · 许可证](/versions/v8/paper)）。

一句话：博客叙事是「更简洁 + 更快 + 更聪明」，代码事实是「换两个积木 + 加一份单 yaml + 保留 v8 头与 NMS」——博客把「换两个积木」包装成了「100% Faster, 100% Smarter」。

> **v1~v10 读者路标**：v10 读者的 NMS-free 部署路径在 v11 快照里没了——见 [v10 论文精读](/versions/v10/paper) 第 3 节；v11 的「端到端」这个词没有出现在博客里，README 也没有；部署侧仍然要跑 `non_max_suppression`。

## 3. C3k2：C2f 的 shortcut 变体（不是「Attention 分流」）

C3k2 是 v11 的骨干内瓶，替换 v8 的 [C2f](/glossary#c2f) 与 v10 的 C2f 群。它的整个类定义非常克制：

```python
class C3k2(C2f):
    """C3k2 CSP Bottleneck block with 2 Convolutions and n Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g=g, k=(3 if c3k else 1), reg=reg)
            for _ in range(n)
        )
```

<SrcLink file="v11:ultralytics/nn/modules/block.py" :line="723" />

关键事实——**请务必记住，v8.3.0 快照里的 C3k2 是这样的**：

1. **`class C3k2(C2f)`**：C3k2 直接继承 C2f、**没有独立 forward**——它的 forward 完全复用 C2f 的残差 + concat 流程，唯一区别是内瓶 Bottleneck 的 k 与 shortcut 由 C3k2 自己设定；
2. **无 Attention、无 PSA、无 C2fCIB**：**「C3k2 含 Attention 分流」是社区常见的误读**——v11 快照里的 C3k2 里没有任何 Attention 相关代码；注意力能力在骨干里的落点是骨干倒数第二层的 C2PSA（见下节），不是 C3k2；
3. **`c3k` 参数控制内瓶 kernel 大小**：`c3k=True` 时内瓶 Bottleneck 的 k=(3,3)，`c3k=False` 时 k=(1,1)——yaml 里骨干首两段 C3k2 传 `c3k=False`（k=1）、后面两段传 `c3k=True`（k=3）；
4. **`e=0.5` 是 v11 的默认宽度系数**：内瓶通道 `c = int(c2 * e)`，与 v8 的 C2f 相同；
5. **`shortcut=True` 是 v11 的默认**：C2f 里 shortcut=True 用于骨干（跳过中间层的残差连接），C3k2 在骨干与 neck 里默认都是 True；

yaml 里 C3k2 的 8 处使用（骨干 4 段 + neck 4 段）：

```yaml
# yolo11.yaml L15-L27 骨干
16: [-1, 1, Conv, [256, 3, 2]]           # P2/4
17: [-1, 1, Conv, [512, 3, 2]]           # P3/8
18: [-1, 1, C3k2, [512, False, 0.25]]    # 首段：c3k=False（k=1），e=0.25
19: [-1, 1, Conv, [512, 3, 2]]           # P4/16
20: [-1, 1, C3k2, [512, False, 0.25]]    # 次段：c3k=False（k=1），e=0.25
21: [-1, 1, Conv, [1024, 3, 2]]          # P5/32
22: [-1, 1, C3k2, [1024, True]]          # 第三段：c3k=True（k=3），e 默认 0.5
23: [-1, 1, Conv, [1024, 3, 2]]
24: [-1, 1, C3k2, [1024, True]]          # 第四段：c3k=True（k=3）
25: [-1, 1, SPPF, [1024, 5]]             # 骨干倒数第二层，v8 沿用
26: [-1, 1, C2PSA, [1024]]               # 骨干尾：C2PSA（下节展开）
```

<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="15" :end="27" />

```yaml
# yolo11.yaml L30-L47 neck（此处摘关键 4 段 C3k2）
- [-1, 1, Upsample, [None, 2, None]]
- [[-1, 6], 1, Concat, [1]]              # P5 上行 → concat 到 P4
- [-1, 1, C3k2, [512, True]]             # L13
- [-1, 1, Upsample, [None, 2, None]]
- [[-1, 4], 1, Concat, [1]]              # P4 上行 → concat 到 P3
- [-1, 1, C3k2, [256, True]]             # L16
- [-1, 1, Conv, [256, 3, 2]]             # P3 → P4 下采样
- [[-1, 13], 1, Concat, [1]]
- [-1, 1, C3k2, [512, True]]             # L19
- [-1, 1, Conv, [512, 3, 2]]             # P4 → P5 下采样
- [[-1, 10], 1, Concat, [1]]
- [-1, 1, C3k2, [1024, True]]            # L22
- [[16, 19, 22], 1, Detect, []]          # 单头 Detect
```

<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="30" :end="47" />

设计动机（**基于代码事实的推断，博客未直接声明**）：

- **`c3k=False → True` 的浅深分层**：P3/P4 高分辨率用 k=1 内瓶（参数省），P5/P6 低分辨率用 k=3 内瓶（感受野扩）——**浅层省参数，深层扩感受野**，是 v11 骨干的核心结构取舍；
- **v10 的效率积木没进 v11**：v10 的 SCDown（空间-通道解耦下采样）、C2fCIB（大核颈）、PSA（部分自注意力）**在 v11 快照里全部没有**——v11 的下采样仍是 Conv 3×3 s2，v11 的 P5 融合仍是一个 C3k2（无大核），v11 的注意力只在骨干尾的 C2PSA；
- **为什么「换 C3k2 而不是保留 C2f」**：博客未说；代码层面最直接的观察是——C3k2 让「同一层内瓶选 k=1 或 k=3」成为 yaml 的一行参数，v10 要靠 SCDown + C2fCIB 组合才能表达「下采样降维 + 大核」两件事。C3k2 的抽象层更高，代价是**每个 C3k2 内瓶都是 Bottleneck 的 shortcut 变体**——计算模式更接近 v8。

> **v1~v10 读者路标**：v8 读者的 C2f 是「每层 Bottleneck 内瓶 + 每层残差连接 + 每 2 层 concat」，C3k2 只改了内瓶 Bottleneck 的 k 与 shortcut 默认——读 v11 骨干时把「C2f 的父类流程」当底表，只盯 C3k2 的 `c3k / e / shortcut` 三个参数怎么传；[v10 读者](/versions/v10/index) 请留意——v10 的效率三件套（SCDown / C2fCIB / PSA）在 v11 快照里**一个都没保留**。

## 4. C2PSA：把注意力塞进骨干尾

C2PSA 是 v11 骨干尾的模块，替换 v10 的「SPPF + 独立 PSA」两件套。定义在 `block.py` L1002-L1041 <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="1002" :end="1041" />：

```python
class C2PSA(C2f):
    """C3kSATA attention block for YOLO11 architecture."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g=g, e=e)
        self.attn = Attention(c2,  # use final c2 dim
                              head_dim=64,   # head dim
                              num_heads=8,
                              attn_ratio=0.5,
                              qkv_bias=True
                              )
```

**关键事实**：

- **同样继承 C2f**：C2PSA 也不是独立 forward，靠 C2f 的 forward 骨架 + 在 `cv1` 后插入 `self.attn`——**C2f 的残差支路 + Attention + FFN 拼在一个 CSP 里**；
- **Attention 参数**：`head_dim=64, num_heads=8, attn_ratio=0.5`——`attn_ratio=0.5` 意味着 Attention 只在**一半通道**上跑（key_dim 减半、qkv 1×1 合并出 Q/K/V），另一半直通；
- **qkv_bias=True**：q/k/v 三个投影都带 bias——与 v10 的 Attention 略有不同（v10 里 qkv_bias 也是 True，见 `block.py` L866 <SrcLink file="v11:ultralytics/nn/modules/block.py" :line="866" />）；
- **落点**：整个网络**只在骨干倒数第二层放 1 个 C2PSA**——SPPF 之后、neck 之前；

```yaml
# yolo11.yaml L25-L26
- [-1, 1, SPPF, [1024, 5]]
- [-1, 1, C2PSA, [1024]]
```

<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="25" :end="26" />

与 v10 的对照：v10 是「SPPF → PSA」两节点，中间一次残差连接；v11 是「SPPF → C2PSA」两节点，C2PSA 内部已含残差连接与 Attention——**减少一次 concat、减少一次残差相加**，骨干尾从「C2f 残差 + PSA 独立块」变成「C2PSA 一个块」。

设计动机（**代码事实的推断，博客未直接声明**）：

- **注意力只在最深层放**：20×20 的 token 数最少，Attention 的 O(n²) 成本可控；v10 的 PSA 也放在同位置，v11 把「SPPF + PSA」合并成一个 C2PSA，是**结构紧凑化**而非新机制；
- **attn_ratio=0.5** 沿用 v10：只有半通道跑注意力，另一半直通——**计算量减半**；
- **v11 快照没有独立的 C2fCIB 大核颈**：v10 的 P5 融合用 `C2fCIB [1024, True, True]`（大核 RepVGGDW），v11 的 P5 融合仍是普通 C3k2 <SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="38" />——大核深度卷积在 v11 里退场。

> **v1~v10 读者路标**：v10 读者的 PSA 是 [v10 论文精读](/versions/v10/paper) 第 4 节讲的「部分自注意力」；v11 的 C2PSA 沿用了 attn_ratio=0.5 的设计，但把注意力**内嵌进 C2f 骨架**——注意力仍是可选支路，不是全通道；[v8 读者](/versions/v8/index) 请留意——v8 的骨干尾是 SPPF 单层，v11 是「SPPF + C2PSA」两层。

## 5. 五档参数与性能表（yolo11.yaml L8-L12 权威）

v11 的参数组织是「**单文件 scales 表**」——一个 `yolo11.yaml` 定义五档，n/s/m/l/x 只在 scales 表里体现，结构层面完全一致：

```yaml
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="8" :end="12" />

**官方 mAP50-95 表（COCO val，输入 640；博客与 README 声明口径）**

| 模型 | Depth | Width | Max Ch | #Layers | #Params | FLOPs | mAP50-95 |
|------|-------|-------|--------|---------|---------|-------|----------|
| YOLO11n | 0.50 | 0.25 | 1024 | **319** | **2.62M** | **6.6G** | **39.5%** |
| YOLO11s | 0.50 | 0.50 | 1024 | **319** | **9.46M** | **21.7G** | **47.0%** |
| YOLO11m | 0.50 | 1.00 | 512 | **409** | **20.11M** | **68.5G** | **51.5%** |
| YOLO11l | 1.00 | 1.00 | 512 | **631** | **25.37M** | **87.6G** | **53.4%** |
| YOLO11x | 1.00 | 1.50 | 512 | **631** | **56.97M** | **196.0G** | **54.7%** |

**读表要点**：

- **n 与 s 深度相同、宽度不同**：n 与 s 都是 depth=0.50，只是 width 0.25 vs 0.50——**深度不涨，只涨宽度**；
- **s → m 深度不涨、宽度翻倍**：s → m 从 width 0.50 到 1.00，但 max_channels 从 1024 降到 512——这是**通道上限的变化**，不是结构变化；
- **m → l 深度翻倍**：m → l 深度 0.50 → 1.00，宽度不变——**结构层数从 409 涨到 631**；
- **l → x 深度相同、宽度涨 50%**：l → x 深度都是 1.00，宽度 1.00 → 1.50——参数量从 25.37M 涨到 56.97M（+124%），mAP 从 53.4 涨到 54.7（+1.3）——**边际收益递减**；
- **对比 v10**：v11 有 5 档，v10 有 6 档（多 B 档）；v11-n 2.62M vs v10-N 2.3M（+14%），v11-x 56.97M vs v10-X 29.5M（+93%）——v11 的整体容量显著更大，x 档尤其明显；
- **对比 v8**：博客口径 v11 同档 mAP50-95 提升 0.8~2.3 个百分点（v8-n 37.3 / v8-s 44.7 / v8-m 50.2 / v8-l 52.5 / v8-x 53.9）——**这是博客声明，snapshot 内不可复核**。

> **v1~v10 读者路标**：v10 读者的六档 scales 是每档一份 yaml 各带独立 scales（如 n 档 `[0.33, 0.25, 1024]`、b 档 `[1.00, 1.00, 1024]`）；v11 用一份 scales 表集中定义——**扩档只需在 scales 表里加一行**，不用新建 yaml；[复合缩放](/glossary#compound-scaling-复合缩放)在 v11 里被单文件集中化了。

## 6. Detect 头与 NMS：结构性回退

v11 的 head.py 里 Detect 是 v8 的原版 <SrcLink file="v11:ultralytics/nn/modules/head.py" :line="21" :end="68" />：

```python
class Detect(nn.Module):
    """YOLOv8 Detect head for detection models.
    Parameters
    ----------
    nc : int
        Number of classes.
    ch : tuple[int]
        Input channels at each scale.
    anchors: bool, optional
        flag to (MI?)
    """
    dynamic = False
    shape = None
    anchors = torch.empty(0)  # anchors
    stride = torch.empty(0)  # mean distance of each pixel from image border
    ...
```

yaml L47 就是 `Detect`，不是 `v10Detect`：<SrcLink file="v11:ultralytics/cfg/models/11/yolo11.yaml" :line="47" />

**关键事实**：

- **`Detect` 类是 v8 原版**：`class Detect(nn.Module)` 的 cv2/cv3 结构、bias_init、forward 三态（train/eval/export）**完全与 v8 一致**；
- **v10Detect 在 v11 快照里不存在**：v10 的 `class v10Detect(Detect)`、`one2one_cv2 / one2one_cv3 = copy.deepcopy(...)`、`xi.detach()` 分支、`v10postprocess` 后处理——**在 v11 快照中一个都没保留**；
- **损失是 v8DetectionLoss**：`class v8DetectionLoss(DetectionLoss)` <SrcLink file="v11:ultralytics/utils/loss.py" :line="157" />——v10 的 `v10DetectLoss` 也不存在；
- **TAL 分配器沿用了**：`class TaskAlignedAssigner` <SrcLink file="v11:ultralytics/utils/tal.py" :line="13" />，但默认参数与 v8 不同——**topk=13**（v8 是 10）、**α=1.0**（v8 是 0.5）、β=6.0；
- **导出走 NMS**：`model.export()` 走 v8 通用路径，ONNX / TRT 图里含 `non_max_suppression` 或等价节点——与 v10 的「导出图无 NMS」相反。

设计动机（**代码事实 + 博客缺失**）：

- **博客对 NMS 只字未提**：v10 立起来的 NMS-free 部署路径在 v11 里没延续——博客叙事是「更简洁的架构」，代码事实是**回归 v8 单头 + NMS**；
- **TAL 参数调高**：topk 从 10 涨到 13、α 从 0.5 涨到 1.0——这是**训练期标签分配策略的调整**，博客未说明动机，代码里可以看到 tal.py L13 的默认构造参数已改；
- **回归的理由猜测**：v10 是 THU-MIG 独立 fork，v11 是 ultralytics 主线——**官方选择回主线、不回 fork**；NMS-free 部署在 v11 快照里不是设计选择，是**没继承 v10 的路径**。

> **v1~v10 读者路标**：v10 读者部署侧的「导出图无 NMS」优势在 v11 快照里没了——请回到 [v8 架构](/versions/v8/architecture) 的 NMS 收口段；v11 的后处理链完全同 v8：`predict()` 走 `non_max_suppression`、`export()` 走 `exporter.py` 的 v8 通用路径。

## 7. 与 v10 的结构性对比

v11 是「**ultralytics 主线在 v10 出现之后的官方结构性回退**」——两者无继承关系。逐项对照：

| 维度 | YOLOv10 | YOLOv11 |
|------|---------|---------|
| 作者 | 清华 THU-MIG | **ultralytics 官方（Glenn Jocher & Jing Qiu）** |
| 仓库 | THU-MIG/yolov10 独立仓库 | **ultralytics/ultralytics 主线 tag `v8.3.0`** |
| 基座 | ultralytics 8.1.34 fork | **ultralytics 8.3.0 主线** |
| 论文 | arXiv:2405.14447 | **无（博客 + README）** |
| 检测头 | v10Detect 双头（one2many + one2one deepcopy） | **Detect 单头（v8 沿用）** |
| 标签分配 | TAL topk=10 + topk=1（同分配器两次实例化） | **TAL topk=13，α=1.0 / β=6.0** |
| 推理后处理 | **v10postprocess（无 NMS）** | **NMS（回归 v8）** |
| 骨干内瓶 | C2f + C2fCIB（大核 RepVGGDW） | **C3k2（C2f 的 shortcut 变体）** |
| 下采样 | **SCDown（空间-通道解耦）** | **Conv 3×3 s2（回归 v8）** |
| 骨干尾 | SPPF + PSA | **SPPF + C2PSA** |
| 分类头 | 深度可分离两段 | **DW-separable 两段（沿用 v10 思路，无 legacy 开关）** |
| 档位 | n/s/m/**b**/l/x 六档 | **n/s/m/l/x 五档**（去 B） |
| 结构文件 | yolov10{nsblmx}.yaml 六份各带 scales | **yolo11.yaml 单份，五档 scales 集中** |
| cfg 目录 | `models/v10/` | **`models/11/`（去 v 前缀，全站独一份）** |
| 参数量 | N 2.3M / X 29.5M | **N 2.62M / X 56.97M**（x 档 +93%） |
| 官方 mAP | N 38.5 / X 54.4 | **N 39.5 / X 54.7** |
| 延迟列 | T4 TRT FP16 全档 | **无（README 不给延迟）** |
| 顶层 Python 类 | `from ultralytics import YOLOv10` | **`from ultralytics import YOLO`（无 YOLOv11 类）** |

一句话：**v10 是「NMS-free + 效率积木」的产品化 fork；v11 是「C3k2 + C2PSA + 单 yaml」的主线更新**——两者在同一时期（2024-05 与 2024-09）分别代表两个方向的探索。

> **v1~v10 读者路标**：v10 读者如要迁移到 v11，需**撤销所有 v10 专属代码路径**——`from ultralytics import YOLOv10` 改为 `from ultralytics import YOLO`、`v10Detect` 相关配置改为 `Detect`、`v10DetectLoss` 相关损失调用改为 v8DetectionLoss、`v10postprocess` 后处理改为常规 NMS；v10 的效率积木（SCDown / C2fCIB / PSA）在 v11 里全部没继承。

## 8. 局限与批判性评述

博客没写的、README 没写的、但读代码能看到的：

1. **无 arXiv 论文**：这是 YOLO 系列自 v2 起的第一次「无论文发布」——博客不是可引用的学术论文；想写综述、写论文、做 baseline 时，v11 只能引博客 + README，学术可引用性弱于 v10；
2. **精度数字不可复核**：博客 mAP50-95 表（39.5 / 47.0 / 51.5 / 53.4 / 54.7）**snapshot 内无实验脚本、无训练日志、无预训练权重的 eval 结果**——是「作者宣称」而非「社区可复现」；
3. **同档对比不可复核**：博客口径「v11 相对 v8 同精度延迟降低 1.8 倍、同档 mAP 提升 0.8~2.3 个百分点」——**snapshot 内既没有延迟基准也没有同档对比脚本**；
4. **无延迟列**：v11 README 快照没有像 v10 那样给 T4 TRT FP16 延迟，也没有像 v8 那样给 CPU ONNX + A100 TRT 双列——**跨版本比延迟一律不可比**；
5. **命名坑是历史遗留**：cfg 目录 `models/11/`（去 v 前缀）与 v8/v9/v10 的 `models/v8/`、`models/v9/`、`models/v10/` 目录规则不一致——**这是全站独一份的例外**，读代码时容易踩坑；
6. **C3k2 名字容易被误读**：社区流传「C3k2 含 Attention 分流」的说法在 v11 快照里**不成立**——C3k2 就是 C2f 的子类只改 `__init__`；注意力能力全部在 C2PSA 里；
7. **回归 NMS 是「没继承」不是「选择回退」**：博客对 NMS 只字未提；v11 单头 + NMS 的形态与 v10 的 NMS-free 部署形态**并存**——用户要 NMS-free 部署还得用 v10；
8. **AGPL-3.0 沿袭**：v11 与 v8 / v10 一致——商用闭源仍受双轨约束，选型清单第一行照旧是许可证（背景见 [v8 论文精读 · 许可证](/versions/v8/paper)）。

> **v1~v10 读者路标**：v8 的「以代码为论文」在 v11 里进一步弱化——博客不写的方法学就没有；v10 的「有 arXiv 论文 + 完整实验表」反而是这一代 YOLO 里最完备的。历史问题在 v11 里以「博客声明 + snapshot 不可复核」的形态回归——见 [v8 痛点追踪](/versions/v8/painpoints)。

## 9. 发布清单（供查证）

- **仓库**：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Tag**：`v8.3.0`（tag 对象 sha `a46b49a27a97d44c642882a405c8b57979e6d368`，消息 `ultralytics 8.3.0 YOLO11 Models Release (#16539)`）
- **Commit**：`6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- **日期**：2024-09-30（tagger 与 release 落地同日）
- **PR**：#16539
- **作者**：Glenn Jocher & Jing Qiu
- **许可证**：AGPL-3.0
- **快照规模**：664 文件
- **版本标识**：`__version__ = "8.3.0"` 写在 `ultralytics/__init__.py` L3 <SrcLink file="v11:ultralytics/__init__.py" :line="3" />

## 延伸阅读

- [Overview 概览页](/versions/v11/index)
- [Architecture 逐层图解](/versions/v11/architecture)
- [理论 Concept](/versions/v11/concepts)
- [源码页：yolo11.yaml](/versions/v11/source/yolo11-yaml) · [head.py](/versions/v11/source/head-py) · [loss.py](/versions/v11/source/loss-py) · [tal.py](/versions/v11/source/tal-py)
- [v8 论文精读（以代码为论文）](/versions/v8/paper) · [v10 论文精读](/versions/v10/paper)
- [系列家族对照](/global/family) · [术语表](/glossary)
- 官方博客：[YOLO11: 100% Faster, 100% Smarter](https://www.ultralytics.com/blog/yolo11-100-faster-100-smarter)（博客，非论文）
- 源码入口：[ultralytics/ultralytics @ v8.3.0](https://github.com/ultralytics/ultralytics/tree/v8.3.0)
