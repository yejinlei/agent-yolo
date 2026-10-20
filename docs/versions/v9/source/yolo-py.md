# models/yolo.py

这是 YOLOv9 快照 `models/yolo.py` 的源码浏览页：**七套检测头同文件 + BaseModel/DetectionModel + parse_model 建网流水线**。763 行，v7 同款布局（v7 把 Model 与五种头塞在 models/yolo.py；v8 拆去了 nn/tasks.py + head.py，v9 又回到单文件）。v9 的主角是 **DualDDetect**——yolov9-c/e 双头检测的落点：cv2/cv3 主分支 + cv4/cv5 辅助分支 + 双 DFL，训练返回两路特征给双头损失，推理返回双路输出。

> 快照路径：`source/yolov9/snapshot/models/yolo.py`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v9:models/yolo.py" />

**阅读路线**：

- [L29 Detect](#L29)：v5-v8 谱系的单尺度头，gelan 系列不用它——留作兼容
- [L78-124 DDetect](#L78-L124)：gelan 系列的正主（`gelan-c.yaml` 收尾），单头单 DFL
- [L127-187 DualDetect](#L127-L187)：双套 cv 分支 + 双 DFL 的第一版——中间卷积**无**分组、通道公式**无** make_divisible
- [L190-256 DualDDetect](#L190-L256)：yolov9-c/e 的正主，DDetect 与 DualDetect 的合体（下方重点卡片）
- [L259-332 TripleDetect](#L259-L332) / [L335 TripleDDetect](#L335)：三头变体（`train_triple.py` 配套），结构同双头再扩一路
- [L419 Segment](#L419)、[L469 BaseModel](#L469)、[L525 DetectionModel](#L525)：v7/v8 遗产段
- 建网核心 parse_model：[L697-700 CBLinear](#L697-L700)、[L701-702 CBFuse](#L701-L702)、[L704-709 Detect 系列统一分支](#L704-L709)——CBNet 积木进 yaml 的通道
- [L549-569 bias_init 分派](#L560)：单头与双/三头各走一条 stride 反推 + bias 初始化路径

## 关键片段逐行走读

> 已读过 [v7 yolo.py](/versions/v7/source/yolo-py)/[v8 tasks.py](/versions/v8/source/tasks-py)？v7 读者只需关注 DualDDetect 与 parse_model 的 CBNet 分支；v8 读者注意：这里没有 scales 复合缩放（v9 每档独立 yaml），parse_model 只做通道透传。

### DDetect（L78-124）：gelan 系列的单头 DFL 头

<SrcSnippet file="v9:models/yolo.py" :start="78" :end="124" hl="89-91,95-100,102-115" title="DDetect：单头 DFL 检测头" />

::: details 逐行注释（L78–L124）
- **L89**：`self.nl = len(ch)` —— 检测层数 = 输入路数（gelan-c.yaml 传 3 路）。
- **L90-91**：`reg_max = 16`、`no = nc + reg_max*4` —— 每锚点输出 = 4×16 维 ltrb 分布 + nc 类分数，无 objectness（v8 口径的延续）。
- **L95**：通道公式 —— 回归分支 c2 = `make_divisible(max(ch[0]//4, 64, 16), 4)`（**4 对齐**），分类分支 c3 = `max(ch[0], min(nc*2, 128))`。
- **L96-97**：`cv2` 回归分支三级连 —— Conv3×3 → **Conv3×3 g=4** → 1×1 groups=4 输出 64 维分布；分组卷积是 v7 头风格的残留，也是与 DualDetect 的第一处差异。
- **L98-100**：`cv3` 分类分支三级连 + DFL 模块（reg_max=16 期望值积分）。
- **L102-107**：`forward` 训练分支 —— 直接返回 cat 后的原始特征（cv2/cv3 拼接，L106-107），解码交给损失层；这是 anchor-free 头的标准分工。
- **L108-115**：`forward` 推理分支 —— make_anchors 生成网格 → DFL 积分 → dist2bbox 解码 → sigmoid 分类，`return y if self.export else (y, x)`。
- **L117-124**：`bias_init` —— box 分支 bias 置 1.0（L123），cls 分支 bias 按 `log(5/nc/(640/s)²)` 初始化（L124，640 图约 5 个目标的先验）。
:::

### DualDetect（L127-187）：双头第一版（无分组卷积）

<SrcSnippet file="v9:models/yolo.py" :start="127" :end="187" hl="138,144-155,164-175" title="DualDetect：双套 cv 分支 + 双 DFL" />

::: details 逐行注释（L127–L187）
- **L138**：`self.nl = len(ch) // 2` —— 6 路输入 = 3 主 + 3 辅，对半分。
- **L144-145**：通道公式与 DDetect 同式，但**没有 make_divisible**，且直接用 `max(...)`——与 DualDDetect 的第二处差异。
- **L146-153**：`cv2/cv3` 主分支吃 `ch[:self.nl]`，`cv4/cv5` 辅助分支吃 `ch[self.nl:]`——两套独立的三级连，中间 Conv **无** g=4 分组。
- **L154-155**：`self.dfl` 与 `self.dfl2` —— 主辅各一个 DFL，不共享。
- **L157-165**：`forward` 训练分支 —— d1/d2 两列表分别收集主、辅路的 cv 输出，返回 `[d1, d2]`（L164-165）。
- **L166-175**：`forward` 推理分支 —— 主辅两路各自 DFL+dist2bbox 解码，`y = [主, 辅]` 一起返回（与 DualDDetect 相同的双路设计）。
- **L177-187**：`bias_init` —— 主辅两套 bias 各初始化一遍。
- 定位：DualDetect 是 yolov9-t/s 档与消融实验用的轻量双头；yolov9-c/e 换用下面的 DualDDetect。
:::

### DualDDetect（L190-256）：yolov9-c/e 的双头正主

<SrcSnippet file="v9:models/yolo.py" :start="190" :end="256" hl="201-218,227-238,239-244" title="DualDDetect：主辅双分支 + 双 DFL" :max-height="620" />

::: details 逐行注释（L190–L256）
- **L201**：`self.nl = len(ch) // 2` —— yolov9-c.yaml 传 6 路 `[31, 34, 37, 16, 19, 22]`：前 3 路是辅助分支（A3/A4/A5），后 3 路是主分支（P3/P4/P5）。
- **L202-203**：`reg_max = 16`、`no = nc + reg_max*4` —— 与 DDetect 同口径。
- **L207-208**：通道公式 —— `make_divisible(..., 4)` 版本，DDetect 同款（DualDetect 反而无）。
- **L209-212**：`cv2/cv3` 主分支 —— 只吃 `ch[:self.nl]`（16/19/22 的通道），cv2 中间 Conv **g=4** 分组。
- **L213-216**：`cv4/cv5` 辅助分支 —— 只吃 `ch[self.nl:]`（31/34/37 的通道），cv4 中间 Conv 同样 g=4——主辅结构完全对称。
- **L217-218**：`dfl/dfl2` —— 主辅各一个 DFL。
- **L220-226**：`forward` 逐尺度循环 —— d1 收主路 cat、d2 收辅路 cat，i 与 nl+i 配对。
- **L227-228**：训练分支返回 `[d1, d2]` —— 供 `loss_tal_dual.py` 的双 assigner 各吃一路（见 [loss 源码页](/versions/v9/source/loss-tal-dual-py)）。
- **L229-231**：推理分支的锚点重建只在输入尺寸变化时执行（dynamic/shape 缓存）。
- **L233-238**：推理双路解码 —— 主路 dbox+cls、辅路 dbox2+cls2 各自 DFL/dist2bbox/sigmoid，`y = [主, 辅]` 一起返回（L237-238）。
- **L239-244**：注释代码 —— 第一版 `return (y, d2)` 只留辅助路输出（L239-240），第二版 `[(y1, y2), (d1, d2)]` 又改回双路；辅助主导训练时期的历史痕迹，也是 [v9 痛点页](/versions/v9/painpoints) 痛点 7「推理仍输出双路」的出处。
- **L246-256**：`bias_init` —— 主（cv2/cv3，L251-253）辅（cv4/cv5，L254-256）两套 bias 各自按 stride 初始化，与 DualDetect 同型。
:::

### parse_model 的 CBNet 与 Detect 分支（L693-715）

<SrcSnippet file="v9:models/yolo.py" :start="693" :end="715" hl="697-709" title="parse_model：CBLinear / CBFuse / Detect 系列" />

::: details 逐行注释（L693–L715）
- **L697-700**：CBLinear 分支 —— `c2 = args[0]` 直接把 yaml 里的**通道拆分表**（如 `[256, 512, 512]`）当输出"通道"记账，不做 width 缩放；c1 取 from 层通道后原样透传。下游层引用它时拿到的是这张表。
- **L701-702**：CBFuse 分支 —— `c2 = ch[f[-1]]`，输出通道取 from 列表最后一层（目标尺度）的通道。
- **L704-709**：Detect 系列统一分支 —— `args.append([ch[x] for x in f])` 把 6 路 from 层的通道附给头；对 DualDDetect 来说就是主辅各 3 路的通道列表。**没有 anchors 参数**——L706-707 的注释残留（`# number of anchors`）显示 v7 时代头吃 anchor 配置的管道，v9 已拆。
- **L666-667**（卡片外）：`no = na*(nc+5)` 的 anchors 遗留计算仍在 parse_model 开头，只用于 `if c2 != no` 的"是否输出层"判断——yaml 里 `anchors: 3` 字段 vestigial 但仍参与这个比较。
- **L710-715**：Contract/Expand（L710-713）与默认 `c2 = ch[f]`（L714-715）。
:::

### bias_init 的分派（L549-569）

<SrcSnippet file="v9:models/yolo.py" :start="549" :end="569" hl="551,555-559,560,564-569" title="DetectionModel：单头/双头 bias_init 分派" />

::: details 逐行注释（L549–L569）
- **L550**：`m = self.model[-1]` —— 最后一层即检测头。
- **L551**：单头分派 —— `isinstance(m, (Detect, DDetect, Segment))` 走这条：gelan 系列的 DDetect 在此。
- **L552-555**：256 像素假输入跑一次前向，按输出特征图尺寸反推 stride（8/16/32）。
- **L559**：`m.bias_init()` —— 单头 bias 初始化，仅此一次。
- **L560**：双/三头分派 —— `isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect))` 走另一条：yolov9 系列的 DualDDetect 在此。
- **L564**：`forward = lambda x: self.forward(x)[0]` —— 假输入也要过一次完整前向（训练态返回 `[d1, d2]`，取 [0] 量 stride）。
- **L565-569**：同样的 stride 反推 + `m.bias_init()`（双头版内部主辅各刷一遍）。
- 两条路径几乎复制粘贴——单库多头的代价，与 [v8 tasks.py 的 stride 构建](/versions/v8/source/tasks-py#L189-L197) 同思路。
:::

## 与 v7/v8 的对比

| 维度 | v7 models/yolo.py | v8 tasks.py | v9 models/yolo.py |
|------|-------------------|-------------|-------------------|
| 头的住址 | 五头同文件 | 头在 nn/modules/head.py | 七头回同文件（v7 布局回归） |
| 检测头家族 | IDetect 等（anchor 系） | Detect（anchor-free 单头） | Detect/DDetect/DualDetect/DualDDetect/TripleDetect/TripleDDetect |
| 双头支持 | 无 | 无 | 主辅双分支 + 双 DFL + 双 assigner 配套 |
| 分组卷积 | 头内常见 | 无 | cv2/cv4 中间层 g=4（DDetect/DualDDetect） |
| anchor 参数 | parse_model 传入 anchors | 无 | yaml `anchors: 3` 残留但头不吃 |
| 缩放 | depth/width_multiple | + max_channels 复合缩放 | depth/width_multiple（每档独立 yaml，无 scales） |
| CBNet 分支 | 无 | 无 | CBLinear/CBFuse 专设分支 |

结构配置（这些头怎么被 yaml 点名）见 [yolov9-c.yaml 源码页](/versions/v9/source/yolov9-c-yaml)，积木实现见 [models/common.py 源码页](/versions/v9/source/common-py)，双头损失见 [loss_tal_dual.py 源码页](/versions/v9/source/loss-tal-dual-py)。
