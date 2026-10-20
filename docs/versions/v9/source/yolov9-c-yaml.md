# models/detect/yolov9-c.yaml

这是 YOLOv9 快照 `models/detect/yolov9-c.yaml` 的源码浏览页：**124 行的双头检测配置**——backbone 0-9、head 主干 10-22、**多级可逆辅助分支 23-37**、DualDDetect 收尾。对比 v8 的 46 行：v9 没有 v8 式 `scales` 复合缩放（每档独立 yaml），多出来的近 80 行几乎全花在辅助分支上——PGI 的结构落点就在这段 yaml 里。

> 快照路径：`source/yolov9/snapshot/models/detect/yolov9-c.yaml`。点击左侧行号可复制/分享行锚点。

<SourceViewer file="v9:models/detect/yolov9-c.yaml" />

**阅读路线**：

- [L4-6 nc/depth/width](#L4-L6)：nc=80，depth/width_multiple 均 1.0——无 scales 复合缩放，v9 各档（t/s/m/c/e）各写各的 yaml
- [L11 anchors: 3](#L11)：v7 字段残留——parse_model 里只剩 `no = na*(nc+5)` 的判断用途，DualDDetect 不吃 anchor（见 [yolo-py 源码页](/versions/v9/source/yolo-py#L666-L667)）
- backbone [L14-44](#L14-L44)：Silence 占位 → 双 Conv 下采样 → RepNCSPELAN4 ×4，ADown 全程降采样
- head 主干 [L47-78](#L47-L78)：SPPELAN 收尾 + FPN/PAN 双向融合，出 P3/P4/P5 主特征（层 16/19/22）
- 辅助分支 [L81-116](#L81-L116)：CBLinear 从骨干 5/7/9 抽头，CBFuse 三级回注，出 A3/A4/A5 辅助特征（层 31/34/37）——PGI 主张的结构化载体
- 检测头 [L120-124](#L120-L124)：`[[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]]`——辅助 3 路在前、主 3 路在后

## 关键片段逐行走读

> 已读过 [v7 yolov7.yaml](/versions/v7/source/yolov7-yaml)/[v8 yolov8.yaml](/versions/v8/source/yolov8-yaml)？v7 读者注意：anchor 组没了（只剩字段残留）、MP-Conv 换成 ADown；v8 读者只需关注 L81 起的辅助分支段——这是 v8 配置里完全不存在的部分。

### backbone 段（L14-44）：Silence 占位 + ADown 降采样

<SrcSnippet file="v9:models/detect/yolov9-c.yaml" :start="14" :end="44" hl="16,19,22,25,28,31,34,37,40,43" title="yolov9-c.yaml backbone：层 0-9" />

::: details 逐行注释（L14–L44）
- **L16**：`[-1, 1, Silence, []]` —— 第 0 层是**恒等层**：不改数据，只为给"原始输入"一个可被 from 引用的层号（辅助分支 L89 的 from 0 指向它）。
- **L19**：`Conv [64, 3, 2]` → 层 1，P1/2。
- **L22**：`Conv [128, 3, 2]` → 层 2，P2/4。
- **L25**：`RepNCSPELAN4 [256, 128, 64, 1]` → 层 3 —— args 依次是 c2=256、c3=128、c4=64、c5=1（重复数），对应 [common.py L582](/versions/v9/source/common-py#L582) 的形参表。
- **L28**：`ADown [256]` → 层 4，P3/8 —— 均值/极值两路下采样（见 [common-py 源码页](/versions/v9/source/common-py#L70-L83)）。
- **L31**：`RepNCSPELAN4 [512, 256, 128, 1]` → 层 5。
- **L34**：`ADown [512]` → 层 6，P4/16。
- **L37**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 7。
- **L40**：`ADown [512]` → 层 8，P5/32。
- **L43**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 9 —— 骨干收尾；width_multiple=1.0，通道按 yaml 字面值走。
:::

### head 主干段（L47-78）：SPPELAN + FPN/PAN 双向融合

<SrcSnippet file="v9:models/detect/yolov9-c.yaml" :start="47" :end="78" hl="50,53,54,57,60,61,64,67,68,71,74,75,78" title="yolov9-c.yaml head 主干：层 10-22" :max-height="560" />

::: details 逐行注释（L47–L78）
- **L50**：`SPPELAN [512, 256]` → 层 10 —— SPP-ELAN 感受野收尾（c2=512、c3=256），接在骨干层 9 之后。
- **L53-54**：Upsample ×2 + `Concat [-1, 7]` —— FPN 上行第一步，接骨干 P4（层 7）。
- **L57**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 13。
- **L60-61**：再 Upsample + `Concat [-1, 5]` —— 接骨干 P3（层 5）。
- **L64**：`RepNCSPELAN4 [256, 256, 128, 1]` → 层 16 —— **P3/8 主特征**（small），输出通道 256。
- **L67-68**：`ADown [256]` + `Concat [-1, 13]` —— PAN 下行第一步，回接 head P4（层 13）→ 层 17-18。
- **L71**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 19 —— **P4/16 主特征**（medium）。
- **L74-75**：`ADown [512]` + `Concat [-1, 10]` —— 回接 SPPELAN 输出（层 10）→ 层 20-21。
- **L78**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 22 —— **P5/32 主特征**（large）。
- 至此主分支与 v8 的 FPN/PAN 拓扑同构——v9 的结构新意全部在下面这段。
:::

### 辅助分支段（L81-116）：CBLinear 抽头 + CBFuse 回注

<SrcSnippet file="v9:models/detect/yolov9-c.yaml" :start="81" :end="116" hl="84,85,86,89,92,95,98,99,102,105,106,109,112,113,116" title="yolov9-c.yaml 辅助分支：层 23-37" :max-height="620" />

::: details 逐行注释（L81–L116）
- **L84-86**：三根 CBLinear 抽头 —— `[5, 1, CBLinear, [[256]]]`（层 23）、`[7, 1, CBLinear, [[256, 512]]]`（层 24）、`[9, 1, CBLinear, [[256, 512, 512]]]`（层 25）：从骨干层 5/7/9 各用一个卷积输出多路通道备用（见 [common-py 源码页](/versions/v9/source/common-py#L631-L639) 的 split 实现）。**越深抽头拆得越多**——供不同尺度的 CBFuse 各取一路。
- **L89**：`[0, 1, Conv, [64, 3, 2]]` → 层 26 —— 辅助主干的起点是 **from 0**（Silence，即原始输入）：不走骨干主路，从输入重新下采样。
- **L92**：`Conv [128, 3, 2]` → 层 27，P2/4。
- **L95**：`RepNCSPELAN4 [256, 128, 64, 1]` → 层 28 —— 与骨干层 3 同构。
- **L98**：`ADown [256]` → 层 29，P3/8。
- **L99**：`[[23, 24, 25, -1], 1, CBFuse, [[0, 0, 0]]]` → 层 30 —— 取层 23/24/25 的**第 0 路**特征，interpolate 到层 29 的尺寸逐元素相加（见 [common-py 源码页](/versions/v9/source/common-py#L641-L650)）。
- **L102**：`RepNCSPELAN4 [512, 256, 128, 1]` → 层 31 —— **A3 辅助特征**。
- **L105-106**：`ADown [512]` → 层 32 + `CBFuse [24, 25]` 取各自身第 1 路 → 层 33。
- **L109**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 34 —— **A4 辅助特征**。
- **L112-113**：`ADown [512]` → 层 35 + `CBFuse [25]` 取第 2 路 → 层 36。
- **L116**：`RepNCSPELAN4 [512, 512, 256, 1]` → 层 37 —— **A5 辅助特征**。
- 全段只有 CBLinear/CBFuse 两处带参数/路由的"接头"，其余是骨干的复制品——可逆分支的训练期算力就花在这里，部署期整段可手动裁剪（见 [v9 痛点页](/versions/v9/painpoints) 痛点 7）。
:::

### 检测头行（L120-124）：DualDDetect 六路输入

<SrcSnippet file="v9:models/detect/yolov9-c.yaml" :start="120" :end="124" hl="123" title="yolov9-c.yaml：DualDDetect 收尾" />

::: details 逐行注释（L120–L124）
- **L123**：`[[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]]` —— 六路 from：**前 3 路（31/34/37）是辅助 A3/A4/A5，后 3 路（16/19/22）是主 P3/P4/P5**。DualDDetect 里 `ch[:nl]` 反而切出主分支——顺序别记反。
- 参数只有 `[nc]` —— anchors 时代的参数表只剩类别数，6 路通道由 parse_model 的 Detect 分支自动附上（[yolo-py 源码页](/versions/v9/source/yolo-py#L704-L709)）。
- 对照：`gelan-c.yaml` 的收尾无辅助分支、单头单 DFL：

```yaml
# gelan-c.yaml head 收尾（论文消融的 GELAN 基线）
[[15, 18, 21], 1, DDetect, [nc]]
```

两份 yaml 的差集 = CBLinear/CBFuse 辅助段 + DualDDetect vs DDetect——论文"GELAN 结构涨点"与"PGI 辅助涨点"的消融接口就在这里。
:::

## 与 v7/v8 的对比

| 维度 | v7 yolov7.yaml | v8 yolov8.yaml | v9 yolov9-c.yaml |
|------|----------------|----------------|------------------|
| 行数 | 140 行 | 46 行 | 124 行（近半是辅助分支） |
| anchor | 三组写死 | 无 anchor 字段 | `anchors: 3` 字段残留，头不吃 |
| 骨干 | ELAN 多路 Concat + MP | C2f ×4 + SPPF | Silence + RepNCSPELAN4 ×4 |
| 下采样 | MaxPool + Conv | Conv s2 | ADown（均值+极值两路） |
| 感受野 | SPPCSPC（层 51） | SPPF（层 9） | SPPELAN（层 10） |
| Neck | PANet + RepConv | Upsample/Concat + C2f | 同构 v8，积木换 RepNCSPELAN4 |
| 辅助分支 | 无（aux head 在训练代码） | 无 | CBLinear/CBFuse 多级可逆分支（层 23-37） |
| 检测头 | IDetect，[nc, anchors] | Detect，[nc] | DualDDetect，[nc]，六路输入 |
| 缩放 | depth/width_multiple | scales 五档复合 | depth/width_multiple（每档独立 yaml） |

积木实现见 [models/common.py 源码页](/versions/v9/source/common-py)，建网与双头见 [models/yolo.py 源码页](/versions/v9/source/yolo-py)，双头损失见 [loss_tal_dual.py 源码页](/versions/v9/source/loss-tal-dual-py)，逐层解读见 [YOLOv9 layers](/versions/v9/layers)。
