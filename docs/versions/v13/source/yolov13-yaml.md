# ultralytics/cfg/models/v13/yolov13.yaml

v13 的四档结构清单：**50 行单份 yaml，`n/s/l/x` 四档通过内嵌 `scales` 块（L2-L7）在同一份 yaml 里表达**——沿用 v11/v12 的「一份 yaml 走多档」组织方式；目录名 `v13/` 带 v 前缀，与 v12 一致。结构上的核心变化是：**骨干 9 层保持 A2C2f 内嵌 Attention 的思路，但下采样改用 DSConv、C3k2 中段换成 DSC3k2；Head 首个 block 变成 HyperACE，之后是 FPN+PAN + 7 条 FullPAD_Tunnel 门控连接**。

> 快照路径：`source/yolov13/snapshot/ultralytics/cfg/models/v13/yolov13.yaml`（50 行，commit `73289949`）。点击左侧行号可复制/分享行锚点。

**这个文件在 v13 里的角色**：**50 行 yaml 里塞了 v13 全部 10 个新概念**——`DSC3k2 × 7` + `DSConv × 2` + `HyperACE × 1` + `DownsampleConv × 1` + `FullPAD_Tunnel × 7`——是 v13「DS 深度可分离 + HyperACE 超图融合 + FullPAD 门控连接」三主线的唯一映射。整张 yaml 32 层（v12 是 22 层），新增的 10 层全在 Head。

## 命名坑：目录名 `v13/` 带 v 前缀

v9 是 `ultralytics/cfg/models/9/`，v10 是 `ultralytics/cfg/models/v10/`，v11 又变回 `ultralytics/cfg/models/11/`（不带 v），v12/v13 都回归到 `ultralytics/cfg/models/v12|v13/`（带 v）。**五代 yaml 目录命名风格并不统一**——查代码时最好直接读 yaml 目录列表，不要靠「数字前/无 v」的直觉判断。

这个坑体现在三处：

1. **文件路径**：写代码时用 `yolov13.yaml`（`v13/` 目录下的完整文件名），别写成 `13/yolov13.yaml` 或 `v13/yolov13.yaml`；
2. **`guess_model_scale` 正则**：<SrcLink file="v13:ultralytics/nn/tasks.py" :line="1106" /> 里的 `yolo[v]?\d+([nslmx])`——**同时容忍带 v 与不带 v 两种写法**，所以 `yolov13n` 与 `yolo13n` 都能匹配；
3. **档位解析**：档位后缀（n/s/l/x）从 yaml 内嵌 `scales:` 查表，与 v11/v12 完全同款机制；注意 **v13 没有 m 档**——只有 n/s/l/x 四档。

**给读者的建议**：查 v13 源码时目录名用 `v13/`；文件名是 `yolov13.yaml`；档位只有 n/s/l/x。

## 关键行段速查

| 行段 | 内容 | 看点 |
|------|------|------|
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="1" /> | nc: 80 | 类别数 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="2" :end="7" /> | scales | **四档 compound scaling：n/s/l/x（无 m 档）**；l/x 档 `max_channels=512`（v12 是 1024） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="9" :end="19" /> | backbone | **9 层**：Conv / **DSC3k2** / **DSConv s2** / **A2C2f 内嵌 Attention** |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="22" /> | 层 9 **HyperACE** | **v13 招牌**：`[4, 6, 8], 512, 8, True, True, 0.5, 1, "both"` |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="24" /> | 层 11 DownsampleConv | v13 新：AvgPool 2 + 1×1 conv |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="25" :end="27" /> | 层 12-14 FullPAD_Tunnel × 3 | 骨干 P4/P3/P5 → 颈部 L11 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="32" /> | 层 18 FullPAD_Tunnel | 颈部内部（L17 + HyperACE） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="38" /> | 层 23 FullPAD_Tunnel | 颈部内部（L21 + Conv 1×1） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="43" /> | 层 27 FullPAD_Tunnel | 颈部→头（L26 + HyperACE） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="48" /> | 层 31 FullPAD_Tunnel | 颈部→头（L30 + DownsampleConv） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="50" /> | 层 32 Detect | 三尺度收尾 `[23, 27, 31]` |

<SourceViewer file="v13:ultralytics/cfg/models/v13/yolov13.yaml" />

## scales 块逐行解读（L2-L7）

```yaml
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]   # Nano
  s: [0.50, 0.50, 1024]   # Small
  l: [1.00, 1.00, 512]    # Large
  x: [1.00, 1.50, 512]    # Extra Large
```

- **无 m 档**：v12 有 `n/s/m/l/x` 五档，v13 只有 `n/s/l/x` 四档（跳过了 m）；
- **l/x 档 `max_channels=512`**（v12 是 1024）——**这是 v13 相对 v12 的一个隐性改动**：width=1.5 时通道翻倍上限被压低到 512，防止 x 档通道数失控；
- **depth 系数**：n/s 都是 0.50，l/x 都是 1.00——**Depthwise 缩放只按两档**，n/s 用 0.50 缩放 DSC3k2 的 repeats，l/x 保持原始 repeats=2/4。

## backbone 段逐层解读（L11-L19）

| 行号 | 层号 | 类型 | 说明 |
|------|-----|------|------|
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="11" /> | 0 | `Conv 64 3 2` | stem 下采样 · P1/2 · 64×320×320 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="12" /> | 1 | `Conv 128 3 2` | P2/4 · 128×160×160 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="13" /> | 2 | **`DSC3k2 256 False 0.25 ×2`** | 骨干浅层 · **dsc3k=False 走 DSBottleneck 内瓶** · e=0.25 · 256×160×160 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="14" /> | 3 | `Conv 256 3 2` | P3/8 · 256×80×80 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="15" /> | 4 | **`DSC3k2 512 False 0.25 ×2`** | 骨干 P3 · dsc3k=False · e=0.25 · 512×80×80 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="16" /> | 5 | **`DSConv 512 3 2`** | P4/16 · **v13 用 DSConv 替代 v12 的 Conv s2** · 512×40×40 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="17" /> | 6 | `A2C2f 512 True 4 ×4` | **v12 保留** · P4/16 · **a2=True · area=4 局部 Attention** |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="18" /> | 7 | **`DSConv 1024 3 2`** | P5/32 · v13 用 DSConv 替代 v12 的 Conv s2 · 1024×20×20 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="19" /> | 8 | `A2C2f 1024 True 1 ×4` | **v12 保留** · P5/32 · **a2=True · area=1 全局 Attention** |

**骨干的三处 v13 特征**：

1. **浅层两处 C3k2 换 DSC3k2**（#2/#4）：`dsc3k=False` 走 DSBottleneck 单段内瓶，参数与计算量约为 C3k2 的一半；
2. **两处下采样换 DSConv**（#5/#7）：`DWConv(3, s=2) + PWConv(1×1)`，stride 塞进 DWConv 内部；
3. **两处 A2C2f 完全保留**（#6/#8）：v12 引入的骨干注意力块原样沿用——**v13 没有替换 A2C2f**，而是**在其后加 HyperACE** 做骨干/颈部主通路的高阶融合。

## head 段逐层解读（L22-L50）

| 行号 | 层号 | 类型 | 说明 |
|------|-----|------|------|
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="22" /> | 9 | **`HyperACE [4,6,8] 512 8 True True 0.5 1 "both"`** | **v13 招牌**：fuse+chunk(3)+2×C3AH+DSC3k 并行 → 512 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="23" /> | 10 | `nn.Upsample [None, 2, "nearest"]` | HyperACE 上采样到 P4 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="24" /> | 11 | **`DownsampleConv []`** | AvgPool(2) + Conv(1×1, 2×c) · 512→1024 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="25" /> | 12 | **`FullPAD_Tunnel [6, 11]`** | gate * 骨干 P4 A2C2f → L11 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="26" /> | 13 | **`FullPAD_Tunnel [4, 11]`** | gate * 骨干 P3 DSC3k2 → L11 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="27" /> | 14 | **`FullPAD_Tunnel [8, 11]`** | gate * 骨干 P5 A2C2f → L11 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="29" /> | 15 | `nn.Upsample [None, 2, "nearest"]` | L11 上采样到 P3 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="30" /> | 16 | `Concat [-1, 12]` | cat 骨干 P4 tunnel 结果 → 1536 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="31" /> | 17 | **`DSC3k2 512 True ×2`** | **dsc3k=True 走 DSC3k 骨架** · 512×80×80 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="32" /> | 18 | **`FullPAD_Tunnel [-1, 9]`** | gate * 颈部 L9 HyperACE → L17 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="34" /> | 19 | `nn.Upsample [None, 2, "nearest"]` | L17 上采样到 P2 尺度 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="35" /> | 20 | `Concat [-1, 13]` | cat 骨干 P3 tunnel 结果 → 768 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="36" /> | 21 | **`DSC3k2 256 True ×2`** | P3 检测输入 · **stride 8** |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="37" /> | 22 | `Conv 256 1 1` | 1×1 通道规整 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="38" /> | 23 | **`FullPAD_Tunnel [21, 22]`** | gate * L22 → L21（颈部内部再融） |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="40" /> | 24 | `Conv 256 3 2` | PAN 下行 · 回到 P4 分辨率 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="41" /> | 25 | `Concat [-1, 18]` | cat 颈部内部 FullPAD tunnel → 768 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="42" /> | 26 | **`DSC3k2 512 True ×2`** | P4 检测输入 · **stride 16** |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="43" /> | 27 | **`FullPAD_Tunnel [-1, 9]`** | gate * 颈部 L9 HyperACE → L26 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="45" /> | 28 | `Conv 512 3 2` | PAN 下行 · 回到 P5 分辨率 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="46" /> | 29 | `Concat [-1, 14]` | cat 骨干 P5 tunnel 结果 → 1536 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="47" /> | 30 | **`DSC3k2 1024 True ×2`** | P5 检测输入 · **stride 32** |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="48" /> | 31 | **`FullPAD_Tunnel [-1, 11]`** | gate * 颈部 L11 DownsampleConv → L30 |
| <SrcLink file="v13:ultralytics/cfg/models/v13/yolov13.yaml" :line="50" /> | 32 | `Detect [23, 27, 31]` | 三尺度收尾 · **输入位置在 FullPAD tunnel 之后** |

## 7 条 FullPAD_Tunnel 一览表

| 序号 | 层号 | from 索引 | 语义 | 类别 |
|------|------|-----------|------|------|
| 1 | #12 | [6, 11] | 骨干 P4 → 颈部 L11 | 骨干↔颈部 |
| 2 | #13 | [4, 11] | 骨干 P3 → 颈部 L11 | 骨干↔颈部 |
| 3 | #14 | [8, 11] | 骨干 P5 → 颈部 L11 | 骨干↔颈部 |
| 4 | #18 | [-1, 9] | 颈部 L17 + HyperACE | 颈部内部 |
| 5 | #23 | [21, 22] | L21 + L22 Conv 1×1 | 颈部内部 |
| 6 | #27 | [-1, 9] | 颈部 L26 + HyperACE | 颈部→头 |
| 7 | #31 | [-1, 11] | 颈部 L30 + DownsampleConv | 颈部→头 |

**分布规律**：前 3 条走「骨干 → 颈部枢纽 L11」，中间 2 条走「颈部内部再融」，后 2 条走「颈部 → Detect 之前」——**每个尺度都至少被一条 FullPAD tunnel 触达**，Detect 的三个输入都不是"干净"的 DSC3k2 输出。

## 论文消融：FullPAD 单独增益（YOLOv13-S）

| 骨干↔颈部 | 颈部内部 | 颈部→头 | AP50:95 |
|-----------|---------|---------|---------|
| 0 | 0 | 0 | 47.1 |
| 1 | 0 | 0 | 47.8 |
| 0 | 1 | 0 | 47.6 |
| 0 | 0 | 1 | 47.7 |
| 1 | 1 | 1 | **48.0** |

三条单独消融增益都在 0.5-0.7 AP 之间；**联合启用 111 才把增益叠加到 0.9 AP**。这也是 v13 把 7 条 FullPAD tunnel 一次性引入 yaml 的原因。

## 与 v12 yaml 的规模对比

| 维度 | v12 yaml | **v13 yaml** |
|------|---------|-------------|
| 文件行数 | 45 | **50** |
| 档位 | n/s/m/l/x（5 档） | **n/s/l/x（4 档，无 m）** |
| l/x 档 max_channels | 1024 | **512** |
| 骨干层数 | 9 | 9 |
| Head 层数 | 13 | **23** |
| 总层数 | 22 | **32** |
| 新引入模块 | A2C2f / AAttn / ABlock | **DSC3k2 / DSC3k / DSBottleneck / HyperACE / FullPAD_Tunnel / DownsampleConv / AdaHyperedgeGen / AdaHGConv / AdaHGComputation / C3AH / FuseModule** |
| Detect 位置 | L45 | L50 |

## 给读者的建议

- **查源码**：目录名 `v13/`；文件名 `yolov13.yaml`；档位 `n/s/l/x`；
- **想从 v12 快速迁移**：把骨干 C3k2 换成 DSC3k2、Conv s2 换成 DSConv s2，其余（A2C2f、Detect、损失、TAL）完全一致；
- **Head 从零读**：v13 的 Head 首个 block 是 HyperACE（不是 Upsample/Concat），之后是 7 条 FullPAD_Tunnel 门控连接——**建议从 [architecture 逐层图解](/versions/v13/architecture) 和 [HyperACE 概念页](/versions/v13/concepts) 入手**。

## 参考

- [YOLOv13 架构与层速查](/versions/v13/layers)
- [Architecture 逐层图解](/versions/v13/architecture)
- [理论 Concept](/versions/v13/concepts)
- [block.py 源码页](/versions/v13/source/block-py)
- [conv.py 源码页](/versions/v13/source/conv-py)
- Yang W. *YOLOv13.* arXiv:2506.17733v2, 2025-11. https://arxiv.org/abs/2506.17733
