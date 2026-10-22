# nn/modules/head.py

**这个文件在 v26 里的角色**：v26 的头文件——重头戏。集中定义了 End2End 双分支 `Detect`（`copy.deepcopy` 复制出 one2one 分支）、9 种任务头（`Detect / Segment26 / OBB26 / Pose26 / Depth / SemanticSegment / Classify / YOLOEDetect / YOLOESegment26`）、辅助组件（`BNContrastiveHead`、`SAVPE`、`LRPCHead`），以及 v26 独有的 `_grouped_topk` 加速算子。

## End2End 双分支（核心机制）

### `_grouped_topk`

<SrcLink file="v26:nn/modules/head.py" :line="92" :end="92"/> `@staticmethod _grouped_topk(x, k, groups=8)`——v26 新增的分组 topk：**把 anchor 维度切成 `groups` 份，每份各取 top-k，再全局取 top-k**；比单次 `topk(k)` 更快，因为分组后每组数据量小 8 倍。核心循环在 <SrcLink file="v26:nn/modules/head.py" :line="95" :end="102"/>：先 `reshape` 分块，逐块 `topk(k)`，`flatten` 后再取一次 topk，最后把局部索引还原成全局索引（`winners // k * size + index.flatten(1).gather(1, winners)`）。

### `Detect.__init__` 的 `end2end=True` 分支

<SrcLink file="v26:nn/modules/head.py" :line="108" :end="108"/> `Detect.__init__(nc, reg_max, end2end, ch)`；<SrcLink file="v26:nn/modules/head.py" :line="139" :end="139"/> `self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()`——**`reg_max=1` 时 `DFL` 失效**；<SrcLink file="v26:nn/modules/head.py" :line="141" :end="143"/> `if end2end: self.one2one_cv2 = copy.deepcopy(self.cv2); self.one2one_cv3 = copy.deepcopy(self.cv3)`——**这里就是 End2End 双分支的物理入口**。

### `Detect.end2end` property

<SrcLink file="v26:nn/modules/head.py" :line="155" :end="158"/> `end2end` getter：`return one2one_cv2 is not None and (_end2end or cv2 is None)`——**融合后 cv2 被置 None 时自动 fallback 到 one2one**。
<SrcLink file="v26:nn/modules/head.py" :line="160" :end="167"/> setter：带日志警告的两向切换。

### `Detect.forward`

<SrcLink file="v26:nn/modules/head.py" :line="180" :end="204"/> forward 主流程：
- <SrcLink file="v26:nn/modules/head.py" :line="194" :end="194"/> 先跑 `forward_head(x, **self.one2many)` 得到 `one2many` preds。
- <SrcLink file="v26:nn/modules/head.py" :line="195" :end="198"/> 如果有 `one2one_cv2`，`x_detach = [xi.detach() for xi in x] if self.training else x`——**训练时 detach backbone 输出，防止 one2one 梯度污染 backbone**——然后跑一次 one2one 分支，输出 dict `{"one2many": ..., "one2one": ...}`。
- <SrcLink file="v26:nn/modules/head.py" :line="201" :end="203"/> 推理时根据 `self.end2end` 选择输出分支，再 `postprocess` 做 top-k。

### `Detect.postprocess` 中的 `_grouped_topk`

<SrcLink file="v26:nn/modules/head.py" :line="282" :end="287"/> 三次调用 `_grouped_topk`：
- <SrcLink file="v26:nn/modules/head.py" :line="282" :end="282"/> `agnostic_nms=True` 时先按类别维度 topk 得到 index。
- <SrcLink file="v26:nn/modules/head.py" :line="285" :end="285"/> `agnostic_nms=False` 时先取每 anchor 的最大类分（`scores.max(-1)`），走分组 topk 得到 index。
- <SrcLink file="v26:nn/modules/head.py" :line="287" :end="287"/> 最终选 top-k。

## 9 种 v26 任务头

| 类 | 起止行 | 用途 | 关键继承 |
| --- | --- | --- | --- |
| <SrcLink file="v26:nn/modules/head.py" :line="38" :end="296"/> `Detect` | 主线检测 | — | nn.Module |
| <SrcLink file="v26:nn/modules/head.py" :line="298" :end="391"/> `Segment` | 实例分割基础 | Detect | Detect |
| <SrcLink file="v26:nn/modules/head.py" :line="393" :end="462"/> `Segment26` | v26 实例分割，End2End 支持 | Segment | Segment |
| <SrcLink file="v26:nn/modules/head.py" :line="464" :end="536"/> `OBB` | 旋转框基础 | Detect | Detect |
| <SrcLink file="v26:nn/modules/head.py" :line="538" :end="570"/> `OBB26` | v26 旋转框，End2End 支持 | OBB | OBB |
| <SrcLink file="v26:nn/modules/head.py" :line="572" :end="661"/> `Pose` | 关键点基础 | Detect | Detect |
| <SrcLink file="v26:nn/modules/head.py" :line="663" :end="783"/> `Pose26` | v26 关键点 + RealNVP | Pose | Pose |
| <SrcLink file="v26:nn/modules/head.py" :line="785" :end="870"/> `Depth` | 单目深度 | — | nn.Module |
| <SrcLink file="v26:nn/modules/head.py" :line="872" :end="932"/> `Classify` | 图像分类 | — | nn.Module |
| <SrcLink file="v26:nn/modules/head.py" :line="1091" :end="1336"/> `YOLOEDetect` | 开集检测 | Detect | Detect |
| <SrcLink file="v26:nn/modules/head.py" :line="1338" :end="1459"/> `YOLOESegment` | 开集分割基础 | YOLOEDetect | YOLOEDetect |
| <SrcLink file="v26:nn/modules/head.py" :line="1461" :end="1526"/> `YOLOESegment26` | v26 开集分割 | YOLOESegment | YOLOESegment |
| <SrcLink file="v26:nn/modules/head.py" :line="1926" :end="1975"/> `SemanticSegment` | 语义分割 | — | nn.Module |

## 各 26 版本头的 `end2end=True` 分支

| 头 | end2end 分支行段 | deepcopy 的层 |
| --- | --- | --- |
| <SrcLink file="v26:nn/modules/head.py" :line="393" :end="393"/> `Segment26` | 继承自父类 <SrcLink file="v26:nn/modules/head.py" :line="345" :end="346"/> | cv2, cv3, cv4（mask）；`Segment26` 自身仅换 `Proto → Proto26` (<SrcLink file="v26:nn/modules/head.py" :line="434" :end="434"/>) |
| <SrcLink file="v26:nn/modules/head.py" :line="538" :end="538"/> `OBB26` | 继承自父类 <SrcLink file="v26:nn/modules/head.py" :line="501" :end="502"/> | cv4（angle） |
| <SrcLink file="v26:nn/modules/head.py" :line="663" :end="663"/> `Pose26` | <SrcLink file="v26:nn/modules/head.py" :line="711" :end="714"/> | cv4, cv4_kpts, cv4_sigma |
| <SrcLink file="v26:nn/modules/head.py" :line="1091" :end="1091"/> `YOLOEDetect` | <SrcLink file="v26:nn/modules/head.py" :line="1159" :end="1161"/> | cv3, cv4 |
| <SrcLink file="v26:nn/modules/head.py" :line="1461" :end="1461"/> `YOLOESegment26` | <SrcLink file="v26:nn/modules/head.py" :line="1504" :end="1505"/> | cv5 |

## `Pose26` 与 `RealNVP`

<SrcLink file="v26:nn/modules/head.py" :line="701" :end="701"/> `Pose26.__init__` 关键行：`self.flow_model = RealNVP()`——把 <SrcLink file="v26:nn/modules/block.py" :line="2045" :end="2045"/> 可逆耦合流嵌入头部，用于对 kpt 输出做非参数化自适应变换。

## `Depth` 头

<SrcLink file="v26:nn/modules/head.py" :line="785" :end="785"/> `class Depth(nn.Module)`；<SrcLink file="v26:nn/modules/head.py" :line="808" :end="808"/> `__init__(c_mid=256, ch)`——`c_mid` 是瓶颈通道数；<SrcLink file="v26:nn/modules/head.py" :line="837" :end="837"/> `forward` 返回 `{"depth": pred_depth}`；损失走 <SrcLink file="v26:utils/loss.py" :line="1169" :end="1169"/> `DepthLoss26`（SILog + L1 梯度）。

## `SemanticSegment` 头

<SrcLink file="v26:nn/modules/head.py" :line="1926" :end="1926"/> `class SemanticSegment(nn.Module)`；<SrcLink file="v26:nn/modules/head.py" :line="1947" :end="1947"/> `__init__(nc=19, ch)`——默认 Cityscapes 19 类；损失走 <SrcLink file="v26:utils/loss.py" :line="1389" :end="1389"/> `SemanticSegmentationLoss`（CE `ignore_index=255` + Dice）。

## `YOLOEDetect` 关键行

- <SrcLink file="v26:nn/modules/head.py" :line="1123" :end="1123"/> `__init__(nc, embed=512, with_bn=True, reg_max=16, end2end=False, ch=())`。
- <SrcLink file="v26:nn/modules/head.py" :line="1158" :end="1158"/> `self.cv4 = nn.ModuleList(BNContrastiveHead(embed) for _ in ch)`——**用 BNContrastiveHead 替代普通分类头**，把 BN 的 `weight` 作为类别 embedding。
- <SrcLink file="v26:nn/modules/head.py" :line="1164" :end="1164"/> `self.savpe = SAVPE(ch, c3, embed)`——注入视觉 prompt。
- <SrcLink file="v26:nn/modules/head.py" :line="1259" :end="1261"/> forward 里的分支选择：`cv2 = self.one2one_cv2 if self.end2end else self.cv2`。

## SourceViewer

<SourceViewer file="v26:nn/modules/head.py"/>

## 参考

- [End2End 概念](../concepts.md)
- [Losses](../losses.md)
- [Block 源码](./block-py.md)
- [Loss 源码](./loss-py.md)
- [TAL 源码](./tal-py.md)
- [主线 YAML](./yolo26-yaml.md)
- [v26 主页](../index.md)
