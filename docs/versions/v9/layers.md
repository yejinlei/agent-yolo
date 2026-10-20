# YOLOv9 层结构速查

> 以 `models/detect/yolov9-c.yaml` 全文 124 行为准：backbone + head 共 **39 个顶层条目（#0~#38）**——主干 #0~#22、多级可逆辅助分支 #23~#37、DualDDetect #38。通道数为 yaml 声明值（depth/width_multiple 均为 1.0，无缩放），@640 输入。配置原文见 <SrcLink file="v9:models/detect/yolov9-c.yaml" :line="1" :end="124" />。

> **读者路标**：想先看模块内部数据流（RepNCSPELAN4 / ADown / 辅助分支图解）与整体拓扑，去 [YOLOv9 架构总览](/versions/v9/architecture)；本页是查表页——层号、参数、通道一张表对完。

## 1. 逐层明细表（yolov9-c.yaml #0~#38，640 输入）

| # | from | module | 参数（yaml 原文） | 输出（@640） | 说明 |
|-----|------------|--------------|----------------------|--------------------|------|
| 0 | -1 | `Silence` | [] | 3×640×640 | 恒等层 · 辅助分支抽头锚点 |
| 1 | -1 | `Conv` | 64, 3, 2 | 64×320×320 | stem 下采样 · P1/2 |
| 2 | -1 | `Conv` | 128, 3, 2 | 128×160×160 | P2/4 |
| 3 | -1 | `RepNCSPELAN4` | 256, 128, 64, 1 | 256×160×160 | elan-1 块 |
| 4 | -1 | `ADown` | 256 | 256×80×80 | avg-conv down · P3/8 |
| 5 | -1 | `RepNCSPELAN4` | 512, 256, 128, 1 | 512×80×80 | elan-2 块 · CBLinear #23 抽头 |
| 6 | -1 | `ADown` | 512 | 512×40×40 | P4/16 |
| 7 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×40×40 | CBLinear #24 抽头 · FPN 回连点 |
| 8 | -1 | `ADown` | 512 | 512×20×20 | P5/32 |
| 9 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×20×20 | 骨干出口 · CBLinear #25 抽头 |
| 10 | -1 | `SPPELAN` | 512, 256 | 512×20×20 | elan-spp 块 · PAN 顶点 |
| 11 | -1 | `nn.Upsample` | None, 2, 'nearest' | 512×40×40 | FPN 上行 |
| 12 | [-1, 7] | `Concat` | 1 | 1024×40×40 | cat 骨干 P4 |
| 13 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×40×40 | |
| 14 | -1 | `nn.Upsample` | None, 2, 'nearest' | 512×80×80 | |
| 15 | [-1, 5] | `Concat` | 1 | 1024×80×80 | cat 骨干 P3 |
| 16 | -1 | `RepNCSPELAN4` | 256, 256, 128, 1 | 256×80×80 | **P3 检测输入 · stride 8** |
| 17 | -1 | `ADown` | 256 | 256×40×40 | PAN 下行 |
| 18 | [-1, 13] | `Concat` | 1 | 768×40×40 | cat 头部 P4 |
| 19 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×40×40 | **P4 检测输入 · stride 16** |
| 20 | -1 | `ADown` | 512 | 512×20×20 | PAN 下行 |
| 21 | [-1, 10] | `Concat` | 1 | 1024×20×20 | cat SPPELAN 输出 |
| 22 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×20×20 | **P5 检测输入 · stride 32** |
| 23 | 5 | `CBLinear` | [[256]] | tuple · 1 份 256×80×80 | 辅助路由 · 从骨干 #5 抽头 |
| 24 | 7 | `CBLinear` | [[256, 512]] | tuple · 256+512 | 从骨干 #7 抽头拆 2 份 |
| 25 | 9 | `CBLinear` | [[256, 512, 512]] | tuple · 256+512+512 | 从骨干 #9 抽头拆 3 份 |
| 26 | 0 | `Conv` | 64, 3, 2 | 64×320×320 | 辅助主干起点 · 从原始输入重跑 |
| 27 | -1 | `Conv` | 128, 3, 2 | 128×160×160 | |
| 28 | -1 | `RepNCSPELAN4` | 256, 128, 64, 1 | 256×160×160 | |
| 29 | -1 | `ADown` | 256 | 256×80×80 | avg-conv down · P3/8 |
| 30 | [23, 24, 25, -1] | `CBFuse` | [[0, 0, 0]] | 256×80×80 | 三份"第 1 拆"逐元素相加 |
| 31 | -1 | `RepNCSPELAN4` | 512, 256, 128, 1 | 512×80×80 | **A3 辅助检测输入** |
| 32 | -1 | `ADown` | 512 | 512×40×40 | P4/16 |
| 33 | [24, 25, -1] | `CBFuse` | [[1, 1]] | 512×40×40 | 两份"第 2 拆"相加 |
| 34 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×40×40 | **A4 辅助检测输入** |
| 35 | -1 | `ADown` | 512 | 512×20×20 | P5/32 |
| 36 | [25, -1] | `CBFuse` | [[2]] | 512×20×20 | "第 3 拆"相加 |
| 37 | -1 | `RepNCSPELAN4` | 512, 512, 256, 1 | 512×20×20 | **A5 辅助检测输入** |
| 38 | [31, 34, 37, 16, 19, 22] | `DualDDetect` | nc | 6 路 · 每位置 144 通道 | 双头收官 · A3/A4/A5 + P3/P4/P5 |

- 重复数外层全为 1：`parse_model` 的深度增益 `n = max(round(n × gd), 1) if n > 1`（<SrcLink file="v9:models/yolo.py" :line="676" />）在本 yaml 无事可做——RepNCSPELAN4 的"c5 重复"写死在参数元组第 4 位（模块内部展开）
- 宽度增益 gw=1.0：`c2 = make_divisible(c2 × 1.0, 8)` 原样通过（<SrcLink file="v9:models/yolo.py" :line="681" :end="685" />）——v9 无 v8 式 scales 复合缩放，**一档一份独立 yaml**
- `from` 支持负索引与显式层号：#12/#15 回连骨干 #7/#5，#21 回连 #10；#23/#24/#25 与 #26 是辅助分支的"隔空取物"（取骨干 #5/#7/#9 与 Silence #0）
- CBLinear 的"输出"是**多份张量的 tuple**：表中通道列写拆分方案；CBFuse 输出 = from 末位（ADown 路）的通道——相加不改通道（<SrcLink file="v9:models/yolo.py" :line="701" :end="702" />）
- Concat 通道求和（<SrcLink file="v9:models/yolo.py" :line="691" :end="692" />）；Upsample 不改通道；DualDDetect 的 6 路通道表由 parse_model append 进 args（<SrcLink file="v9:models/yolo.py" :line="704" :end="705" />）

## 2. 输出张量布局（双头 ×3 尺度）

| 模式 | 返回 | 形状 | 出处 |
|------|------|------|------|
| 训练 | `[d1, d2]` 两个三尺度列表 | d1/d2 各 (bs, 144, 80/40/20) | <SrcLink file="v9:models/yolo.py" :line="227" :end="228" /> |
| 推理 | `(y, [d1, d2])` 元组 | y = [辅路解码 (bs, 84, 8400), 主路解码 (bs, 84, 8400)] | <SrcLink file="v9:models/yolo.py" :line="233" :end="238" /> |

- 每位置 144 = \(4 \times 16 + 80\)，与 v8 相同：64 维 box 距离分布（[DFL](/glossary#dfl-distribution-focal-loss) 期望值积分解码）+ 80 类分数，**无 objectness**
- 预测数 8400 = \(80^2 + 40^2 + 20^2\)（stride 8/16/32，640 输入），[anchor](/glossary#anchor-锚框-先验框) 即网格中心——v8 的 anchor-free 公式全盘继承
- 主路/辅路各一个 DFL（dfl/dfl2）；`val_dual.py` L199 取 `preds[0][1]` 即**主路**解码输出做验证——辅路输出只在训练期有意义
- 训练态 d1/d2 分别进 `loss_tal_dual.py` 的两套 [TAL](/glossary#tal-task-alignment-learning) 分配（双路损失详见[源码走读](/versions/v9/code-walkthrough)）

## 3. 层类型统计（39 条账本）

| 类型 | 条数 | 层号分布 |
|------|------|----------|
| `Conv` | 4 | 1, 2（主干）· 26, 27（辅助） |
| `RepNCSPELAN4` | 12 | 3, 5, 7, 9（骨干）· 13, 16, 19, 22（头）· 28, 31, 34, 37（辅助） |
| `ADown` | 8 | 4, 6, 8, 17, 20（主干 5 条）· 29, 32, 35（辅助 3 条） |
| `CBLinear` | 3 | 23, 24, 25 |
| `CBFuse` | 3 | 30, 33, 36 |
| `Silence` | 1 | 0 |
| `SPPELAN` | 1 | 10 |
| `nn.Upsample` | 2 | 11, 14 |
| `Concat` | 4 | 12, 15, 18, 21 |
| `DualDDetect` | 1 | 38 |

- 合计 4+12+8+3+3+1+1+2+4+1 = **39**，与 yaml 逐条对得上
- 对比 v8：条目 23 → 39，多出的 16 条（辅助分支 15 + 双头增路）全是训练期基建——**主干本体 #0~#22 与 v8"骨干 10 条 + 头 13 条"的节奏几乎一致**
- 模块家族从 v8 的 block.py/head.py 收回 v7 式单文件 `models/common.py`（1200 行）——仓库是 YOLOv7 代码库 fork 的直接证据

## 4. 模块速查（作用 + 装配规则）

| 模块 | 位置 | 一句话作用 | yaml 使用 |
|------|------|-----------|-----------|
| `RepNCSPELAN4` | <SrcLink file="v9:models/common.py" :line="580" :end="598" /> | GELAN 核心块：cv1 升维 → chunk 两半 → cv2/cv3 两级级联 → 4 路 concat → cv4 收拢 | 12 |
| `ADown` | <SrcLink file="v9:models/common.py" :line="70" :end="83" /> | avgpool 预平滑 → chunk 两半 → 一半 conv3s2 / 一半 maxpool+conv1 → concat | 8 |
| `AConv` | <SrcLink file="v9:models/common.py" :line="60" :end="67" /> | ADown 简化版：avgpool + 单根 conv3s2 | 0 |
| `SPPELAN` | <SrcLink file="v9:models/common.py" :line="563" :end="577" /> | SPP-ELAN：cv1 → 串行 3 个 SP(5) → 4 路 concat → cv5（SPPF 同款拓扑） | 1 |
| `CBLinear` | <SrcLink file="v9:models/common.py" :line="631" :end="639" /> | 一个 conv 输出 sum(c2s) 通道再 split 成多份存档（CBNet 系） | 3 |
| `CBFuse` | <SrcLink file="v9:models/common.py" :line="641" :end="650" /> | 多份存档 interpolate 到目标尺寸后逐元素相加（CBNet 系） | 3 |
| `Silence` | <SrcLink file="v9:models/common.py" :line="554" :end="558" /> | 恒等层，给辅助分支提供原始输入抽头 | 1 |
| `ImplicitA` | <SrcLink file="v9:models/common.py" :line="605" :end="613" /> | 可学习加性先验 `implicit + x`（v7 隐式知识遗产） | 0 |
| `ImplicitM` | <SrcLink file="v9:models/common.py" :line="616" :end="624" /> | 可学习乘性先验 `implicit × x`（v7 隐式知识遗产） | 0 |
| `RepConvN` | <SrcLink file="v9:models/common.py" :line="86" :end="111" /> | v7 RepConv 的 N 版：conv3×3 + conv1×1 训练并行、部署折叠 | 0 |
| `DDetect` | <SrcLink file="v9:models/yolo.py" :line="78" :end="125" /> | gelan 系单头 DFL 头（gelan-c.yaml 用） | 0 |
| `DualDDetect` | <SrcLink file="v9:models/yolo.py" :line="190" :end="256" /> | yolov9 系双头：6 路输入、cv2/cv3 + cv4/cv5、双 DFL | 1 |

- `AConv` / `ImplicitA` / `ImplicitM` / `RepConvN` 在 yolov9-c.yaml 里次数为 0——隐式先验是 v7 遗产，v9 主配置不用；RepConvN/AConv 留给自定义配置与 hub 配置
- **parse_model 装配规则**（<SrcLink file="v9:models/yolo.py" :line="659" :end="727" />）：
  - Conv 家族（Conv/AConv/ADown/RepNCSPELAN4/SPPELAN）走 `c1, c2 = ch[f], args[0]` + 宽度增益（L677-685）
  - `CBLinear`：`c2 = args[0]`（c2s 列表本身）、`c1 = ch[f]`（L697-700）——from 指向骨干层号，直接读那层通道
  - `CBFuse`：`c2 = ch[f[-1]]`（L701-702）——输出通道取 from 列表末位（ADown 路）
  - 检测头家族：`args.append([ch[x] for x in f])`（L704-705）——DualDDetect 收到 6 路 512 通道表
  - savelist：显式层号 from 自动进 save（L722），CBLinear 的"隔空抽头"与 Concat 回连全靠它

## 5. gelan-c.yaml：拆掉辅助分支的对照组

gelan-c.yaml（80 行）= yolov9-c.yaml 的"纯主分支版"，论文消融实验里的 GELAN 基线——整网 **23 条（#0~#22）**：

| 维度 | gelan-c.yaml | yolov9-c.yaml |
|------|--------------|----------------|
| 层 0 | `Conv` 64 s2（P1/2） | `Silence`（恒等锚点） |
| 骨干 | #0~#8（9 条） | #0~#9（10 条，整体后移 1 号） |
| 头 | #9~#21，P3/P4/P5 = #15/#18/#21 | #10~#22，P3/P4/P5 = #16/#19/#22 |
| 辅助分支 | 无 | #23~#37 共 15 条 |
| 检测头 | `[[15, 18, 21], 1, DDetect, [nc]]` 单头（<SrcLink file="v9:models/detect/gelan-c.yaml" :line="79" />） | `[[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]]` 双头 |
| 训练入口 | train.py + loss_tal.py 单路损失 | train_dual.py + loss_tal_dual.py 双路损失 |

- gelan 系**不需要 Silence**：没有辅助分支就没有"从原始输入重新出发"的抽头需求，层 0 直接是 Conv——这也是判断一份 v9 yaml 有没有辅助分支的最快特征
- 除层号平移外，主干与头结构和 yolov9-c 完全同构（同通道、同重复数）——"辅助分支带来的增益"由此可做受控对照
- 同目录还有 `gelan.yaml` / `gelan-e.yaml` / `yolov9.yaml`（base 与 e 档版本）

## 6. yolov9-e.yaml：辅助分支的完全体

e 档（144 行，#0~#48 共 49 条）与 c 档同思路、更宽更深，五处关键差异：

- **五级拆分**：CBLinear 从骨干 #1/#3/#5/#7/#9 五级抽头（#10~#14），拆分方案 `[64]` → `[64, 128]` → `[64, 128, 256]` → `[64, 128, 256, 512]` → `[64, 128, 256, 512, 1024]`（<SrcLink file="v9:models/detect/yolov9-e.yaml" :line="46" :end="50" />）——c 档只从 5/7/9 三级抽
- **五处 CBFuse**（#16/#18/#21/#24/#27）：从 P1 尺度就开始融合（<SrcLink file="v9:models/detect/yolov9-e.yaml" :line="54" />），c 档从 P3 才开始——辅助覆盖越深，"可逆"保护越早
- **更宽更深**：骨干后段 1024 通道（#7/#9 `[1024, 512, 256, 2]`），RepNCSPELAN4 的 c5 重复数 1 → 2
- **双 SPPELAN**：辅助分支与主分支各一个（#29 辅 / #36 主），c 档只有主分支一个
- **DualDDetect 输入 [35, 32, 29, 42, 45, 48]**（<SrcLink file="v9:models/detect/yolov9-e.yaml" :line="143" />）——A3/A4/A5 = #35/#32/#29，P3/P4/P5 = #42/#45/#48
- README 口径：YOLOv9-E 58.1M 参数 / 192.5 GFLOPs / 55.6 AP（对照 C 档 25.5M / 102.8G / 53.0，COCO val 640）

## 7. 与 v8 层结构对比

| 维度 | v8（yolov8.yaml） | v9（yolov9-c.yaml） |
|------|--------------------|----------------------|
| 顶层条目 | 23 条（#0~#22） | 39 条（#0~#38），多出的 16 条全在辅助分支与双头 |
| 缩放 | scales 五档共用一份 yaml | 无 scales，一档一 yaml，gd/gw=1.0 |
| 骨干块 | C2f（n+2 路 concat） | RepNCSPELAN4（两级级联 4 路 concat） |
| 下采样 | Conv 3×3 s2 | ADown（avgpool 开路双路） |
| 颈入口 | SPPF | SPPELAN |
| 颈聚合 | Upsample/Conv s2 + Concat ×4 | Upsample/ADown + Concat ×4（拓扑同构） |
| 训练专属层 | 无 | CBLinear ×3 + CBFuse ×3 + 辅助主干 6 条 |
| 检测头 | Detect 单头 3 入口 | DualDDetect 双头 6 入口 |
| 推理输出 | (bs, 84, 8400) | y = [辅路, 主路] 两套 (bs, 84, 8400)，验证取主路 |
| 损失 | 单套 TAL + 三分量 | 双套独立 TAL，主头全权重 + 辅头 ×0.25 |
| 模块之家 | nn/modules/block.py + head.py | models/common.py + models/yolo.py（v7 布局） |
| 装配器 | nn/tasks.py parse_model | models/yolo.py parse_model（<SrcLink file="v9:models/yolo.py" :line="659" :end="727" />） |

## 延伸阅读

- [YOLOv9 架构总览](/versions/v9/architecture) — 全图与三大模块图解
- [YOLOv9 源码走读](/versions/v9/code-walkthrough) — parse_model / DualDDetect / 双路损失逐行走读
- [yolov9-c.yaml 源码页](/versions/v9/source/yolov9-c-yaml) — 结构配置原文
- [v8 层结构](/versions/v8/layers) — C2f 时代对照
- [术语表](/glossary) — GELAN/PGI 相关术语索引
