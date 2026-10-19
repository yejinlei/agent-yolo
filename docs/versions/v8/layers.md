# YOLOv8 逐层图解

> 以 `ultralytics/models/v8/yolov8.yaml` 全文 46 行为准：backbone + head 共 **23 个顶层条目（#0~#22）**——yaml 行内注释只标到 #21，Detect 落在 #22。本页通道数为 yaml 声明的**基准值**（缩放前），各档实际宽度见 §4。配置原文见 <SrcLink file="v8:ultralytics/models/v8/yolov8.yaml" :line="1" :end="46" />。

> **读者路标**：想先看整体数据流与三大模块图解，去 [YOLOv8 架构总览](/versions/v8/architecture)；C2f 与 v7 ELAN 的渊源对照见 [v7 逐层图解](/versions/v7/layers)。

## 1. 逐层明细表（yolov8.yaml #0~#22，基准宽度 @640 输入）

| # | from | module | 参数（yaml 原文） | 输出（基准 @640） | 说明 |
|-----|------------|--------------|----------------------|--------------------|------|
| 0 | -1 | `Conv` | 64, 3, 2 | 64×320×320 | stem 下采样 · P1/2 |
| 1 | -1 | `Conv` | 128, 3, 2 | 128×160×160 | P2/4 |
| 2 | -1 | `C2f` | 128, True ×3 | 128×160×160 | 骨干 C2f 带 shortcut |
| 3 | -1 | `Conv` | 256, 3, 2 | 256×80×80 | P3/8 |
| 4 | -1 | `C2f` | 256, True ×6 | 256×80×80 | FPN 回连点二 |
| 5 | -1 | `Conv` | 512, 3, 2 | 512×40×40 | P4/16 |
| 6 | -1 | `C2f` | 512, True ×6 | 512×40×40 | FPN 回连点一 |
| 7 | -1 | `Conv` | 1024, 3, 2 | 1024×20×20 | P5/32 |
| 8 | -1 | `C2f` | 1024, True ×3 | 1024×20×20 | |
| 9 | -1 | `SPPF` | 1024, 5 | 1024×20×20 | 骨干出口 · PAN 顶点 |
| 10 | -1 | `nn.Upsample` | None, 2, nearest | 1024×40×40 | FPN 上行 |
| 11 | [-1, 6] | `Concat` | 1 | 1536×40×40 | cat 骨干 P4 |
| 12 | -1 | `C2f` | 512 ×3 | 512×40×40 | 头部 C2f 无 shortcut |
| 13 | -1 | `nn.Upsample` | None, 2, nearest | 512×80×80 | |
| 14 | [-1, 4] | `Concat` | 1 | 768×80×80 | cat 骨干 P3 |
| 15 | -1 | `C2f` | 256 ×3 | 256×80×80 | **P3 检测输入 · stride 8** |
| 16 | -1 | `Conv` | 256, 3, 2 | 256×40×40 | PAN 下行 |
| 17 | [-1, 12] | `Concat` | 1 | 768×40×40 | cat 头部 P4 |
| 18 | -1 | `C2f` | 512 ×3 | 512×40×40 | **P4 检测输入 · stride 16** |
| 19 | -1 | `Conv` | 512, 3, 2 | 512×20×20 | PAN 下行 |
| 20 | [-1, 9] | `Concat` | 1 | 1536×20×20 | cat SPPF 输出 |
| 21 | -1 | `C2f` | 1024 ×3 | 1024×20×20 | **P5 检测输入 · stride 32** |
| 22 | [15, 18, 21] | `Detect` | nc | 三尺度 144 通道 | anchor-free 三头 · 收官 |

- 重复数 ×3/×6 是基准值：depth 缩放后 n/s 档变 1/2、m 档变 2/4、l/x 档不变（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="493" /> 深度增益、<SrcLink file="v8:ultralytics/nn/tasks.py" :line="501" :end="503" /> 注入 args）
- 骨干 C2f 的 `True` 是 shortcut；头部 C2f 不传、走签名默认 `False`（<SrcLink file="v8:ultralytics/nn/modules/block.py" :line="172" />）——同一模块两种用法
- `from` 支持负索引（上一层）与显式层号：#11/#14 回连骨干 #6/#4，#17/#20 回连头部 #12 与骨干 #9
- Concat 输出通道 = 两路之和（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="515" :end="516" />）；Upsample 保持通道不变

## 2. 输出张量布局（8400×144 从哪来）

| 模式 | 返回 | 形状 | 出处 |
|------|------|------|------|
| 训练 | 三尺度特征列表 | (bs, 144, 80, 80) / (bs, 144, 40, 40) / (bs, 144, 20, 20) | <SrcLink file="v8:ultralytics/nn/modules/head.py" :line="48" :end="49" /> |
| 推理 | (y, x) 元组 | y = (bs, 84, 8400) | <SrcLink file="v8:ultralytics/nn/modules/head.py" :line="54" :end="62" /> |

- 每尺度 144 = $4 \times 16 + 80$：box 分支 4 边 × 16 bin 的距离分布 + cls 分支 80 类
- 预测数 8400 = $80^2+40^2+20^2$ = 6400+1600+400（stride 8/16/32，640 输入）——**anchor-free 每格 1 个预测**，[anchor](/glossary#anchor-锚框-先验框) 点就是网格中心
- 推理解码：split 出 64 维分布 → DFL 期望值积分得 4 个 ltrb → `dist2bbox` 转 xywh、乘 stride → 与 80 维 sigmoid 分数拼成 (bs, 84, 8400)；[DFL](/glossary#dfl-distribution-focal-loss) 期望值积分与 dist2bbox 的实现见[架构总览 §4](/versions/v8/architecture)
- **没有 objectness 通道**：84 = 4 + 80，分类分数即置信度，NMS 阶段直接用——对比 v5/v7 的 85 = 4 + 1 + 80
- 训练输出交给 [TAL](/glossary#tal-task-alignment-learning) 做标签分配，再进三分量损失（详见[损失函数详解](/versions/v8/losses#_1-损失总装-box-cls-dfl-三分量-loss-core)）

## 3. 模块速查（参数含义 + 装配规则）

| 模块 | 位置 | 关键参数 | yaml 使用次数 |
|------|------|----------|----------------|
| `Conv` | <SrcLink file="v8:ultralytics/nn/modules/conv.py" :line="26" :end="43" /> | c1, c2, k, s——Conv+BN+[SiLU](/glossary#silu-swish)，autopad | 7（骨干 5 + 头 2） |
| `C2f` | <SrcLink file="v8:ultralytics/nn/modules/block.py" :line="169" :end="189" /> | c1, c2, n, shortcut, e=0.5——[CSP](/glossary#csp-跨阶段部分连接) 式 split + n×Bottleneck + 全拼接 | 8（骨干 4 + 头 4） |
| `SPPF` | <SrcLink file="v8:ultralytics/nn/modules/block.py" :line="120" :end="135" /> | c1, c2, k=5——串行 3 次 MaxPool 等效 [SPP](/glossary#spp-空间金字塔池化) 5/9/13 | 1（#9） |
| `Bottleneck` | <SrcLink file="v8:ultralytics/nn/modules/block.py" :line="271" :end="283" /> | C2f 内部件（e=1.0、双 3×3、可带残差） | 0（不独立出现） |
| `DFL` | <SrcLink file="v8:ultralytics/nn/modules/block.py" :line="18" :end="36" /> | c1=16——权重锁死的期望值积分 conv | 0（Detect 内置） |
| `nn.Upsample` | torch 原生 | scale_factor=2, mode='nearest' | 2（#10/#13） |
| `Concat` | torch 原生 | dim=1 | 4（#11/#14/#17/#20） |
| `Detect` | <SrcLink file="v8:ultralytics/nn/modules/head.py" :line="22" :end="71" /> | nc, ch；reg_max=16、no=144 | 1（#22） |

block.py 还备有 C1/C2/C3/C3x/Proto/RepC3 等模块（<SrcLink file="v8:ultralytics/nn/modules/block.py" :line="13" :end="15" /> 的 `__all__`），yolov8.yaml 一律不用——Proto 服务分割头，C 系列留给其他配置。

**parse_model 装配规则**（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="462" :end="535" />）：

- 每层四元组 `[from, repeats, module, args]`；`nn.Upsample` 这类 torch 原生模块由 `getattr(torch.nn, ...)` 解析（L487）
- 深度增益：`n = max(round(n × depth), 1) if n > 1 else n`（L493）——全 yaml 只有 4 个 C2f 的重复数 3/6/6/3 参与缩放
- 宽度增益 + 通道上限：`c2 = make_divisible(min(c2, max_channels) × width, 8)`（L498）——先夹 max_channels 再乘 width
- C2f/C3 家族把缩放后的 n 插到 args 第 3 位，随后 n 置 1、不再外层重复（L501-503）
- Detect 家族把三路输入的通道表 append 进 args（L517-518）；Concat 通道求和（L515-516）
- 构建收尾：用 256×256 假输入前向一次推算 stride 并调用 `bias_init`（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="189" :end="197" />）；总装入口 `DetectionModel`（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="173" />）

## 4. 五档对照表（n/s/m/l/x）

| scale | depth / width / max_channels | 层数 | 参数 | GFLOPs | mAP50-95 |
|-------|------------------------------|------|------|--------|----------|
| n | 0.33 / 0.25 / 1024 | 225 | 3.2M | 8.7 | 37.3 |
| s | 0.33 / 0.50 / 1024 | 225 | 11.2M | 28.6 | 44.9 |
| m | 0.67 / 0.75 / 768 | 295 | 25.9M | 78.9 | 50.2 |
| l | 1.00 / 1.00 / 512 | 365 | 43.7M | 165.2 | 52.9 |
| x | 1.00 / 1.25 / 512 | 365 | 68.2M | 257.8 | 53.9 |

- 参数/GFLOPs/[mAP](/glossary#map-mean-average-precision)（COCO val2017，单模型单尺度）为快照 README 口径；yaml 注释里的精确参数（n：3,157,200）与 GFLOPs（n：8.9）与 README 有微小出入——口径差异见[指标与测速口径](/versions/v8/painpoints#_7-指标与测速口径-pain-bench-caliber)
- 速度参考（README）：CPU ONNX 80.4→479.1 ms、A100 TRT 0.99→3.53 ms（n→x）
- 各档骨干通道（stem→末级）：n 16/32/64/128/256 · s 32/64/128/256/512 · m 48/96/192/384/576 · l 64/128/256/512/512 · x 80/160/320/640/640——l/x 的 1024 被 max_channels=512 夹住，是"先夹再乘"最直观的例证
- 这套缩放是 v8 式 [compound scaling](/glossary#compound-scaling-复合缩放)：只缩 C2f 重复数与通道宽，不分 block/transition（<SrcLink file="v8:ultralytics/models/v8/yolov8.yaml" :line="6" :end="12" />）

## 5. 与 v7 层结构对比

| 维度 | v7（yolov7.yaml） | v8（yolov8.yaml） |
|------|--------------------|--------------------|
| 顶层条目 | 106 层（#0~#105） | 23 条（#0~#22），模块粒度更粗 |
| 骨干块 | [ELAN](/glossary#elan-e-elan-高效层聚合)：Conv 级联 + 4 路 Concat 隔一取一 | C2f：split + n×Bottleneck + 全拼接 |
| stem | Conv 3×3 s1（32ch） | Conv 3×3 s2（64ch），无独立 s1 stem |
| 下采样 | MP 块（MaxPool + 双路 Conv Concat） | 纯 Conv 3×3 s2 |
| 颈入口 | SPPCSPC（并联 5/9/13 + CSP 旁路） | SPPF（串行 3 次 k=5） |
| 颈聚合 | 6 路 Concat ELAN ×4 + PAN 下行 MP 块 | FPN 上行 + PAN 下行，C2f 融合 |
| 检测前收口 | RepConv ×3 | 无——C2f 直出进 Detect |
| 检测头 | IDetect：anchor-based，每格 3 anchors，ia/im 隐式知识 | Detect：anchor-free 解耦双分支，无 objectness |
| anchor 声明 | yaml 显式 9 组 anchors | 无 anchor 概念，网格中心即 anchor point |
| 推理输出 | (bs, 25200, 85) | (bs, 84, 8400) |
| 缩放方式 | depth/width_multiple=1.0，变体各开 yaml | scales 五档共用一份 yaml |
| 模块之家 | models/common.py + models/yolo.py | nn/modules/block.py + nn/modules/head.py |
| 装配器 | models/yolo.py `parse_model` | nn/tasks.py `parse_model`（<SrcLink file="v8:ultralytics/nn/tasks.py" :line="462" :end="535" />） |

## 延伸阅读

- [YOLOv8 架构总览](/versions/v8/architecture)
- [理论 Concept](/versions/v8/concepts)
- [损失函数详解](/versions/v8/losses)
- [yolov8.yaml 源码页](/versions/v8/source/yolov8-yaml)
- [v7 架构与层速查（对比）](/versions/v7/architecture)
- [v7 逐层图解（对比）](/versions/v7/layers)
