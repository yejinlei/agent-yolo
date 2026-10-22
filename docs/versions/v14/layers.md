# YOLOv14 架构与层速查

> 首发载体：`zhangcbb/yolov14` commit `0b8610d`（2026-08），对应 arXiv:2608.04720v3（README 自称 YOLO-Omni）。以 [yolov14.yaml](/versions/v14/source/yolov14-yaml) 为基准（**cfg 目录 `cfg_models_v14/`，与 v13 的 `v13/` 不同——v14 首次不带版本子目录层级**；5 档 scales：n/s/m/l/x，v13 缺的 m 档在 v14 补回），本页通道数为**缩放前基准值**（width=1.0）。总览页见 [YOLOv14 架构总览](/versions/v14/architecture)。

> **v1~v13 读者路标**：这张 yaml 是 [yolov13.yaml](/versions/v13/layers) 的替代谱系——**骨干 12 层（v13 是 9 层）加了 3 处新锚点：#0 CircularConv 环形卷积入口 + #2 ViewEmbedding 视角嵌入 + #3/#7 DomainAdaptiveLayer 域适应**；**骨干深段 #9/#11 保留 A2C2f 骨架但内嵌注意力从 AAttn 换成 DeformableAAttn**；**Head 13 层**（v13 是 23 层，v14 大幅精简）——**用 DeformableA2C2f ×3 替代了 v13 的 HyperACE + 7× FullPAD_Tunnel**。总层数 25（v13 是 32）——**v14 更简洁但多域鲁棒**。

## 1. 全网络层一览表（yolov14.yaml L29-L62，基准宽度 @640 输入）

> **v1~v13 读者路标**：这张表可与 [v13 逐层图解 §1](/versions/v13/layers) 逐层对照——**骨干 9 层扩到 12 层，主要是加了 3 处锚点**；**Head 23 层精简到 13 层，最大差异是去掉了 HyperACE + FullPAD_Tunnel 全部 8 处，换成 3 个 DeformableA2C2f**。Detect 位置从 yaml L50 挪到 L62。

### 1.1 Backbone（yaml L29-L42，12 层）

| 行号 | 层号 | 层类型 | 参数（yaml 原文） | 输出（基准 @640） | 说明 |
|-----|-----|-----------|--------------------|--------------------|------|
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="31" /> | 0 | `CircularConv` | 64, 3, 2 | 64×320×320 | **v14 新：P1/2 stem · 环形 padding（全景/鱼眼 x=W-1 ↔ x=0 相接）** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="32" /> | 1 | `Conv` | 128, 3, 2, 1, 2 | 128×160×160 | P2/4 · 标准 stride-2 卷积 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="33" /> | 2 | `ViewEmbedding` | 128, 6, 64 | 128×160×160 | **v14 新：Multi-View Conditioning · 6 类视角码字 · embed_dim=64** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="34" /> | 3 | `DomainAdaptiveLayer` | 128 | 128×160×160 | **v14 新：Game2Real AdaIN · P2 后第一次域对齐** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="35" /> | 4 | `C3k2` | 256, **False**, 0.25 ×2 | 128×160×160 | v11~v13 保留 · P2 · C3k2 e=0.25 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="36" /> | 5 | `Conv` | 256, 3, 2, 1, 4 | 256×80×80 | P3/8 · group=4 分组卷积 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="37" /> | 6 | `C3k2` | 512, **False**, 0.25 ×2 | 256×80×80 | v11~v13 保留 · P3 · C3k2 e=0.25 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="38" /> | 7 | `DomainAdaptiveLayer` | 512 | 256×80×80 | **v14 新：Game2Real AdaIN · P3 后第二次域对齐** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="39" /> | 8 | `Conv` | 512, 3, 2 | 512×40×40 | P4/16 · 下采样 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="40" /> | 9 | `DeformableA2C2f` | 512, **True, 4** ×4 | 512×40×40 | **v14 新：P4/16 · a2=True · area=4 · D-AAttn ×2/组** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="41" /> | 10 | `Conv` | 1024, 3, 2 | 1024×20×20 | P5/32 · 下采样 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="42" /> | 11 | `DeformableA2C2f` | 1024, **True, 1** ×4 | 1024×20×20 | **v14 新：P5/32 · a2=True · area=1 · D-AAttn ×2/组** |

**骨干关键节奏**：

- **3 处 v14 新锚点**（#0 CircularConv / #2 ViewEmbedding / #3/#7 DomainAdaptiveLayer）——**每个都有明确的多域鲁棒职能**；
- **#0 CircularConv 替代 Conv**：环形 padding 让 x=W-1 与 x=0 在卷积时相连——**对全景等距投影至关重要**（否则左右两侧会产生人为的边界响应）；
- **#2 ViewEmbedding 在 P2 首次出现后**：视角码字尽早注入，让后续 11 层都能看到——**「多视角共骨干」的入口**；
- **#3/#7 DomainAdaptiveLayer 双位置**：P2 后一次（浅层几何对齐）+ P3 后一次（中层语义对齐）——**梯度反转的域对抗损失 L_adv 依赖这两处的 `_domain_logits`**；
- **#9/#11 保留 A2C2f 骨架**：`area=4`（局部）+ `area=1`（全局）——**v12/v13 骨干深段的两处 Attention 位置与语义完全一致，只是把 AAttn 换成 DeformableAAttn**；
- **C3k2 两处（#4/#6）与 v13 一致**：`e=0.25, n=2, shortcut=False`——**浅层瓶颈用 C3k2 而非 DSC3k2**，v13 的 DSC3k2 在 v14 骨干浅层不再使用；
- **无 SPPF / 无 C2PSA / 无 HyperACE**：v13 的 HyperACE 与 FullPAD_Tunnel 完全撤出，v14 head 走**标准 FPN+PAN 骨架 + DeformableA2C2f 内嵌**。

### 1.2 Head（yaml L45-L62，13 层）

| 行号 | 层号 | 层类型 | 参数（yaml 原文） | 输出（基准 @640） | 说明 |
|-----|-----|-----------|--------------------|--------------------|------|
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="46" /> | 12 | `nn.Upsample` | None, 2, nearest | 1024×40×40 | P5 上采样到 P4 尺度 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="47" /> | 13 | `Concat` | `[-1, 9]` | 1536×40×40 | cat backbone P4（512）+ 上采样 P5（1024） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="48" /> | 14 | `DeformableA2C2f` | 512, **False, -1** ×2 | 512×40×40 | **v14 新：a2=False → 走 C3k · area=-1 无效 · 骨架层** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="50" /> | 15 | `nn.Upsample` | None, 2, nearest | 512×80×80 | P4 上采样到 P3 尺度 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="51" /> | 16 | `Concat` | `[-1, 6]` | 768×80×80 | cat backbone P3（256）+ 上采样 P4（512） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="52" /> | 17 | `DeformableA2C2f` | 256, **False, -1** ×2 | 256×80×80 | **v14 新：a2=False → C3k · P3 检测输入 · stride 8** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="54" /> | 18 | `Conv` | 256, 3, 2 | 256×40×40 | PAN 下行 · P3 → P4 分辨率 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="55" /> | 19 | `Concat` | `[-1, 14]` | 768×40×40 | cat head P4（512）+ 下行 P3（256） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="56" /> | 20 | `DeformableA2C2f` | 512, **False, -1** ×2 | 512×40×40 | **v14 新：a2=False → C3k · P4 检测输入 · stride 16** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="58" /> | 21 | `Conv` | 512, 3, 2 | 512×20×20 | PAN 下行 · P4 → P5 分辨率 |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="59" /> | 22 | `Concat` | `[-1, 11]` | 1536×20×20 | cat backbone P5（1024）+ 下行 P4（512） |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="60" /> | 23 | `C3k2` | 1024, **True** ×2 | 1024×20×20 | **v14：P5 检测输入 · stride 32 · 走 C3k2 而非 DeformableA2C2f** |
| <SrcLink file="v14:cfg_models_v14/yolov14.yaml" :line="62" /> | 24 | `Detect` | `[17, 20, 23], nc` | — | Detect(P3, P4, P5)——三尺度收尾 |

**Head 关键节奏**：

- **标准 FPN 上行（#12-#17）+ PAN 下行（#18-#23）**——**v13 的 HyperACE + FullPAD_Tunnel 骨架完全消失**，v14 head 是「v10/v11 时代的经典 FPN+PAN 骨架」；
- **5 个融合块（#14/#17/#20/#23）里 4 个是 DeformableA2C2f**：
  - #14：`[512, False, -1] ×2` → **a2=False 走 C3k**（不是 D-AAttn），**area=-1 表示无效**
  - #17：`[256, False, -1] ×2` → 同 #14
  - #20：`[512, False, -1] ×2` → 同 #14
  - #23：`[1024, True] ×2` → **C3k2**（不是 DeformableA2C2f，最后一层是普通 C3k2）
- **DeformableA2C2f 实际启用 D-AAttn 的位置**：只有 backbone #9/#11（a2=True）真正跑 D-AAttn——**yaml 里共 5 处 DeformableA2C2f，但 head 里 3 处都是 a2=False 走 C3k**（回退到 C3k2 的普通结构）；
- **head 里 4 处融合块只有 2 处真正的注意力**（backbone #9/#11）——**这是 v14「几何鲁棒」主要发生在骨干深段而非 head 的证据**；
- **Detect 输入三处**：#17（P3, stride 8）、#20（P4, stride 16）、#23（P5, stride 32）——**与 v11~v13 完全一致**；
- **P5-large 用 C3k2 而非 DeformableA2C2f**：这是**唯一一处「P5-large 收尾不用注意力」**的设计——**保证 DFL 头前的特征有稳定的低阶表示**；
- **yaml 里 DynamicScaleRouter 零实例化**：论文 §3.5 描述的「head 首块前的轻量路由网」**未出现在任何一份 yaml**，`paper.md` §7 已列为最大偏差项。

### 1.3 Head DeformableA2C2f 参数解码

head 里 3 处 DeformableA2C2f 都是 `[C, False, -1] ×2`——**为什么 area=-1、a2=False？**

回看 [DeformableA2C2f.__init__](file:///f:/src/agent-yolo/source/yolov14/snapshot/nn/modules/block.py)（block.py L1686-1702）：

```python
def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True, window_size=8):
    ...
    self.m = nn.ModuleList(
        nn.Sequential(
            DeformableABlock(c_, num_heads, mlp_ratio, area, window_size, shift_size=0),
            DeformableABlock(c_, num_heads, mlp_ratio, area, window_size, shift_size=shift),
        )
        for _ in range(n)
    )
```

**a2 参数在 v14 里被静默忽略**——`DeformableA2C2f` 永远走 `DeformableABlock`（不检查 a2），这与 v12/v13 `A2C2f` 的行为**不同**（v13 a2=False 会走 C3k 分支）。**这是 v14 与 v13 的一个隐藏差异**：

| 位置 | v13 A2C2f (a2=False) | v14 DeformableA2C2f (a2=False) |
|------|---------------------|--------------------------------|
| 内部模块 | C3k | **DeformableABlock**（忽略 a2） |
| Attention | 无 | **D-AAttn 仍在**（Fast Mode 下才跳过形变） |
| area 参数 | 无效（走 C3k） | 传给 D-AAttn（但 area=1 时无 area 分组） |

**结论**：v14 head 里 3 处 DeformableA2C2f 虽然 yaml 写 a2=False、area=-1，**但实际都会走 D-AAttn**（因为 a2 被忽略）——**这是 v14 的一个「隐藏注意力层」**，实际启用 D-AAttn 的位置是**骨干 #9/#11 + head #14/#17/#20 共 5 处**（论文 §3.2 说「×6」是把 backbone #9/#11 各拆成 2 个 DeformableABlock 后计数，与 yaml 层级不一致）。

## 2. 六份 yaml 变体结构对比

v14 首发六份 yaml 变体，每份对应一个目标域——**它们不是「同一模型的不同参数」，而是「同骨干骨架 + 不同模块启用」**。

| 变体 | CircularConv | ViewEmbedding | DomainAdaptiveLayer | DeformableA2C2f | SphereAAttn | 目标域 |
|------|-------------|--------------|---------------------|-----------------|-------------|--------|
| `yolov14.yaml` | ✅ #0 | ✅ #2 | ✅ #3, #7 | ✅ #9, #11 + head ×3 | ❌ | 全量 |
| `yolov14-panorama.yaml` | ✅ 全部下采样 | ❌ | ❌ | ❌（用 A2C2f） | ✅ | 360° 全景 |
| `yolov14-deformable.yaml` | ❌（用 Conv） | ❌ | ❌ | ✅ 全部 | ❌ | 鱼眼广角 |
| `yolov14-multiview.yaml` | ❌ | ✅ | ❌ | ❌（用 A2C2f） | ❌ | 无人机/BEV |
| `yolov14-game2real.yaml` | ❌ | ❌ | ✅ ×2 | ❌（用 A2C2f） | ❌ | 游戏画面 |
| `yolov14-adaptive.yaml` | ❌ | ✅ | ✅ | ✅ 全部 | ❌ | 自适应多域 |

**六份变体的共同点**：
- 都保留 C3k2 浅层（#4/#6）
- 都用 Detect 单头（`head.py` 零改动）
- 都基于 ultralytics 8.3.x 的 tasks.py
- 档位都是 n/s/m/l/x 五档

**六份变体的差异点**：
- `yolov14.yaml` 全量版：3 个锚点（ViewEmbedding + DomainAdaptiveLayer ×2）+ 5 处 DeformableA2C2f + 1 处 CircularConv——**最复杂、参数最多、COCO mAP 最高**；
- `yolov14-panorama.yaml`：全部下采样用 CircularConv + 1 处 SphereAAttn——**为等距投影专门设计，极地/赤道分开处理**；
- `yolov14-deformable.yaml`：全部下采样用标准 Conv + 骨干/head 全铺 DeformableA2C2f——**几何鲁棒最强，面向鱼眼广角**；
- `yolov14-multiview.yaml`：只有 ViewEmbedding——**多视角鲁棒最强，面向无人机/BEV**；
- `yolov14-game2real.yaml`：只有 DomainAdaptiveLayer ×2——**风格鲁棒最强，面向游戏画面**；
- `yolov14-adaptive.yaml`：ViewEmbedding + DomainAdaptiveLayer ×2 + DeformableA2C2f——**「adaptive」= 全锚点但不用 CircularConv，面向自适应多域**。

> **yaml 变体详细对比**：见 [variants 源码页](/versions/v14/source/yolov14-variants)。

## 3. 与 v13 结构对比

| 维度 | v13 | **v14** | 差异 |
|------|-----|---------|------|
| 总层数 | 32 | **25** | **-7 层** |
| Backbone 层数 | 9 | **12** | +3 层（3 处新锚点） |
| Head 层数 | 23 | **13** | **-10 层**（去 HyperACE + 7× FullPAD_Tunnel） |
| 骨干入口 | Conv | **CircularConv** | v14 panorama 用环形 |
| 骨干浅层融合 | DSC3k2 | **C3k2** | 从 DSC3k2 回到 C3k2 |
| 骨干下采样 | DSConv s2 | **Conv s2**（panorama 用 CircularConv） | 从 DSConv 回到 Conv |
| 骨干深段注意力 | A2C2f（AAttn） | **DeformableA2C2f（D-AAttn）** | AAttn → D-AAttn |
| Head 首块 | HyperACE | **Upsample + Concat + DeformableA2C2f** | 从超图融合回到 FPN+PAN |
| FullPAD_Tunnel | ×7 | **0** | 完全撤出 |
| ViewEmbedding | — | **×1**（#2） | 新增 |
| DomainAdaptiveLayer | — | **×2**（#3, #7） | 新增 |
| SphereAAttn | — | **×1**（panorama 变体专用） | 新增 |
| Detect | Detect | Detect | **完全相同** |
| scales 档数 | 4（n/s/l/x） | **5（n/s/m/l/x）** | v14 补回 m 档 |
| mlp_ratio | 1.2 | **2.0** | FFN 加倍 |
| 许可证 | Apache-2.0 | **AGPL-3.0** | 首次同版本双许可证 |

**核心洞察**：v14 不是「v13 的加强版」而是**「v13 的替代谱系」**——**v14 走的是 v12 的结构谱系（C3k2 + Conv s2 + FPN+PAN），但引入了 v13 完全没做的四方向鲁棒**。

## 4. 五档 scales 详解

| 档位 | depth | width | max_channels | 特征 |
|------|-------|-------|-------------|------|
| n | 0.50 | 0.25 | 1024 | 参数 5.0M 级 |
| s | 0.50 | 0.50 | 1024 | 参数 9.2M 级 |
| **m** | 0.50 | 1.00 | 512 | 参数 ~17M 级 · **v14 补回** |
| l | 1.00 | 1.00 | 512 | 参数 ~35M 级 |
| x | 1.00 | 1.50 | 512 | 参数 ~55M 级 |

- **v13 只有 4 档**（n/s/l/x，缺 m）——**v14 首次补回 m 档**，让参数-精度曲线在中间有一档可选项；
- **width=1.0 时 max_channels 从 1024 降到 512**（m/l/x 档）：这是**深度-宽度-通道三者之间的耦合**——m/l/x 档用较小的 max_channels 抑制 P5 通道爆炸；
- **depth=0.50**：所有骨干/neck 的 A2C2f/C3k2 n 值都乘 0.5——**v14 s 档骨干深段 #9/#11 的 n 从 4 缩到 2**（每个 DeformableA2C2f 内 2 组 DeformableABlock → 1 组）；
- **width=0.50/0.25**：所有 Conv/A2C2f 的通道乘 0.5/0.25——**v14 s 档骨干 P5 从 1024 缩到 512**；
- **v14 全档 `mlp_ratio=2.0`**：v13 l/x 档是 1.5，v14 统一加倍——**FFN 更宽，D-AAttn 每头 head_dim 更小但表达力不亏**。

## 5. 参数量与计算量对比

| 版本 | 档 | Params | FLOPs @640 | T4 TensorRT FP16 | FPS | COCO mAP |
|------|---|--------|-----------|-----------------|-----|---------|
| v12 | s | 8.3M | 18.4G | 2.82 ms | 356 | 47.1 |
| v13 | s | 9.0M | 20.8G | 2.98 ms | 336 | 48.0 |
| **v14** | **s** | **9.2M** | **21.5G** | **2.91 ms** | **344** | **49.1** |
| v14 | n | 5.0M | 12.4G | 2.34 ms | 428 | 45.2 |
| v14 | m | 17M | 39.1G | 5.21 ms | 192 | 50.7 |
| v14 | l | 35M | 78.2G | 10.4 ms | 96 | 51.9 |
| v14 | x | 55M | 118G | 16.8 ms | 60 | 52.6 |

**要点**：
- **v14-s 比 v13-s 多 0.2M 参数**（+2.2%）——**主要来自 3 处新锚点（ViewEmbedding ~4K + DomainAdaptiveLayer ×2 ~25K）**；
- **v14-s 比 v13-s 快 0.07 ms**——**因为 v14 用 Conv s2 下采样（v13 用 DSConv s2）+ 无 HyperACE**；
- **v14-m 首次出现**——**参数 ~17M，介于 s 与 l 之间**，填补了 v13 的空档；
- **v14-x 参数 55M**——**v13-x 是 60M**，v14 反而更轻（因为无 FullPAD_Tunnel 权重）；
- **v14-n 参数 5.0M**——**v13-n 是 6.2M**，v14 更轻（因为无 HyperACE + FullPAD_Tunnel）。

> **详细性能数据**：见 [paper.md §5 关键指标](/versions/v14/paper#关键指标)。

## 6. Fast Mode 与部署开关

v14 引入 `YOLOV14_FAST` 环境变量作为**推理侧总开关**——**训练模式下所有模块全开，推理模式下部分模块被短路**。

| 模块 | Fast Mode 关闭 | 训练模式行为 | 推理模式行为 |
|------|---------------|-------------|-------------|
| `DeformableAAttn` | ✅（跳过 offset_conv + grid_sample） | 计算 Δ + warp QKV | 直接做 AAttn（无 warp） |
| `ViewEmbedding` | ✅（return x） | concat + 1×1 投影 | 直接返回 x |
| `DomainAdaptiveLayer` | ✅（return x） | AdaIN + GRL | 直接返回 x |
| `SphereAAttn` | ❌（无 fast 分支） | 完整计算 | 完整计算 |
| `DynamicScaleRouter` | ❌（yaml 未启用） | — | — |

**要点**：
- **Fast Mode 是「推理侧性能保底」**：v14-s 的 2.91 ms 是在 Fast Mode 下测得的——**训练模式的实际延迟约 3.5 ms**；
- **Fast Mode 关闭形变与视角注入**：让骨干退化为「v12/v13 A2C2f 骨干」——**训练学到的是「视角无关 + 域不变」的骨干，推理时可以按需选择是否用这些鲁棒特性**；
- **Fast Mode 环境变量是 zhangcbb 自定义**：`is_fast_mode()` 读 `YOLOV14_FAST=1`——**非 ultralytics 官方约定**，官方 ultralytics 8.3.x 无此机制。

> **读者路标**：Fast Mode 在 [concepts.md §1/§3/§4](/versions/v14/concepts#1-deformableaattn形变注意力concep-tdeformableaattn) 三个模块的实现里都有分支，本节是索引。
