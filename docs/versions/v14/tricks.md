# YOLOv14 训练技巧

> **v1~v13 读者路标**：v14 是「v13 结构 + 多域增强四件套」的第三方学术版，配方层**大幅简化**——README L167-176 只给 `epochs=300, imgsz=640`（无 batch、无 mosaic/mixup/copy_paste 分档），比 v13 的「600 epoch + batch 256 + 四档 mixup/copy_paste」极简到只剩两行；所有真正的「训练技巧」集中在**结构层的可开关模块组合**（[DeformableAAttn](/glossary#deformableaattn) / [ViewEmbedding](/glossary#viewembedding) / [DomainAdaptiveLayer](/glossary#domainadaptivelayer) / [DynamicScaleRouter](/glossary#dynamicscalero) 的四段正交组合）、**6 类视角码字枚举**（`pinhole=0 / fisheye=1 / panoramic=2 / drone=3 / bev=4 / ground=5`，`ViewEmbedding.VIEW_TYPES` 常量表）、**Game2Real 三层消融**（数据层 `GameCharacterStylization` +11.7 AP / 特征层 `DomainAdaptiveLayer` +6.5 AP / 目标层 `DomainAdversarialLoss` +7.3 AP，相对 v12s baseline 24.1 AP）、**4 条退火 schedule**（λ_reg 50ep、GRL λ 100ep、τ 每 iter 指数退火、`set_epoch` 单点下发）、**6 份 yaml 变体的场景路由**、**Fast Mode `YOLOV14_FAST=1` 推理侧三开关关闭**。本页 9 节盘点这些训练侧事实。

## 全部条目速查

| 条目 | 位置 | 说明 | 推理是否零开销 |
|------|------|------|----------------|
| **官方配方极简：epochs=300 · imgsz=640** | README L167-176 | 无 batch / 无 mosaic / 无 mixup / 无 copy_paste 分档；`model.train(data, epochs=300, imgsz=640)` | 是（纯配置） |
| **6 份 yaml 变体（1 全量 + 5 场景）** | `cfg_models_v14/*.yaml` | `yolov14.yaml` / `-adaptive` / `-deformable` / `-game2real` / `-multiview` / `-panorama` | 是（选模型即定结构） |
| **档位回到五档：n/s/m/l/x** | 6 份 yaml L16-22 | 与 v13 相同 `depth/width/max_channels`（v13 无 m 档，v14 补回） | 否（建网期） |
| **A2C2f `mlp_ratio=2.0`**（v14 版） | block.py L1390-L1445 | v13 是 1.2；FFN 加宽 67%，五档统一 | 是（结构） |
| **D-AAttn `window_size=8, shift=4`** | block.py L1447-L1520 | Swin 风格 shifted local window；每层 **+4.7% 参数 / +4.1% FLOPs** | 是（结构） |
| **`ViewEmbedding.VIEW_TYPES` 6 类枚举** | block.py L1717 | `pinhole=0 / fisheye=1 / panoramic=2 / drone=3 / bev=4 / ground=5` 硬编码 | 是（结构） |
| **`ViewEmbedding embed_dim=64`** | block.py L1721 | yaml 第 2 位固定 64，`Embedding(6, 64)`；推理可切视角 | 是（结构） |
| **`DynamicScaleRouter`（1.8K 参数 / 0.06 ms）** | block.py L1736-L1773 | MLP gate 学 P3/P4/P5 权重；**6 份 yaml 全未实例化**（README 声称存在但 yaml 层未用） | 否（yaml 未启用） |
| **Game2Real 消融三层** | README L113-124 | `+GameCharStylization` +11.7 / `+DomainAdaptiveLayer` +6.5 / `+DomainAdversarialLoss` +7.3（v12s baseline 24.1 mAP） | 见下 |
| **`GameCharacterStylization` 四算子** | README L117 | posterisation（bit 3-6）/ unsharp / saturation boost（×1.5-1.8）/ contrast | 是（训练期） |
| **DomainAdaptiveLayer 用 AdaIN** | block.py L1871-L1893 | 特征统计对齐 game→real，2 层可堆栈（`yolov14.yaml` #3/#7） | 是（推理期 AdaIN 前向保留） |
| **4 条退火 schedule 单点下发** | loss.py L933-L950 | `v14DetectionLoss.set_epoch()` 一次性同步 λ_reg / GRL λ / τ | 是（训练期） |
| **λ_reg 线性退火 0.01→0.001（50ep）** | loss.py L879-L896 | `frac = min(epoch/50, 1)`，线性插值 | 是（训练期） |
| **GRL λ sigmoid 退火 0→1（100ep）** | block.py L63-L69 | `2/(1+exp(-10·p))-1`，DANN 经典形式；λ=0 时 GRL 退化 | 是（训练期） |
| **τ 指数退火 0.05→0.5（β=0.01，每 iter）** | loss.py L760-L777 | `tau_min + (tau_max-tau_min)·exp(-β·t)`；`step_update()` 每次前向调一次 | 是（训练期） |
| **`lambda_adv = lambda_cross = 0.05`** | loss.py L923 | 与论文 §3.7 一致（**注意**：`DomainAdversarialLoss.__init__` 内层默认 `lambda_domain=0.1` 被外层显式覆盖） | 是（初始化） |
| **三分量增益 7.5 / 0.5 / 1.5 沿用 v8** | loss.py L178, L258-L262 | 从 v8 到 v14 六版一字不差 | 是（初始化） |
| **TAL `topk=10, α=0.5, β=6.0` 沿用 v8** | loss.py L178 | 显式覆盖 tal.py 类默认 `topk=13, α=1.0`；v8 以来未修的隐性坑 | 是（初始化） |
| **`YOLOV14_FAST=1` Fast Mode** | block.py L16-L37 | 推理侧跳过 scene classifier / DomainAdaptiveLayer / ViewEmbedding；缓存到全局 `_FAST_MODE_CACHE` | **是（推理期专门优化）** |
| **`_FAST_MODE_CACHE` 环境变量缓存** | block.py L13, L27-L30 | 首次 `is_fast_mode()` 后缓存，避免热路径重复读 env | 是 |
| **`reset_fast_mode_cache()`** | block.py L33-L36 | 测试用；改 env 后需主动调用刷新 | — |
| **`batch` 字典按需启用 3 辅助损失** | loss.py L957-L976 | `offset/offset_init/domain_logits/domains/features/view_ids` 缺任一即跳过 | 是（训练期） |
| **`loss_items` 仍返回 3 维 `[box, cls, dfl]`** | loss.py L983 | 辅助损失不进 items——调试必须单独 hook 各 batch 键 | 是（训练期） |
| **COCO 单域训练建议关 adv/cross** | index.md §快速上手 | 默认 λ_adv=λ_cross=0.05 对纯 COCO 略掉分 | 否（需手动关闭） |
| **官方权重尚未发布（IN PROGRESS）** | README L71-L77 | 5 个变体 checkpoint 全待发布；基准数据集也未开源 | — |
| **许可证 AGPL-3.0（回切）** | LICENSE | 商用需遵守 copyleft；v13 是 Apache-2.0 | — |

## 1. 官方配方：300 epoch + imgsz 640（`trick-official-recipe`）

> **v1~v13 读者路标**：v12 首次引入「600 epoch + batch 256 + 分档 mixup/copy_paste」；v13 完全沿用；**v14 大幅回退**——README 只给 2 个参数（`epochs=300, imgsz=640`），无 batch、无 mosaic、无分档增强。这是「多域鲁棒」路线的一次配方简化：作者把「精细调参」的活儿外包给了**结构层的场景路由 + 自适应增强**。

**官方训练命令**（README L166-176，逐行照抄）：

```python
from ultralytics import YOLO

# 训练 Game2Real 变体（游戏画面检测）
model = YOLO("ultralytics/cfg/models/yolo-game2real.yaml")
model.train(data="coco.yaml", epochs=300, imgsz=640)

# 训练全量 Omni 变体（多域鲁棒）
model = YOLO("ultralytics/cfg/models/yolo-adaptive.yaml")
model.train(data="coco.yaml", epochs=300, imgsz=640)
```

**与 v12 / v13 官方的逐键对比**：

| 键 | v12 官方 | v13 官方 | **v14 官方** | 变化 |
|----|---------|---------|------------|------|
| epochs | 600 | 600 | **300** | **砍半** |
| batch | 256 | 256 | **未指定**（走 ultralytics default.yaml） | **消失** |
| imgsz | 640 | 640 | **640** | 同 |
| scale 增强 | 分档 0.5/0.9 | 分档 0.5/0.9（四档） | **未指定** | **消失** |
| mosaic | 1.0 | 1.0 | **未指定** | 消失（走 default） |
| mixup | 分档 0.0~0.2 | 分档 0.0~0.2（四档） | **未指定** | **消失** |
| copy_paste | 分档 0.1~0.6 | 分档 0.1~0.6（四档） | **未指定** | **消失** |
| close_mosaic | 10 | 10 | **未指定** | 消失（走 default） |
| optimizer | auto | auto | **未指定** | 消失（走 auto） |
| λ_adv / λ_cross | — | — | **0.05 / 0.05（默认生效）** | **新增** |
| λ_reg | — | — | **0.01（默认生效，50ep 退火）** | **新增** |

**为什么砍到 300 epoch**（本站推断，作者未明说）：

1. **v14 引入的 3 个可退火超参**（λ_reg 50ep / GRL λ 100ep / τ β=0.01）在 300 epoch 内都能完全收敛——不需要像 v12/v13 那样训练 600 epoch；
2. **多域鲁棒靠「场景路由 + 结构模块」而不是「更长训练」**——作者选择把精度预算花在结构创新上，而不是训练时长上；
3. **300 epoch 与 v8 官方默认一致**（v8 README 是 300）——v14 其实是**回到了 v8 训练周期的起点**，同时补上 v12/v13 才有的分档增强的替代品（结构层场景路由）；
4. **README 明确说「Benchmark datasets 也在 IN PROGRESS」**——四基准数据集当前未开源，配方简化也侧面说明作者还在快速迭代期，配方本身也未最终定型。

**推理开销**：零开销（训练期配置）。

## 2. 6 份 yaml 变体：场景路由的静态层（`trick-yaml-variants`）

> **v1~v13 读者路标**：v8/v11/v13 都是「单份 yaml × 五档复合缩放」；v12 是「单份 yolov12.yaml × 四档」；**v14 首次引入「6 份 yaml × 五档」的双维度结构**——场景维度（6 份）× 尺寸维度（n/s/m/l/x）= **30 种可能组合**，是 YOLO 系列历史上模型矩阵最庞大的一次。

**6 份 yaml 变体的模块差异**：

| yaml | 关键模块 | 目标场景 | 参数增量 |
|------|---------|---------|---------|
| `yolov14.yaml`（全量） | CircularConv + ViewEmbedding + DomainAdaptiveLayer×2 + DeformableA2C2f×6 + C3k2×4 | 通用（4 基准 + COCO） | 全量 |
| `yolov14-adaptive.yaml` | `AdaptiveAugmentPolicy`（README 声称，代码未实现） | 自动场景检测 | 同全量 |
| `yolov14-deformable.yaml` | DeformableA2C2f×6（无 DAL / VE / CircularConv） | 鱼眼 / 广角 | D-AAttn 部分 |
| `yolov14-game2real.yaml` | DomainAdaptiveLayer×2（A2C2f 用非形变版） | 游戏画面检测 | DAL 部分 |
| `yolov14-multiview.yaml` | ViewEmbedding + CrossViewLoss（无 DAL / D-AAttn） | 无人机 / BEV / 多视角 | VE 部分 |
| `yolov14-panorama.yaml` | SphereAAttn + CircularConv（无 DAL / VE） | 360° 全景 | Sphere 部分 |

**四要点**：

1. **档位五档齐全**（`n/s/m/l/x`），五档 `depth/width/max_channels` 与 v13 一致——`n:[0.50, 0.25, 1024]` / `s:[0.50, 0.50, 1024]` / `m:[0.50, 1.00, 512]` / `l:[1.00, 1.00, 512]` / `x:[1.00, 1.50, 512]`（`yolov14.yaml` L16-L22）；v14 补回 v13 缺失的 m 档；
2. **s 档参数量注释**（`yolov14-deformable.yaml` L13-L17）：n=2.6M / s=9.2M / m=19.8M / l=26.7M / x=59.6M——是**唯一一份 yaml 里把参数量写进注释**的；
3. **选错变体直接掉精度**：`yolov14-game2real.yaml` 里没有 `ViewEmbedding`，用它训练无人机场景会失去视角嵌入；`yolov14-panorama.yaml` 没有 DAL，用它训练游戏画面会失去域适应——**场景路由的静态选择必须与训练数据匹配**；
4. **`yolov14-adaptive.yaml` 声称 `AdaptiveAugmentPolicy` 但代码未实现**——快照里 grep `AdaptiveAugmentPolicy` 只在 yaml 注释里出现一次，实际是死配置；README 声称它是「auto-detects scene type and applies optimal aug」，实际是纯注释（见 [痛点页 §X](/versions/v14/painpoints)）。

**推理开销**：零开销（yaml 是建网期选择）。

## 3. 6 类视角码字枚举（`trick-view-types`）

> **v1~v13 读者路标**：v13 之前所有版本的模型都对「视角」零感知——一张 COCO 图片无论正拍、俯视、鱼眼、全景，进入网络前是同一个 `[B, 3, 640, 640]` 张量；**v14 是 YOLO 系列首个在骨干入口注入「视角信息」的版本**——通过 `ViewEmbedding` 把 6 类视角的 embedding 注入 P2 特征。

**代码事实**（<SrcLink file="v14:nn/modules/block.py" :line="1710" :end="1731" />）：

```python
class ViewEmbedding(nn.Module):
    VIEW_TYPES = {"pinhole": 0, "fisheye": 1, "panoramic": 2, "drone": 3, "bev": 4, "ground": 5}

    def __init__(self, c1, num_views=6, embed_dim=64):
        super().__init__()
        self.view_embed = nn.Embedding(num_views, embed_dim)
        self.proj = Conv(c1 + embed_dim, c1, 1)

    def forward(self, x, view_ids=None):
        if is_fast_mode():
            return x
        B, C, H, W = x.shape
        if view_ids is None:
            view_ids = x.new_zeros(B, dtype=torch.long)
        emb = self.view_embed(view_ids).view(B, -1, 1, 1).expand(-1, -1, H, W)
        return self.proj(torch.cat([x, emb], dim=1))
```

**6 类视角码字**：

| 码字 | 语义 | 常见数据源 |
|------|------|-----------|
| `pinhole=0` | 标准针孔相机 | COCO / VOC / Pascal 等传统检测数据集 |
| `fisheye=1` | 鱼眼广角 | 车载前视 / 全景相机 |
| `panoramic=2` | 360° 等距投影 | 全景相机 / Matterport |
| `drone=3` | 无人机航拍 | VisDrone / UAV 数据集 |
| `bev=4` | Bird's-Eye-View 俯视 | 自动驾驶 BEV 检测 |
| `ground=5` | 地面视角 | 地面机器人 / 消费级拍照 |

**五要点**：

1. **枚举是硬编码常量**：`VIEW_TYPES` 是类属性，**不能扩展**——想加第 7 类视角（比如「水下」「太空」）必须改源码，或把 `num_views` 从 6 提到 7（此时 `nn.Embedding(7, 64)` 多一个可学习向量，但没有语义映射）；
2. **`view_ids=None` 时默认走 `pinhole=0`**：L1728-L1729 静默把 `view_ids` 填 0——**如果忘记传 view_ids，模型会按针孔视角训/推理**，其他视角的场景会掉精度；
3. **embedding 与特征 concat 后再 1×1 Conv 投影**：`x.shape = [B, C, H, W]`，`emb.shape = [B, 64, H, W]`，`cat` 后 `[B, C+64, H, W]`，`Conv(C+64, C, 1)` 投回 `[B, C, H, W]`——**通道数守恒**，yaml 里 P2 从 128 进从 128 出；
4. **推理时可切视角**：训练时用真实视角标签（比如 `data` 字典里的 `view_type` 键），推理时可以直接指定——如果部署到多源混合场景，**每张图片带一个视角标签**即可让模型自适应；
5. **`embed_dim=64` 是硬编码**：yaml 第 2 位固定 64，`Embedding(6, 64)` 一共 **384 个参数**（可忽略不计）；调大 embed_dim 会增加 `Conv(C+64, C, 1)` 的输入维度（`C=128` 时 `192×128×1×1 = 24.6K` 参数，`C=128, embed_dim=256` 时 `384×128 = 49.2K`）。

**推理开销**：`Embedding + concat + Conv(1×1)` ≈ **可忽略**（<0.01 ms / <10K 参数）；Fast Mode 下直接跳过。

## 4. Game2Real 消融三层（`trick-game2real-ablation`）

> **v1~v13 读者路标**：v13 之前所有版本的基准都是 COCO 标准针孔图像，「游戏画面检测」这类跨域任务是**完全未覆盖的场景**；**v14 首次系统化地做跨域检测（Game2Real）**——论文给出 3 层消融（数据 / 特征 / 目标），每一层单独贡献 6.5~11.7 AP，累计 +25.5 AP（24.1 → 50.2，相对 v12s baseline）。

**消融数据**（README L113-124，YOLOv12s baseline = 24.1 mAP on Game）：

| 层 | 模块 | 类型 | 累计 AP | 增量 |
|----|------|------|--------|------|
| 0 baseline | YOLOv12s 原版 | — | 24.1 | — |
| +1 数据层 | `GameCharacterStylization` | 数据增强 | 35.8 | **+11.7** |
| +2 特征层 | `DomainAdaptiveLayer`（AdaIN） | 结构模块 | 42.3 | **+6.5** |
| +3 目标层 | `DomainAdversarialLoss`（GRL） | 损失函数 | 49.6 | **+7.3** |
| +完整 | + `yolov14-game2real.yaml` 全结构 | — | **50.2** | +0.6（额外结构增益） |

**三层机制详解**：

**层 1：数据层 `GameCharacterStylization`（+11.7 AP）**

`README L117` 描述的四算子组合：
- `posterisation`（bit depth 3-6）——模拟游戏的 posterise 渲染效果
- `unsharp masking`——锐化边缘，模仿游戏画面的边缘强调
- `saturation boost`（×1.5-1.8）——提升饱和度
- `contrast adjustment`——调整对比度

**这一层贡献最大（+11.7 AP）**，因为它是「让训练数据的分布靠近游戏画面」的第一步——**如果没有这一层，后面两层再强大也无法弥补数据分布的巨大鸿沟**。

> **注意**：`GameCharacterStylization` 类在快照源码中**未实现**（grep 只在 README 里出现），是**论文声称但代码未落地的模块**——需要用户自己按 README 描述实现（见 [痛点页 §X](/versions/v14/painpoints)）。

**层 2：特征层 `DomainAdaptiveLayer`（+6.5 AP）**

<SrcLink file="v14:nn/modules/block.py" :line="1871" :end="1893" />

核心机制：**AdaIN（Adaptive Instance Normalisation）**
- 训练时：把 `game` 域特征的第二/三阶矩（mean / std）向 `real` 域对齐
- 推理时：按输入实际域做归一化

**在 `yolov14.yaml` 里出现 2 处**（backbone #3 P2 后 + #7 P4 后），全量版是双层叠加；`yolov14-game2real.yaml` 里也是 2 层（#2 + #6）。

**层 3：目标层 `DomainAdversarialLoss`（+7.3 AP）**

`loss.py L815-L859` 的 DANN 变体：
- 域判别器（`DomainAdaptiveLayer.domain_classifier`）学习区分 game/real
- GRL（`GradientReverse`）在特征提取器到判别器之间插入，反向传播时**翻转梯度符号**并乘 λ
- 特征提取器因此**学会产生无法被判别器区分的域不变特征**

**四要点**：

1. **消融增量呈「倒 U 形」——数据层贡献最大**：这是跨域任务的普遍规律，「数据对齐 > 特征对齐 > 目标对齐」；
2. **三层可以独立叠加**：`DomainAdaptiveLayer` 单独使用（无 GRL）也能拿到 +6.5 AP，`DomainAdversarialLoss` 单独使用（无 DAL）也能拿到 +7.3 AP——但两者叠加会有约 +1.2 AP 的相互增益（24.1 → 35.8 → 42.3 → 49.6 vs 单独叠加 24.1+6.5+7.3=37.9）；
3. **训练时 `batch["domains"]` 必须正确标注**：0=real，1=game；训练数据混合真实照片 + 游戏画面（或经 stylization 处理的真实照片）时，标签必须对应——**如果 label 反了，GRL 会让特征学到「域相关」而不是「域不变」，效果比不训还差**；
4. **推理时 GRL 自动关闭**：`GradReverse.backward` 只在反向传播时生效，前向是恒等——推理完全零开销，且不需要 domain 标签。

**推理开销**：`DomainAdaptiveLayer` 前向（AdaIN）≈ 每层 <0.05 ms（LayerNorm 级开销）；`GradReverse` 前向零开销。

## 5. 4 条退火 schedule 单点下发（`trick-schedules`）

> **v1~v13 读者路标**：v8-v13 只有 `warmup_lr` + `cosine/linear` 学习率 schedule 两条曲线；**v14 首次出现 4 条独立的超参 schedule 曲线**——λ_reg（50 epoch 线性退火）、GRL λ（100 epoch sigmoid 退火）、τ（每 iter 指数退火）、tal_topk（无退火）；**所有 schedule 都从 `v14DetectionLoss.set_epoch()` 单点下发**，避免多路 schedule 各写各的导致不一致。

**4 条 schedule 对照表**：

| schedule | 初值 → 终值 | 时域 | 函数形式 | 代码位置 |
|---------|-----------|------|---------|---------|
| λ_reg | 0.01 → 0.001 | 50 epoch 线性退火 | `frac = min(epoch/50, 1)`；`λ = 0.01 + (0.001-0.01)·frac` | loss.py L879-L896 |
| GRL λ | 0 → 1 | 100 epoch sigmoid 退火 | `2/(1+exp(-10·p))-1`, `p = min(epoch/100, 1)` | block.py L63-L69 |
| τ（NT-Xent 温度） | 0.05 → 0.5 | **每 iteration 指数退火** | `0.05 + (0.5-0.05)·exp(-0.01·t)` | loss.py L760-L777 |
| tal_topk | 10（无退火） | — | — | loss.py L178 |

**关键实现**（<SrcLink file="v14:utils/loss.py" :line="932" :end="950" />）：

```python
def set_epoch(self, epoch):
    """Update training schedules for this epoch."""
    self._epoch = epoch
    self.offset_loss.set_epoch(epoch)
    try:
        from ultralytics.nn.modules.block import grl_lambda_schedule, DomainAdaptiveLayer
        model = getattr(self, "model", None)
        if model is not None:
            for m in model.modules():
                if isinstance(m, DomainAdaptiveLayer):
                    m.grl_lambda = grl_lambda_schedule(epoch, max_epochs=100)
    except Exception:
        pass
```

**四要点**：

1. **λ_reg 用线性插值**——不是 cosine、不是指数，最简单的线性——50 epoch 内从 0.01 减到 0.001；训练 300 epoch 时**后 250 epoch λ_reg 完全恒定在 0.001**；
2. **GRL λ 用 sigmoid 形式**（`2/(1+exp(-10·p))-1`）——是 DANN（Ganin et al. 2016）原论文的公式，`p=0` 时 λ=0、`p=1` 时 λ≈1，`p=0.5` 时 λ≈0.76——**训练前 50 epoch 判别器几乎不向特征提取器反传梯度**，让特征提取器先在检测损失上稳定，后期判别器再介入对抗；
3. **τ 是唯一「每 iter 退火」的**——不是每 epoch；`step_update()` 在 `v14DetectionLoss.__call__` 末尾 L979 每步调一次；`β=0.01` 意味着 `t≈100` iter 时 τ≈0.065，`t≈500` iter 时 τ≈0.10，`t≈2000` iter 时 τ≈0.30；**每 64 batch 一次 epoch 时**，300 epoch ≈ 300×64=19200 iter，τ 最终稳定在 `0.05 + 0.45·exp(-0.01·19200) ≈ 0.05`（`exp(-192)` 是几乎 0）；
4. **`try: except: pass` 静默吞异常**——如果 `DomainAdaptiveLayer` 导入失败或 `model` 属性不存在，`set_epoch` 静默失败；**这是 v14 训练调试的第一个坑**（见 [痛点页 §X](/versions/v14/painpoints)）。

**⚠ 论文-代码偏差**：论文 §3.7 明确写「τ = 0.05→0.5 adaptive (**per-epoch**)」，但代码里 `step_update()` 在 `__call__` 末尾**每 iter 调一次**——τ 实际退火速度比论文描述**快 batch_size 倍**。要按论文复现，需在 trainer 里改成「每 epoch 调一次 `step_update`」。

**推理开销**：零开销（训练期配置）。

## 6. 三分量增益 + TAL 参数沿用 v8（`trick-loss-gains-inherit-v8`）

> **v1~v13 读者路标**：v13 损失侧对 v8 零改动；**v14 的三分量与 TAL 分配器同样一字不差沿用 v8**——`7.5/0.5/1.5` 增益、`topk=10, α=0.5, β=6.0`——这是从 v8 到 v14 六个版本（v8/v9/v11/v12/v13/v14）里连续 6 版没动过的默认值。

**代码事实**（<SrcLink file="v14:utils/loss.py" :line="176" :end="180" />）：

```python
class v8DetectionLoss(nn.Module):
    def __init__(self, model, tal_topk=10):
        super().__init__()
        ...
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
```

- **`topk=10, α=0.5, β=6.0`** 在 `v8DetectionLoss.__init__` 显式传参，覆盖 tal.py 类默认（`topk=13, α=1.0`）——**这是 v8 以来 6 版未修过的隐性坑**，v14 继续沿用；
- **三分量增益**（`loss[0]*=7.5; loss[1]*=0.5; loss[2]*=1.5`）在 `compute_loss` 里硬编码；
- **`loss_items` 返回 3 维 `[box, cls, dfl]`**（L983）——辅助损失不进 items，调试时必须单独 hook batch 里的 `offset/domain_logits/features` 键。

**推理开销**：零开销（初始化）。

## 7. `YOLOV14_FAST=1` Fast Mode：推理侧三开关（`trick-fast-mode`）

> **v1~v13 读者路标**：v13 之前所有版本的「训练模型 = 推理模型」，训练与推理走同一份 forward；**v14 首次引入「训练/推理 forward 分支」**——通过环境变量 `YOLOV14_FAST=1` 在推理期关闭三段场景相关模块（Scene Analysis、DomainAdaptiveLayer 的 domain_classifier、ViewEmbedding），README 声称「保留 70% 跨域增益，降低延迟」。

**代码事实**（<SrcLink file="v14:nn/modules/block.py" :line="16" :end="37" />）：

```python
_FAST_MODE_CACHE: bool | None = None

def is_fast_mode() -> bool:
    """Check if YOLOv14 Fast Mode is enabled (§4.4)."""
    global _FAST_MODE_CACHE
    if _FAST_MODE_CACHE is None:
        _FAST_MODE_CACHE = os.environ.get("YOLOV14_FAST", "0") == "1"
    return _FAST_MODE_CACHE

def reset_fast_mode_cache() -> None:
    """Clear the fast-mode cache so the next call re-reads the env var."""
    global _FAST_MODE_CACHE
    _FAST_MODE_CACHE = None
```

**Fast Mode 影响的模块**：

| 模块 | 位置 | 关闭行为 | 代码是否实现 |
|------|------|---------|------------|
| Scene Analysis（启发式场景分类） | README §Architecture Overview 声称 | 直接返回默认场景标签，不做 heuristic 分析 | ❌ 快照里 grep 不到 `SceneAnalysis` 类 |
| DomainAdaptiveLayer | `DomainAdaptiveLayer.forward` L1946 | 跳过 domain classifier 分支；AdaIN 前向保留 | ✓ |
| ViewEmbedding | `ViewEmbedding.forward` L1725-L1726 | 直接返回 `x`，跳过 embedding 注入 | ✓ |

> ⚠ **Scene Analysis 未实现**：README L84-L91 把 Scene Analysis 列为流水线第 1 阶段，但快照源码里 grep 不到 `SceneAnalysis` 类；`YOLOV14_FAST=1` 对它的影响无从验证。

**使用方式**：

```bash
# 训练时不设环境变量
python train.py model=yolov14.yaml

# 推理时开启 Fast Mode
export YOLOV14_FAST=1
python infer.py model=yolov14.pt
```

```python
import os
os.environ["YOLOV14_FAST"] = "1"
from ultralytics import YOLO
model = YOLO("yolov14.pt")
# 若需要重读环境变量：
from ultralytics.nn.modules.block import reset_fast_mode_cache
reset_fast_mode_cache()
```

**四要点**：

1. **环境变量是全局生效**——一旦设置 `YOLOV14_FAST=1`，**所有已加载的模型都会跳过这三段**；
2. **`_FAST_MODE_CACHE` 是缓存**——首次调用 `is_fast_mode()` 后不再读 env，性能友好；但要改环境变量需主动 `reset_fast_mode_cache()`；
3. **README 声称「保留 70% 跨域增益」**——具体数字来自论文消融（未在快照中提供详细表格）；
4. **训练时不要开 Fast Mode**——如果训练期间 `YOLOV14_FAST=1`，`ViewEmbedding` 与 `DomainAdaptiveLayer` 都不参与计算，模型永远学不到跨域特征——**这是 v14 部署的一个隐蔽坑**（见 [痛点页 §X](/versions/v14/painpoints)）。

**推理开销**：Fast Mode 开启时**降低**——跳过 ViewEmbedding 的 `Conv(1×1)` + 跳过 Scene Analysis 的 heuristic + 跳过 Domain classifier 的 forward。

## 8. 训练命令与常见场景（`trick-commands`）

> **v1~v13 读者路标**：v14 没有独立的 `YOLOv14` 顶层类，`from ultralytics import YOLO` 后传 yaml 路径即可；**但 6 份 yaml 变体的选择比 v12/v13 更讲究**——选错变体直接掉精度。

**官方复现配方**（300 epoch，需多卡）：

```bash
# 通用全量（推荐作为起点）
yolo detect train \
  model=yolov14.yaml \
  data=coco.yaml \
  epochs=300 imgsz=640 \
  device=0,1,2,3

# 游戏画面专用
yolo detect train \
  model=yolov14-game2real.yaml \
  data=game_dataset.yaml \
  epochs=300 imgsz=640
```

**单卡小数据集微调**：

```python
from ultralytics import YOLO
model = YOLO('yolov14.yaml')
results = model.train(
    data='mydataset.yaml',
    epochs=100, batch=32, imgsz=640,
    lr0=0.001, patience=20,
    lambda_adv=0.0, lambda_cross=0.0,  # 单域微调关闭域/视角损失
)
```

**推理与 Fast Mode**：

```python
import os
os.environ["YOLOV14_FAST"] = "1"  # 生产部署打开
from ultralytics import YOLO
model = YOLO('yolov14.pt')
preds = model('delta_force_screenshot.jpg')
```

**训练前的环境自检**：

```python
import torch
import os
from ultralytics.nn.modules.block import is_fast_mode, reset_fast_mode_cache
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("fast_mode:", is_fast_mode())  # 训练前应为 False
reset_fast_mode_cache()  # 若改过 env 需手动刷新
```

**注意事项**：

- **官方权重尚在 IN PROGRESS**（README L71-L77）——5 个变体的 checkpoint 均未公开发布，需自行训练或等 ETA；
- **四基准数据集未开源**——Game/Fisheye/Drone/Panorama 的自建评测集当前未公开，无法在本站或第三方完全复现论文数字；
- **不要动 loss 权重默认值**——`lambda_adv=0.05 / lambda_cross=0.05 / lambda_reg=0.01` 是作者调好的，改动需要重新消融；
- **COCO 单域训练建议关 adv/cross**——`lambda_adv=lambda_cross=0.0`（否则纯 COCO 训练会略掉 0.1-0.3 AP）；
- **AGPL-3.0 许可**——商用需遵守 copyleft，闭源产品不能直接集成 v14 代码。

## 9. 与 v8 / v9 / v10 / v11 / v12 / v13 配方完整对比（`trick-compare`）

> **v1~v13 读者路标**：把 7 代配方摆在一起，v14 的位置很清楚——**配方极简回到 v8**，**结构层首次引入 6 类视角枚举 + 场景路由 + 4 条退火 schedule**，**推理侧首次引入训练/推理分支（Fast Mode）**。

| 项目 | v8 | v9 | v10 | v11 | v12 | v13 | **v14** |
|------|----|----|-----|-----|-----|-----|--------|
| 官方 epochs | 300 | 300 | 300 | 500 | **600** | **600** | **300** |
| 官方 batch | 16-64 | 16-64 | 16-64 | 16-64 | **256** | **256** | **未指定** |
| mosaic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **未指定** |
| mixup | 0.0 | 0.0 | 0.0 | 0.0 | 分档 0.0~0.2 | 分档 0.0~0.2 | **未指定** |
| copy_paste | 0.0 | 0.0 | 0.0 | 0.0 | 分档 0.1~0.6 | 分档 0.1~0.6 | **未指定** |
| 损失增益 | 7.5/0.5/1.5 | 7.5/0.5/1.5 | 6×1.0 | 7.5/0.5/1.5 | 7.5/0.5/1.5 | 7.5/0.5/1.5 | **7.5/0.5/1.5** |
| TAL topk | 10 | 10+10 | 10+1 | 13（类默认） | 10（显式） | 10（显式） | **10（显式）** |
| **辅助损失** | — | — | — | — | — | — | **L_adv+L_cross+L_reg（4 新类）** |
| **可退火超参** | 无 | 无 | 无 | 无 | 无 | 无 | **λ_reg/GRL λ/τ 三条 + 学习率** |
| **视角感知** | — | — | — | — | — | — | **ViewEmbedding 6 类码字** |
| **域适应** | — | — | — | — | — | — | **DomainAdaptiveLayer AdaIN + GRL** |
| **yaml 数量** | 5 档（1 yaml） | 5 档 | 5 档 | 1（5 档） | 1（5 档） | 1（4 档，无 m） | **6 yaml × 5 档 = 30 组合** |
| **推理/训练分支** | — | — | — | — | — | — | **YOLOV14_FAST=1** |
| 许可证 | AGPL | AGPL | AGPL | AGPL | AGPL | Apache-2.0 | **AGPL（回切）** |
| 训练入口 | YOLO | YOLOv9 | YOLOv10 | YOLO | YOLO | YOLO | **YOLO** |
| FlashAttention 依赖 | — | — | — | — | 硬依赖 | 硬依赖 | **硬依赖** |
| 档位 | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/l/x（无 m） | **n/s/m/l/x（补回 m）** |

**四句话总结**：

1. **v14 训练配方大幅简化，回到 v8 极简**——300 epoch + imgsz 640，无 batch/mosaic/mixup/copy_paste 分档；作者把「精细调参」外包给「结构层场景路由」；
2. **v14 首次引入 6 类视角枚举 + 场景路由**——`ViewEmbedding.VIEW_TYPES` 硬编码 6 类码字，6 份 yaml 变体覆盖 4 种专门场景 + 通用全量 + adaptive（后者的 AdaptiveAugmentPolicy 未实现）；
3. **v14 首次引入 4 条独立退火 schedule**——λ_reg（线性 50ep）/ GRL λ（sigmoid 100ep）/ τ（指数每 iter）/ tal_topk（无退火），全部从 `v14DetectionLoss.set_epoch()` 单点下发；
4. **v14 首次引入推理/训练分支**——`YOLOV14_FAST=1` 环境变量在推理期跳过三段场景模块，是 YOLO 系列首个「训练与推理走不同 forward」的版本。

## 10. 常见坑与反常行为（`trick-common-pitfalls`）

> 本节汇总 v14 训练过程中最容易踩到的坑，详见 [痛点页](/versions/v14/painpoints)。

**坑 1：`batch` 字典缺键导致辅助损失静默不生效**

`v14DetectionLoss.__call__` 里三个辅助损失都是「key 存在才启用」（L957-L976），如果 trainer 没在 batch 里放 `offset/offset_init/domain_logits/domains/features/view_ids` 中的任何一个，**对应损失直接跳过，训练不报错但也不学到跨域特征**。

**坑 2：`DomainAdversarialLoss.__init__` 的 `lambda_domain` 默认 0.1 被外层 0.05 覆盖**

`DomainAdversarialLoss.__init__(lambda_domain=0.1)` L838 里的 0.1 是**静态默认**，但外层 `v14DetectionLoss.__init__(lambda_adv=0.05)` L924 会把它覆盖为 0.05——**读源码看 0.1 会以为是 0.1，实际生效的是 0.05**。

**坑 3：τ 单位偏差（论文 per-epoch vs 代码 per-iteration）**

论文 §3.7 写「τ = 0.05→0.5 adaptive (**per-epoch**), β=0.01」，但代码里 `step_update()` 在 `__call__` 末尾每 iter 调一次（L979）——τ 实际退火速度比论文快 batch_size 倍（batch=32 时快 32 倍）。要按论文复现需改 trainer 让 `step_update()` 每 epoch 调一次。

**坑 4：`view_ids=None` 时静默走 `pinhole=0`**

`ViewEmbedding.forward` L1728-L1729 在 `view_ids=None` 时把视角标签填 0（pinhole）——如果训练多视角数据集但忘记传 `view_ids`，模型会当成全部是 pinhole 训练，多视角效果为零。

**坑 5：`YOLOV14_FAST=1` 在训练时误开**

如果训练期间 `YOLOV14_FAST=1` 已设，`ViewEmbedding.forward` 与 `DomainAdaptiveLayer.forward` 都会跳过核心计算——**模型永远学不到跨域特征**，且不会有任何警告。生产环境部署脚本如果 `export YOLOV14_FAST=1` 但被训练脚本继承，会静默训出「跨域无效」的模型。

**坑 6：`AdaptiveAugmentPolicy` 未实现**

`yolov14-adaptive.yaml` L5 注释声称「AdaptiveAugmentPolicy: auto-detects scene type and applies optimal aug」，但代码快照中 grep 不到该类实现——**该 yaml 实际行为与 `yolov14.yaml` 全量版一致**（README 声称的自适应增强是死配置）。

**坑 7：官方权重与四基准数据集未开源**

README roadmap L71-L77 显示权重与数据集均「IN PROGRESS」——用户无法直接下载权重验证数字，也无法在自己的机器上复现 Game/Fisheye/Drone/Panorama 四基准的官方 mAP。

**坑 8：GRL λ=0 时特征提取器收不到 L_adv 梯度**

`grl_lambda_schedule(epoch=0)` 返回 0（sigmoid 在 p=0 处取 0）——训练前若干 epoch（大约 20-30 epoch 才升到 0.1 以上）GRL 反传的实际梯度接近 0，特征提取器这段时间只学 `L_det`——这是**设计意图**（先稳定检测损失，再引入对抗），但如果用户想「更快看到域适应效果」，可以缩短 GRL schedule 的 `max_epochs` 参数。

## 延伸阅读

- [理论 Concept](/versions/v14/concepts)（D-AAttn / ViewEmbedding / DomainAdaptiveLayer / SphereAAttn 四个概念的机制解析）
- [损失函数](/versions/v14/losses)——v14DetectionLoss + 三分量 + L_adv/L_cross/L_reg
- [反向传播推导](/versions/v14/loss-gradient)——三条独立梯度通路（GRL 反向 + offset 正向 + cross 正向）
- [痛点追踪](/versions/v14/painpoints)——训练侧 8 大坑的详细分析
- [YOLOv13 训练技巧](/versions/v13/tricks)——配方层最复杂的一版，作为对照
- [YOLOv12 训练技巧](/versions/v12/tricks)——分档 mixup/copy_paste 的起源
- [YOLOv8 训练技巧](/versions/v8/tricks)——v14 训练配方回归的直接出处
- [术语表](/glossary)——TAL、DFL、GRL、NT-Xent、AdaIN、DANN、AAttn、D-AAttn、Fast Mode 等

**参考**：

- Jia J. Z. *YOLOv14: Adaptive Real-Time Object Detection for Diverse Imaging Conditions.* arXiv:2608.04720v3, 2026-08-14.
- zhangcbb/yolov14. https://github.com/zhangcbb/yolov14 @ commit `0b8610d`
- README L166-176（官方训练命令）· L113-124（Game2Real 消融）· L134-L135（`AdaptiveAugmentPolicy` 声称）
- block.py L16-L37（Fast Mode 缓存）· L63-L69（`grl_lambda_schedule`）· L1710-L1731（`ViewEmbedding`）· L1736-L1773（`DynamicScaleRouter`）
- loss.py L760-L777（`CrossViewConsistencyLoss.temperature_at`）· L838-L859（`DomainAdversarialLoss`）· L879-L911（`OffsetRegularizationLoss`）· L911-L983（`v14DetectionLoss`）
