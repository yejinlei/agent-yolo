# YOLO26 训练技巧

> **v1~v14 读者路标**：v14 是「v13 结构 + 多域增强四件套」的第三方学术版，配方层大幅简化；**v26 是 Ultralytics 官方 2026 快照**，把「训练策略」放到了「E2E 双分支权重退火」（`E2ELoss.decay` 0.8→0.1）与「任务专用损失类」两处。v26 的配方比 v14 更完整——官方给 **12 份 yaml 变体、6 个任务方向、五档 scales、11 项官方权重**——训练策略集中在**E2E 权重退火**（唯一时间相关权重）、**6 任务损失类**（每个任务有独立配方）、**五档 scales 复合缩放**（depth × width × max_channels）、**Fast Mode = 推理时只跑 one2one**（无环境变量，写在 `Detect.__init__` 里）。本页 9 节盘点这些训练侧事实。

## 全部条目速查

| 条目 | 位置 | 说明 | 推理是否零开销 |
|------|------|------|----------------|
| **11 份 yaml 变体（1 检测 + 6 任务 + 3 ED + 1 TVP）** | `cfg_models_26/*.yaml` | `yolo26.yaml` / `-cls` / `-seg` / `-obb` / `-pose` / `-depth` / `-segsemantic` / `-ed` / `-ed-seg` / `-ed-obb` / `-tvp` | 是（选模型即定结构） |
| **档位五档：n/s/m/l/x** | yaml L11-L17 | 复合缩放 `[depth, width, max_channels]`；n 档 2.57M 参数、x 档 59.0M 参数 | 否（建网期） |
| **`end2end: True` E2E 双分支** | yaml L9 | Detect 内部 `copy.deepcopy` 复制 cv2/cv3；推理时只跑 one2one | **是（推理期自动切换）** |
| **`reg_max: 1` DFL → Identity** | yaml L10 | bbox 回归从 64 bins 缩到 4 值 raw logits；DFL 分支梯度为 0 | 是（结构） |
| **E2ELoss 权重退火 0.8→0.1** | loss.py L1296 | `decay(t) = max(1-t/(T-1),0)·0.7+0.1`；训练 500 epoch 时 w_o2m 每 epoch 递减 0.0014 | 否（训练期） |
| **E2EDetectLoss 等权版** | loss.py L1263-L1280 | `loss = loss_o2m + loss_o2o`；不推荐用于生产 | 否（训练期） |
| **三分量增益 7.5/0.5/1.5 沿用 v8** | loss.py L252-L256 | 从 v8 到 v26 七版一字不差 | 是（初始化） |
| **TAL `topk=10 α=0.5 β=6.0` 沿用 v8** | loss.py L159 | 显式覆盖 tal.py 类默认 `topk=13, α=1.0`；v8 以来未修的隐性坑 | 是（初始化） |
| **Pose RLE + RealNVP Flow** | loss.py L810-L932 | Normalizing Flow 学不确定性；`flow_model` 可选；`_rle_zero` 兜底保计算图 | 否（推理保留 flow） |
| **Depth SILog + 4-尺度梯度 L1** | loss.py L1169-L1260 | `silog_lambda=1.0` scale-invariant；`keep` mask 过滤 LiDAR 稀疏图 | 是（推理用 pred_depth） |
| **Semantic CE + Cityscapes 权重** | loss.py L1389-L1442 | `nc=1` BCE / `nc>1` CE(ignore=255)；`persistent=False` buffer | 是（推理用 pred_cls） |
| **TVP 动态 nc 调整** | loss.py L1318-L1388 | `vp_criterion.nc` 按 prompt 数量动态扩展 | 是（推理用） |
| **官方权重已发布** | 官方仓库 | 6 任务 × 5 档 = 30+ checkpoint 全部可用；`yolo26n.pt` 直接下载 | — |
| **5 档 scales 详细参数** | yaml L13-L17 | n: 260 layers, 2.57M params, 6.1 GFLOPs；x: 392 layers, 59.0M, 209.5 GFLOPs | 否（建网期） |
| **`cross_attn=True` 只在 #22 P5** | yaml L50 | C3k2 唯一启用跨阶段注意力位置 | 是（结构） |
| **`e=0.5` 只在 #22 P5** | yaml L50 | C3k2 唯一启用压缩 Bottleneck 位置 | 是（结构） |
| **`A2C2f` 保留但 yaml 不用** | block.py L1804 | 类定义在，但 yaml 里 0 处实例化（v11 的模块） | 是（yaml 未启用） |
| **`AAttn` 保留但 yaml 不用** | block.py L1447 | 类定义在，但 yaml 里 0 处实例化 | 是（yaml 未启用） |
| **`SAVPE` 新模块** | block.py L1930 | v26 独有 Spatial Attention View-aware Pooling | 是（yaml 未启用） |
| **`RealNVP` 新模块** | block.py L2045 | Normalizing Flow；供 Pose RLE 使用 | 否（推理保留） |
| **`Proto26` 新模块** | block.py L1998 | 实例分割掩膜头，继承 v10 Proto | 是（yaml 未启用） |
| **`YOLOEDetect` 新头** | head.py L1091 | 开放词汇检测；`text_model: mobileclip2:b` | 是（yaml 未启用） |
| **AGPL-3.0 许可证** | LICENSE | 商用需遵守 copyleft | — |

## 1. 官方配方：`model.train(data, epochs=300, imgsz=640)`（`trick-official-recipe`）

> **v1~v14 读者路标**：v12/v13/v14 都用「600 epoch + batch 256 + 分档 mixup/copy_paste」或极简配方；**v26 官方 README 给的是「标准 Ultralytics 训练命令」**——`model.train(data='coco128.yaml', epochs=300, imgsz=640)`，无特殊 batch、无分档增强。这是「官方快照」的标准配方。

**官方训练命令**（README 逐行照抄）：

```python
from ultralytics import YOLO

# 加载官方权重
model = YOLO("yolo26n.pt")

# 训练
results = model.train(
    data="coco128.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    device=0
)

# 验证
metrics = model.val(data="coco128.yaml", imgsz=640)

# 导出
model.export(format="onnx")  # 或 "torchscript" / "tflite" / "engine"
```

**要点**：
- **300 epoch**：与 v14 一致，是 Ultralytics 默认；
- **imgsz=640**：COCO 标准分辨率；
- **batch=16**：单卡 T4 默认；
- **device=0**：单 GPU；多卡用 `device=[0,1,2,3]` 或 `device='cuda'`。

## 2. 五档 scales 复合缩放（`trick-scales`）

> **v1~v14 读者路标**：v13 只有 4 档（n/s/l/x），v14 补回 m 档；**v26 五档 scales 与 v14 一致**——`[depth, width, max_channels]` 三元组复合缩放。

**五档 scales 详细参数**（yaml L13-L17）：

| 档位 | depth | width | max_ch | layers | Params | GFLOPs | 用途 |
|------|-------|-------|--------|--------|--------|--------|------|
| n | 0.50 | 0.25 | 1024 | 260 | 2.57M | 6.1 | 边缘设备 |
| s | 0.50 | 0.50 | 1024 | 260 | 10.01M | 22.8 | 移动端 |
| m | 0.50 | 1.00 | 512 | 280 | 21.90M | 75.4 | 桌面 GPU |
| l | 1.00 | 1.00 | 512 | 392 | 26.30M | 93.8 | 高精度 |
| x | 1.00 | 1.50 | 512 | 392 | 58.99M | 209.5 | 云端 |

**复合缩放机制**：

```python
# 简化示意（实际在 ultralytics.nn.tasks 里）
def scale_depth(ch, d):
    """depth 缩放：C3k2 n 值乘 d"""
    return max(1, int(round(ch * d)))

def scale_width(ch, w, min_ch=8):
    """width 缩放：通道乘 w，至少 min_ch"""
    return max(min_ch, int(ch * w))

def scale_channels(ch, w, max_ch):
    """通道上限：不超过 max_ch"""
    return min(ch * w, max_ch)
```

**要点**：
- **n 档 max_channels=1024**：允许通道爆炸到 1024——**因为 width=0.25 时通道不会超过 1024**；
- **m/l/x 档 max_channels=512**：抑制 P5 通道爆炸——**因为 width=1.0/1.5 时通道会超过 1024**；
- **depth=0.50**：C3k2 n 值从 2 缩到 1——**v26 s 档骨干 #2/#4 的 n 从 2 缩到 1**；
- **depth=1.00**：C3k2 n 值不变——**v26 l/x 档用满深度**。

## 3. E2E 双分支权重退火（`trick-e2e-annealing`）

> **v1~v14 读者路标**：v14 有 3 条退火 schedule（λ_reg / GRL λ / τ），v26 只有 1 条（E2E decay）——**v26 是 v8 以来唯一用「权重退火」作为训练策略的版本**。

**E2ELoss.decay 公式**（<SrcLink file="v26:utils/loss.py" :line="1296" :end="1298" />）：

```python
def decay(self, x):
    # x 是当前 epoch
    return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m
```

**退火曲线**：

| Epoch | 0 | 100 | 200 | 300 | 400 | 500 |
|-------|---|-----|-----|-----|-----|-----|
| w_o2m | 0.8 | 0.66 | 0.52 | 0.38 | 0.24 | **0.10** |
| w_o2o | 0.2 | 0.34 | 0.48 | 0.62 | 0.76 | **0.90** |

**训练策略**：
- **训练初期（0-100 ep）**：one2many 主导（w_o2m=0.8→0.66）——密集监督让模型学「所有 anchor 的粗略分配」；
- **训练中（100-300 ep）**：两分支平衡（w_o2m=0.66→0.38）——one2many 与 one2one 各贡献一半；
- **训练后期（300-500 ep）**：one2one 主导（w_o2m=0.38→0.1）——模型学「每 anchor 的最优决策」；
- **推理时**：只跑 one2one 分支——one2many 被剪掉，NMS 免费。

**要点**：
- **权重退火是「时间相关权重」**——`w_o2m` 是 epoch 的函数，不是模型参数；
- **500 epoch 时每 epoch 递减 0.0014**——线性退火；
- **`decay` 函数在 `E2ELoss.__call__` 里每次调用**——不需要 `set_epoch()`（v14 的做法是 `set_epoch` 单点下发）。

## 4. 三分量增益 + TAL 分配器（v8 沿用）

> **v1~v14 读者路标**：从 v8 到 v26 七版一字不差——**YOLO 系列的检测分支损失公式从未被改过一个字节**。

**三分量增益**（<SrcLink file="v26:utils/loss.py" :line="252" :end="256" />）：

```python
loss[0] *= self.hyp.box   # 7.5
loss[1] *= self.hyp.cls   # 0.5
loss[2] *= self.hyp.dfl   # 1.5
```

**TAL 分配器**（<SrcLink file="v26:utils/loss.py" :line="159" :end="161" />）：

```python
self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
```

**要点**：
- **`topk=10`**：one2many 分支用 10 个正样本 anchor；
- **`α=0.5`**：TAL 分配器的对齐分数指数；
- **`β=6.0`**：TAL 分配器的置信度阈值；
- **v8 以来未修的隐性坑**：`topk=10, α=0.5, β=6.0` 显式覆盖 tal.py L29 类默认 `topk=13, α=1.0`——**七版没改过**。

## 5. 六任务专用配方（`trick-6-task-presets`）

> **v1~v14 读者路标**：v14 只有检测任务，v26 首发 6 个任务方向——**每个任务有独立 yaml 和 loss 类**。

### 5.1 Detection（yolo26.yaml）

```python
model = YOLO("yolo26n.pt")  # 官方权重
results = model.train(
    data="coco.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **COCO mAP**：n 40.9 / s 48.6 / m 53.1 / l 55.0 / x 57.5
- **E2E mAP**：n 40.1 / s 47.8 / m 52.5 / l 54.4 / x 56.9
- **速度**：n 38.9ms CPU / 1.7ms T4 TensorRT10

### 5.2 Segmentation（yolo26-seg.yaml）

```python
model = YOLO("yolo26n-seg.pt")
results = model.train(
    data="coco-seg.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **COCO box mAP**：n 39.6 / x 56.5
- **COCO mask mAP**：n 33.9 / x 47.0
- **Proto26 掩膜头**：继承 v10 Proto

### 5.3 Pose（yolo26-pose.yaml）

```python
model = YOLO("yolo26n-pose.pt")
results = model.train(
    data="coco-pose.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **COCO mAP**：n 57.2 / x 71.6
- **RLE + RealNVP Flow**：不确定性估计
- **`kpt_shape=[17, 3]`**：17 关键点 × 3 值（x, y, visibility）

### 5.4 Depth（yolo26-depth.yaml）

```python
model = YOLO("yolo26n-depth.pt")
results = model.train(
    data="nyudepthv2.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **NYU Depth V2 Δ1**：n 0.882 / x 0.933
- **SILog + 4-尺度梯度 L1**：尺度不变深度估计
- **`silog_lambda=1.0`**：NYU 用尺度不变

### 5.5 Semantic Segmentation（yolo26-segsemantic.yaml）

```python
model = YOLO("yolo26n-semantic.pt")
results = model.train(
    data="cityscapes.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **Cityscapes mIoU**：n 78.3 / x 83.6
- **CE + Cityscapes 权重**：19 类长尾加权
- **`ignore_index=255`**：Cityscapes 无效标签

### 5.6 OBB（yolo26-obb.yaml）

```python
model = YOLO("yolo26n-obb.pt")
results = model.train(
    data="dota-v1.yaml",
    epochs=300,
    imgsz=640,
    batch=16
)
```

- **DOTAv1 mAP**：n 52.4 / x 56.7
- **OBB26 头**：10 值回归（4×2 顶点 + rotation）

### 5.7 Classification（yolo26-cls.yaml）

```python
model = YOLO("yolo26n-cls.pt")
results = model.train(
    data="imagenet.yaml",
    epochs=100,
    imgsz=224,
    batch=32
)
```

- **ImageNet top1**：n 71.4 / x 79.9
- **Classify 头**：全局池化 + Linear
- **imgsz=224**：ImageNet 标准分辨率

## 6. Fast Mode = 推理时只跑 one2one（`trick-fast-mode`）

> **v1~v14 读者路标**：v14 用 `YOLOV14_FAST=1` 环境变量控制 Fast Mode，v26 **没有环境变量**——**推理策略写在 `Detect.__init__` 里**。

**Detect.__init__ 的 E2E 切换**（<SrcLink file="v26:nn/modules/head.py" :line="38" :end="154" />）：

```python
def __init__(self, ch=(256, 512, 1024), nc=80, end2end=True):
    self.end2end = end2end
    self.cv2 = nn.ModuleList(...)  # one2many
    self.cv3 = nn.ModuleList(...)  # one2many
    if end2end:
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
```

**推理时自动切换**（<SrcLink file="v26:nn/modules/head.py" :line="200" :end="250" />）：

```python
def forward(self, x):
    if self.training:
        # 训练：返回两分支
        o2m = self.forward_one2many(x)
        o2o = self.forward_one2one(x)
        return o2m, o2o
    else:
        # 推理：只跑 one2one
        return self.forward_one2one(x)
```

**要点**：
- **推理时参数量减半**：one2many 分支被剪掉（从 ~10M 到 ~5M）；
- **推理时无 NMS**：topk=1 TAL 保证每 anchor 一个正样本，不需要后处理 NMS；
- **无环境变量开关**：`Detect.__init__` 里 `if self.end2end:` 分支控制；
- **与 v14 对比**：v14 的 Fast Mode 是「跳过 scene classifier / DomainAdaptiveLayer / ViewEmbedding」，v26 的 Fast Mode 是「跳过 one2many 分支」。

## 7. 官方权重清单（`trick-official-checkpoints`）

> **v1~v14 读者路标**：v14 官方权重尚未发布（README 说 IN PROGRESS），v26 **30+ checkpoint 全部可用**——`yolo26n.pt` 直接下载。

**官方权重清单**：

| 模型 | 大小 | 任务 | COCO mAP |
|------|------|------|----------|
| yolo26n.pt | 5.4 MB | Detection | 40.9 |
| yolo26s.pt | 21.9 MB | Detection | 48.6 |
| yolo26m.pt | 46.8 MB | Detection | 53.1 |
| yolo26l.pt | 55.8 MB | Detection | 55.0 |
| yolo26x.pt | 122.9 MB | Detection | 57.5 |
| yolo26n-seg.pt | 6.2 MB | Segmentation | 39.6/33.9 |
| yolo26n-pose.pt | 6.5 MB | Pose | 57.2 |
| yolo26n-depth.pt | 6.4 MB | Depth | Δ1=0.882 |
| yolo26n-semantic.pt | 6.3 MB | Semantic | mIoU=78.3 |
| yolo26n-obb.pt | 6.4 MB | OBB | 52.4 |
| yolo26n-cls.pt | 5.8 MB | Classification | 71.4 |
| yolo26n-ed.pt | 41.5 MB | Open-Vocab Detect | — |
| yolo26n-ed-seg.pt | 42.3 MB | Open-Vocab Seg | — |
| yolo26n-tvp.pt | 41.9 MB | TVP | — |

**下载方式**：

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")  # 自动下载
```

## 8. 与 v14 训练技巧对比

| 维度 | v14 | **v26** |
|------|-----|---------|
| 官方配方 | epochs=300, imgsz=640（极简） | **epochs=300, imgsz=640, batch=16**（标准） |
| yaml 变体数 | 6（1 全量 + 5 场景） | **11**（1 检测 + 6 任务 + 3 ED + 1 TVP） |
| 档位 scales | 5（n/s/m/l/x） | **5**（n/s/m/l/x） |
| 官方权重 | **未发布** | **30+ checkpoint 全部可用** |
| 训练策略 | 4 条退火 schedule | **1 条退火**（E2E decay） |
| 任务方向 | 1（仅检测） | **7**（Detect/Seg/Pose/Depth/Semantic/OBB/CLS） |
| Fast Mode | `YOLOV14_FAST=1` 环境变量 | **推理时自动切换**（无环境变量） |
| batch 字典开关 | 有（4 键） | **无**（每任务有专用 loss 类） |
| 三分量增益 | 7.5/0.5/1.5 | **完全相同** |
| TAL 分配器 | topk=10 α=0.5 β=6.0 | **完全相同** |
| reg_max | 1（DFL → Identity） | **1**（DFL → Identity） |
| 许可证 | AGPL-3.0 | **AGPL-3.0** |

## 9. 论文-代码偏差（训练侧）

v26 训练侧 README 与代码偏差 **3 处**：

1. **E2E 权重退火未描述**：README 只说 `end2end: True`，没说 `E2ELoss.decay` 的 0.8→0.1 曲线——**读者实测时以代码为准**；
2. **`E2EDetectLoss` vs `E2ELoss` 两套并存**：README 只提 `E2ELoss`，没说还有等权版 `E2EDetectLoss`——**推荐用 `E2ELoss`（退火版）**；
3. **`flow_model` 兜底未描述**：README 没提 `_rle_zero` 兜底（`pred_kpt[..., :0].sum() + sum(p[..., :0].sum() for p in self.flow_model.parameters())`）——**这是为了让 BaseTrainer 不崩溃的 workaround**。

## 想快速对上位读者的话

- **v8 读者**：三分量、TAL、增益、配方与 v8 一字不差，直接套 v8 心智模型；再补 `E2ELoss.decay` 权重退火即可；
- **v11/v12 读者**：`reg_max=1` 让 DFL 退化为 Identity，box 回归从 64 bins 缩到 4 值——**比 v11/v12 更快更简单**；
- **v13 读者**：v13 训练侧零改动（与 v8 完全相同），v26 在其上包了一层 `E2ELoss`——**三分量走 super()、E2E 走权重退火**；
- **v14 读者**：v14 用「多域鲁棒辅助损失」做训练策略，v26 用「E2E 权重退火 + 任务专用损失类」——**v26 的训练策略更简单、更通用**；
- **想关掉所有 E2E 特性**：把 `end2end: True` 改为 `end2end: False`——v26 训练**退化到 v8**，方便对照实验。

## 参考

- [v26 架构总览](/versions/v26/architecture)
- [v26 概念详解](/versions/v26/concepts)
- [v26 损失函数](/versions/v26/losses)
- [v26 反向传播](/versions/v26/loss-gradient)
- [v14 训练技巧](/versions/v14/tricks)
- [v13 训练技巧](/versions/v13/tricks)
- [v8 训练技巧](/versions/v8/tricks)
- <SrcLink file="v26:utils/loss.py" :line="1263" :end="1315" />（E2EDetectLoss / E2ELoss）
- <SrcLink file="v26:nn/modules/head.py" :line="38" :end="154" />（Detect E2E 双分支）
- Ultralytics YOLO26 官方文档：https://docs.ultralytics.com/models/yolo26
