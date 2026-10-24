# 高级训练技巧 · 让模型多涨 2-5% mAP

> **一句话**：把训练从"能跑"到"能涨点"的方法论——EMA、冻结/解冻、多尺度输入、学习率调度、超参搜索、损失调整，一套组合拳把 v8-s 从 50 拉到 53+。

- ⏱ 预计阅读：30 分钟
- 📌 前置阅读：[train.md](/recipes/train)（先掌握基础流程）
- 🔗 相关：[compare-matrix](/compare-matrix) · [quantization.md](/recipes/quantization) · [data-annotation.md](/recipes/data-annotation)

---

## 目录

- [训练涨点的六大杠杆](#训练涨点的六大杠杆)
- [杠杆 1：冻结/解冻策略](#杠杆-1冻结解冻策略)
- [杠杆 2：学习率调度](#杠杆-2学习率调度)
- [杠杆 3：EMA（Exponential Moving Average）](#杠杆-3emaexponential-moving-average)
- [杠杆 4：多尺度输入与 SAHI](#杠杆-4多尺度输入与-sahi)
- [杠杆 5：损失函数调整](#杠杆-5损失函数调整)
- [杠杆 6：超参搜索](#杠杆-6超参搜索)
- [组合拳：三种经典配方](#组合拳三种经典配方)
- [常见坑](#常见坑)

---

## 训练涨点的六大杠杆

按"性价比"排序（涨点 / 训练成本）：

| 杠杆 | mAP 提升 | 成本 | 推荐优先级 |
|------|---------|------|-----------|
| 冻结/解冻 | +1-3% | 零 | ⭐⭐⭐ |
| 学习率调度 | +0.5-1% | 零 | ⭐⭐⭐ |
| EMA | +0.5-1% | 零 | ⭐⭐⭐ |
| 损失调整 | +1-2% | 中 | ⭐⭐ |
| 多尺度输入 | +0.5-2% | 中 | ⭐⭐ |
| 超参搜索 | +2-5% | 高 | ⭐ |

**顺序建议**：先做零成本的（1-3），再做中等成本的（4-5），最后考虑搜索。

---

## 杠杆 1：冻结/解冻策略

### 三种模式

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

# 模式 1：全冻结，只训 head（少样本推荐）
model.train(
    data="mydata.yaml",
    epochs=50,
    freeze=12,           # 冻结前 12 层（backbone 大部分）
    lr0=1e-3,
    batch=32,
)

# 模式 2：冻结 backbone，训 neck + head（中等样本）
model.train(
    data="mydata.yaml",
    epochs=100,
    freeze=3,           # 只冻结最底层几个
    lr0=1e-4,           # 学习率更低
)

# 模式 3：从头训（数据 >10 万张）
model.train(
    data="mydata.yaml",
    epochs=300,
    pretrained=False,   # 不用 COCO 预训练
    lr0=1e-3,
)
```

### 何时用哪种

| 数据量 | 冻结策略 | 学习率 | Epochs |
|-------|---------|-------|--------|
| <500 张 | freeze=12 | 1e-3 | 50-100 |
| 500-5000 张 | freeze=3 | 1e-4 | 100-200 |
| 5000-50000 张 | freeze=0（默认） | 1e-4 | 200-300 |
| >50000 张 | 从头训 | 1e-3 | 300-500 |

### 分段解冻（进阶）

```python
# Ultralytics 原生支持
model.train(
    data="mydata.yaml",
    epochs=200,
    unfreeze="12",      # 从冻结层开始解冻
)
```

先冻结训 100 epoch，再解冻训 100 epoch。适合数据中等且变化大的场景。

---

## 杠杆 2：学习率调度

Ultralytics 支持三种：

```python
model.train(
    data="mydata.yaml",
    scheduler="cosine",   # 余弦退火，默认
    # scheduler="linear", # 线性衰减
    # scheduler="constant", # 恒定（少样本）
)
```

### 各调度器特点

| 调度器 | 特点 | 适用 |
|-------|------|------|
| **cosine** | 前期快后期慢，收敛最好 | 大部分场景 |
| **linear** | 线性下降，简单可预测 | 中等规模数据 |
| **constant** | 恒定学习率 | 少样本、后期微调 |
| **one_cycle** | 上升-下降曲线 | 特定场景（少用） |

### 自定义学习率曲线（进阶）

```yaml
# hyp.yaml
lr0: 0.01
lrf: 0.01          # 最终学习率 = lr0 * lrf = 0.0001
warmup_epochs: 3.0 # 预热阶段
warmup_momentum: 0.8
warmup_bias_lr: 0.1
```

**关键洞察**：
- 学习率太低 → 收敛慢，永远找不到最优
- 学习率太高 → 震荡，可能不收敛
- **`warmup_epochs` 很重要**：前 3 epoch 学习率从 0 升到 lr0，防止爆炸

### Cosine 调度的隐藏优势

Cosine 相比 Linear 的优势：
- **后期学习率更低**，能精修最优解
- **前期学习率下降更平缓**，模型有充分探索空间
- **对长训练友好**，300+ epochs 依然能提升

**实测**：v8-s 在 COCO 上，Cosine 比 Linear 多 0.3-0.5% mAP。

---

## 杠杆 3：EMA（Exponential Moving Average）

### 原理

EMA 是 PyTorch 中"权重滑动平均"，训练时保存一份权重的历史平均，推理时用 EMA 权重。

```python
model.train(
    data="mydata.yaml",
    ema=True,           # 默认已开启
    ema_beta=0.9999,    # 越大越平滑，越小响应快
)
```

### EMA 的收益

| 训练 Epochs | EMA 相对提升 |
|-------------|-------------|
| <50 | +0.2-0.3% |
| 50-200 | +0.5-1.0% |
| >200 | +1.0-1.5% |

训练越久，EMA 优势越大，因为它能过滤掉"末期震荡"。

### EMA Beta 的选择

- **0.9999**（默认）：训练 >100 epoch 推荐
- **0.999**：训练 <50 epoch 或数据波动大
- **0.99**：极少用，只在极端场景

### Ultralytics 已内置

Ultralytics 从 v8 起默认开启 EMA，无需手动设置。**唯一要确认的**：加载权重时选 `last.pt` 还是 `best.pt`。

```python
# last.pt 是最新 EMA 权重
# best.pt 是训练过程中 mAP 最高的 EMA 权重
# 生产用 best.pt，评估用 last.pt
```

---

## 杠杆 4：多尺度输入与 SAHI

### 训练时多尺度（Ultralytics 默认开启）

```python
model.train(
    data="mydata.yaml",
    imgsz=640,
    rect=True,          # 矩形批处理（按宽高比分组）
    multi_scale=True,   # 每 epoch 随机 ±50% 缩放
)
```

### 推理时多尺度

**问题**：训练用 640，推理时大图小目标漏检。

**方案 1：高分辨率推理**
```python
model.predict(img, imgsz=1280)  # 慢但准
```

**方案 2：SAHI（Slicing Aided Hyper Inference）**
- 把大图切成小块（640×640），每块独立推理
- 结果后处理合成

```bash
pip install sahi
```

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detector = AutoDetectionModel.from_config({
    "type": "YOLOv5",
    "model_path": "yolov8s.pt",
})

sliced_prediction = get_sliced_prediction(
    image="large_scene.jpg",
    detection_model=detector,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

**SAHI 收益**：
- 小目标 mAP +3-8%
- 推理慢 3-5×（因为切块多）
- **只在有明确小目标需求的场景用**

### 不同任务的多尺度策略

| 任务 | 建议 |
|------|------|
| Detect | 训练多尺度，推理固定 640 或 1280 |
| Segment | 分辨率敏感，训练和推理同尺度 |
| Pose | 关键点位置敏感，训练分辨率 ≥640 |
| OBB | 旋转框角度敏感，避免极端尺度变化 |
| Depth | 固定分辨率即可 |

---

## 杠杆 5：损失函数调整

Ultralytics v8 默认损失函数：

| 目标 | 损失 | 权重 |
|------|------|------|
| 分类 | BCEWithLogits | 0.5 |
| 回归 | DFL + CIoU | 7.5 |
| 分割 | BCE (mask) | 1.5 |
| 姿态 | DFL + cls | 7.5 / 0.5 |

### 常见调整

```yaml
# hyp.yaml
box: 7.5       # 默认，回归损失权重
cls: 0.5       # 分类损失权重
dfl: 1.5       # DFL 分布损失权重
```

### 什么时候调

- **误检多**：降低 `cls` 权重（0.3-0.4）
- **漏检多**：提高 `cls` 权重（0.6-0.8）
- **框不紧**：提高 `box` 权重（8-10）
- **回归不稳定**：降低 `dfl` 权重（1.0）

### 特殊场景：小目标损失加权

```yaml
# hyp.yaml 中修改
val_gamma: 1.0  # 小目标惩罚

# 或自定义
nms_rotated: True  # OBB 专用
```

### 类别不平衡的处理

如果类别数量差异大（如 100 张类 A vs 10000 张类 B）：

**方法 1：过采样少数类**
```python
model.train(
    data="mydata.yaml",
    oversample=0.5,     # 少数类过采样比例
)
```

**方法 2：类别权重**
```python
# dataset.yaml
weights:
  0: 3.0    # 类 0 权重 3.0（提升权重，让损失更大）
  1: 1.0
  2: 0.5
```

---

## 杠杆 6：超参搜索

### Ultralytics 内置的遗传算法

```python
from ultralytics.utils import default_cfg

# 加载 COCO 预训练
from ultralytics import YOLO
results = YOLO("yolov8s.pt").train(
    data="mydata.yaml",
    evolve=10,          # 进化代数
    generations=10,     # 迭代次数
    population=50,      # 种群大小
    device=0,
)
```

**成本估算**：10 代 × 50 种群 × 100 epochs ≈ 500 次训练。GPU 时间约 3-5 天。

### 更实用的方案：网格搜索（选几个关键点）

只调最有影响的 5 个参数：

```python
import itertools
from ultralytics import YOLO

configs = {
    "lr0": [1e-3, 3e-4, 1e-4],
    "batch": [32, 64],
    "imgsz": [640, 960],
    "epochs": [100],
    "warmup_epochs": [3],
}

results = []
for combo in itertools.product(*configs.values()):
    model = YOLO("yolov8s.pt")
    r = model.train(
        data="mydata.yaml",
        lr0=combo[0],
        batch=combo[1],
        imgsz=combo[2],
        epochs=combo[3],
        warmup_epochs=combo[4],
    )
    results.append((combo, r.metrics.box.map))

best = max(results, key=lambda x: x[1])
print(f"Best: {best}")
```

### 何时值得调

| 场景 | 值得吗 |
|------|-------|
| 数据 <500 张，简单任务 | 不值得，直接跑默认 |
| 中等数据 + 有明确目标 | 值得，选 3-5 个关键点 |
| 数据充足 + 追求极致 | 值得，全参数搜索 |

---

## 组合拳：三种经典配方

### 配方 A：少样本快速起步（<500 张）

```python
model.train(
    data="mydata.yaml",
    epochs=100,
    freeze=12,
    lr0=1e-3,
    batch=16,
    imgsz=640,
    mosaic=0.0,          # 少样本时关闭 Mosaic
    mixup=0.5,
    augment=True,        # 强增强
)
```

**预期提升**：默认 50% → 55-58% mAP

### 配方 B：标准微调（500-5000 张）

```python
model.train(
    data="mydata.yaml",
    epochs=200,
    freeze=3,
    lr0=3e-4,
    batch=32,
    imgsz=640,
    scheduler="cosine",
    ema=True,
    mixup=0.5,
    mosaic=1.0,
)
```

**预期提升**：默认 50% → 53-56% mAP

### 配方 C：大规模从头（>50000 张）

```python
model.train(
    data="mydata.yaml",
    epochs=300,
    pretrained=False,
    lr0=1e-3,
    batch=64,
    imgsz=640,
    scheduler="cosine",
    ema=True,
    augment=True,
    # 后期关闭增强
    final_augment=False,
)
```

**预期提升**：从零 40% → 55-58% mAP（相当于微调）

### 配方 D：小目标优化（遥感/无人机）

```python
model.train(
    data="mydata.yaml",
    epochs=200,
    imgsz=1280,           # 高分辨率
    rect=True,
    multi_scale=True,
    hsv_v=0.7,
    mosaic=1.0,
    scale=0.5,
    box=7.5,
    cls=0.8,              # 提高分类权重
    lr0=3e-4,
    freeze=3,
)

# 推理时
from sahi import AutoDetectionModel
# 见上节 SAHI 部分
```

**预期提升**：小目标 mAP +3-8%

---

## 常见坑

- **学习率太高 + 冻结太浅**：loss 直接爆炸，NaN。要么降学习率，要么加深冻结。
- **EMA 关闭后精度崩**：忘记 EMA 就是"末期震荡"，mAP 掉 0.5-1.5%。
- **Mosaic 训到最后一刻**：Mosaic 是训练技巧，最后 10-20 epoch 建议关掉。Ultralytics 默认会做，但自定义时要注意。
- **imgsz 训练 640 推理 1280**：训练和推理分辨率不匹配，模型不适应。要么都 1280，要么推理用 SAHI。
- **过拟合到验证集**：mAP 在验证集上升、测试集不动，就是过拟合。加正则、减训练轮数、加数据增强。
- **`freeze=12` 训 50 epoch 就停**：解冻层还没充分学习，应该训 100+ epochs。
- **混合数据源但没打标签**：从不同源合并数据时，标注规范不统一，模型学混乱。
- **调参顺序错**：先调 lr0 再调 batch，容易漏掉最优。按 lr0 → batch → imgsz → 增强顺序调。
- **超参搜索太早做**：先手动试 3-5 组，找到合理范围再搜索。一上来就跑 1000 组合是浪费。
- **忽略 `val_gamma` / `val` 参数**：验证时 NMS 参数与推理不一致，评估虚高。用 `yolo val` 而非自定义。

---

## 下一步

- 学基础训练流程 → [train.md](/recipes/train)
- 学数据准备 → [data-annotation.md](/recipes/data-annotation)
- 学部署 → [edge-deploy.md](/recipes/edge-deploy)
- 学量化 → [quantization.md](/recipes/quantization)
