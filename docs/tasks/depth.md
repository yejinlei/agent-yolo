# 单目深度估计（Depth） · YOLO 第 8 任务

> **一句话**：YOLO26 起官方把 **Monocular Depth Estimation（MDE）** 纳入多任务统一 backbone，与 detect/segment/pose/obb/rotated/classify/pose 并列——**一张 RGB 图，输出一个像素级深度图**，让"距离"变成可推理的量。

## 目录

- [何时用 Depth](#何时用-depth)
- [5 档模型](#5-档模型)
- [vs Depth Anything V2](#vs-depth-anything-v2)
- [两种深度头：log-depth vs bounded sigmoid](#两种深度头log-depth-vs-bounded-sigmoid)
- [calibrate() · 闭式解 log-affine](#calibrate--闭式解-log-affine)
- [一键脚本与训练](#一键脚本与训练)
- [常见坑](#常见坑)
- [反哺信号](#反哺信号)
- [参考](#参考)

---

## 何时用 Depth

**必用 Depth 的场景**：
- **无人配送 / AGV**：单相机方案下的**障碍物距离估计**（不用双目/LiDAR）
- **无人机航测**：地面高程 + 相对高度
- **机器人抓取**：末端机械臂的近距离深度感知
- **VR / AR 平面重建**：需要真实尺度
- **视频 3D 增强**：深度作为视频扩散模型的条件输入

**不必用 Depth 的场景**：
- **只需要框**：Detect 就够
- **有 LiDAR / 双目**：直接测距比 MDE 稳得多
- **绝对精度 > 20% 需求**：MDE 本质是"感知距离"，不是"测量距离"

**参考**：[案例 · 3D BEV](/cases/3d-bev) · [案例 · 无人机 OBB](/cases/drone-obb)

---

## 5 档模型

Ultralytics 官方提供 **yolo26{n,s,m,l,x}-depth.pt**，参数量从 6.3M → 57M，**共享 detect 的 backbone**，深度头挂在 top features 上。

| 版本 | 参数 | 输入分辨率 | KITTI δ1 | 延迟 (Jetson Orin) | 推荐场景 |
|------|------|----------|---------|------------------|---------|
| **yolo26n-depth** | 6.3M | 256×192 | 0.91 | **~13ms** | 边缘 / 手机端 |
| **yolo26s-depth** | 23.5M | 256×192 | 0.94 | ~22ms | **默认甜点** |
| yolo26m-depth | 32M | 512×384 | 0.95 | ~45ms | 服务器 |
| yolo26l-depth | 46M | 512×384 | 0.96 | ~78ms | 高精度场景 |
| yolo26x-depth | 57M | 1024×768 | 0.965 | ~140ms | 离线研究 |

> 官方页：<https://docs.ultralytics.com/tasks/depth>

**默认选 yolo26s-depth**：精度差 0.5pt 换来 2× 延迟，多数场景不值。

---

## vs Depth Anything V2

MDE 领域最强的对手是 **Depth Anything V2（DNOV2）**，Ultralytics 官方做了正面对比：

| 模型 | 参数量 | KITTI δ1 | 延迟 (Orin) | 相对加速 |
|------|-------|---------|-----------|---------|
| Depth Anything V2 Small | 24.8M | 0.88 | 100ms | 1.0× |
| Depth Anything V2 Base | 97.5M | 0.90 | 210ms | 1.0× |
| **yolo26n-depth** | **6.3M** | **0.91** | **13ms** | **7.7×** |
| **yolo26s-depth** | **23.5M** | **0.94** | **22ms** | **4.5×** |

关键结论：
- **精度反超**：yolo26n-depth (6.3M) δ1=0.91 > DNOV2 Small (24.8M) δ1=0.88——**小一档还更准**
- **速度 7.7×**：yolo26n 在边缘设备上才是可实时
- **代价**：DNOV2 依赖大规模自监督预训练，泛化更稳；yolo26-depth 依赖 KITTI 等监督数据微调，跨域稍弱

一句话：**要精度+泛化选 DNOV2，要速度+可部署选 yolo26-depth**。

---

## 两种深度头：log-depth vs bounded sigmoid

这是 MDE 领域**最容易踩的坑**，也是 U-Net / DNOV2 论文里反复讨论的建模差异。

### log-depth（对数深度）

输出 `log(D)`，无界，反推 `D = exp(out)`。

```python
# Ultralytics 默认头
out = model(x)          # out ∈ [-∞, +∞]
depth = torch.exp(out)  # depth ∈ (0, +∞)
```

- ✅ **优势**：动态范围大，1m → 100km 都能表示
- ❌ **劣势**：无界优化不稳，KITTI 80m 数据上容易发散

### bounded sigmoid（有界）

输出 `sigmoid(out) * max_depth`，强制在 `[0, max_depth]`。

```python
out = model(x)
depth = torch.sigmoid(out) * 80.0   # 80m KITTI
```

- ✅ **优势**：优化稳，KITTI δ1=**0.08**（有界时）
- ❌ **劣势**：**bounded head 在 80m 尺度上崩溃**——sigmoid 饱和后梯度消失，>40m 区域全部预测错误

### 生产决策

| 场景 | 推荐头 | 理由 |
|-----|-------|-----|
| KITTI 城市驾驶（0-80m） | log-depth + `calibrate()` | 有界头在 80m 崩 |
| 无人机低空（0-100m） | bounded `max_depth=120` | 有界优化更稳 |
| 室内 AGV（0-20m） | bounded `max_depth=25` | 尺度可控 |
| 通用推理 | log-depth | 无需预知场景深度上限 |

---

## calibrate() · 闭式解 log-affine

`calibrate()` 是 Ultralytics 提供的**零训练标定**接口——用一小批带 GT 的样本，闭式解出 log-affine 变换 `log(D) = a · out + b`，把 MDE 从"感知距离"拉回"测量距离"。

```python
from ultralytics import YOLO

model = YOLO("yolo26s-depth.pt")
model.calibrate(
    calib_path="./kitti_depth_calib/",  # 目录：.png + .gt_depth.npy
    max_depth=80.0,
)
# 标定后 model 保存 calib.yaml 一起，加载即生效
```

**为什么闭式解有效**：
- MDE 输出的 `out` 在 log 空间近似**线性偏离** GT log-depth
- 闭式解 `a,b = argsort(min ||a*out + b - log(D_gt)||²)` 是一次最小二乘
- 30 张样本就能收敛，无需重训

**常见收益**：KITTI δ1 从 0.94 → **0.97**（几乎对齐 Ground Truth）。

---

## 一键脚本与训练

### 推理

```bash
yolo predict \
  model=yolo26s-depth.pt \
  source=street.jpg \
  task=depth
```

### Python API

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26s-depth.pt")
results = model.predict("street.jpg", task="depth")

depth_map = results[0].depth   # [H, W]，米
plt.imshow(depth_map, cmap="magma")
plt.colorbar(label="Depth (m)")
plt.show()
```

### 训练

```bash
yolo train \
  model=yolo26s-depth.pt \
  task=depth \
  data=kitti_depth.yaml \
  epochs=30 \
  imgsz=256
```

数据 yaml 结构：

```yaml
path: /data/kitti_depth
train: train.txt    # 每行：<img_path> <depth_path>
val: val.txt
```

### 导出 ONNX / TensorRT

```bash
yolo export \
  model=yolo26s-depth.pt \
  format=onnx \
  dynamic=True \
  opset=17
```

---

## 常见坑

1. **max_depth 忘设 → 深度尺度漂**：KITTI 上 max_depth=80m，换成 Cityscapes 要 120m，不更新会让所有远处目标变近
2. **log-depth 忘 `exp()`**：直接拿 out 当深度图，数值范围在 [-2, 3] 左右，看起来"错得离谱"
3. **图像尺寸不匹配训练**：训练 256×192 推理 640×480 → 深度图分辨率不对，插值后边缘错位
4. **calibrate() 样本量 < 10**：闭式解噪声大，参数 a 波动 ±0.1
5. **视频流场景忘 warmup**：第一帧深度是黑图，需要 3-5 帧稳定
6. **深度 vs 距离**：深度是**相机到像素的欧氏距离**，不是**地面上的水平距离**——平地俯视 30° 时差 15%
7. **反射面深度崩**：玻璃、金属反光面深度全错，需要结合 semantic mask 屏蔽
8. **和 detect 混用时的顺序**：先跑 detect 拿到 object 中心再算"到目标的距离"比整帧深度可靠

---

## 反哺信号

本页内容触发以下**跨案例反哺**：

- **PainPoint 新增**：`pain-bounded-head-long-range`（有界头在 >40m 崩） → 挂到 [YAML PainPoints](/ontology/entity-types#painpoint)
- **Trick 新增**：`trick-log-affine-calibration`（`calibrate()` 闭式解） → [Tricks](/ontology/entity-types#trick)
- **Metric 强化**：现有 `metric-abs-rel-err` 与 `metric-δ1` 已在 [Metrics](/ontology/entity-types#metric)，本页新增 `metric-δ2`（阈值 2 时的正确率）与 `metric-inv-err`（1/d 空间误差，KITTI 惯例）
- **模型关联**：yolo26-depth 归入 [versions/v26/index](/versions/v26/index)，标为多任务之一
- **案例回链**：[3D BEV 案例](/cases/3d-bev) 可考虑"6 目环视 + MDE"作为 LiDAR 稀疏时的补充

---

## 参考

- Ultralytics Depth 官方页：<https://docs.ultralytics.com/tasks/depth>
- Depth Anything V2 (Yang et al., 2024)：<https://arxiv.org/abs/2406.09414>
- Monocular Depth Estimation survey (Li et al., 2022)：<https://arxiv.org/abs/2209.12239>
- KITTI Depth Dataset：<http://www.cvlibs.net/datasets/kitti/eval_depth.php>
- KITTI 评测协议（δ1 / Abs Rel / inv_err）：<http://www.cvlibs.net/datasets/kitti/eval_depth_result.php>
- Site 关联：[v26 概览](/versions/v26/index) · [3D BEV 案例](/cases/3d-bev) · [横向对比](/cases/compare)

**最后更新：2026-10-25**
