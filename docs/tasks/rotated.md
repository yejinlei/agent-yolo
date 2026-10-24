# 旋转检测（Rotated Detection） · YOLO 任务全景

> **一句话**：旋转检测是 OBB 的"进阶版"——**同时支持任意角度 + 任意形状**，用 5 或 6 个参数描述目标（含角度 + 可选的 4 点多边形拟合），特别适合**密集 + 旋转 + 微小目标**的极端场景。

## 目录

- [Rotated vs OBB](#rotated-vs-obb)
- [YOLO 家族支持情况](#yolo-家族支持情况)
- [旋转检测的两种实现](#旋转检测的两种实现)
- [典型场景](#典型场景)
- [训练配置](#训练配置)
- [调优要点](#调优要点)
- [常见坑](#常见坑)

---

## Rotated vs OBB

"Rotated" 这个概念在不同语境下有不同含义，容易混淆：

| 术语 | 含义 | YOLO 语境下的实现 |
| --- | --- | --- |
| **OBB** | 定向框，4 参数 + 角度（5 参数） | YOLOv8-OBB 官方实现 |
| **Rotated** | 广义"旋转检测"，含 OBB + 更复杂形状 | 通常是 OBB 的别名，或含 Shapelet 输出 |
| **RotBox** | 旋转框库（RotateBoxLib） | Ultralytics 依赖的旋转 IoU/NMS 库 |

**结论**：**90% 情况下 "Rotated Detection" = OBB**。剩下 10% 是特定研究项目（如 DOTA 数据集的 Shapelet 扩展），工业界不常用。

**如果你不确定要不要用 Rotated，先看 [OBB 任务](/tasks/obb)**——OBB 覆盖绝大多数需求。

---

## YOLO 家族支持情况

| 版本 | Rotated 支持 | 备注 |
| --- | --- | --- |
| v1-v4 | ❌ | 无支持 |
| v5-Rotated | ✅（社区版） | [xuebinqin/YOLOv5-Rotated](https://github.com/xuebinqin/YOLOv5-Rotated) |
| **v8-OBB/Rotated** | ✅ | Ultralytics 官方 |
| v11-Rotated | ✅ | 实验性 |
| v26 | ✅ | 内嵌多任务统一 backbone |
| YOLO-World | ❌ | 只做 HBB 开放词表 |
| YOLOE | ❌ | 只做 HBB 分割 |
| YOLO-Master | ✅ | 多任务统一 |

**推荐**：**v8-OBB** 已足够，无需再单独找 "Rotated" 版本。

---

## 旋转检测的两种实现

### 实现 A：OBB 角度回归（主流）

**核心思路**：在 Detect 头后加角度回归分支，输出 `[x, y, w, h, θ]` 5 参数。

**代表**：YOLOv5-OBB、YOLOv8-OBB

**优点**：
- 结构简单，工程友好
- IoU 计算快（RotateBoxLib 已优化）
- 部署成熟

**缺点**：
- 只能表达矩形目标
- 角度回归和位置/尺寸回归相互耦合

### 实现 B：Shapelet 多形状（研究向）

**核心思路**：输出关键点（shapelet points），每个关键点有角度权重，通过凸包拟合任意形状。

**代表**：S2ANet（研究项目）、DOTA Shapelet 扩展

**优点**：
- 表达力强，可拟合非矩形（船体、不规则建筑）
- 训练目标灵活

**缺点**：
- 训练复杂，收敛慢
- 部署困难
- 生态几乎为零

**结论**：**除非有特殊形状需求，否则用实现 A（OBB）**。

---

## 典型场景

**Rotated / OBB 的极端场景**：

### 1. 密集微小目标

**场景**：航拍港口车辆（<30px），目标密集重叠。

**痛点**：
- 目标 <20px 时，HBB 完全无法区分
- 密集目标之间角度接近，容易混淆

**方案**：
- YOLOv8-OBB @ 1280+ 输入
- 加 P2 head（stride=4）
- 关闭所有几何增强

### 2. 极远目标 + 极近目标共存

**场景**：航拍机场，近景飞机（200×50px）+ 远景飞机（20×5px）。

**痛点**：
- 远景目标 20px 被 P5 head（stride=32）漏检
- 近景目标大，占据画面 1/3

**方案**：
- 多尺度输入（640/1280/2560 分别处理）
- 或超分辨 + OBB 后融合

### 3. 360° 全角度

**场景**：军事侦察坦克，任意朝向。

**痛点**：
- 角度回归困难（0° 和 360° 是同一个角度）
- 角度损失函数需要特殊处理（sin/cos 参数化）

**方案**：
- YOLOv8-OBB 默认 sin/cos 参数化，可用
- 或改用 GIoU-D（距离 IoU）作为主损失

---

## 训练配置

### 数据集

- **DOTA**：最权威的遥感旋转目标数据集，15 类目标
- **DOTA-v1.0 / v1.5**：不同年份版本，v1.5 是最新
- **HRSC2016**：船舶检测，128k 目标
- **DIOU-RotBox**：研究项目专用
- **自定义**：无人机/卫星采集

### YAML 示例

```yaml
# rotated_data.yaml — 与 OBB 格式相同
path: /data/dota
train: images/train
val: images/val
nc: 15
names:
  - plane
  - ship
  - car
  - ...

# 关键点：4 点多边形顺序一致
```

### 训练配置

```yaml
# 与 OBB 配置基本一致
imgsz: 1280
epochs: 200
batch: 16
box: 7.5
cls: 0.5
dfl: 1.5

augment:
  - random_rotation: 0
  - perspective: 0.0
  - affine: 0.0
  - flipud: 0.0
  - fliplr: 0.0
  - mosaic: 0.2
  - mixup: 0.0
```

---

## 调优要点

### 1. 角度的 sin/cos 参数化

**YOLOv8-OBB 使用 sin/cos 参数化角度**：
- `sin(θ) ∈ [-1, 1]`
- `cos(θ) ∈ [-1, 1]`
- 输出后 `θ = atan2(sin, cos)`

**优点**：
- 天然解决角度环绕问题（0° 和 360° 相同）
- 梯度连续（L1 损失在 ±π/2 处会不连续）

**注意**：
- 如果训练数据角度分布不均匀，考虑角度平衡采样
- 推理时确保 `atan2` 输出范围与 GT 一致

### 2. IoU 损失的选择

**旋转 IoU 有几种变体**：

| IoU 类型 | 说明 | 速度 | 精度 |
| --- | --- | --- | --- |
| **GIoU-Rot** | 基础版，最常用 | 快 | 中 |
| **DIoU-Rot** | 距离感知 | 稍慢 | 高 |
| **CIoU-Rot** | 含长宽比 | 稍慢 | 高 |
| **WIoU-Rot** | 权重分配 | 中 | 高 |

**YOLOv8-OBB 默认用 DIoU-Rot**——精度和速度的平衡。

### 3. NMS 的旋转版本

**普通 HBB NMS 不支持 OBB**，需要旋转 NMS。

**实现**：
- Ultralytics 内建（推荐）
- RotateBoxLib（第三方）
- MMDet（mmdetection 的 OBB 支持）

---

## 常见坑

### ❌ "我用 YOLOv8 做旋转检测"

**不对**。v8 默认是 Detect（HBB），要做旋转检测必须用 `YOLOv8OBB` 类：

```python
from ultralytics import YOLOv8OBB  # 正确
# 或者
model = YOLO('yolov8s-obb.pt')    # 正确
# 不要
model = YOLO('yolov8s.pt')        # 这是 Detect 版本
```

### ❌ "OBB 用 Detect 的 NMS"

**不对**。OBB 必须用**旋转 NMS**，否则 IoU 计算错误。

**修复**：
- 使用 Ultralytics 默认推理（已内建旋转 NMS）
- 或者用 RotateBoxLib

### ❌ "旋转目标训练用随机旋转增强"

**大坑**（同 OBB 的坑）——随机旋转会污染角度信号。

**修复**：所有几何增强关闭：
```yaml
augment:
  - random_rotation: 0
  - perspective: 0.0
  - affine: 0.0
  - flipud: 0.0
  - fliplr: 0.0
```

### ❌ "旋转检测用 CIoU"

**CIoU 是 HBB 的 IoU**，旋转场景要用 **CIoU-Rot** 或 **DIoU-Rot**。

**修复**：Ultralytics 已内建，无需手动改。

---

## 更多资源

- [OBB 任务完整指南](/tasks/obb) — 覆盖 90% 需求
- [案例 · 无人机 OBB](/cases/drone-obb) — 生产级案例
- [案例 · 3D BEV](/cases/3d-bev) — 3D 旋转目标
- [DOTA 数据集](https://github.com/WXIN-LEE/DOTA) — 遥感旋转数据集
- [RotateBoxLib](https://github.com/leeseungjin/rotbox) — 旋转框库
