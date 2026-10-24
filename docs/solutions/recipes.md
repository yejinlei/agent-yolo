# Ultralytics Solutions · 即用配方

Ultralytics 官方提供的 **17 个预置应用**（Solutions），定位是"**检测/分割/姿态已训好，只需画个 ROI 就能拿到业务结果**"。它们与本站的 [实战案例](/cases) 是**互补**关系：

- **cases/（深度案例）**：完整数据 → 训练 → 部署链路，含踩坑与反哺信号
- **solutions/（本页配方）**：不训模型，直接用官方权重，把常见业务逻辑封装成 CLI 命令

适用场景：**POC 验证 · 演示 Demo · 内部评估 · 快速出结果**，不适合长期生产（缺少事件去重、时序判定、隐私合规等工程细节）。

## 官方 17 个 Solution 全景

| 类别 | Solution | CLI | 依赖模型 |
|------|---------|-----|---------|
| 计数 | Object Counting | `yolo solutions count` | detect / seg |
| 计数 | Queue Management | `yolo solutions queue` | detect |
| 计数 | Workout Monitoring | `yolo solutions workout` | pose |
| 计数 | Analytics | `yolo solutions analytics` | detect + track |
| 计数 | Count-in-Zone | `yolo solutions trackzone` | detect + track |
| 安防 | Security Alarm | — | detect（+track） |
| 安防 | VisionEye | `yolo solutions visioneye` | detect + track |
| 场景 | Parking Management | — | detect + track |
| 场景 | Speed Estimation | `yolo solutions speed` | detect + track |
| 场景 | Heatmap | `yolo solutions heatmap` | detect / track |
| 视觉 | Object Crop | `yolo solutions crop` | seg |
| 视觉 | Image Blur | `yolo solutions blur` | seg |
| 视觉 | Object Search | `yolo solutions isearch` | seg |
| 视觉 | Object Segmentation | `yolo solutions isegment` | seg |
| 检索 | Similarity Search | `yolo solutions similarity-search` | CLIP + FAISS |
| 通用 | Inference | `yolo solutions inference` | detect |
| 通用 | Vision | `yolo solutions vision` | detect |

> 官方 URL：<https://docs.ultralytics.com/solutions>

本页挑 8 个高频配方讲透：**Object Counting · Security Alarm · Parking · Workout · Speed · Queue · Heatmap · Similarity Search**。

## 通用前置

```bash
pip install ultralytics opencv-python
```

所有 solution 都基于 `YOLO(...)` 实例，共享以下属性（结果对象）：

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")   # 或 yolo26s-seg.pt / yolo26s-pose.pt

results = model.track(source="video.mp4", stream=True)
# results[0].boxes, .masks, .keypoints, .names, .boxes.id  ← Solution 全部在这层读
```

关键点：**Solutions 不是独立任务**，而是"读 results → 加 ROI/计数/告警/裁剪"的后处理。所以选任务模型时看的是**检测目标形状**，不是 Solution 本身。

---

## 1 · Object Counting（人员/车辆/物体计数）

**场景**：停车场入口数车、商超数人流、工厂数良品/次品。ROI 内累计计数 + 可视化。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `show_count=True` | 输出累计计数 | 必填 |
| `show_fps=True` | 显示推理 FPS | 可选 |
| `line_color=[192,57,43]` | 计数线颜色 | 默认 |
| `line_thickness=3` | 线宽 | 默认 |
| `save=False` | 是否保存结果图 | False |

### 一键脚本

```bash
yolo solutions count \
  model=yolo26n.pt \
  source=parking_entry.mp4 \
  classes=2   # 2 = car in COCO
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounting

model = YOLO("yolo26n.pt")
obj_counter = ObjectCounting(model=model, show_count=True)

for result in model.track("parking_entry.mp4", stream=True, classes=[2]):
    obj_counter.start_process(
        source="parking_entry.mp4",
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        line=[(0, 320), (640, 320)],   # 左右两点定义计数线
        classes=[2],
    )
```

### 常见坑

1. **计数线方向影响 in/out 判定**：从左到右 vs 从右到左，`in_count` / `out_count` 会互换——先看 3 帧再决定线端点顺序
2. **跨框目标计数偏 1**：目标跨线瞬间可能被计 2 次或漏计，用 `track_ids` 而非 `boxes` 判定时会稳
3. **多路并行**：Solutions 内部 `start_process` 是单线程，多路要开多进程

关联案例：[安防监控](/cases/security-monitoring) · [零售货架](/cases/retail-shelf)

---

## 2 · Security Alarm（安防告警）

**场景**：周界入侵 · 无人区闯入 · 关键区域停留过久。核心是"**N 次检测同一区域 → 触发告警**"，天然去抖动。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `line` | 计数线（同上） | 必填 |
| `class_id` | 触发类别 | 0 = person |
| `email_to` | 告警收件人 | 可选，需 smtp 配置 |
| `mail_server` / `mail_password` | SMTP 账号 | 必填 |
| `min_continuous` | 连续多少帧算触发 | 5（默认建议） |

### 一键脚本

```bash
# 官方无独立 CLI，走 SecurityAlarm 类
python security_alarm.py --model yolo26n.pt --source camera.mp4
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import SecurityAlarm

model = YOLO("yolo26n.pt")
alarm = SecurityAlarm(
    model=model,
    line=[(0, 480), (1280, 480)],
    class_id=[0],   # person
    email_to=["ops@example.com"],
    mail_server="smtp.example.com",
    mail_password="***",
    min_continuous=5,
)

for i, result in enumerate(model.track("camera.mp4", stream=True)):
    if alarm.start_process(
        source="camera.mp4",
        frame_idx=i,
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        count_type="in",   # 只统计进入
    ):
        alarm.process_alert()   # 触发邮件 + 截图
        print(f"⚠ frame={i} 触发告警")
```

### 常见坑

1. **min_continuous=1 会疯狂告警**：单个阴影、飞鸟都触发邮件——生产上至少 5
2. **SMTP 未配置时静默失败**：`process_alert` 返回 False 但不抛异常，要主动检查
3. **跨视角重复告警**：同一目标经过 2 个摄像头会告警 2 次，需在业务层去重（track_id + 时间窗）
4. **邮件模板写死在库内**：想让中文邮件要 override `process_alert`

关联案例：[安防监控](/cases/security-monitoring)

---

## 3 · Parking Management（车位管理）

**场景**：停车场车位占用/空闲可视化，支持多车位独立计数、占用率统计。核心机制：**每个车位一个 polygon，独立计算目标中心是否落在多边形内**。

### 关键参数

| 参数 | 说明 |
|-----|-----|
| `car_positions` | list of polygon（每车位一个） |
| `class_ids` | 目标类别（默认 car + truck + bus） |
| `save=True` | 保存每帧可视化 |

### JSON 配置坐标

```json
{
  "camera": "parking_cam_01.mp4",
  "positions": [
    [[80, 400], [220, 400], [220, 500], [80, 500]],
    [[240, 400], [380, 400], [380, 500], [240, 500]],
    [[400, 400], [540, 400], [540, 500], [400, 500]]
  ],
  "class_ids": [2]
}
```

### 一键脚本

```python
import json
from ultralytics import YOLO
from ultralytics.solutions import ParkingManagement

with open("parking.json") as f:
    cfg = json.load(f)

model = YOLO("yolo26n.pt")
parking = ParkingManagement(show=False)

for result in model.track(cfg["camera"], stream=True, classes=cfg["class_ids"]):
    parking.start_process(
        source=cfg["camera"],
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        car_positions=cfg["positions"],
        class_ids=cfg["class_ids"],
    )
```

### 常见坑

1. **坐标必须按视频分辨率标定**：1080P 摄像头画到 720p 上会全部错位
2. **多边形必须凸多边形**：斜停车位的 L 形会被判错
3. **车辆部分进入车位时中心点可能还在外面**：判定用中心点，非 IoU——需要"半占"状态时得自己改逻辑
4. **车位相邻时目标可能同时命中两个**：用最近的 polygon 而非"是否命中"

关联案例：[安防监控](/cases/security-monitoring)

---

## 4 · Workout Monitoring（健身计数）

**场景**：俯卧撑 / 深蹲 / 引体向上计数，基于**两个关键点的夹角**判定动作完成。适合健身房 / 健身 App / 教学视频分析。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `key_points` | 定义动作的 3 个关键点索引 | 俯卧撑 `[15,13,11]`；深蹲 `[11,13,15]` 或 `[13,11,9]` |
| `up_angle` | 动作"上"阈值（度） | 150 |
| `down_angle` | 动作"下"阈值（度） | 90 |
| `show_angle=True` | 显示夹角 | 可选 |

关键点索引参考 COCO 17 keypoint：`0=nose, 5=左肩, 6=右肩, 11=左髋, 12=右髋, 13=左膝, 14=右膝, 15=左踝, 16=右踝`。

### 一键脚本

```bash
yolo solutions workout \
  model=yolo26s-pose.pt \
  source=workout.mp4 \
  key_points="[6,5,11]" \
  up_angle=150 \
  down_angle=90
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import WorkoutMonitoring

model = YOLO("yolo26s-pose.pt")
workout = WorkoutMonitoring(show_angle=True, view_img=True)

for result in model.predict("workout.mp4", stream=True):
    workout.start_process(
        im0=result[0],
        source="workout.mp4",
        kpts=result[0].keypoints,
        key_points=[11, 13, 15],   # 髋-膝-踝 → 深蹲
        up_angle=150,
        down_angle=90,
    )
```

### 常见坑

1. **关键点顺序决定动作方向**：`[肩,肩,髋]` 和 `[髋,肩,肩]` 算出的夹角完全相反
2. **遮挡后角度归零**：手挡住膝盖会让 `down_angle` 判为"未完成"，需要置信度阈值 < 0.3 时跳过
3. **视角必须水平**：俯视 60° 时 90° 阈值会失效，需要重标定
4. **单人假设**：多人同时做动作，Solutions 内部只取置信度最高的一个人

关联案例：[MOT 深度对比](/cases/mot)（姿态 + 跟踪组合）

---

## 5 · Speed Estimation（速度估算）

**场景**：高速公路超速检测 · 工厂传送带速度监控 · 无人机测速。核心公式：**速度 = 位移（像素）× 像素到米的换算 × 帧率**。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `max_hist` | 保留多少帧历史 | 5 |
| `meter_per_pixel` | 像素到米换算系数 | 0.05（0.05 m/pix） |
| `show_hist=True` | 显示轨迹 | 可选 |

### 一键脚本

```bash
yolo solutions speed \
  model=yolo26n.pt \
  source=highway.mp4 \
  classes=2 \
  max_hist=5 \
  meter_per_pixel=0.05
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import SpeedEstimation

model = YOLO("yolo26n.pt")
speed_est = SpeedEstimation(max_hist=5, show_hist=True)

for result in model.track("highway.mp4", stream=True, classes=[2]):
    speed_est.start_process(
        source="highway.mp4",
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        meter_per_pixel=0.05,
        classes=[2],
    )
```

### 常见坑

1. **meter_per_pixel 是常量**：俯视/斜视镜头下不同深度的像素尺寸不同，误差 30%+ 常见
2. **max_hist 太小 → 速度抖**：3 帧历史会让速度曲线高频震荡，5 是甜蜜点
3. **fps 影响**：默认假设 30 fps，15 fps 视频要传 `fps=15` 参数
4. **静态目标也出速度**：噪声导致的框抖动会算出假速度，`conf=0.5` + 位移阈值可缓解

关联案例：[无人机 OBB](/cases/drone-obb) · [安防监控](/cases/security-monitoring)

---

## 6 · Queue Management（排队长度）

**场景**：银行/机场/政务大厅排队长度监控，超过阈值触发排队告警。基于**多边形 ROI 内目标数** 判定排队。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `line` | 排队检测多边形 | list of points |
| `class_ids` | 目标类别 | [0] |
| `alert_threshold` | 排队人数阈值 | 10 |

### 一键脚本

```bash
yolo solutions queue \
  model=yolo26n.pt \
  source=queue.mp4 \
  line="[[0,640],[640,640],[640,480],[0,480]]" \
  class_ids=0 \
  alert_threshold=10
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import Queue

model = YOLO("yolo26n.pt")
queue = Queue(show_count=True)

line = [[0, 640], [640, 640], [640, 480], [0, 480]]

for result in model.track("queue.mp4", stream=True, classes=[0]):
    queue.start_process(
        source="queue.mp4",
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        line=line,
        classes=[0],
        alert_threshold=10,
    )
```

### 常见坑

1. **多边形顺序必须闭合**：`[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` 首尾不闭合会算错
2. **拥挤时目标重叠检测漏**：YOLO 对密集小目标 recall 会掉到 60%，建议叠加 `BoT-SORT`
3. **排队方向不同阈值不同**：横向 vs 纵向队列单位长度的"人数密度"差 2-3 倍
4. **进出 ROI 边界抖动**：人在多边形边缘会反复进出，加 5 帧平滑

关联案例：[安防监控](/cases/security-monitoring)

---

## 7 · Heatmap（热力图）

**场景**：商场人流热力 · 直播间观众分布 · 无人机地面密度图。将一段时间内所有目标位置加权渲染成红色热区。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `alpha=0.5` | 热力图透明度 | 0.4-0.6 |
| `show_fps=True` | 显示 FPS | 可选 |
| `save=True` | 保存热力图 | False |

### 一键脚本

```bash
yolo solutions heatmap \
  model=yolo26n.pt \
  source=mall.mp4 \
  alpha=0.5 \
  save=True
```

### Python API

```python
from ultralytics import YOLO
from ultralytics.solutions import Heatmap

model = YOLO("yolo26n.pt")
heatmap = Heatmap(alpha=0.5, view_img=True)

for result in model.predict("mall.mp4", stream=True, classes=[0]):
    heatmap.start_process(
        im0=result[0].plot(),
        result_objects=result[0].boxes,
        classes=[0],
    )
```

### 常见坑

1. **热力图不是累积的**：默认每帧独立渲染，累积需自己缓存 box 中心
2. **alpha=1 会盖住原图**：0.4-0.6 是甜蜜点
3. **多目标同框 → 红点堆积**：一个 box 只贡献一个热区点，密集人群下会低估
4. **不区分类别**：默认所有 class 叠加，混合热力图易误读——需要按 class 分别画

关联案例：[智慧农业](/cases/agriculture) · [安防监控](/cases/security-monitoring)

---

## 8 · Similarity Search（相似检索 · CLIP + FAISS）

**场景**：以文搜图 / 以图搜图（"找红色轿车" "找出这个人的所有抓拍"）。**唯一用多模态大模型的 Solution**，其他都是 CV。

### 关键参数

| 参数 | 说明 | 常用值 |
|-----|-----|-------|
| `source_img` | 参考图 | 必填 |
| `search_dir` | 搜索目录 | 必填 |
| `similarity_search_type` | `text_to_image` / `image_to_image` | 默认 text_to_image |
| `query` | 文字查询（仅 text_to_image） | 必填 |
| `clip_model` | CLIP 模型路径 | 默认 openai/clip-vit-base-patch32 |
| `faiss_index` | FAISS 索引（缓存） | 可选，加速二次查询 |

### 一键脚本

```bash
yolo solutions similarity-search \
  source=img.jpg \
  search_dir="./gallery" \
  similarity_search_type=text_to_image \
  query="red sports car"
```

### Python API

```python
from ultralytics.solutions import SimilaritySearch

ss = SimilaritySearch(
    source_img="query.jpg",
    similarity_search_type="text_to_image",
    query="red sports car",
    save=True,
)
ss.start_process(search_dir="./gallery")
```

### 常见坑

1. **CLIP 对细粒度识别弱**："红色轿车"能找出红色车辆，但"丰田卡罗拉 vs 本田思域"分辨不清
2. **中英文混用效果差异大**：CLIP base 主要英文训练，中文 query 建议走 `clip-vit-large`
3. **FAISS 索引要持久化**：不缓存的话每次全量重算 embedding，1 万图要 5-10 分钟
4. **不是 YOLO 类**：SimilaritySearch 完全不依赖 YOLO 权重，独立 CLIP 模型，pip 需额外 `transformers faiss-cpu`

关联案例：[多模态融合](/cases/multimodal) · [零售货架](/cases/retail-shelf)

---

## 8 配方横向对比

| 配方 | 依赖任务 | 是否需 track | 常见输入 | 主要坑 |
|-----|---------|------------|---------|-------|
| Object Counting | detect / seg | 否 | 视频 | 计数线方向 |
| Security Alarm | detect | 否 | 视频/摄像头 | 抖动告警 |
| Parking | detect + track | 是 | 视频 | 坐标标定 |
| Workout | pose | 否 | 视频 | 关键点顺序 |
| Speed | detect + track | 是 | 视频 | 像素-米换算 |
| Queue | detect | 否 | 视频 | 多边形闭合 |
| Heatmap | detect | 否 | 视频 | 不累积 |
| Similarity Search | CLIP + FAISS | — | 图片目录 | 中文弱 |

## 何时该跳出 Solutions 自己写

Solutions 是"够用"封装，遇到下面 3 种情况要切回 [cases/](/cases) 或 [recipes/](/recipes/train) 自建：

1. **需要跨摄像头去重**：Solutions 每路独立跑，全局 track_id 冲突
2. **需要时序行为判定**：跌倒/徘徊/徘徊后离开等——Solutions 只做**当前帧**决策
3. **需要合规与隐私**：脱敏、加密、审计日志、数据保留策略——Solutions 全不管

一句话：**Solutions 是"POC 加速器"，cases 是"生产教科书"，两者互补不打架。**

---

## 参考

- Ultralytics Solutions 官方文档：<https://docs.ultralytics.com/solutions>
- Ultralytics Track 模式：<https://docs.ultralytics.com/modes/track>
- Ultralytics CLI Reference：<https://docs.ultralytics.com/yolov5/tutorials/cli>
- CLIP (Radford et al., 2021)：<https://arxiv.org/abs/2103.00020>
- FAISS (Johnson et al., 2017)：<https://github.com/facebookresearch/faiss>

**最后更新：2026-10-25**
