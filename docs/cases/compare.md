# 12 大案例横向对比

12 个案例跨 8 大 domain，同一套 YOLO 家族在完全不同的场景下表现出截然不同的形态。**这张表是选型的起点**——先找到你所在场景，再看它对应的模型/硬件/踩坑模式。

> **Batch 4 更新**：新增 MOT 深度、视频流检测优化、多模态融合 3 个案例，反哺信号从 27 条扩到 42 条，踩坑模式扩到 11 种。

## 一句话选型速查

| 你的场景 | 起点案例 | YOLO 版本 | 关键 trick |
|---------|---------|-----------|------------|
| 手机/树莓派实时检测 | [edge-deployment](/cases/edge-deployment) | v11n / v8n | INT8 + QAT |
| PCB/面板质检 | [industrial-defect](/cases/industrial-defect) | v13s + SAM | QAT + 规则引擎 |
| 车道线/物体追踪 | [autonomous-lane](/cases/autonomous-lane) | v11x 双头 | 光流一致性 + P2 |
| 货架 3000 SKU | [retail-shelf](/cases/retail-shelf) | v8s + LoRA | 分层路由 + Prompt 库 |
| 皮肤/病灶分割 | [medical-lesion](/cases/medical-lesion) | v8m + SAM | Tversky + 联邦 |
| 安防事件检测 | [security-monitoring](/cases/security-monitoring) | v8s + BoT-SORT | 事件冷却分区 |
| 大田作物病害 | [agriculture](/cases/agriculture) | v8n + SAM | 多光谱 + 光照鲁棒 |
| 港口车辆俯视 | [drone-obb](/cases/drone-obb) | v8-OBB | 禁旋转增强 + DEM |
| 自动驾驶 3D | [3d-bev](/cases/3d-bev) | v11-3D + BEVFormer | 时序 Transformer + 地图 |
| 仓库叉车调度 | [mot](/cases/mot) | v8s + BoT-SORT | 全局 ReID + MOTIF |
| 64 路 4K 视频流 | [video-stream](/cases/video-stream) | v8n + NGF | CUDA stream 并行 + 背压 |
| 商场开放词表 | [multimodal](/cases/multimodal) | YOLO-World + Qwen2-VL | 文本 prompt + 4 张精标注 |

## 版本选择矩阵

```mermaid
flowchart TB
  A[场景] --> B{算力预算}
  B -->|< 3 TOPS| C[v5n/v8n/v11n]
  B -->|3-10 TOPS| D[v8s/v11s/v13s]
  B -->|10-50 TOPS| E[v8m/v11m]
  B -->|> 50 TOPS| F[v8x/v11x]
  C --> G{输入尺寸}
  G -->|< 640| H[v8n + P3]
  G -->|640| I[v8n + P3+P4]
  G -->|> 640| J[v8n + P3+P4+P5]
  D --> K{多任务?}
  K -->|检测| L[v13s 单头]
  K -->|检测+分割| M[v11x 双头]
  K -->|3D 位姿| N[v11-3D + BEVFormer]
  K -->|开放词表| O[YOLO-World]
  K -->|视频跟踪| P[v8s + BoT-SORT]
```

| 案例 | 主模型 | 输入尺寸 | 参数量 | FP16 延迟 | 备注 |
|-----|-------|---------|-------|-----------|------|
| edge-deployment | v11n | 640 | 2.6M | 12ms (Orin) | 极致压缩 |
| industrial-defect | v13s | 1280 | 11.2M | 45ms (Jetson Orin) | 高分辨率 |
| autonomous-lane | v11x 双头 | 1920 | 68.7M | 78ms (Orin AGX) | 6 目融合 |
| retail-shelf | v8s + ResNet-50 | 640 + 224 | 11.2M + 25.6M | 32ms (GPU) | 分层路由 |
| medical-lesion | v8m + SAM | 1024 | 25.9M + 91M | 320ms (A100) | SAM 慢 |
| security-monitoring | v8s + BoT-SORT | 640 | 11.2M | 21ms (A10) | 多路 |
| agriculture | v8n | 640 | 3.6M | 28ms (RPI5) | 太阳能 |
| drone-obb | v8-OBB | 1920 | 11.2M | 1800ms (A100) | 大图 |
| 3d-bev | v11-3D + BEVFormer | 900×6 | 68.7M + 32M | 87ms (Orin AGX) | 时序 |
| **mot** | v8s + BoT-SORT | 640 | 11.2M + 18M ReID | 24ms (A10) | IDF1=0.87 |
| **video-stream** | v8n + NGF | 640 | 3.6M × 64 路 | 720ms p50 (4×4090) | 端到端 |
| **multimodal** | YOLO-World + Qwen2-VL | 640 + text | 11M + 80B (LLM) | 42ms + 1200ms | 两段 |

## 硬件/成本速查

| 案例 | 典型硬件 | 单次推理成本 | 年运维/单路 | 备注 |
|-----|---------|------------|-----------|------|
| edge-deployment | 树莓派 5 | ¥0.0001 | ¥28/月 | 极致边缘 |
| industrial-defect | Jetson Orin | ¥0.0005 | ¥300/月 | 工业精度 |
| autonomous-lane | Orin AGX | ¥0.002 | ¥1200/月 | 车端昂贵 |
| retail-shelf | RTX 3090 | ¥0.0003 | ¥500/月 | 服务器 |
| medical-lesion | A100-40G | ¥0.01 | ¥5000/月 | SAM 慢 |
| security-monitoring | A10 × 40 | ¥0.0002 | ¥28/月每路 | 规模效应 |
| agriculture | RPI5 + 太阳能 | ¥0.0001 | ¥120/亩 | 野外 |
| drone-obb | A100-40G | ¥0.02 | ¥850/次 | 飞行成本 |
| 3d-bev | Orin AGX 64GB | ¥0.002 | ¥8000/月 | 车端 |
| **mot** | A10 + Kafka | ¥0.0003 | ¥35/月每路 | ReID 开销 |
| **video-stream** | 4× RTX 4090 | ¥0.00005/路 | ¥120/路 | 64 路摊销 |
| **multimodal** | A100 + vLLM | ¥0.05 | ¥8000/月 | LLM 贵 |

## 输入尺寸对比

```
640 ──┬── v8n, v11n (边缘/安防/农业)
     │   ├── v8s + BoT-SORT (MOT)
     │   ├── v8n × 64 路 (video-stream)
     │   └── YOLO-World (multimodal)
     │
     ├── v8s (零售/工业低配)
     │
     ├── v8-OBB (drone 局部 tile)
     │
1024 ──┴── v8m (医疗)
     │
     ├── v11-3D (BEV 每相机)
     │
1280 ─── v13s (工业高清)
     │
1920 ─── v11x 双头 (自动驾驶)
     ├── v8-OBB 大图 (drone 全景)
```

**规律**：
- < 640：几乎必然是边缘场景
- 640：8 个案例的甜蜜点（含 MOT / video-stream / multimodal）
- 640-1024：常规场景甜蜜点
- > 1024：只有高精度或大分辨率需求（航拍、车道远距）

## Loss 函数对比

| 案例 | 主 Loss | 特殊项 |
|-----|--------|--------|
| edge-deployment | CIoU + BCE + DFL | - |
| industrial-defect | CIoU + Focal + Tversky | + SAM 一致性 |
| autonomous-lane | CIoU + BCE + Dice | + 光流一致性 (0.15) |
| retail-shelf | Softmax CE | + LoRA 正则 |
| medical-lesion | Tversky (α=0.3, β=0.7) | + EWC 抗遗忘 |
| security-monitoring | CIoU + BCE + DFL | + ReID InfoNCE |
| agriculture | CIoU + BCE + Dice | + SAM 一致性 |
| drone-obb | CIoU + AngleLoss + BCE | + OBB overlap |
| 3d-bev | 3D Center + Size + Rot + Vel | + BEV consistency |
| **mot** | CIoU + BCE + DFL | + ReID InfoNCE (0.1) + MOTIF |
| **video-stream** | CIoU + BCE + DFL | + 抽帧率自适应 |
| **multimodal** | L1 + CIoU + DFL (World) | + CLIP 对比 (0.5) |

## 反哺信号汇总

跨 12 个案例共发现 **42 条反哺信号**，按类别聚合：

### 新增 PainPoints (14 条)

| PainPoint | 首个来源 | 影响案例 |
|----------|---------|---------|
| `pain-class-imbalance-indefect` | industrial | industrial, agriculture |
| `pain-mask-edge-fragmentation` | autonomous | autonomous, 3d-bev |
| `pain-temporal-inconsistency` | autonomous | autonomous, 3d-bev, security |
| `pain-sofmax-huge-num-classes` | retail | retail |
| `pain-data-leakage-medical` | medical | medical |
| `pain-confidence-miscalibration` | medical | medical, autonomous |
| `pain-multi-camera-id-switch` | security | security, 3d-bev |
| `pain-hbb-angle-blindness` | drone-obb | drone-obb, 3d-bev |
| `pain-light-invariance-outdoor` | agriculture | agriculture, security, autonomous |
| `pain-camera-lidar-time-skew` | 3d-bev | 3d-bev, autonomous |
| **`pain-reid-cross-camera-drift`** | mot | mot, security, 3d-bev |
| **`pain-gpu-oom-under-backpressure`** | video-stream | video-stream, security |
| **`pain-llm-latency-budget-blowup`** | multimodal | multimodal, medical |
| **`pain-open-vocab-zero-shot-recall`** | multimodal | multimodal, retail |

### 新增 Tricks (20 条)

| Trick | 首个来源 |
|-------|---------|
| `trick-mask-non-overlap-aug` | industrial |
| `trick-optic-flow-consistency-loss` | autonomous |
| `trick-lora-prompt-incremental` | retail |
| `trick-roi-padding-expansion` | retail |
| `trick-tversky-loss-recall-first` | medical |
| `trick-federated-doctor-loop` | medical |
| `trick-event-cooldown-per-zone` | security |
| `trick-slowfast-candidate-only` | security |
| `trick-multispectral-band-fusion` | agriculture |
| `trick-edge-cloud-candidate-only` | agriculture, drone-obb |
| `trick-obb-no-rotation-aug` | drone-obb |
| `trick-orthomosaic-dem-correction` | drone-obb |
| `trick-temporal-buffe-on-gpu` | 3d-bev |
| `trick-open-drive-fusion` | 3d-bev |
| **`trick-global-reid-embedding-pool`** | mot |
| **`trick-motif-metric-learning`** | mot |
| **`trick-cuda-stream-per-worker`** | video-stream |
| **`trick-adaptive-frame-throttle`** | video-stream |
| **`trick-yoloworld-zero-shot-boot`** | multimodal |
| **`trick-llm-tool-loop-agent`** | multimodal |

### 新增 Metrics (8 条)

| Metric | 首个来源 |
|-------|---------|
| `metric-false-positive-rate` | industrial |
| `metric-jitter-cm` | autonomous |
| `metric-time-to-online` | retail |
| `metric-physician-review-time` | medical |
| `metric-false-positive-events-per-hour` | security |
| `metric-daily-farmer-action` | agriculture |
| `metric-angle-error-deg` | drone-obb, 3d-bev |
| `metric-bev-coverage-m2` | 3d-bev |
| **`metric-idf1`** | mot |
| **`metric-end2end-latency-p50-p99`** | video-stream |
| **`metric-time-to-add-sku`** | multimodal |

## 踩坑模式聚合

跨案例的**共性踩坑**（≥3 案例都踩过）：

| 踩坑模式 | 影响案例 | 通用解 |
|---------|---------|-------|
| **FP16 敏感层退化** | industrial, autonomous, 3d-bev | 数值关键层强制 FP32 |
| **小目标丢失** | industrial, medical, 3d-bev, drone-obb | P2 head (stride=4) |
| **光照突变** | agriculture, security, autonomous, 3d-bev | 动态白平衡 + 归一化 |
| **数据不均衡** | industrial, retail, medical | mixup + copy_paste + 类加权 |
| **灾难性遗忘** | retail, medical | LoRA + Replay + EWC |
| **多传感器同步** | autonomous, 3d-bev | PTP 802.1AS + 时钟重同步 |
| **隐私合规** | medical, security, agriculture | 联邦 + 脱敏 + 分区存储 |
| **量化精度损失** | industrial, edge-deployment, security | QAT 而非 PTQ |
| **ID 切换（跨摄像头/遮挡）** | security, mot, 3d-bev, autonomous | 全局 ReID + ReID 周期性重训 |
| **GPU OOM / 背压失效** | video-stream, security, 3d-bev | 事件引擎背压 + CUDA stream 隔离 |
| **LLM / 大模型延迟爆炸** | multimodal, medical | 分层触发（候选 → LLM 只在必要时） |

## 何时选哪个案例作起点

**决策树**：

```
你的任务
├── 2D 目标检测
│   ├── 静态（图像/截图）
│   │   ├── 边缘硬件 → edge-deployment
│   │   ├── 高分辨率 → industrial-defect
│   │   └── 千级类别 → retail-shelf
│   └── 动态（视频/流）
│       ├── 事件报警 → security-monitoring
│       ├── 大田环境 → agriculture
│       ├── 高路数大规模 → video-stream
│       └── 多机融合 → 3d-bev
├── 2D 分割
│   ├── 时序敏感 → autonomous-lane
│   └── 精度敏感 → medical-lesion
├── 3D 位姿
│   └── 多相机 → 3d-bev
├── 多目标跟踪
│   └── 需要稳定 ID → mot
├── 开放词表 / 新类快速上线
│   └── 无需重训 → multimodal
└── 文本 + 图像混合问答
    └── LLM Agent → multimodal
```

## 12 案例关系图

```mermaid
flowchart LR
  EDGE[edge-deployment] -.通用底座.-> ALL[其他所有]

  IND[工业质检] <--> AGRI[智慧农业]
  IND <--> MED[医疗]

  AUTO[自动驾驶] <--> SEC[安防]
  AUTO <--> BEV[3D BEV]

  RETAIL[零售] <--> SEC

  AGRI <--> DRONE[无人机 OBB]

  SEC <--> MOT[多目标跟踪]
  MOT <--> VSTREAM[视频流]
  SEC <--> VSTREAM

  RETAIL <--> MM[多模态]
  MED <--> MM
  MM -.文本-视觉对齐.-> RETAIL

  classDef core fill:#fbb,stroke:#888
  class EDGE core
  classDef new fill:#e0f0ff,stroke:#06c,stroke-width:2px
  class MOT,VSTREAM,MM new
```

**阅读顺序建议**：
- **入门**：先看 [edge-deployment](/cases/edge-deployment) 理解基础，再看一个具体场景
- **进阶**：找最接近你场景的案例，看它的 YAML 和踩坑
- **专家**：看 [反哺信号汇总](#反哺信号汇总) 和 [踩坑模式聚合](#踩坑模式聚合)，学跨案例的通用模式

## 12 案例踩坑速查（按踩坑模式索引）

反过来看——**已经遇到某个坑了**，直接搜下面这张表找到同类案例的解法：

| 你遇到的坑 | 首先看 | 次看 |
|-----------|-------|------|
| 跨摄像头 ID 切换、ID 频繁跳变 | [mot](/cases/mot) | [security-monitoring](/cases/security-monitoring) |
| 视频流 OOM / worker 被 kill | [video-stream](/cases/video-stream) | [security-monitoring](/cases/security-monitoring) |
| 新 SKU / 新类别上线周期太长 | [multimodal](/cases/multimodal) | [retail-shelf](/cases/retail-shelf) |
| 小目标漏检、精度上不去 | [industrial-defect](/cases/industrial-defect) | [medical-lesion](/cases/medical-lesion) |
| 边缘设备跑不动、延迟太高 | [edge-deployment](/cases/edge-deployment) | [agriculture](/cases/agriculture) |
| 隐私合规 / 数据出域 | [medical-lesion](/cases/medical-lesion) | [security-monitoring](/cases/security-monitoring) |
| 多相机/多传感器时间不同步 | [3d-bev](/cases/3d-bev) | [autonomous-lane](/cases/autonomous-lane) |
| 光照变化导致精度掉 | [agriculture](/cases/agriculture) | [drone-obb](/cases/drone-obb) |
| LLM 响应太慢 | [multimodal](/cases/multimodal) | - |
| 64 路以上流拉不动 | [video-stream](/cases/video-stream) | [security-monitoring](/cases/security-monitoring) |

## 案例横向指标矩阵

12 个案例的**核心 KPI** 横向对齐，方便一眼看哪个案例离你的目标最近：

| 案例 | 核心 KPI | 目标值 | 实测 | 达标 |
|-----|---------|-------|-----|------|
| edge-deployment | FPS @ Orin | ≥ 30 | 82 FPS | ✅ |
| industrial-defect | 缺陷召回率 | ≥ 98% | 99.2% | ✅ |
| autonomous-lane | 定位误差 (cm) | < 5 | 3.2 | ✅ |
| retail-shelf | SKU 识别准确率 | ≥ 99.5% | 99.7% | ✅ |
| medical-lesion | Dice (病灶分割) | ≥ 0.85 | 0.89 | ✅ |
| security-monitoring | 事件误报/小时 | < 5 | 3.2 | ✅ |
| agriculture | 病害 F1 | ≥ 0.90 | 0.92 | ✅ |
| drone-obb | 角度误差 (°) | < 3 | 1.8 | ✅ |
| 3d-bev | BEV 覆盖 (m²) | ≥ 1600 | 2050 | ✅ |
| **mot** | IDF1 | ≥ 0.85 | 0.87 | ✅ |
| **video-stream** | 端到端延迟 p99 | < 1.2s | 1.05s | ✅ |
| **multimodal** | 新 SKU 上线时间 | < 3 天 | 2.5 天 | ✅ |

## 未覆盖的高价值场景

以下场景值得作为 v3 案例补充：

| 场景 | 潜在价值 | 复杂度 |
|-----|---------|-------|
| 视频姿态估计 (Pose) | 健身/康复/工业巡检 | ⭐⭐⭐ |
| 关键点检测 (Keypoint) | 生物计数/机器人手眼 | ⭐⭐⭐ |
| 全景/立体视觉 | AR 导航/室内定位 | ⭐⭐⭐⭐ |
| 遥感地表变化检测 | 城市/环境/灾害 | ⭐⭐⭐⭐ |
| 视频异常检测 (Anomaly) | 通用安防 | ⭐⭐⭐⭐⭐ |
| 医学影像 3D CT/MRI | 器官/肿瘤分割 | ⭐⭐⭐⭐⭐ |
| 工业 AGV 调度 | 工厂物流 + MOT + 路径 | ⭐⭐⭐⭐ |
| 边缘多模态 VLM 推理 | 手机侧 LLM 融合 | ⭐⭐⭐⭐⭐ |

## 下一步

- **场景迁移模板**：给一个"从 A 案例迁移到 B 案例"的操作手册
- **踩坑案例库**：把每个案例的失败复盘抽成独立页
- **量化对比**：把 12 案例的定量结果做成可交互筛选表格
