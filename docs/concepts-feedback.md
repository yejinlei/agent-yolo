# 反哺信号汇总 · 全库扫描结果

> 自动生成，请勿手改。运行 `node scripts/sync-feedback.js` 重新生成。
> 最后更新：2026-10-25
> 扫描范围：`docs/**/*.md` · 共扫描 **341** 个 md 文件，检出 **42** 条信号（分布在 **14** 个文件）

## 分类速览

| 类型 | 数量 |
|-----|-----|
| PainPoint | 14 |
| Trick | 19 |
| Metric | 9 |
| **合计** | **42** |

## PainPoint（14 条）

| 名称 | 说明 | 来源 |
|------|------|------|
|`pain-3d-nms-overslow`| 3D NMS 计算昂贵 | `[cases/3d-bev.md](/cases/3d-bev)` |
|`pain-bounded-head-long-range`| 有界头在 >40m 崩 | `[tasks/depth.md](/tasks/depth)` |
|`pain-camera-lidar-time-skew`| 多传感器时钟偏差 | `[cases/3d-bev.md](/cases/3d-bev)` |
|`pain-class-imbalance-indefect`| PCB 缺陷天然不均衡，通用 cls 损失失效 | `[cases/industrial-defect.md](/cases/industrial-defect)` |
|`pain-cnn-vs-transformer-deployment`| N/S 与 M/L 双族架构是折中 | `[models/v27-preview.md](/models/v27-preview)`<br>`[versions/deltas.md](/versions/deltas)` |
|`pain-confidence-miscalibration`| 医疗置信度必须校准 | `[cases/medical-lesion.md](/cases/medical-lesion)` |
|`pain-data-leakage-medical`| 医疗标注一致性陷阱 | `[cases/medical-lesion.md](/cases/medical-lesion)` |
|`pain-hbb-angle-blindness`| 水平框对任意角度目标无能为力 | `[cases/drone-obb.md](/cases/drone-obb)` |
|`pain-light-invariance-outdoor`| 户外光照剧烈变化难对抗 | `[cases/agriculture.md](/cases/agriculture)` |
|`pain-mask-edge-fragmentation`| FP16 下分割 mask 边缘锯齿 | `[cases/autonomous-lane.md](/cases/autonomous-lane)` |
|`pain-multi-camera-id-switch`| 多摄像头 ID 切换率难以接受 | `[cases/security-monitoring.md](/cases/security-monitoring)` |
|`pain-sofmax-huge-num-classes`| 千级类别 softmax 训练崩塌 | `[cases/retail-shelf.md](/cases/retail-shelf)` |
|`pain-temporal-inconsistency`| 相邻帧预测抖动 | `[cases/autonomous-lane.md](/cases/autonomous-lane)` |
|`pain-wasm-sharedarraybuffer-cors`| 浏览器需要 COOP/COEP header 才能开 SharedArrayBuffer | `[interactive/compare.md](/interactive/compare)` |

## Trick（19 条）

| 名称 | 说明 | 来源 |
|------|------|------|
|`trick-browser-onnx-warmup`| 首次推理会因 JIT 慢，预热一次即可稳定 | `[interactive/compare.md](/interactive/compare)` |
|`trick-browser-ort-wasm-threads`| `numThreads=4` + `proxy=true` 是稳定组合，2 线程以上需 SAB | `[interactive/compare.md](/interactive/compare)` |
|`trick-edge-cloud-candidate-only`| 边缘粗筛 + 云精定位，省带宽 90% | `[cases/agriculture.md](/cases/agriculture)` |
|`trick-event-cooldown-per-zone`| 事件冷却按 zone 分区，不按全局 | `[cases/security-monitoring.md](/cases/security-monitoring)` |
|`trick-federated-doctor-loop`| 联邦 + 医师闭环合规方案 | `[cases/medical-lesion.md](/cases/medical-lesion)` |
|`trick-foreground-alignment`| Query-based 与 one-to-one 的精度补偿 | `[models/v27-preview.md](/models/v27-preview)` |
|`trick-log-affine-calibration`| `calibrate()` 闭式解 | `[tasks/depth.md](/tasks/depth)` |
|`trick-lora-prompt-incremental`| LoRA + Prompt 支持新类免重训 | `[cases/retail-shelf.md](/cases/retail-shelf)` |
|`trick-mask-non-overlap-aug`| 增强时保证 mask 不交叠 | `[cases/industrial-defect.md](/cases/industrial-defect)` |
|`trick-multispectral-band-fusion`| 多光谱融合捕捉早期病害 | `[cases/agriculture.md](/cases/agriculture)` |
|`trick-obb-no-rotation-aug`| OBB 训练禁用旋转增强 | `[cases/drone-obb.md](/cases/drone-obb)` |
|`trick-open-drive-fusion`| 高精地图融合过滤和稳定 | `[cases/3d-bev.md](/cases/3d-bev)` |
|`trick-optic-flow-consistency-loss`| 光流一致性正则项 | `[cases/autonomous-lane.md](/cases/autonomous-lane)` |
|`trick-orthomosaic-dem-correction`| 正射必须做 DEM 校正 | `[cases/drone-obb.md](/cases/drone-obb)` |
|`trick-roi-padding-expansion`| ROI 外扩保留边缘特征 | `[cases/retail-shelf.md](/cases/retail-shelf)` |
|`trick-slowfast-candidate-only`| 只对候选事件跑行为识别，省 20× 算力 | `[cases/security-monitoring.md](/cases/security-monitoring)` |
|`trick-temporal-buffe-on-gpu`| BEV 时序 buffer 存 GPU 加速 | `[cases/3d-bev.md](/cases/3d-bev)` |
|`trick-tversky-loss-recall-first`| α<β Tversky 惩罚漏检 | `[cases/medical-lesion.md](/cases/medical-lesion)` |
|`trick-version-delta-card`| 版本差异卡片：每代 3 行的极简形式，避免读者陷入版本页细节 | `[versions/deltas.md](/versions/deltas)` |

## Metric（9 条）

| 名称 | 说明 | 来源 |
|------|------|------|
|`metric-angle-error-deg`| 角度误差（度），OBB 必备指标 | `[cases/drone-obb.md](/cases/drone-obb)` |
|`metric-bev-coverage-m2`| BEV 有效覆盖面积 | `[cases/3d-bev.md](/cases/3d-bev)` |
|`metric-browser-inference-ms`| WASM 单张 640×640 端到端耗时 | `[interactive/compare.md](/interactive/compare)` |
|`metric-daily-farmer-action`| 每日农户可执行动作数（业务指标） | `[cases/agriculture.md](/cases/agriculture)` |
|`metric-false-positive-events-per-hour`| 每小时误报事件数，事件级比帧级更贴近运营 | `[cases/security-monitoring.md](/cases/security-monitoring)` |
|`metric-false-positive-rate`| 工业质检场景 FP 比漏检更致命 | `[cases/industrial-defect.md](/cases/industrial-defect)` |
|`metric-jitter-cm`| 跨帧位置抖动，单位 cm | `[cases/autonomous-lane.md](/cases/autonomous-lane)` |
|`metric-physician-review-time`| 医师复核时间（秒） | `[cases/medical-lesion.md](/cases/medical-lesion)` |
|`metric-time-to-online`| 新类上线时间（分钟） | `[cases/retail-shelf.md](/cases/retail-shelf)` |

---

## 使用方式

- 想查某个 trick 在哪些案例用过 → 直接搜索本页
- 想追踪某个痛点的演进 → 先看 [cases/compare.md](/cases/compare) 的横向对比
- 想新增信号 → 在案例页 / 任务页 / 配方页末尾的「反哺信号」章节加一行，再运行 `node scripts/sync-feedback.js`

