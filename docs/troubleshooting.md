# 故障排查手册 · 症状 → 原因 → 修复

> **定位**：本站 9 个实战案例里积累的踩坑、教训、真实故障汇总成一页速查表。每条都有案例来源（🟡），可回读原文复核。
>
> **使用方式**：按**症状**（不是"某版 YOLO 报某错"这种技术口径）搜——先找"精度突然掉了"、"边缘部署 NaN"、"跨摄像头 ID 切换"这类业务症状，再看具体修复。
>
> **数据来源**：全部来自 [industrial-defect](/cases/industrial-defect) · [edge-deployment](/cases/edge-deployment) · [autonomous-lane](/cases/autonomous-lane) · [medical-lesion](/cases/medical-lesion) · [retail-shelf](/cases/retail-shelf) · [3d-bev](/cases/3d-bev) · [drone-obb](/cases/drone-obb) · [agriculture](/cases/agriculture) · [security-monitoring](/cases/security-monitoring) 9 个案例。

## 目录

- [A. 训练 & 增强](#a-训练--增强)
- [B. 量化 & 精度](#b-量化--精度)
- [C. 导出 & 部署](#c-导出--部署)
- [D. 后处理 & 推理](#d-后处理--推理)
- [E. 数据 & 分布漂移](#e-数据--分布漂移)
- [F. 多目标跟踪](#f-多目标跟踪)
- [G. 3D / BEV / OBB 专项](#g-3d--bev--obb-专项)
- [H. 医疗合规专项](#h-医疗合规专项)
- [I. 隐私与法规](#i-隐私与法规)
- [J. 灾难性遗忘与持续学习](#j-灾难性遗忘与持续学习)
- [K. 硬件 & 传感器异常](#k-硬件--传感器异常)
- [L. 快速诊断卡](#l-快速诊断卡)

---

## A. 训练 & 增强

### A.1 医疗/生物类任务里开了 `mixup` 或 `copy_paste` 增强

**症状**：训练 loss 曲线漂亮，上线后召回崩 30%+；合成出"半良性半恶性"怪病灶。

**原因**：`mixup` 把两张图像素级融合，产生解剖上不可能存在的中间态；`copy_paste` 如果不控 overlap，会把"焊点上方有划痕=少锡"这种假关联写进网络。

**修复**：
- 医疗、3D、OBB、PCB 俯拍 —— **`mixup: 0.0`**，`copy_paste: 0.0~0.1` 且加 `copy_paste_non_overlap: true`。
- 需要补样本时用 `copy_paste_classes_per_image: 2~3` 限制单次拼接数量，避免 Precision 掉 1–2pt。

**来源**：[medical-lesion](/cases/medical-lesion)、[industrial-defect](/cases/industrial-defect)、[agriculture](/cases/agriculture)

### A.2 相机固定视角却开了随机旋转 / 透视 / 仿射

**症状**：OBB 精度降 3–5pt；3D 模型对外参极其敏感；PCB/俯拍场景学到"翻转后的世界"。

**原因**：几何增强把"角度是相对量"这个信号污染了；OBB 的 `rotate` 让网络学不到相对旋转。

**修复**：
- OBB：**`random_rotation: 0`**，用多尺度 `scale: [0.5, 1.0, 1.5]` 替代；`mosaic: 0.2` 以下；`mixup: 0.0`。
- 3D BEV：**关闭 `rotate`、`perspective`、`affine`**，只允许 `brightness / contrast / gaussian_noise`。
- PCB 工业：**`perspective: 0.0`**（俯拍固定）、**`flipud: 0.0`**（PCB 有极性）、`fliplr: 0.0`（元件有左右）。
- 车道分割：**`fliplr: 0.0`**（道路有左右方向语义）、`perspective: 0.0`、`rotate: 0.0`。

**来源**：[drone-obb](/cases/drone-obb)、[3d-bev](/cases/3d-bev)、[industrial-defect](/cases/industrial-defect)、[autonomous-lane](/cases/autonomous-lane)

### A.3 分辨率不足吃小目标

**症状**：输入 640 吃 1–2px 划痕；`<100px` 小病灶 Recall 从 0.95 掉到 0.42；输入 320 时 `<16×16` 目标 Recall 60%→32%。

**原因**：P5 stride=32 的池化把细目标抹掉；`<16×16` 像素的框 P2/P3 head 也接不到有效激活。

**修复**：
- 工业划痕：`imgsz: 1280` + 关 P5 head（显存 +40%），v8s@640 → v8s@1280 划伤 Recall 41%→68%。
- 医疗 6mm 病灶：**加 P2 head**（stride=4），Recall 62%→89%。
- 边缘设备：v8n 输入 320 崩 → v8s + 384 或多尺度 head。

**代价**：1536→2048 提升 `<0.5` mAP 但推理 +120%，**边际收益下降快**。

**来源**：[industrial-defect](/cases/industrial-defect)、[medical-lesion](/cases/medical-lesion)、[edge-deployment](/cases/edge-deployment)

### A.4 分类头 softmax 前景太少直接崩

**症状**：车道分割 mask 前景 `<3%`，用 BCE 全部预测为 0；mAP 卡 0.42 不动。

**原因**：BCE 在极端不平衡下主导类完全压制少数类。

**修复**：Dice(0.4) + Focal(α=0.75, γ=2) + Tversky 组合，mAP 0.42→0.55。**医疗 Tversky 记得 `α<β`（如 0.3/0.7），惩罚漏检**，Recall +3.8pt。

**来源**：[autonomous-lane](/cases/autonomous-lane)、[medical-lesion](/cases/medical-lesion)

### A.5 类别爆炸（8000 类直接 fine-tune head）

**症状**：前 5 epochs 退化为永远预测最常见类，loss 从 2.1 反弹到 14.7，epoch 40 崩塌。

**原因**：梯度消失 + 长尾分布，softmax 学不到稀有类。

**修复**：**两级架构** —— v8s 粗检 50 品类 → ResNet-50 512×512 分类 8 分支，每支 100–300 类 + 路由层。Top-1 41.2%→86.8%。

**来源**：[retail-shelf](/cases/retail-shelf)

### A.6 同类近似 SKU 用普通 CE

**症状**：3 种薯片口味稳定互误，precision 卡 90% 上不去。

**修复**：center loss(0.3) + triplet(0.1)，或加层级关系让网络先学区分层。

**来源**：[retail-shelf](/cases/retail-shelf)

### A.7 冻结 backbone 全程

**症状**：多卡 DDP 训练早期 loss spike；时序不一致导致梯度异常。

**修复**：**前 12 epoch 冻结 backbone**，之后解冻。3D/BEV 尤其需要。

**来源**：[3d-bev](/cases/3d-bev)

### A.8 patience 太宽松

**症状**：训练跑到 60 epoch，mAP 只在 0.80→0.85 之间小幅抖动。

**修复**：`patience: 15` 起早停，几乎无损；多训 40h 换 5pt 不值。

**来源**：[agriculture](/cases/agriculture)

---

## B. 量化 & 精度

### B.1 v8 DFL 层 FP16 下 NaN

**症状**：TensorRT FP16 推理偶发 NaN，或置信度全部变 0。

**原因**：DFL 里的 `F.softmax` 加权平均在 FP16 下 exp 溢出（`exp(x)` 在 x>11 就溢出）。

**修复**：`mark_output_precision(dfl_layer, FLOAT32)` —— **v6 / v8 DFL 层强制 FP32**，其它层 FP16 或 INT8 皆可。

**来源**：[edge-deployment](/cases/edge-deployment)

### B.2 INT8 后 NMS IoU 阈值漂移

**症状**：INT8 上线后误报暴涨 3–5 倍，但同一模型 FP32 完全没问题。

**原因**：量化让置信度整体偏移 0.05–0.1，同一 IoU 阈值下的框过滤数完全不同。

**修复**：
- FP32 `IoU 0.45` → INT8 `0.55`
- FP32 `IoU 0.70` → INT8 `0.75`
- **INT8 后必须重新校准 NMS，不能沿用 FP32 的阈值**。

**来源**：[industrial-defect](/cases/industrial-defect)、[edge-deployment](/cases/edge-deployment)

### B.3 INT8 校准集只有"干净良品"

**症状**：QAT 上线后准确率反降 2%；PTQ 掉 4.2%。

**原因**：校准集分布不覆盖缺陷 / 边缘 / 夜间，量化参数偏到错误区间。

**修复**：**校准集必须包含缺陷 / 低光 / 极端样本**。加入 100 张含缺陷校准图后准确率回升；PTQ 92%→88% vs QAT 92%→90.5%（QAT 明显更好）。

**来源**：[industrial-defect](/cases/industrial-defect)、[edge-deployment](/cases/edge-deployment)

### B.4 关键层保护清单

| 层 | 原因 | 保护策略 |
|---|---|---|
| **v6/v8 DFL** | FP16 softmax 溢出 | 强制 FP32 |
| **v14 AAttn** | 注意力权重对低精度敏感 | FP16 起 |
| **v13 HyperACE** | 超图聚合数值范围窄 | FP16 起 |
| **NMS IoU 计算** | 阈值需重校准 | 量化后单独走 FP32 CPU NMS |
| **分割 mask 上采样** | FP16 边缘锯齿 | 强制 FP32（BilinearFP32 plugin） |
| **isotonic 校准** | 医疗合规要求 | 独立于主模型，永远 FP32 |

**来源**：[benchmarks](/benchmarks)、各案例综合

### B.5 PTQ vs QAT 什么时候选哪个

**症状**：不知道哪个精度保持更好，想一步到位。

**经验法则**：
- **PTQ 掉 > 3pt** → 上 QAT（多训 1 epoch 换 2–3pt 精度）。
- **PTQ 掉 < 1pt** → 直接用 PTQ，别为省不存在的精度多花时间。
- **医疗 / 安全相关** → **必须 QAT**（PTQ 精度不可控）。
- **BEV / 3D** → **量化敏感，INT8 精度损失 3%+**，先测 PTQ 再决定。

**定量参考**：PTQ vs QAT 320 mAP50 41.2 vs 42.5。

**来源**：[edge-deployment](/cases/edge-deployment)、[benchmarks](/benchmarks)

---

## C. 导出 & 部署

### C.1 ONNX opset 太旧

**症状**：`yolo export --format onnx` 报 op 不支持或 DFL 导出失败。

**修复**：**v8 需 ONNX opset ≥ 12**，默认导出通常够用；显式指定 `--opset 12` 保险。

**来源**：[edge-deployment](/cases/edge-deployment)

### C.2 TensorRT `--maxBatchSize` 一味加大

**症状**：maxBatch=16 单帧 FPS 从 48→29，显存 512MB→2.1GB。

**修复**：**单路永远 `maxBatchSize=1`**。多路时才考虑加大，且必须实测收益。

**来源**：[edge-deployment](/cases/edge-deployment)

### C.3 TensorRT `--minShapes/optShapes/maxShapes` 与 forward stride 不匹配

**症状**：1280×736 输入下任何 shape mismatch 直接崩。

**修复**：三个 shape 都必须是 stride 倍数（如 32 的倍数），且不越界。

**来源**：[autonomous-lane](/cases/autonomous-lane)

### C.4 `--workspace` 太小 Jetson 自动降级

**症状**：Jetson Orin Nano 上 INT8 建 engine 突然降级成 FP32，速度翻倍变慢。

**修复**：**`--workspace 4GB`** 防止自动降级。

**来源**：[edge-deployment](cases/edge-deployment)

### C.5 CUDA / TensorRT 版本乱配

**症状**：JetPack 6.0 = CUDA 12.1 + TRT 8.6 锁死；混装直接崩。

**修复**：
- **JetPack 6.0** = CUDA 12.1 + TRT 8.6
- **CUDA 12.4** 需 TRT 9.0 / 9.2
- **不要手动升级 CUDA** 或 TRT，装 JetPack 官方包。

**来源**：[edge-deployment](/cases/edge-deployment)

### C.6 TensorRT NMS plugin 有 bug

**症状**：TRT 内置 NMS plugin 输出偶发错位或空。

**修复**：**生产环境独立跑 CPU NMS**（如 Python 或 OpenCV 实现），别用 TRT 内嵌 plugin。

**来源**：[edge-deployment](/cases/edge-deployment)

### C.7 校验精度只用 `torch.no_grad() + eval()`

**症状**：PyTorch 里 mAP 0.55，导出 ONNX 后 0.42，没人发现直到部署。

**修复**：**必须用导出后的 ONNX 与 torch 比对**：`np.allclose(ort_out, torch_out, atol=1e-4)`。差异超阈值说明导出有问题。

**来源**：[edge-deployment](/cases/edge-deployment)

### C.8 MobileNet-V3 替 v8n backbone

**症状**：320 输入下反而慢 18%；医学影像 Top-1 76%，比 v8m 低 5–8pt。

**原因**：深度可分离卷积在小尺寸下 CUDA 核利用率低；小病灶在 MobileNet 上损失严重。

**修复**：**别在小目标 / 医学场景换 MobileNet backbone**，v8n 或 v8m 直接够用。

**来源**：[edge-deployment](/cases/edge-deployment)、[medical-lesion](/cases/medical-lesion)

### C.9 TFLite 后端默认 CPU

**症状**：RPi5 上 v11n INT8 只有 15 FPS，社区跑 34 FPS。

**修复**：开 **XNNPACK delegate** 快 2.3×。

**来源**：[edge-deployment](/cases/edge-deployment)

---

## D. 后处理 & 推理

### D.1 全局 confidence 阈值

**症状**：车类总漏检、人却误报——因为类别置信度分布不同。

**修复**：**per-class threshold** 或 class-conditional DFL。

**来源**：[edge-deployment](/cases/edge-deployment)

### D.2 人群场景 NMS IoU 太小

**症状**：IoU 0.5 把 5 个人合并成 1 个。

**修复**：人群场景 `IoU: 0.65`。

**来源**：[security-monitoring](/cases/security-monitoring)

### D.3 车道用标准 NMS

**症状**：长条状车道跨类 IoU 极低被合并成一段。

**修复**：车道**关闭 NMS 或 `IoU=0.05`**，或换 NMS-free 头。

**来源**：[autonomous-lane](/cases/autonomous-lane)

### D.4 FP16 分割 mask 边缘锯齿

**症状**：车道抖动 FP16 全量 6cm vs FP32+DFL 2cm；softmax 低激活区梯度丢失。

**修复**：**分割 head softmax 强制 FP32**，或改用 sigmoid+threshold；加 BilinearFP32 plugin。

**来源**：[autonomous-lane](/cases/autonomous-lane)

### D.5 SAM prompt 模式选错

**症状**：PCB 金色焊盘 SAM point prompt 失效；小病灶 SAM 训练数据偏大覆盖 3× 实际面积。

**修复**：
- **PCB / 反光表面**：polygon prompt + 前置 CLAHE 直方图均衡化。
- **医疗小病灶**：v8m bbox 作 box prompt，不用 point。
- **补 300 例 `<10mm` fine-tune SAM**，让模型认识小目标。

**来源**：[industrial-defect](/cases/industrial-defect)、[medical-lesion](/cases/medical-lesion)

### D.6 YOLO-World prompt 只用类别词

**症状**：跨 SKU 泛化差，召回低 12%。

**修复**：加视觉锚点，如 `"thin irregular dark line on green PCB surface"` 而非裸 `scratch`。

**来源**：[industrial-defect](/cases/industrial-defect)

---

## E. 数据 & 分布漂移

### E.1 实验室 → 产线光照漂移

**症状**：产线 LED 频闪让同一模型 mAP 掉 8pt。

**修复**：**200 张产线实拍图 fine-tune backbone 前 3 层 5 epochs**（约 3 分钟），冻结 head。

**来源**：[industrial-defect](/cases/industrial-defect)

### E.2 训练集视角单一

**症状**：训练 15° 视角，测试 30° 召回掉 25%。

**修复**：训练时 `perspective: 0.10`（默认 0.0）强制扭曲 + `flipud: 0.5`（促销堆头倒置）。

**来源**：[retail-shelf](/cases/retail-shelf)

### E.3 场景覆盖率 vs 平均精度

**症状**：训练集隧道样本仅 0.8%，隧道内 3D 框 mAP 从 0.82→0.11。

**修复**：**场景重采样 5×** + 补 500 段隧道样本；训练前先做场景分布审计。

**来源**：[3d-bev](/cases/3d-bev)

### E.4 训练集全亚洲人、测试集 30% 欧美

**症状**：色素差异致黑色素瘤漏检 25%。

**修复**：**种族/性别/年龄分布覆盖 min 5%/组**，训练前做多样性审计。

**来源**：[medical-lesion](/cases/medical-lesion)

### E.5 静态光照归一化

**症状**：白平衡不同时间差 >30%。

**修复**：**逐帧动态估计**（每 5 分钟重估）。农业多光谱还需每波段 `percentile clip [2, 98]` 归一化。

**来源**：[agriculture](/cases/agriculture)

### E.6 相机 / 分辨率变更

**症状**：医院 A 换相机 2048→1024，6mm 病灶图上变 3px 单点故障，Recall 从 96%→64%。

**修复**：
- **部署侧分辨率下限硬检**（`<1600` 边长报警）。
- **200 张模拟降分辨率图域适应**。

**来源**：[medical-lesion](/cases/medical-lesion)

### E.7 相机曝光 AE 反应慢

**症状**：亮度突降 90% 时 AE <500ms 反应，算法已崩。

**修复**：**亮度突变 >50% 强制重置 AE**；3D 检测加 `degrade_mode` 降级路径。

**来源**：[3d-bev](/cases/3d-bev)

### E.8 EXIF 未处理

**症状**：手机拍照 Orientation=90/180 全图颠倒，模型全错。

**修复**：**PIL `ImageOps.exif_transpose`** 或 `exif_transpose` 算子在数据管道最前面。

**来源**：[retail-shelf](/cases/retail-shelf)

### E.9 多光谱通道对齐

**症状**：R/G/B/NIR 通道对齐差 1–2px，果实分级误差 5–8%。

**修复**：**逐波段对齐后 percentile clip 归一化**。

**来源**：[agriculture](/cases/agriculture)

---

## F. 多目标跟踪

### F.1 跨摄像头 ID 切换

**症状**：入侵者在 45s 内穿过 3 个摄像头，事件引擎只报 1 起入侵，另外 2 个认为新目标。

**原因**：BoT-SORT ReID embedding 只在同机数据上训，跨摄像头 embedding 空间不一致。

**修复**：**全局 ReID 库**（3000 例跨摄像头样本，10 天标注）+ **MOTIF 融合**。

**来源**：[security-monitoring](/cases/security-monitoring)

### F.2 事件冷却 30s 全局

**症状**：同目标反复入侵被吞掉。

**修复**：**按 polygon 分区冷却**，避免全局冷却压制真事件。

**来源**：[security-monitoring](/cases/security-monitoring)

### F.3 事件引擎按摄像头独立判断

**症状**：摄像头交界处（重叠 `<30%`）漏关联同一入侵。

**修复**：**改全局事件图**，让跨摄像头目标能合并。

**来源**：[security-monitoring](/cases/security-monitoring)

### F.4 SlowFast 训练集偏斜

**症状**：SlowFast 训练集全快速动作，"缓慢倒下"样本缺失导致跌倒漏报。

**修复**：**人工合成慢动作样本**加入训练集。

**来源**：[security-monitoring](/cases/security-monitoring)

### F.5 SlowFast 全量跑成本 20×

**症状**：单帧推理预算爆了。

**修复**：**只对候选事件跑 SlowFast**（~500 事件/日），延迟可控。

**来源**：[security-monitoring](/cases/security-monitoring)

### F.6 跌倒报警无静止确认

**症状**：跌倒动作刚结束就报警，人立即站起误报。

**修复**：**`post_fall_still: 5s`** 才推 P0，跌倒端到端 3.2s 内报警。

**来源**：[security-monitoring](/cases/security-monitoring)

---

## G. 3D / BEV / OBB 专项

### G.1 外参标定误差被忽略

**症状**：外参 0.1° 误差 → 100m 处 17cm 定位误差；相机曝光延迟 200μs → 5m 定位误差。

**修复**：**定期重标**；相机曝光延迟纳入标定流程。

**来源**：[3d-bev](/cases/3d-bev)

### G.2 LiDAR 与相机延迟 >50ms

**症状**：急转弯融合撕裂。

**修复**：**PTP 时钟同步 ±1ms**，超出丢帧绝不撕裂。

**来源**：[autonomous-lane](/cases/autonomous-lane)

### G.3 时序 buffer 被污染

**症状**：BEVFormer 时序 buffer 被隧道入口低光帧污染，检测数从 50 崩到 3。

**修复**：
- **时序一致性 loss**（光流一致性 0.15 权重，零运行时成本）。
- **历史帧质量过滤**：亮度突降帧不进入 buffer。

**来源**：[3d-bev](/cases/3d-bev)、[autonomous-lane](/cases/autonomous-lane)

### G.4 8 帧时序 buffer 引入延迟

**症状**：800ms 历史窗口本身不增延迟，但**历史污染当前**。

**修复**：接入 Kalman 平滑位姿；`degrade_mode` 无 2D 降级路径。

**来源**：[3d-bev](/cases/3d-bev)

### G.5 OBB 角度符号翻转（CVAT ↔ Ultralytics）

**症状**：CVAT CCW 负角导入 Ultralytics CW 正角，4000 张角度全变号，mAP 0.80→0.21。

**修复**：**`angle *= -1`**；DOTA `(h,w)` vs Ultralytics `(w,h)` 必须转换，否则长集装箱识别成短。

**来源**：[drone-obb](/cases/drone-obb)

### G.6 OBB 权重档位选错

**症状**：OBB 用 s 权重，2048 分辨率下 3–6° 偏差 vs x/l 权重 `>15°`。

**修复**：**港口 / 大目标 OBB 用 x 或 l 权重**；`imgsz: 1536`（比 1024 mAP 0.76→0.82）。

**来源**：[drone-obb](/cases/drone-obb)

### G.7 OBB 旋转精度 rpl 默认太低

**症状**：R@95° 0.78 业务不可用。

**修复**：港口需 **`rpl: 3.0`** 提到 0.90；`scale=0.5, shear=5, degrees=30` 提 2–3 mAP。

**来源**：[drone-obb](/cases/drone-obb)

### G.8 正射地形起伏区未做 DEM 校正

**症状**：正射误差从 0.5cm 恶化到 3m。

**修复**：Agisoft **`auto_dem: true`**；RTK 无人机 + DEM 联合校正。

**来源**：[drone-obb](/cases/drone-obb)

### G.9 大图切块 overlap 不足

**症状**：卡车跨 tile 边界被切断。

**修复**：tile 1280 时 `overlap: 96px` 以上。

**来源**：[drone-obb](/cases/drone-obb)

### G.10 3D NMS 慢

**症状**：3D NMS 200 目标 O(n²) 慢到不能上生产。

**修复**：**GPU 加速 + class-aware + 分层 NMS**；OBB 多边形 NMS `iou_threshold: 0.7`。

**来源**：[3d-bev](/cases/3d-bev)、[drone-obb](/cases/drone-obb)

### G.11 多相机白平衡差异

**症状**：融合特征不一致，3D 检测抖动。

**修复**：**颜色一致性预处理**（CLAHE / 白平衡对齐）。

**来源**：[3d-bev](/cases/3d-bev)

---

## H. 医疗合规专项

### H.1 softmax 原始分数直接显示给医师

**症状**：0.9 ≠ 90%；模型置信度未校准导致临床误判。

**修复**：**isotonic regression 校准**（每 1000 例重训，ECE `<0.05`）。Platt scaling 在样本 `<1000` 时不如 isotonic。

**isotonic 校准集**：小病灶占比 `<8%` 会反校准；**必须按病灶大小分层采样每档 500 例**。

**来源**：[medical-lesion](/cases/medical-lesion)

### H.2 Grad-CAM 与 mask 混为一谈

**症状**：边界模糊病灶 heatmap 与 mask 不一致。

**修复**：**mask = 边界**、**heatmap = 决策依据**，两个互补不替代；UI 上分开展示。

**来源**：[medical-lesion](/cases/medical-lesion)

### H.3 FP16 分割亚毫米 GGO 结节

**症状**：细节丢失，边界模糊。

**修复**：**分割 head 强制 FP32** 或用 INT8 QAT 而非 PTQ。

**来源**：[medical-lesion](/cases/medical-lesion)

### H.4 Tversky α/β 反了

**症状**：惩罚漏检权重反了，导致医疗场景漏检率上升。

**修复**：**医疗必须 α<β**（如 0.3/0.7，惩罚 FP 更弱），Recall +3.8pt。

**来源**：[medical-lesion](/cases/medical-lesion)

### H.5 增强方向违反生理语义

**症状**：`flipud: 0.5` 上下翻转肝脏，`hsv_h: 0.5` 色相变导致组织类型混淆。

**修复**：
- **`flipud: 0.0`**
- **`hsv_h: 0.0`**（色相 = 组织类型）
- **`perspective: 0.0`**、**`rotate: 0.0`**

**来源**：[medical-lesion](/cases/medical-lesion)

---

## I. 隐私与法规

### I.1 原图上传云端未脱敏

**症状**：GDPR 罚款最高年营收 4%；摄像头数据进云端前没做处理。

**修复**：
- **推理前 blur 人脸**
- **ReID 只传 sha256 哈希**（不上传 embedding）
- **原图 `retention_days: 0`**（本地不留存）

**来源**：[security-monitoring](/cases/security-monitoring)

---

## J. 灾难性遗忘与持续学习

### J.1 LoRA 微调无 replay

**症状**：老 SKU 准确率掉 2–3%；LoRA 只解参数效率不解遗忘。

**修复**：**200 张老 SKU regularizer 或 replay buffer** + LoRA + EWC 一起用。

**来源**：[retail-shelf](/cases/retail-shelf)

### J.2 双十一 280 新品单次重训

**症状**：新品类 96%→42%，老品掉到 68%（灾难性遗忘）。

**修复**：**LoRA + Replay + EWC**，新品 90%、老品 92%（+24pt）。

**来源**：[retail-shelf](/cases/retail-shelf)

### J.3 LoRA rank 只调大不解遗忘

**症状**：只 rank=64 无 EWC，遗忘问题依旧。

**修复**：**遗忘根因是分类头梯度冲击**，LoRA + Replay + EWC 三件套缺一不可。

**来源**：[retail-shelf](/cases/retail-shelf)

### J.4 周增病例全量 fine-tune

**症状**：老病例下降（灾难性遗忘）。

**修复**：**200 张 replay buffer + LoRA 冻结主干 + EWC penalty=0.1**。

**来源**：[medical-lesion](/cases/medical-lesion)

### J.5 联邦训练按样本量加权

**症状**：小中心权重 `<5%` 长尾分布学不到。

**修复**：**贡献均衡、小中心梯度权重提 2×**。

**来源**：[medical-lesion](/cases/medical-lesion)

### J.6 混流新 SKU 未做规则引擎分层

**症状**：混流 B 型板上线第 3 天报警率暴涨 37×（SAM box-prompt 在密集元件区被吸走 + 规则引擎阈值按 A 型板定死 + YOLO-World 无 B 板 prompt）。

**修复**：
- **ROI 预定位**
- **补 18 条 B 板 prompt**
- **规则引擎阈值按 SKU 分组自适应**

**来源**：[industrial-defect](/cases/industrial-defect)

---

## K. 硬件 & 传感器异常

### K.1 无人机云台快门太快

**症状**：触发快门模糊致正射伪影。

**修复**：锁定 **`1/1000s`** 快门。

**来源**：[drone-obb](/cases/drone-obb)

### K.2 大风天起飞

**症状**：M300 抗风 12m/s vs 港口 8–15m/s 波动，40 航点全失败。

**修复**：**风速 >6m/s 切地面摄像头降级**。

**来源**：[drone-obb](/cases/drone-obb)

### K.3 暴雨 / 水滴误判

**症状**：暴雨后水滴轮廓误判为病斑，占新增报警 62%（+24×）。

**修复**：**HSV 水滴特征过滤**（高饱和 + 均匀亮度）；告警**气象联动**（暴雨 24h `>20mm` 且告警 `>5×baseline` 30d 均值时自动熔断转人工复核）。

**来源**：[agriculture](/cases/agriculture)

### K.4 多光谱相机挂露水

**症状**：NIR 通道噪声 +40%。

**修复**：**起飞前微振动清洁 + 疏水涂层**。

**来源**：[agriculture](/cases/agriculture)

### K.5 太阳能连续阴天

**症状**：电池只撑 3 天，边缘设备离线。

**修复**：**事件缓存按天而非小时**（离线 14 天）。

**来源**：[agriculture](/cases/agriculture)

### K.6 摄像头曝光 200μs 延迟

**症状**：产生 5m 定位误差（3D 场景）。

**修复**：纳入标定流程；PTP 时钟同步。

**来源**：[3d-bev](/cases/3d-bev)

---

## L. 快速诊断卡

按**症状关键词**反查：

| 症状 | 首查 |
|---|---|
| **推理 NaN / 置信度全 0** | [B.1 DFL FP16 NaN](#b1-v8-dfl-层-fp16-下-nan) |
| **INT8 上线误报暴涨** | [B.2 NMS IoU 阈值漂移](#b2-int8-后-nms-iou-阈值漂移) · [B.3 校准集缺缺陷](#b3-int8-校准集只有干净良品) |
| **小目标 Recall 崩** | [A.3 分辨率不足](#a3-分辨率不足吃小目标) · [A.1 mixup 增强](#a1-医疗生物类任务里开了-mixup-或-copy_paste-增强) |
| **导出后精度暴跌** | [C.7 只用 torch 校验](#c7-校验精度只用-torchnograd--evaleval) · [C.1 opset 太旧](#c1-onnx-opset-太旧) |
| **ONNX / TensorRT 崩** | [C.3 shape 不匹配](#c3-tensorrt--minshapesoptshapesmaxshapes-与-forward-stride-不匹配) · [C.4 workspace 太小](#c4---workspace-太小-jetson-自动降级) · [C.5 版本乱配](#c5-cuda--tensorrt-版本乱配) |
| **Jetson 突然变慢** | [C.4 workspace 降级](#c4---workspace-太小-jetson-自动降级) · [C.8 MobileNet 慢](#c8-mobilenet-v3-替-v8n-backbone) |
| **TFLite 手机慢** | [C.9 默认 CPU 后端](#c9-tflite-后端默认-cpu) |
| **ID 切换频繁** | [F.1 跨摄像头 ReID](#f1-跨摄像头-id-切换) |
| **事件漏关联** | [F.2 冷却 30s 全局](#f2-事件冷却-30s-全局) · [F.3 独立判断](#f3-事件引擎按摄像头独立判断) |
| **3D 定位漂移** | [G.1 外参标定](#g1-外参标定误差被忽略) · [G.2 传感器延迟](#g2-lidar-与相机延迟-50ms) |
| **BEV 时序污染** | [G.3 buffer 污染](#g3-时序-buffer-被污染) · [G.4 历史污染当前](#g4-8-帧时序-buffer-引入延迟) |
| **OBB 角度全错** | [G.5 CVAT 符号翻转](#g5-obb-角度符号翻转cvat--ultralytics) |
| **医疗置信度不可信** | [H.1 isotonic 校准](#h1-softmax-原始分数直接显示给医师) |
| **增强后解剖学不合理** | [A.1 mixup/copy_paste](#a1-医疗生物类任务里开了-mixup-或-copy_paste-增强) · [H.5 生理语义](#h5-增强方向违反生理语义) |
| **换相机后崩** | [E.6 分辨率下限](#e6-相机--分辨率变更) |
| **产线 vs 实验室精度差** | [E.1 光照漂移](#e1-实验室--产线光照漂移) |
| **场景覆盖率不足** | [E.3 隧道样本少](#e3-场景覆盖率-vs-平均精度) |
| **SKU 上新遗忘老品** | [J.2 LoRA+Replay+EWC](#j2-双十一-280-新品单次重训) · [J.6 混流未分层](#j6-混流新-sku-未做规则引擎分层) |
| **GDPR 合规** | [I.1 原图脱敏](#i1-原图上传云端未脱敏) |
| **无人机任务失败** | [K.2 大风](#k2-大风天起飞) · [G.8 无 DEM 校正](#g8-正射地形起伏区未做-dem-校正) |
| **多光谱噪声大** | [K.4 挂露水](#k4-多光谱相机挂露水) · [E.9 通道对齐](#e9-多光谱通道对齐) |

---

## 与其它页面联动

| 相关页面 | 用途 |
|---|---|
| [benchmarks](/benchmarks) | 6 层数字基准，本页的"阈值"从那里读 |
| [choose-a-model](/choose-a-model) | 按算力档位选型，本页按症状修复 |
| [compare-matrix](/compare-matrix) | 结构差异，本页给出结构差异带来的实战后果 |
| [data-flow](/data-flow) | 数据流全景，本页按阶段分节 |
| [concepts](/concepts) | 概念反向索引，本页用词与之一致 |
| [cases/compare](/cases/compare) | 9 案例横向对比，本页是它的时间/症状维度切片 |
| [evolution-story](/evolution-story) | 演进故事，本页每节都是演进中的"坑" |

---

## 数据局限与更新机制

- **可信度分级**：本页全部 🟡（案例实测），无 🟢 官方基准。每条都标注了案例来源，可回读原文复核。
- **未覆盖**：v1–v5 时代的老坑（如 v1 的 7×7 网格限制、v2 的 anchor-free 时代问题）已随版本淘汰，不再单独列。
- **更新节奏**：新案例落地后，把"演进故事与失败复盘"和"常见踩坑 Top N"章节的条目**同步摘录到本页**，保持症状口径一致。
- **不重复**：本页只做"症状 → 修复"聚合，具体背景与量化数据回到原案例看。避免同一坑两处维护。
