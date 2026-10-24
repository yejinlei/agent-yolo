# 概念反向索引

按概念→出处的反向索引。**术语表回答"是什么"，本页回答"哪几个版本用到、哪个案例踩过坑、去哪一页看详解"**。

> **这一页存在的意义**：读者问得最多的是"这个 trick 是哪一代有的？"、"哪些案例遇到过类似坑？"——这两个问题术语表和版本页都不直接回答。**本页把 100+ 概念按 8 大主题索引到版本、案例、教程三处**。

**配合阅读**：症状速查 → [故障排查](/troubleshooting) · 数字量级 → [基准数据](/benchmarks) · 结构差异 → [对比矩阵](/compare-matrix) · 链路视角 → [数据流](/data-flow) · 按场景选型 → [选型决策树](/choose-a-model)

---

## 使用方式

每条概念按下面格式索引：

```
- **概念名**（首次出现版本 / 关键作用一句话）
  - 出现版本：v5, v8, v11, v26
  - 相关案例：edge-deployment, industrial-defect
  - 相关踩坑：INT8 下 X 问题
  - 详解：[versions/vX/concepts](/versions/vX/concepts) 或 [glossary](/glossary#xxx)
```

---

## 1. 检测基础（Detection Basics）

### single-shot

- 一次前向推理出所有框，无需候选区域提议
- 出现版本：v1 ~ v26 全系
- 相关案例：全部 9 个
- 详解：[glossary](/glossary#_1-single-shot)

### bbox（边界框）

- 中心坐标 + 宽高的矩形表示
- 出现版本：v1 ~ v26 全系
- 相关案例：全部 9 个
- 详解：[v1 concepts](/versions/v1/concepts)

### IoU（交并比）

- 两个框的重叠比例，检测精度核心度量
- 出现版本：v1 ~ v26 全系
- 相关案例：全部 9 个
- 详解：[v1 concepts](/versions/v1/concepts)

### NMS（非极大值抑制）

- 推理后处理：同一物体的重复框只留最高分
- 出现版本：v1 ~ v13 全系，v10 / v26 首次实现 NMS-free
- 相关案例：edge-deployment（INT8 下 IoU 阈值需重调）
- 详解：[v10 concepts](/versions/v10/concepts)

### anchor（锚框）

- 预设的典型框模板，网络预测相对偏移
- 出现版本：v2 ~ v5（v6 起 anchor-free）
- 相关案例：retail-shelf（多 SKU 类别时 anchor 聚类重）
- 详解：[v2 concepts](/versions/v2/concepts) / [v6 concepts](/versions/v6/concepts)

### anchor-free

- 网格中心直接回归 ltrb，无锚框模板
- 出现版本：v6 首次，v7 回退，v8+ 主流
- 相关案例：全部 v8+ 案例
- 详解：[v8 concepts](/versions/v8/concepts)

### mAP（均精度均值）

- 检测总分，每个类 AP 的均值
- 出现版本：全系
- 相关案例：全部 9 个（每个案例都报 mAP50 / mAP50-95）
- 详解：[glossary](/glossary#map-mean-average-precision)

### 置信度 / objectness

- "这个框里有物体"的概率
- 出现版本：v1 ~ v9（v10 起 objectness 分支移除）
- 详解：[v1 concepts](/versions/v1/concepts)

---

## 2. 骨干网络（Backbone）

### Darknet-19 / 53

- v2 / v3 的骨干命名（层数）
- 出现版本：v2 / v3 / v4
- 详解：[v3 architecture](/versions/v3/architecture)

### ResNet / 残差连接

- `y = F(x) + x`，梯度抄近路
- 出现版本：v3 首次（23 个 shortcut），v4+ 全系
- 详解：[v3 concepts](/versions/v3/concepts)

### CSP（跨阶段部分连接）

- 梯度分流不重复，计算量降 20%
- 出现版本：v4 首次引入（CSPDarknet53）
- 相关案例：全部 v4+ 案例的骨干基础
- 详解：[v4 concepts](/versions/v4/concepts)

### C2f

- v8 起的骨干模块：cv1 输出多路 concat 到 cv2
- 出现版本：v8 ~ v11（v12 后被 A2C2f 替代 P4/P5）
- 相关案例：全部 v8+ 案例
- 详解：[v8 concepts](/versions/v8/concepts)

### C3k2

- C2f 的紧凑变体，专为边缘瘦身
- 出现版本：v11 / v12 / v26
- 相关案例：edge-deployment / agriculture
- 详解：[v11 concepts](/versions/v11/concepts)

### C2PSA

- C3k2 内嵌 PSABlock 的复合块
- 出现版本：v11 首次，v26 回归
- 详解：[v11 concepts](/versions/v11/concepts)

### GELAN

- 两级级联 RepNCSPELAN4，v9 骨干核心
- 出现版本：v9
- 详解：[v9 concepts](/versions/v9/concepts)

### SCDown / C2fCIB

- v10 的空间-通道解耦下采样 + 紧凑倒置残差块
- 出现版本：v10
- 详解：[v10 layers](/versions/v10/layers)

### A2C2f

- v12 引入注意力的骨干块（AAttn + FFN）
- 出现版本：v12 / v14 / v26
- 相关踩坑：INT8 下 AAttn 不可用（medical-lesion）
- 详解：[v12 concepts](/versions/v12/concepts)

### DSC3k2 / DSConv

- v13 深度可分离卷积体系
- 出现版本：v13
- 相关踩坑：量化友好度较其他版本略低（1-2% 掉点）
- 详解：[v13 concepts](/versions/v13/concepts)

### HyperACE

- v13 超图融合：顶点-边-邻接的三阶关系
- 出现版本：v13
- 相关踩坑：超图层量化不友好，建议 FP16
- 详解：[v13 concepts](/versions/v13/concepts)

### CircularConv / ViewEmbedding

- v14 多相机视角感知：环状卷积 + 视角 embedding
- 出现版本：v14
- 相关案例：暂无独立案例（drone-obb 可参考）
- 详解：[v14 concepts](/versions/v14/concepts)

### DeformableA2C2f

- v14 可变形注意力 CSP 块
- 出现版本：v14
- 详解：[v14 layers](/versions/v14/layers)

### ELAN / E-ELAN

- v7 聚合网络，梯度路径优化
- 出现版本：v7
- 详解：[v7 concepts](/versions/v7/concepts)

### EfficientRep

- v6 RepVGG 重参数化骨干
- 出现版本：v6
- 相关案例：全部 v6+ 案例（部署端优化基础）
- 详解：[v6 concepts](/versions/v6/concepts)

---

## 3. 检测头（Head）

### 单头耦合（v1-v3）

- box 和 class 一起过 1×1 conv
- 出现版本：v1 ~ v3
- 详解：[v1 architecture](/versions/v1/architecture)

### 解耦头（Decoupled Head）

- 分类/回归分两条分支，各自学习
- 出现版本：v4 首次，v5+ 主流
- 详解：[v8 concepts](/versions/v8/concepts)

### 多尺度三头

- P3/P4/P5 三个尺度出检测头，多尺度融合
- 出现版本：v3 ~ v26 全系
- 详解：[v3 concepts](/versions/v3/concepts)

### DualDDetect

- v9 双头 + 双分配器（PSC + ATFL）
- 出现版本：v9
- 详解：[v9 concepts](/versions/v9/concepts)

### NMS-free 头

- v10 训练时保证每个 GT 只出 1 个高分配分框
- 出现版本：v10 / v26
- 相关案例：无（学术验证）
- 详解：[v10 concepts](/versions/v10/concepts)

### End2End 双分支头

- v26 官方落地：`copy.deepcopy(cv2/cv3)` 生成两套并行分支
- 出现版本：v26
- 详解：[v26 concepts](/versions/v26/concepts)

### Proto26

- v26 分割原型：C3k2 + DWConv 组合
- 出现版本：v26
- 详解：[v26 layers](/versions/v26/layers)

### Segment26

- v26 分割头：配合 Proto26 输出 mask coefficient
- 出现版本：v26
- 相关案例：medical-lesion 的 v26 版本可参考
- 详解：[v26 concepts](/versions/v26/concepts)

### OBB26 头

- v26 旋转框头，替代 v8-OBB 的独立模型
- 出现版本：v26
- 相关案例：drone-obb 的 v26 迁移方案
- 详解：[v26 concepts](/versions/v26/concepts)

### Pose26 头

- v26 姿态估计头，直接输出关键点
- 出现版本：v26
- 详解：[v26 layers](/versions/v26/layers)

---

## 4. 分配器（Label Assignment）

### 硬分配（Anchor Matching）

- 每个格子最多 1 个正样本
- 出现版本：v1 ~ v4
- 详解：[v4 concepts](/versions/v4/concepts)

### 多正样本（Multi-positive）

- IoU > 阈值的所有 anchor 都参与监督
- 出现版本：v5
- 详解：[v5 concepts](/versions/v5/concepts)

### SimOTA

- YOLOX 的动态分配：按 cost 分配
- 出现版本：v6 初版
- 详解：[glossary](/glossary#simota)

### TAL（Task Alignment Learning）

- 对齐度量 = 分类^α × IoU^β
- 出现版本：v6 首次，v8 ~ v14 主流
- 关键参数：α=0.5, β=6.0, topk=10（v12 起统一）
- 详解：[v6 losses](/versions/v6/losses) / [v12 losses](/versions/v12/losses)

### 一致性双分配

- one2many(topk=10) + one2one(topk=1) 相加
- 出现版本：v10 / v26
- 详解：[v10 losses](/versions/v10/losses)

### PSC + ATFL（v9 双分配）

- v9 的两级分配器
- 出现版本：v9
- 详解：[v9 concepts](/versions/v9/concepts)

### _grouped_topk

- v26 无 NMS 的分组 top-k 推断算子
- 出现版本：v26
- 详解：[v26 tricks](/versions/v26/tricks)

---

## 5. 损失函数（Loss）

### L1 / L2

- v1/v2 的坐标回归损失
- 出现版本：v1 / v2
- 详解：[v1 losses](/versions/v1/losses)

### BCE / Softmax CE

- 分类损失（二分类/多分类）
- 出现版本：v2 ~ v26 全系
- 相关踩坑：retail-shelf 3000+ 类别时 Softmax CE 显存爆炸
- 详解：[v2 losses](/versions/v2/losses)

### VFL（VarifocalLoss）

- 非对称分类，负样本按预测分降权
- 出现版本：v6 首次，v8+ 主流
- 详解：[v6 losses](/versions/v6/losses)

### CIoU

- 完整 IoU：重叠度 + 中心距 + 长宽比
- 出现版本：v4 首次，v5+ 全系
- 相关案例：全部 9 个案例的坐标损失
- 详解：[v4 concepts](/versions/v4/concepts) / [v5 losses](/versions/v5/losses)

### DFL（Distribution Focal Loss）

- 框边距离建模为分布
- 出现版本：v6 首次，v8 ~ v14 主流
- 相关踩坑：DFL 必须保持 FP32（autonomous-lane 车道线抖动事故）
- v10 / v26 通过 `reg_max=1` 退化（等价去掉 DFL）
- 详解：[v6 losses](/versions/v6/losses)

### Tversky / Dice

- 分割任务的召回优先损失
- 出现版本：v8-seg / v11-seg / v26-seg
- 相关案例：medical-lesion（Tversky α=0.3 β=0.7）
- 详解：[v8 losses](/versions/v8/losses)

### Focal Loss

- 困难样本加权
- 出现版本：v8+（可选）
- 相关案例：industrial-defect（小目标+不平衡）
- 详解：[v8 losses](/versions/v8/losses)

### AngleLoss

- OBB 旋转角损失
- 出现版本：v8-OBB / v11-OBB / v26-OBB
- 相关案例：drone-obb
- 详解：[v8 losses](/versions/v8/losses)

### EWC（Elastic Weight Consolidation）

- 抗灾难性遗忘的权重正则
- 出现版本：v8+ 案例层（非内置）
- 相关案例：medical-lesion（多任务连续训练）
- 详解：[medical-lesion](/cases/medical-lesion)

### 3D 位姿损失

- Center + Size + Rot + Vel 四项组合
- 出现版本：v11-3D 外挂
- 相关案例：3d-bev
- 详解：[3d-bev](/cases/3d-bev)

### SILog / Grad（v26 深度损失）

- 尺度不变对数损失 + 梯度一致性
- 出现版本：v26
- 详解：[v26 losses](/versions/v26/losses)

### E2ELoss / Decay 退火

- v26 端到端损失 + one2many/one2one 权重退火
- 出现版本：v26
- 详解：[v26 losses](/versions/v26/losses)

---

## 6. 训练技巧（Training Tricks）

### Mosaic 增强

- 4 图拼 1，物体数量翻倍
- 出现版本：v4 首次，v5+ 主流
- 相关踩坑：drone-obb 禁旋转 Mosaic
- 详解：[v4 concepts](/versions/v4/concepts)

### close_mosaic

- 训练末期关闭 Mosaic
- 出现版本：v8+
- 相关案例：industrial-defect（末端 10 epoch 关闭）
- 详解：[v8 concepts](/versions/v8/concepts)

### EMA（指数移动平均）

- 训练时维护平滑版参数
- 出现版本：v5+ 全系
- 相关案例：全部 v5+ 案例
- 详解：[v5 tricks](/versions/v5/tricks)

### AMP（自动混合精度）

- fp16/fp32 自动切换，显存减半
- 出现版本：v5+ 全系
- 相关踩坑：medical-lesion（BF16 稳定优于 FP16）
- 详解：[v5 tricks](/versions/v5/tricks)

### autoanchor

- v5 anchor 自动学习：k-means + GA
- 出现版本：v5
- 详解：[v5 concepts](/versions/v5/concepts)

### 可训练免费午餐（bag-of-freebies）

- v7 方法论：训练期涨点，部署不变速
- 出现版本：v7
- 详解：[v7 paper](/versions/v7/paper)

### CosLR / OneCycleLR

- 学习率调度策略
- 出现版本：v5+ 全系
- 详解：[v5 tricks](/versions/v5/tricks)

### mixup / copy_paste

- 数据增强补充手段
- 相关案例：industrial-defect / retail-shelf / medical-lesion
- 详解：[glossary](/glossary#数据增强-jitter)

### LoRA

- 增量微调低秩适配
- 出现版本：v8+ 案例层
- 相关案例：retail-shelf（3000 SKU 增量微调）
- 详解：[retail-shelf](/cases/retail-shelf)

### 域对抗训练（GRL / DomainAdversarialLoss）

- 让骨干学习跨域不变特征
- 出现版本：v14
- 详解：[v14 losses](/versions/v14/losses)

---

## 7. 量化与部署（Quantization & Deployment）

### INT8 PTQ

- 训练后量化，快速但可能掉点
- 相关案例：edge-deployment / agriculture / security-monitoring
- 详解：[edge-deployment](/cases/edge-deployment)

### INT8 QAT

- 量化感知训练，精度高但慢
- 相关案例：industrial-defect / edge-deployment
- 详解：[industrial-defect](/cases/industrial-defect)

### FP16

- 半精度推理，工业主流
- 相关案例：industrial-defect / autonomous-lane / retail-shelf
- 详解：[industrial-defect](/cases/industrial-defect)

### 重参数化（Reparam）

- 训练多分支，推理折叠单路
- 出现版本：v6 / v8-World / v26
- 详解：[v6 concepts](/versions/v6/concepts)

### TensorRT

- NVIDIA 官方推理引擎
- 相关案例：industrial-defect / autonomous-lane / security-monitoring
- 详解：[industrial-defect](/cases/industrial-defect)

### ONNX Runtime

- 跨平台推理框架
- 相关案例：retail-shelf / security-monitoring
- 详解：[retail-shelf](/cases/retail-shelf)

### TFLite

- 移动端/边缘首选
- 相关案例：edge-deployment / agriculture
- 详解：[edge-deployment](/cases/edge-deployment)

### MLC / RKNN / NCNN

- 平台专用推理引擎
- 相关案例：暂无独立案例
- 详解：[glossary](/glossary#_1-anchor-网格中心预测-concept-anchor-free)

---

## 8. 多任务与衍生（Multi-task & Derivatives）

### SAM 集成

- v8 + SAM 大模型组合
- 相关案例：industrial-defect / medical-lesion / agriculture
- 详解：[medical-lesion](/cases/medical-lesion)

### 分层路由（v8s + ResNet）

- 两级分类：粗分类 → 精分类
- 相关案例：retail-shelf（3000 SKU）
- 详解：[retail-shelf](/cases/retail-shelf)

### BoT-SORT / SlowFast

- 多目标追踪 + 事件检测
- 相关案例：security-monitoring
- 详解：[security-monitoring](/cases/security-monitoring)

### BEVFormer 集成

- 6 目环视 3D 感知
- 相关案例：3d-bev
- 详解：[3d-bev](/cases/3d-bev)

### CLIP 集成（YOLO-World）

- 文本-视觉对齐，开放词表检测
- 出现版本：YOLO-World / v26 内嵌 YOLOE
- 详解：[YOLO-World 总览](/derivatives/yoloworld/index)

### RealNVP / BNContrastiveHead

- v26 YOLOE 的文本-图像对齐模块
- 出现版本：v26
- 详解：[v26 concepts](/versions/v26/concepts)

### Prompt / Prompt-Free / Visual Prompt

- YOLOE 三种输入模式
- 出现版本：YOLOE
- 详解：[YOLOE 总览](/derivatives/yoloe/index)

### ES-MoE / ExpertRouter

- v26+ 稀疏专家路由
- 出现版本：YOLO-Master
- 详解：[YOLO-Master 总览](/derivatives/master/index)

---

## 9. 跨案例踩坑聚合索引

按踩坑模式反查案例，快速定位解决方案：

| 踩坑模式 | 影响案例 | 相关概念 | 通用解 |
|---------|---------|---------|-------|
| FP16 敏感层退化 | industrial-defect / autonomous-lane / 3d-bev | DFL / Attention | 数值关键层强制 FP32 |
| 小目标丢失 | industrial-defect / medical-lesion / 3d-bev / drone-obb | P2 head / stride=4 | 加 P2 head 或升分辨率 |
| 光照突变 | agriculture / security-monitoring / autonomous-lane / 3d-bev | 归一化 / 白平衡 | 动态白平衡 + Mixup |
| 数据不均衡 | industrial-defect / retail-shelf / medical-lesion | mixup / copy_paste | 类加权 + 过采样 |
| 灾难性遗忘 | retail-shelf / medical-lesion | LoRA / Replay / EWC | 组合策略 |
| 多传感器同步 | autonomous-lane / 3d-bev | PTP 802.1AS | 时钟重同步 |
| 隐私合规 | medical-lesion / security-monitoring / agriculture | 联邦学习 | 联邦 + 脱敏 |
| 量化精度损失 | industrial-defect / edge-deployment / security-monitoring | INT8 QAT | QAT 而非 PTQ |
| 类别爆炸（3000+ SKU） | retail-shelf | 分层路由 | 两级分类 |
| 旋转框角度盲 | drone-obb / 3d-bev | AngleLoss | 禁旋转 Mosaic + 正射校正 |
| 时序抖动 | autonomous-lane / 3d-bev | 光流一致性 | 光流损失（权重 0.15） |
| 多相机 ID 切换 | security-monitoring / 3d-bev | ReID / BoT-SORT | ReID InfoNCE |

---

## 10. 反向搜索入口

按不同维度快速定位：

- **想看某一代的完整概念列表** → 直接读对应版本的 concepts 页（如 [v1 concepts](/versions/v1/concepts)、[v8 concepts](/versions/v8/concepts)、[v13 concepts](/versions/v13/concepts)）
- **想看某概念的所有相关案例** → 本页对应条目
- **想看术语白话解释** → [glossary](/glossary)
- **想看概念在代码里的实现** → 对应版本的 source 目录（如 [v1 detection-layer](/versions/v1/source/detection-layer)、[v8 tasks.py](/versions/v8/source/tasks-py)）
- **想看概念在实战中踩过什么坑** → 对应案例的"踩坑模式聚合"章节

---

## 三张索引页联动

| 页面 | 索引维度 | 用途 |
|-----|---------|-----|
| **本页（概念反向索引）** | 概念 → 版本/案例 | 从概念出发找出处 |
| **[compare-matrix](/compare-matrix)** | 版本 → 横向对比 | 从版本出发看差异 |
| **[cases/compare](/cases/compare)** | 案例 → 选型速查 | 从场景出发找案例 |

**典型阅读路径**：
1. 遇到一个新概念 → 查本页定位它在哪些版本/案例出现
2. 想看这个概念在哪个版本首次出现 → 读对应版本的 concepts 页
3. 想看这个概念在实战中的坑 → 读对应案例
4. 想看这个概念和其他版本怎么比较 → 读 compare-matrix

---

## 未覆盖但值得关注

以下概念在术语表中有定义，但本站尚无对应版本/案例的详细讲解：

| 概念 | 术语表位置 | 建议补齐方式 |
|-----|-----------|------------|
| RealNVP | [glossary](/glossary#realnvp) | 已在 v26 concepts 详解 |
| 3D 卷积 / 视频检测 | 未收录 | 案例层可扩展 |
| Contrastive Loss | 未收录 | YOLO-World 论文 |
| MoE Router | [glossary](/glossary#es-moe) | YOLO-Master 论文 |
| Diffusion Det | 未收录 | 与 YOLO 无关 |
| 自监督学习 | 未收录 | 案例层可扩展 |
