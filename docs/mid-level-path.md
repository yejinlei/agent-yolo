# 中阶学习路线 · 3 天吃透 YOLO 家族

> **定位**：给已经读完 [小白入门](/for-beginners) 或自己有 YOLO 使用经验、想系统吃透整个家族的读者。
>
> **目标读者**：读过至少一篇目标检测论文、能看懂 Python、有 YOLOv5+ 使用经验
>
> **完成后能得到**：能对任意一个 YOLO 版本做"5 分钟定位 + 判断它适不适合我的场景"的能力

## 路线概览

```
Day 1（主线演进）      Day 2（衍生分支 + 任务扩展）      Day 3（工程落地）
├─ v1 → v3 → v4        ├─ YOLO-World（开放词表）          ├─ 训练微调完整流程
├─ v5 → v6             ├─ YOLOE（fuse 架构）              ├─ ONNX / TensorRT 导出
├─ v8（工业主力）      ├─ YOLO-Master（多任务统一）       ├─ INT8 量化实操
└─ v11 → v13 → v14     ├─ 分割 / 姿态 / OBB / 旋转        └─ 硬件选型与踩坑
└─ v26（终态）          └─ NMS-free 端到端               └─ 项目评估标准
```

**总阅读时间**：约 20-30 小时（含实操）

---

## Day 1 · 主线演进（8-10 小时）

### Day 1-1：三个时代的分水岭（3 小时）

**核心问题**：YOLO 家族的架构演进可以分成哪几个"时代"？每个时代解决了上一代的什么痛点？

**阅读顺序**：
1. [小白入门 · 第 2 页 · 六个关键转折](/for-beginners) （15 分钟，建立框架）
2. [演化故事 · 家族总览](/evolution-story) （30 分钟，叙事视角）
3. [对比矩阵 · 骨干演进 + 检测头演进](/compare-matrix) （45 分钟，横向对齐）

**必须记住的三处分水岭**：

- **v5（2020）**：darknet → PyTorch，工程生态从 C++ 切到 Python
- **v6（2022）**：anchor-based → anchor-free + TAL，检测范式革命
- **v8（2023）**：C2f + DFL + Detect，工业界统一答案

### Day 1-2：三个必读版本（4 小时）

**v1（打基础）**：
- [v1 · index](/versions/v1/index) · 5 分钟
- [v1 · concepts](/versions/v1/concepts) · 15 分钟
- [v1 · losses](/versions/v1/losses) · 15 分钟
- 重点看：为什么"grid regression"是 single-shot 检测的起点

**v5（转 PyTorch）**：
- [v5 · index](/versions/v5/index) · 10 分钟
- [v5 · painpoints](/versions/v5/painpoints) · 20 分钟
- [v5 · code-walkthrough](/versions/v5/code-walkthrough) · 40 分钟
- 重点看：yaml 配置驱动 + AGPL 许可证带来的工程红利

**v8（工业主力）**：
- [v8 · index](/versions/v8/index) · 10 分钟
- [v8 · concepts](/versions/v8/concepts) · 20 分钟
- [v8 · losses](/versions/v8/losses) · 20 分钟
- [v8 · loss-gradient](/versions/v8/loss-gradient) · 30 分钟
- [v8 · code-walkthrough](/versions/v8/code-walkthrough) · 60 分钟
- 重点看：C2f 为什么比 C3 好、DFL 为什么有效、TAL topk=10 的取舍

### Day 1-3：现代版本速览（2 小时）

**v11（骨干演进代表）**：
- [v11 · index](/versions/v11/index) · 15 分钟
- [v11 · concepts](/versions/v11/concepts) · 20 分钟
- 重点看：C3k2 相对 C2f 的取舍

**v13（稀疏注意力）**：
- [v13 · index](/versions/v13/index) · 15 分钟
- [v13 · concepts](/versions/v13/concepts) · 20 分钟
- 重点看：DSC3k2 + AAttn 的稀疏化方向

**v26（终态）**：
- [v26 · index](/versions/v26/index) · 20 分钟
- [v26 · concepts](/versions/v26/concepts) · 30 分钟
- [v26 · code-walkthrough](/versions/v26/code-walkthrough) · 40 分钟
- 重点看：End2End 头 + YOLOE 内嵌 + 8 任务统一

**跳过**：v2/v3/v4/v6/v7/v9/v10/v12/v14 的详细概念页——这些版本在主线演进里已经通过 v1/v5/v8/v11/v13/v26 覆盖了核心思想。

---

## Day 2 · 衍生分支 + 任务扩展（6-8 小时）

### Day 2-1：三个衍生分支（3 小时）

**衍生分支是 YOLO 家族"横向生长"的方向**——每个分支解决主线不解决的痛点。

**YOLO-World（开放词表检测）**：
- [YOLO-World · index](/derivatives/yoloworld) · 30 分钟
- [YOLO-World · paper](/derivatives/yoloworld/paper) · 30 分钟
- [YOLO-World · architecture](/derivatives/yoloworld/architecture) · 30 分钟
- 重点看：CLIP 冻结文本嵌入 + BNContrastiveHead 的设计哲学

**YOLOE（fuse 架构）**：
- [YOLOE · index](/derivatives/yoloe) · 30 分钟
- [YOLOE · architecture](/derivatives/yoloe/architecture) · 30 分钟
- [YOLOE · layers](/derivatives/yoloe/layers) · 30 分钟
- 重点看：`fuse()` 后参数共享 vs 训练时独立分支

**YOLO-Master（多任务统一）**：
- [YOLO-Master · index](/derivatives/master) · 30 分钟
- [YOLO-Master · architecture](/derivatives/master/architecture) · 30 分钟
- 重点看：Master 如何协调多个任务头，与 v26 内嵌 YOLOE 的取舍

**衍生 vs 主线的判断标准**：
- 主线（v1→v26）解决"如何把检测做得更好"
- 衍生（World/YOLOE/Master）解决"如何让检测做更多事情"

### Day 2-2：四个任务扩展（3 小时）

**除了 Detect，YOLO 家族能做的任务**：

1. **Segment（实例分割）**：
   - [分割完整指南](/tasks/segment)
   - 核心：C2f 后加一个 mask prototype head，每类一个 mask

2. **Pose（姿态估计）**：
   - [姿态估计指南](/tasks/pose)
   - 核心：在 Detect 头上加关键点回归（K 个 keypoint × [x,y,score]）

3. **OBB（定向框）**：
   - [定向框指南](/tasks/obb)
   - 核心：框参数从 4（x,y,w,h）变成 5（x,y,w,h,θ），角度回归

4. **Rotated / 旋转检测**：
   - [旋转检测指南](/tasks/rotated)
   - 核心：处理小目标密集 + 任意角度场景

**任务选型速查**：
- 只需要检测框 → Detect
- 需要精确轮廓 → Segment
- 需要姿态（人/动物）→ Pose
- 需要倾斜框（遥感/交通）→ OBB
- 需要极端小目标 + 任意角度 → Rotated

### Day 2-3：NMS-free 端到端（1 小时）

**主线视角的另一个重要变化**：v10 首次 NMS-free，v26 官方线 NMS-free。

**为什么重要**：
- 推理链更短（少后处理）
- 阈值不再影响精度
- 更适合实时场景

**权衡**：
- 精度略低（v10 s 档比 v8s 低 2 AP）
- 训练更复杂（需要一致的分配策略）

**参考**：
- [v10 · index](/versions/v10/index) · 20 分钟
- [v26 · code-walkthrough](/versions/v26/code-walkthrough) · End2End 部分 · 20 分钟
- [对比矩阵 · 检测头演进](/compare-matrix#检测头演进) · 20 分钟

---

## Day 3 · 工程落地（6-8 小时）

### Day 3-1：训练微调（3 小时）

**从零开始做一个自己的 YOLO 模型**：

- [训练调优指南](/recipes/train) · 1.5 小时通读
- [训练调优指南 · 少样本微调](/recipes/train#少样本微调) · 30 分钟
- [训练调优指南 · 超参选择](/recipes/train#超参选择) · 30 分钟
- [故障排查 · 训练 & 增强](/troubleshooting#a-训练--增强) · 30 分钟

**必须掌握的操作**：
1. 用 Ultralytics CLI 训练：`yolo detect train data=custom.yaml model=yolov8s.pt epochs=100`
2. 学会看训练曲线：loss / mAP / PR curve
3. 理解"冻结前 N 层"的意义
4. 掌握 `mixup / copy_paste / mosaic` 的开关时机

### Day 3-2：部署导出（3 小时）

**从 PyTorch 权重到边缘设备推理**：

- [ONNX / TensorRT 部署](/recipes/export-onnx) · 1.5 小时通读
- [故障排查 · 量化 & 精度](/troubleshooting#b-量化--精度) · 30 分钟
- [故障排查 · 导出 & 部署](/troubleshooting#c-导出--部署) · 30 分钟
- [边缘部署案例](/cases/edge-deployment) · 30 分钟

**必须掌握的操作**：
1. `yolo export model=yolov8s.pt format=onnx half=True`
2. `yolo export model=yolov8s.pt format=engine half=True device=0`
3. INT8 QAT 的基本流程
4. DFL 层保持 FP32 的原因

**硬件选型**：
- Jetson Orin Nano 8GB → v8s INT8 @ 640
- RK3588 → v8s INT8 @ 640
- RTX 4090 → v8m/l INT8 @ 1280
- 桌面 8GB 显存 → v8n/s @ 640

### Day 3-3：选型与项目评估（2 小时）

- [选型决策树](/choose-a-model) · 30 分钟
- [9 大场景速查](/cases/compare) · 30 分钟
- [基准数据](/benchmarks) · 30 分钟
- [FAQ · 版本选择](/faq#版本选择) · 30 分钟

**项目评估三问**：
1. **精度下限**：这个任务的 SOTA 是多少？我期望多少？
2. **推理预算**：多少 FPS 是可接受的？在什么硬件上？
3. **部署难度**：ONNX 就够吗？需要 TensorRT / INT8 / 结构重参数化吗？

---

## 学习节奏建议

### 每日学习节奏

| 时段 | 内容 | 时长 |
| --- | --- | --- |
| 上午 | 概念页 + 论文精读 | 3 小时 |
| 中午 | 复盘 + 笔记 | 30 分钟 |
| 下午 | 代码走读 + 实操 | 3 小时 |
| 晚上 | 案例阅读 + 踩坑 | 2 小时 |

### 输出物

学完每一阶段后，产出一份**个人笔记**：

- **Day 1 输出**：一张"YOLO 主线演进脑图"，从 v1 到 v26 每个版本的"痛点 → 解决方案"
- **Day 2 输出**：一张"YOLO 家族树"，主线 + 衍生分支 + 任务扩展的完整拓扑
- **Day 3 输出**：一个可运行的 YOLO 项目（训练 → 导出 → 部署的完整 pipeline）

---

## 常见问题

### Q：能不能只读一天？

**可以，但要选重点**。时间紧的话：
1. v8 index + concepts（1 小时）
2. 衍生分支三个 index（1.5 小时）
3. 训练指南 + 部署指南（2 小时）
4. 选型决策树（30 分钟）

**总共 5 小时**，能覆盖 80% 场景。

### Q：能不能只看论文不看代码走读？

**不推荐**。YOLO 家族的很多设计（比如 DFL、TAL、C2f）**在代码里比论文讲得更清楚**——尤其是各版本 yaml 配置的实际数值差异。

### Q：这个路线适合纯研究员吗？

**不太适合**。这个路线偏工程落地，研究员应该：
1. 直接读论文精读版（每版本的 paper.md）
2. 重点关注 loss-gradient 页（损失函数的梯度分析）
3. 跳过训练 / 部署 / 硬件相关章节

### Q：这个路线适合产品经理吗？

**部分适合**。产品经理应该看：
1. 小白入门（5 页）
2. 演化故事（家族总览）
3. 选型决策树
4. 9 大场景速查
5. FAQ

**总共 3-4 小时**，能覆盖业务选型和沟通需求。

---

## 更多资源

- **纯入门**：[小白入门](/for-beginners)
- **术语疑问**：[术语表](/glossary)
- **报错类问题**：[故障排查](/troubleshooting)
- **FAQ 问答**：[FAQ](/faq)
- **实战案例**：[9 大案例](/cases/index)
