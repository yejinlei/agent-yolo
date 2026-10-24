# 版本差异卡片 · 一图看懂 v1→v27 每代核心差异

> 一句话：每代 YOLO 只记一件事——**这一代相对上一代，核心改动是什么、代价是什么、什么时候该用它**。

- ⏱ 预计阅读：5 分钟
- 📌 详细对比：[逐版本 Diff](/versions/version-diffs)（代码级） · [改动清单](/versions/changelog)（一句话列表）
- 🔗 相关：[对比矩阵](/compare-matrix) · [演化故事](/evolution-story) · [选型决策树](/choose-a-model)

---

## 使用方式

每张卡片 3 行：

- **一句话概括**：这一代的核心变化
- **⚙ 关键改动**：结构/训练/部署三类的具体改动
- **⚖ 权衡代价**：升级值得 vs 不值得的判断

---

## 卡片 1 · v1（2015）→ 一切的起点

- **一句话**：证明"单阶段检测也能又快又准"。
- **⚙ 关键改动**：VGG-16 + 7×7 grid + 2 box/grid；一次前向出所有框，去掉 Region Proposal。
- **⚖ 权衡代价**：AP 只有 48% 是硬伤，但把"实时检测"从学术讨论变成工程问题。**今天不用它训练，但必须理解它的思想**。

## 卡片 2 · v2（2016）→ 精度起飞

- **一句话**：加 anchor + BN + log 回归，AP 从 48% → 65.4%。
- **⚙ 关键改动**：Darknet-19 backbone，24×24 grid（预测点 12× 增加），3 scale anchor，首次引入 BatchNorm。
- **⚖ 权衡代价**：小目标召回大幅提升；但引入 anchor 让超参调优变难。**v2 的 anchor 思想一直延续到 v5**，v6 起改回 anchor-free。

## 卡片 3 · v3（2018）→ 多尺度革命

- **一句话**：确立 P3/P4/P5 三尺度范式——**这个范式一直到 v26 都没变**。
- **⚙ 关键改动**：Darknet-53 backbone，FPN 三尺度头，9 anchor（k-means 聚类）。
- **⚖ 权衡代价**：AP 71.2%，但速度掉到 17 FPS。v3 是"研究版"，实际部署几乎都被 v4/v5 替代。

## 卡片 4 · v4（2020）→ 训练技巧大合集

- **一句话**：**不加强度**，只把过去所有训练技巧打包。
- **⚙ 关键改动**：CSPDarknet，Mosaic / CutOut / MMR / SAM / CIoU / EIoU / SWA / AutoAugment；SPP / PANet。
- **⚖ 权衡代价**：AP 从 71.2 → 83.5%，速度 130 FPS。v4 是**手工技巧的巅峰**，之后 v5+ 走向"标准化"。

## 卡片 5 · v5（2021）→ 工业化起点

- **一句话**：把 v4 的所有技巧封装成"能一键跑"的开源工程。
- **⚙ 关键改动**：PyTorch 重写；P3/P4/P5 三头；Focal Loss + DFL；Ultralytics 生态雏形；首次引入 Pose / Segment 任务接口。
- **⚖ 权衡代价**：AP 78.6 (s) → 82.4 (x)。**这是"研究版 → 生产版"的分水岭**，之后所有版本都基于 Ultralytics 生态。

## 卡片 6 · v6（2022 · Meituan）→ anchor-free 回归

- **一句话**：抛弃 anchor，回归 anchor-free；引入 RepVGG 风格分支重参数化。
- **⚙ 关键改动**：anchor-free head，TAL 任务分配器（Task-Aligned Assigner），AFC（辅助分支），RepVGG 结构。
- **⚖ 权衡代价**：同精度比 v5 快 25%。**v6 是学术线**，社区版维护活跃度不如 v5/v8。

## 卡片 7 · v7（2022）→ 复合网络结构

- **一句话**：E-ELAN + 辅助头 + 多尺度标签分配，把网络结构重新组合。
- **⚙ 关键改动**：E-ELAN 复合网络，Reparameterization，SimOTA，EIOU 损失。
- **⚖ 权衡代价**：v7 精度好但**部署复杂**（多分支），实际生产落地少。**v7 是"研究版"，不是"生产版"**。

## 卡片 8 · v8（2023 · Ultralytics）→ 生产基线

- **一句话**：官方重新整理 API，C2f 骨干 + DFL + Anchor-Free，**这是 3 年内最稳的生产基线**。
- **⚙ 关键改动**：C2f（替代 C3/C3k），Decoupled Head，Anchor-Free + DFL，SiLU 激活，统一 tasks API（detect/segment/pose/classify/obb）。
- **⚖ 权衡代价**：AP 44.9 (n) → 53.9 (x)，速度 125-178 FPS。**v8 是本项目案例页 8/12 个的首选项**。

## 卡片 9 · v9（2024 · Peking U）→ 高效推理块

- **一句话**：GELAN 高效推理块，PGWR 权重归一，同精度比 v8 快 20%。
- **⚙ 关键改动**：GELAN（Generalized ELAN），PGWR（Precision-Guided Weight Reduction）。
- **⚖ 权衡代价**：精度不输 v8，速度略快。**但 v9 是学术线，官方 API 生态不如 v8/v11**。

## 卡片 10 · v10（2024 · Tsinghua）→ NMS-free

- **一句话**：**NMS-free** —— 训练时直接教网络"输出不重叠框"。
- **⚙ 关键改动**：双头架构（分类头 + 回归头），NMS-free 训练，推理无后处理。
- **⚖ 权衡代价**：延迟比 v8 低 40%+，但**训练不收敛风险更高**，需要更多数据。**边缘设备推荐**。

## 卡片 11 · v11（2024 · Ultralytics）→ C2PSA + 多任务

- **一句话**：C2PSA（PSA 注意力）+ 多任务（seg/pose/obb/classify），**Ultralytics 官方第 2 代**。
- **⚙ 关键改动**：C2PSA（Partial Self-Attention），统一 8 任务 API，参数比 v8 少 22% 但精度持平。
- **⚖ 权衡代价**：AP 43.5 (n) → 52.4 (x)，速度 120+ FPS。**当前生产主流**，本项目 4/12 案例首选。

## 卡片 12 · v12（2025）→ Attention 骨干

- **一句话**：引入注意力 backbone，为纯视觉任务做结构升级。
- **⚙ 关键改动**：Attention-based backbone（替代 C2f），保持 anchor-free + DFL。
- **⚖ 权衡代价**：精度略高于 v11，速度持平。**社区版**，Ultralytics 官方未发布。

## 卡片 13 · v13（2025）→ 工业高清首选

- **一句话**：面向工业高清场景的结构优化，1280/1920 输入表现最好。
- **⚙ 关键改动**：优化大分辨率下的 backbone/neck 结构，改善高分辨率梯度。
- **⚖ 权衡代价**：AP 46 (s) → 53 (x)。**本项目 industrial-defect 案例首选**（1280 输入）。

## 卡片 14 · v14 / Omni（2026）→ 多视角融合

- **一句话**：Omni 面向多相机/多视角场景，学术线但架构野心大。
- **⚙ 关键改动**：Omni head 支持多视角融合，跨摄像头一致性损失。
- **⚖ 权衡代价**：适合自动驾驶、安防多目。**社区版**，Ultralytics 官方未发布。

## 卡片 15 · v26（2026 · Ultralytics）→ 8 任务统一

- **一句话**：**8 任务统一 backbone**（detect/segment/pose/obb/classify/depth/segment-sem/cls-tracker），取消 x 档，n/s/m/l 四档标准化。
- **⚙ 关键改动**：统一 backbone，取消 x 档（合并到 l），新增 depth 任务，新增 solutions 一键 API，新增 TrackTrack 默认 tracker。
- **⚖ 权衡代价**：AP 42 (n) → 51 (l)，速度 130+ FPS。**未来 2 年的生产基线**，本项目 3/12 案例 + depth 任务首选。

## 卡片 16 · v27（Preview · 未 GA）→ Query-based 双族

- **一句话**：N/S 双族 CNN + M/L Query-based（UltraViT），**取消 x 档**。
- **⚙ 关键改动**：Foreground Alignment Supervision 抹平 one-to-many vs one-to-one 差距，UltraViT backbone（Query-based 族）。
- **⚖ 权衡代价**：Preview 阶段，权重和精度数字都未 GA。**雷达页详见** [v27-preview](/models/v27-preview)。

---

## 三张速查表

### 按用途选版本

| 用途 | 推荐版本 | 理由 |
|------|---------|------|
| 生产稳定首选 | v8 / v11 / v26 | Ultralytics 官方基线 |
| 极致边缘（<3 TOPS） | v10 / v11n / v26n | NMS-free 或最紧凑 |
| 工业高清（1280+） | v13 | 大分辨率优化最好 |
| 多任务统一 | v26 | 8 任务一套 backbone |
| 多视角融合 | v14 Omni | 学术线，跨摄像头 |
| 学术前沿 | v27 Preview | 双族架构，未来基线 |

### 按时间选

```
2015 ── v1 ─────────────── 单阶段革命
2018 ── v2, v3 ─────────── 精度起飞 + 多尺度
2020 ── v4 ─────────────── 训练技巧巅峰
2022 ── v5, v6, v7 ─────── 工业化 + 学术分岔
2023 ── v8 ─────────────── Ultralytics 生产基线
2024 ── v9, v10, v11 ───── 结构创新 + 官方迭代
2025 ── v12, v13 ───────── 高清 / Attention
2026 ── v14, v26, v27* ── Omni + 官方基线 + Preview
          (* 未 GA)
```

### 三代范式

| 时代 | 版本 | 核心特征 |
|------|-----|---------|
| Darknet 时代 | v1-v4 | CifarNet / Darknet-53，anchor-based |
| Transformer 时代 | v5-v11 | ResNet / CSP / C2f，anchor-free + DFL |
| Query-based 时代 | v12-v27 | Attention backbone + Query-based + 多任务统一 |

---

## 常见问答

**Q：我该升级到哪一版？**
A：如果现在是 v8/v11 → 观察 v26 GA 后决定；如果还在 v5 之前 → 直接跳到 v11 或 v26，中间版本不用看。

**Q：v10 和 v26 都号称"小模型最强"，选谁？**
A：v10 是 NMS-free 极简化，适合**边缘小场景**；v26 是多任务统一，适合**要多任务/要官方支持**。看你的场景是否需要 v26 的多任务能力。

**Q：v27 现在能上生产吗？**
A：**不能**。Preview 阶段精度数字未 GA，权重也未官方发布。看 [v27-preview](/models/v27-preview) 的跟踪 checklist，等 GA 后再评估。

**Q：为什么要跳过 v9 / v12 / v14 看卡片？**
A：这几版是**学术线**，官方不维护。看卡片了解思想即可，但生产落地不建议。

---

## 反哺信号

- **Trick 新增**：`trick-version-delta-card`（版本差异卡片：每代 3 行的极简形式，避免读者陷入版本页细节）
- **PainPoint 关联**：`pain-cnn-vs-transformer-deployment`（Query-based 部署复杂，N/S CNN 与 M/L Query-based 是折中，见 [v27-preview](/models/v27-preview)）

---

**最后更新：2026-10-25**
