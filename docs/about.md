# 关于本项目

> **一句话总结**：本项目用「本体论（Ontology）」+「文档站」的方式，把 YOLO 家族从零散论文整理成一张可查询、可追溯、可持续维护的知识网络。

- ⏱ 预计阅读：5 分钟
- 🔗 相关：[本体 Schema](/ontology/schema) · [GitHub 仓库](https://github.com/yejinlei/agent-yolo) · [贡献指南](https://github.com/yejinlei/agent-yolo#contributing)

---

## 目录

- [为什么要做这个项目](#为什么要做这个项目)
- [方法论：本体论怎么落地](#方法论本体论怎么落地)
- [内容分层：读者怎么用](#内容分层读者怎么用)
- [数据基线与准确性承诺](#数据基线与准确性承诺)
- [项目结构](#项目结构)
- [与同类项目的区别](#与同类项目的区别)
- [路线图](#路线图)
- [参与项目](#参与项目)

---

## 为什么要做这个项目

YOLO 系列从 2016 年到 2025 年出了 26+ 个主线版本 + 数十个衍生分支。信息高度碎片化：

- **Ultralytics 官方文档**只讲 v5-v11 官方实现，历史版本靠社区整理
- **arXiv 论文**关注学术创新，工程细节略过
- **GitHub README**是"能跑起来"的最低说明，缺上下文
- **知乎 / 公众号 / B 站**大量二传，事实错误多、结论互相矛盾

本项目想解决的不是"再写一遍教程"，而是**把 YOLO 全家族的知识编成一张网**，任何一句话都能追溯到：

- 是哪篇论文提出的
- 具体在哪个代码文件的哪几行
- 影响了哪些后续版本
- 落地到具体场景时要注意什么

---

## 方法论：本体论怎么落地

"本体论（Ontology）"原本是哲学 / 知识工程术语，指的是**用形式化语言描述某领域的概念、关系和约束**。放到本项目里，就是给 YOLO 家族定义一套"实体类型 + 关系类型"的数据模型。

### 四层知识闭环

```mermaid
flowchart LR
    L1["L1 论文层<br/>Paper"] --> L2["L2 理论层<br/>Concept"]
    L2 --> L3["L3 实现层<br/>Model / Layer / Loss / Trick"]
    L3 --> L4["L4 实战层<br/>CaseStudy"]
```

- **L1 论文层**：每篇论文的原子事实（提出什么、验证什么）
- **L2 理论层**：从论文提炼的知识（如 anchor-free、NMS-free）
- **L3 实现层**：落到具体网络结构（如 C2f、DFL、TAL 分配器）
- **L4 实战层**：某个具体场景踩过的坑（如无人机小目标、遥感 OBB）

四层之间用**明确的关系**连接：`elaborates`（论文阐述理论）、`manifestsIn`（理论体现为实现）、`appliedIn`（实现应用于案例）。

### 为什么不用普通目录？

普通 markdown 目录只能表达"父子关系"。本体的价值在于**关系网**：

- YOLOv8 用了 DFL → DFL 也用在 RetinaNet++ 和 YOLOv11
- CIoU 是 YOLOv4 引入的 → 但原论文更早
- 案例踩坑"随机旋转让 OBB 崩" → 这个坑对所有 OBB 变体都适用

这些跨文档的关系，只有本体才能承载。

### Schema 文件

完整定义见 [ontology/schema.md](/ontology/schema)，包含 17 种实体类型、每种的必填字段、约束和校验规则。

---

## 内容分层：读者怎么用

不同读者只需要看不同部分，不必从头读到尾。

| 读者角色 | 建议路径 | 预计耗时 |
|---------|---------|---------|
| 完全新手 | [for-beginners](/for-beginners) → [timeline](/global/timeline) | 半天 |
| 有 CV 基础 | [compare-matrix](/compare-matrix) → [faq](/faq) → 选一个版本深入 | 1-2 天 |
| 中级研究/工程 | [mid-level-path](/mid-level-path) → 任务/版本专页 → 案例 | 3-5 天 |
| 特定任务落地 | [任务页](/tasks/segment) → 对应案例 → [训练指南](/recipes/train) → [部署指南](/recipes/export-onnx) | 1-2 周 |
| 想引用/研究 | [references](/references) → 本体 schema → GitHub | 按需 |

---

## 数据基线与准确性承诺

本项目对内容准确性有明确标准：

### 数据基线

- **COCO 2017 数据集**：除非特别说明，所有 AP / mAP 数字都指 COCO 2017 val
- **Ultralytics v8.3.0**：官方 API 行为以此版本为基线
- **时间基线**：所有"最新"表述以文档最近更新日期为准，见页脚

### 承诺

1. **可追溯**：每个数字都能追到原始论文或官方仓库
2. **可对比**：同一指标用同一口径，不做苹果对橘子比较
3. **明确标注**：任何"经验估算""社区数据"都会标注，不假装权威
4. **承认不确定**：找不到来源的内容会写"待验证"，不硬编

### 已知局限

- 衍生分支（YOLO-World / YOLOE / YOLO-Master）的更新频率比主线快，本文档可能滞后
- 少数案例的具体数字来自作者本地实验，硬件差异会导致偏差
- 版本号（v15-v25）在官方发布前不预先写文档

---

## 项目结构

```
agent-yolo/
├── docs/                     # VitePress 文档站（本站）
│   ├── versions/             # 主线版本 v1-v14 + v26
│   ├── derivatives/          # 衍生分支 YOLO-World / YOLOE / YOLO-Master
│   ├── cases/                # 实战案例（工业、遥感、医学等）
│   ├── tasks/                # 任务扩展（Segment / Pose / OBB / Rotated）
│   ├── recipes/              # 实操手册（训练 / 部署）
│   ├── global/               # 全局视图（家族、时间线、演化故事）
│   ├── ontology/             # 本体论 schema 和实体说明
│   └── ...                   # FAQ、compare-matrix、glossary、troubleshooting
├── ontology/                 # 本体数据文件（YAML + JSON Schema）
├── scripts/                  # 校验和生成脚本
└── README.md
```

文档站点由 [VitePress 1.6.4](https://vitepress.dev/) 构建，图表由 [Mermaid](https://mermaid.js.org/) 渲染，无外部 API 依赖，所有静态资源都在同一仓库。

---

## 与同类项目的区别

| 项目 | 定位 | 与本项目区别 |
|------|------|-------------|
| [Ultralytics 官方文档](https://docs.ultralytics.com/) | 官方使用手册 | 只覆盖 Ultralytics 出品的 v5-v11 |
| [Ultralytics YOLO Hub](https://hub.ultralytics.com/) | 云端训练平台 | 商业产品，本项目是开源知识 |
| 各类中文博客/CSDN | 二手整理 | 事实错误率高，本项目以原论文为准 |
| CVPR/ICCV 论文集 | 学术论文 | 无落地建议，本项目补 L4 实战层 |

本项目的核心差异：**把历史版本（v1-v4）也纳入统一体系**，同时补上衍生分支和任务扩展，这是任何一家商业文档都不会做的事。

---

## 路线图

### 已完成（Batch 1）

- 主线版本 v1-v14 + v26 详解
- 衍生分支 YOLO-World / YOLOE / YOLO-Master
- 9 个实战案例 + 案例索引
- 任务扩展 4 篇（Segment / Pose / OBB / Rotated）
- 训练指南、部署指南
- FAQ、中阶路线、故障排查
- 本体 Schema v0.3.0

### 计划中（Batch 2）

- 深度案例：多目标跟踪 (MOT)
- 深度案例：视频流检测优化
- 深度案例：多模态融合（文本+图像）
- 每版本"改动清单"页（v1→v2 到底改了什么，逐行 diff）

### 计划中（Batch 3）

- 交互式模型对比（浏览器内跑 ONNX）
- 术语双语对照（中↔英）
- 贡献者指南完善
- 版本时间线的可视化增强

### 已完成（Batch 4）

- 12 案例横向对比（选型速查 + 踩坑聚合）
- 版本矩阵扩充（任务广度对比）
- 时间线可视化（Gantt 全景 + 三时代划分）
- 术语中英对照总表（130 条）

### 已完成（Batch 5 · 2026-10-25）

- MOT 案例扩到 6 tracker + 决策树 + ONNX ReID
- Ultralytics Solutions 配方页（8 配方 · 即用不训）
- 单目深度估计任务页（yolo26-depth 5 档 + calibrate()）
- YOLO27 雷达页（Preview · 未 GA）

### 已完成（Batch 6 · 2026-10-25）

- **交互式模型对比**（`/interactive/compare`）：浏览器内跑 ONNX 推理，对比 yolo26n / yolo11n / yolov8n，onnxruntime-web WASM + SIMD
- **贡献者指南**（`/CONTRIBUTING`）：10 章规范 · 5 类文档边界 · 术语一致性 · PR checklist
- **版本一句话差异卡片**（`/versions/deltas`）：v1→v27 Preview 共 16 张卡片 · 3 张速查表 · 三代范式
- **反哺信号自动同步脚本**（`scripts/sync-feedback.js`）：扫描全站 markdown，生成 `/concepts-feedback` 汇总页（41 条信号 · PainPoint 14 / Trick 18 / Metric 9）

---

## 训练配方约定：三种文档的边界

本仓库有大量"训练 / 使用 / 落地"的文档，为避免读者混淆，统一约定三类文档的边界与命名：

| 目录 | 定位 | 训不训模型 | 典型内容 |
|-----|-----|---------|---------|
| [**cases/**](/cases) | 深度实战案例 | ✅ 从数据到部署全链路 | 具体场景 + 演进故事 + 反哺信号 |
| [**solutions/**](/solutions/recipes) | 官方即用配方 | ❌ 直接用官方权重 | 5-8 行 CLI + 关键参数 + 常见坑 |
| [**recipes/**](/recipes/train) | 训练 / 部署实操手册 | ✅ 训练方法学 | 训练技巧 · 数据标注 · 量化 · 生产化 |
| [**tasks/**](/tasks/segment) | 单任务技术页 | 可训可推理 | 任务定义 · 架构 · 数据集 · 调优 |
| [**versions/**](/versions/v26/index) | 单版本深度页 | 参考实现 | 论文精读 · 源码走读 · 逐层图解 |

**决策速查**：

- 我要"用现成模型快速出结果" → 看 [solutions/recipes](/solutions/recipes)
- 我要"从零训练一个模型" → 看 [recipes/train](/recipes/train)
- 我要"某个具体场景的完整方案" → 看 [cases/](/cases) 里的对应案例
- 我要"某个任务的技术细节" → 看 [tasks/](/tasks/segment) 对应任务
- 我要"某个版本到底改了什么" → 看 [versions/](/versions/v26/index) 对应版本

**避免踩坑的通用约定**：

1. **cases/ 只写"能复现的方案"**——不允许只写"用 XX 模型"这种空话
2. **solutions/ 只写"能直接跑的命令"**——每个配方必须有一键脚本
3. **recipes/ 只写"训练方法学"**——不绑定具体版本或场景
4. **反哺信号（PainPoint / Trick / Metric）只允许出现在 cases/ 与 tasks/**——recipes/ 是"方法学"层，不承担反哺责任

**新增案例时必须做的 4 件事**：
1. 在 [cases/compare.md](/cases/compare) 加入 12 案例横向对比表
1. 在 [compare-matrix.md](/compare-matrix) 的"实战案例与版本匹配"表加一行
3. 在 [cases/index.md](/cases) 的案例索引加一行
4. 在 [.vitepress/config.mjs](https://github.com/yejinlei/agent-yolo/blob/main/docs/.vitepress/config.mjs) 的 sidebar 加链接

**新增任务页时必须做的 3 件事**：
1. 在 [compare-matrix.md](/compare-matrix) 的"任务广度对比"表加一行
2. 在 nav / sidebar 两处都加链接
3. 在至少 1 个案例页里做回链

**新增版本页时必须做的 3 件事**：
1. 在 [compare-matrix.md](/compare-matrix) 的"主线版本对比"表加一行
2. 在 [global/timeline.md](/global/timeline) 的 Gantt 加一个节点
3. 在 [versions/changelog.md](/versions/changelog) 加入改动清单

**新增 solutions 配方时必须做的 3 件事**：
1. 在 [solutions/recipes.md](/solutions/recipes) 加入全景表
2. 加入横向对比表
3. 关联到至少 1 个 cases/ 深度案例

---

## 参与项目

欢迎任何形式的贡献：

- **提 issue**：发现错误、缺少版本、内容过时
- **PR**：新增文档、修正术语、补充案例
- **讨论**：对某个技术点有不同观点，欢迎在 issue 里辩

**贡献流程**：

1. Fork 仓库
2. 新建分支 `docs/{主题}`
3. 修改对应 markdown 文件
4. 本地运行 `npm run docs:build` 验证 Mermaid 无报错
5. 提交 PR

**贡献前请读**：

- [docs/CONTRIBUTING.md](/CONTRIBUTING) — 贡献指南（章节结构 / 术语一致性 / PR checklist）
- [docs/glossary.md](/glossary) — 术语必须与全站一致
- [docs/.vitepress/config.mjs](https://github.com/yejinlei/agent-yolo/blob/main/docs/.vitepress/config.mjs) — 新增页面要更新 nav + sidebar

---

## 常见问题

**Q：为什么有 v1-v14 但没有 v15-v25？**  
A：v15-v25 在 Ultralytics 官方发布之前，本项目不预写文档，避免传播未经核实的信息。v26 有官方发布，所以收录。

**Q：为什么不用飞书/Notion，偏要用 VitePress？**  
A：技术文档需要版本管理、diff、issue 追踪，这些 GitHub 提供的一流能力，飞书/Notion 做不到。

**Q：数据基线会不会过时？**  
A：会。每个数据密集页脚都有"数据基线"提示，读者发现过时可以在 issue 里提出，会尽快更新。

**Q：本体论对我有用吗？**  
A：如果你是纯使用者（跑模型、改参数），本体层不重要，看 [for-beginners](/for-beginners) 就够。如果你是研究员、想扩展知识库、或做二次开发，本体层价值很大。

---

## 下一步

- 想快速上手 → [for-beginners](/for-beginners)
- 想系统学习 → [mid-level-path](/mid-level-path)
- 想查术语 → [glossary](/glossary)
- 想在浏览器内试模型 → [interactive/compare](/interactive/compare)
- 想看版本差异 → [versions/deltas](/versions/deltas)
- 想贡献内容 → [CONTRIBUTING](/CONTRIBUTING)
- 想看本体细节 → [ontology/schema](/ontology/schema)
- 想提意见 → [GitHub Issues](https://github.com/yejinlei/agent-yolo/issues)
