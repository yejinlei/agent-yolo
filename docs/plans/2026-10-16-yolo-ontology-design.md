# YOLO 本体论设计文档

> 创建日期：2026-10-16
> Schema 版本：v0.3.0
> 状态：v1 全实例 + VitePress 站点已上线

---

## 1. 设计目标

构建一个覆盖 YOLO v1-v12 全系列的**结构化知识图谱**，实现四层闭环：

```
L1 论文 (Paper)  →  L2 理论 (Concept)  →  L3 实现 (Model/Layer/Loss/Trick)  →  L4 实战 (CaseStudy)
     ↑                                                                                    │
     └────────────────── 反哺（PainPoint/Trick/Metric） ←──────────────────────────────────┘
```

### 1.1 与传统文档的区别

| 维度 | 传统文档 | 本体论 |
|------|---------|--------|
| 结构 | 线性文章 | 图谱（实体 + 关系） |
| 查询 | 全文搜索 | 实体 ID 查询 |
| 跨版本追踪 | 手动对比 | 关系遍历 |
| 可扩展性 | 改文章 | 加实体 + 加关系 |
| 机器可读 | 否 | YAML + JSON Schema |

### 1.2 与 OWL/RDF 的差异

| 维度 | OWL/RDF | 本体论 |
|------|---------|--------|
| 格式 | RDF Turtle / OWL XML | YAML + JSON Schema |
| 推理 | DL Reasoner | 无（关系显式声明） |
| 目标 | 机器推理 | 人类阅读 + 机器查询 |
| 复杂度 | 高 | 低 |

---

## 2. Schema 设计

### 2.1 实体类型（18 种）

按四层闭环分组：

| 层 | 类型 | ID 前缀 | 说明 |
|----|------|---------|------|
| L0 全局 | Family | `yolo-` | YOLO 家族 |
| L0 全局 | Evolution | `evo-` | 演化事件 |
| L0 全局 | Timeline | `tl-` | 时间线 |
| L1 论文 | Paper | `paper-` | 论文 |
| L2 理论 | Concept | `concept-` | 核心概念 |
| L3 实现 | Model | `yolo-v*` | 模型 |
| L3 实现 | Layer | `layer-` | 网络层 |
| L3 实现 | Loss | `loss-` | 损失函数 |
| L3 实现 | Trick | `trick-` | 训练/推理技巧 |
| L3 实现 | SourceFile | `src-` | 源码文件 |
| L3 实现 | Dataset | `ds-` | 数据集 |
| L3 实现 | Metric | `metric-` | 评价指标 |
| L4 实战 | CaseStudy | `case-` | 案例研究 |
| 外部 | ExternalTechnique | `ext-` | 传统 CV 技术 |
| 外部 | PainPoint | `pain-` | 痛点 |
| 辅助 | Person | `person-` | 人物 |
| 辅助 | Institution | `inst-` | 机构 |
| 辅助 | Tool | `tool-` | 工具框架 |

### 2.2 关系类型（27 种）

分 6 组：

| 组 | 关系 | 语义 |
|----|------|------|
| 四层闭环 | derivedFrom, usesConcept, implements, validates | L1→L2→L3→L4 |
| 组装 | hasLayer, usesLoss, usesTrick, hasBackbone | Model 组装 |
| 演化 | supersedes, forksFrom, outperforms, inspiredBy | 版本演化 |
| 源码 | definedIn, calls, references | 代码引用 |
| 外部 | borrowedFrom, adapts, replaces | 传统 CV 借鉴 |
| 痛点 | solvesPainPoint, exacerbatesPainPoint, exposesPainPoint | 痛点追踪 |

### 2.3 ID 命名约定

- 全部 kebab-case
- type 前缀 + 描述性名称
- 版本号用 `v1` / `v2`，不用 `v01`
- 示例：`yolo-v1`, `loss-coord-l2`, `trick-sqrt-one`, `pain-small-object-recall`

### 2.4 SourceRef（证据类型）

每个实体必须带 `source` 字段，支持 5 种证据：

```yaml
source:
  - kind: paper          # 论文引用
    quote: "..."
  - kind: code           # 源码引用
    path: "source/yolov1/snapshot/src/detection_layer.c"
    lines: "82-83"
  - kind: commit         # Git commit
    repo: "pjreddie/darknet"
    commit: "8322a58"
  - kind: url            # 外部 URL
    url: "https://arxiv.org/..."
  - kind: doc            # 内部文档
    path: "docs/versions/v1/layers.md"
```

---

## 3. 目录结构

```
agent-yolo/
├── ontology/
│   ├── schema/
│   │   └── yolo-ontology.schema.json    ← JSON Schema v0.3.0
│   └── instances/
│       ├── global/                      ← L0 全局实例
│       │   ├── family.yaml
│       │   └── evolutions.yaml
│       └── yolov1/                      ← v1 全套实例
│           ├── model.yaml               ← Model
│           ├── layers.yaml              ← Layer × 29
│           ├── losses.yaml              ← Loss × 6
│           ├── tricks.yaml              ← Trick × 7
│           ├── painpoints.yaml          ← PainPoint × 6
│           ├── concepts.yaml            ← Concept × 9
│           ├── papers.yaml              ← Paper × 4
│           ├── sources.yaml             ← SourceFile × 4
│           ├── datasets.yaml            ← Dataset
│           ├── metrics.yaml             ← Metric
│           ├── external.yaml            ← ExternalTechnique
│           └── relations.yaml           ← 关系实例
├── papers/
│   └── yolov1/
│       └── 2015-yolo-single-shot.md    ← 论文精读独立版
├── source/
│   └── yolov1/
│       ├── snapshot/                    ← darknet 源码快照
│       ├── README.md                    ← 源码概览
│       └── code-walkthrough.md          ← 源码走读
├── experiments/
│   └── README.md                        ← 实验骨架
├── docs/                                ← VitePress 站点
│   ├── index.md                         ← 首页
│   ├── global/                          ← L0 全局页 × 4
│   ├── versions/v1/                     ← v1 页 × 8
│   ├── ontology/                        ← 本体页 × 3
│   ├── cases/                           ← 实战页 × 2
│   ├── plans/                           ← 设计文档
│   │   └── 2026-10-16-yolo-ontology-design.md  ← 本文件
│   └── .vitepress/
│       ├── config.mjs                   ← VitePress 配置
│       └── public/
│           └── logo.svg                 ← 站点 logo
├── .github/workflows/
│   └── deploy.yml                       ← GitHub Actions 部署
└── package.json
```

---

## 4. 实现进度

### 4.1 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| Schema v0.3.0 | ✅ | 18 实体 + 27 关系 |
| YOLOv1 全实例 | ✅ | 12 个 YAML, 156 实体 |
| VitePress 站点 | ✅ | 17 页 |
| GitHub Pages 部署 | ✅ | <https://yejinlei.github.io/agent-yolo/> |
| 论文精读 | ✅ | VitePress + 独立版 |
| 源码走读 | ✅ | VitePress + 独立版 |
| 实验骨架 | ✅ | 3 个实验规划 |
| 设计文档 | ✅ | 本文件 |

### 4.2 待扩展

| 模块 | 优先级 | 说明 |
|------|--------|------|
| YOLOv2 全实例 | 高 | anchor + multi-scale training |
| YOLOv3 全实例 | 高 | FPN + 3 scale head |
| YOLOv4-v7 实例 | 中 | CSP + CIoU + Mosaic |
| YOLOv8-v11 实例 | 高 | anchor-free + DFL |
| YOLOv10-v12 实例 | 中 | NMS-free |
| 更多 CaseStudy | 中 | 工业质检 / 自动驾驶 |
| 中英双语 | 低 | 英文翻译 |

### 4.3 Schema 演进计划

| 版本 | 变更 |
|------|------|
| v0.1 | 初始：7 种实体 |
| v0.2 | 加 PainPoint + ExternalTechnique |
| v0.3.0 | 当前：18 种实体 + 27 种关系 + SourceRef 5 种 |
| v0.4 (计划) | 加 Person + Institution + Tool |
| v0.5 (计划) | 加自动校验脚本（YAML → Schema 验证） |

---

## 5. 设计决策记录

### 5.1 为什么用 YAML 而不是 JSON？

YAML 支持注释、多行字符串、引用，更适合人类编写和阅读。JSON Schema 用于验证。

### 5.2 为什么不用 Neo4j / 知识图谱数据库？

本项目目标是**可读性优先**——YAML + VitePress 让内容对人类友好，同时保持机器可查。不需要图数据库的查询能力（关系是显式声明的）。

### 5.3 为什么不用 Obsidian / Notion？

- Obsidian：双向链接好，但无法做结构化验证
- Notion：封闭格式，无法 git 版本控制
- 本体论：YAML + JSON Schema = 开放 + 可验证 + 可版本控制

### 5.4 为什么是 7×7 而不是其他尺寸？

这是 YOLOv1 的设计选择，不是本体论的选择。本体论忠实记录论文和源码的事实，不做价值判断。

---

## 6. 质量保障

### 6.1 数据一致性

- 每个 YAML 实例的 `source` 字段必须有至少一个证据
- VitePress 页面内容与 YAML 实例一一对应
- 跨版本引用的实体 ID 必须在对应 YAML 中存在

### 6.2 构建验证

```bash
# 本地构建验证
npm run docs:build

# 输出无 dead link 即通过
```

### 6.3 已知限制

- Schema 暂无自动校验脚本（计划 v0.5 加入）
- 跨版本关系（如 v3 `solvesPainPoint` v1 的 `pain-small-object`）目前手动维护
- 实验结果未实跑验证，数值来自论文引用
