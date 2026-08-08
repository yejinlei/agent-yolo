# 小白入门路线 · 5 页看完 YOLO 家族

> 你是刚入门的 CV 小白？这一页给你一条最短的阅读路径。跟着五段走下来，你能看懂后面任何一版 YOLO 的 index 页。
>
> **前置要求**：高中数学 + Python 语法基础。**不要**先看 [术语表](/glossary)——那本书适合"已经入门想查"的人。

---

## 第 1 页 · 目标检测到底在做什么

一句话：**给一张图，模型输出一堆矩形框，每个框告诉你"这是什么、在哪里、有多确定"**。

```
输入：一张 640×640 的照片
输出：[
  {box: [x1,y1,x2,y2], class: "dog",   score: 0.92},
  {box: [x1,y1,x2,y2], class: "cat",   score: 0.87},
  {box: [x1,y1,x2,y2], class: "person",score: 0.78},
  ...
]
```

**三个必须知道的东西**：

1. **bbox（边界框）**：一个矩形，用 `(x, y, w, h)` 表示中心 + 宽高
2. **IoU（交并比）**：两个框重叠程度，0~1，衡量"预测的框和真值框准不准"
3. **NMS（非极大值抑制）**：同一物体报了多次 → 只留分最高那个框，其它压掉

想深入就跳：[术语表·检测基础](/glossary#检测基础) · [v1·concepts](/versions/v1/concepts)

---

## 第 2 页 · 从 v1 到 v8 的六个关键转折

把主线 8 个版本当成一条演化链，每一代只记住**"上一版的一个痛点 + 本版如何解决"**：

| 版本 | 年份 | 一句话概括 | 上一版痛点 |
| --- | --- | --- | --- |
| **v1** | 2015 | **single-shot**：一次前向出所有框 | 两阶段检测太慢 |
| **v2** | 2016 | 加 **anchor** + **BN** + log 回归 | v1 小目标召回差 |
| **v3** | 2018 | 三尺度 **FPN** + **Darknet-53** + logistic 类头 | v2 只有单一尺度 |
| **v4** | 2020 | **CSPDarknet53** + **SPP** + **PANet** + **CIoU** + Mosaic | v3 融合弱、增强少 |
| **v5** | 2020 | darknet C 切到 **PyTorch**，yaml 配置 | v4 训练流程复杂 |
| **v6** | 2022 | **RepVGG** 重参数化 + **anchor-free** + **TAL** + **DFL** | v5 部署慢、正样本分配几何化 |
| **v7** | 2022 | **trainable bag-of-freebies** + **E-ELAN** + compound scaling | v6 只报自家数据集 |
| **v8** | 2023 | **C2f + SPPF + Detect + TAL + DFL** → 工业级归一 | v5/v6/v7 三流派配方不统一 |

**记不住没关系，只记住一条主线**：v1 打基础 → v3/v4 加多尺度融合 → v5 转 PyTorch → v6 上 TAL/DFL → **v8 是"默认答案"**。

深入跳：[v1](/versions/v1/index) · [v3](/versions/v3/index) · [v4](/versions/v4/index) · [v5](/versions/v5/index) · [v6](/versions/v6/index) · [v7](/versions/v7/index) · [v8](/versions/v8/index)

---

## 第 3 页 · YOLOv8 拆解（重点，工业主力）

**v8 是 2023 至今的默认答案**，学会它就能读 v9~v13。四段结构：

```
输入 640×640
  ↓
① Backbone（骨干）：Stem + 4 个 C2f + SPPF
  - C2f：CSP 变体，多条分支细粒度融合
  - SPPF：空间金字塔池化融合，把感受野扩到 5/9/13/17
  ↓
② Neck（颈）：FPN + PANet 双向融合
  - 三个尺度：P3 (小目标) / P4 / P5 (大目标)
  ↓
③ Detect 头（解耦头）：
  - cv2 → 框回归（box + DFL 分布）
  - cv3 → 分类
  ↓
④ 损失：TAL 任务对齐 + DFL 分布焦点
```

**四个必备术语**：

- **C2f**：CSP 的更细粒度版本，v8 骨干主力模块
- **SPPF**：串联式空间金字塔池化，感受野扩得多但参数少
- **TAL**：任务对齐学习，"分类难度 × 回归难度"决定谁是正样本
- **DFL**：把框边长预测当分布来学，比直接回归精度更高

跳：[v8·index](/versions/v8/index) · [v8·architecture](/versions/v8/architecture) · [术语表·C2f](/glossary#c2f-csp-变体) · [TAL](/glossary#tal-任务对齐学习) · [DFL](/glossary#dfl-分布焦点损失)

---

## 第 4 页 · v9~v14 各记一句

从 v8 之后，每一版只做**一件事**——记住这个"只做一件事"就够了：

| 版本 | 年份 | 只做一件事 | 一句吐槽 |
| --- | --- | --- | --- |
| **v9** | 2024 | **PGI + GELAN**：训练时多学、推理不亏 | "训练时多花点成本换精度" |
| **v10** | 2024 | **一致性双分配 → NMS-free** | "训练时就没让重复框出来" |
| **v11** | 2024 | **C3k2 + C2PSA + 单头 Detect** | "骨干换配方，检测头再简化" |
| **v12** | 2024 | **A2C2f 内嵌注意力** + 去 SPPF | "把注意力塞进 C2f" |
| **v13** | 2025 | **HyperACE 超图融合** | "普通图不够，用超图" |
| **v14** | 2026 | **DeformableAAttn + ViewEmbedding + DAL + SphereAAttn** | "鱼眼/无人机/全景/游戏画面全吃" |

**衍生分支**（不属于主线，但很热门）：

- **YOLO-World**（2024）：换类别不用重训——文本当标签
- **YOLOE**（2025）：文本 / 视觉 / 无 prompt 三合一
- **YOLO-Master**（2026）：MoE 稀疏专家 + Transformer 专家

跳：[v9](/versions/v9/index) · [v10](/versions/v10/index) · [v11](/versions/v11/index) · [v12](/versions/v12/index) · [v13](/versions/v13/index) · [v14](/versions/v14/index) · [衍生分支总览](/derivatives/index)

---

## 第 5 页 · 你该做什么？（按场景选路）

**先答三个问题**：

1. **你要做什么？** 学术研究 / 生产部署 / 学习
2. **要检测什么？** 固定类别 / 任意类别（含长尾）
3. **要跑在哪？** GPU 服务器 / 边缘设备（Jetson / 手机）

**推荐路径**：

| 场景 | 起点 | 目标版本 |
| --- | --- | --- |
| 🎓 **学习** | 读 [v1·index](/versions/v1/index) → v2 → v3 → **v8** | 停在 v8，其它版本按需查 |
| 🏭 **生产部署（固定类别）** | 直接看 [v8·index](/versions/v8/index) | **v8** 或 **v11**（更瘦） |
| 📱 **边缘设备（Jetson / 手机）** | 读 [v8](/versions/v8/index) 后跳 [v11](/versions/v11/index) | **v11-n/s** 或 **v12** |
| 🎯 **换类别不想重训** | 读 [YOLOv8](/versions/v8/index) → [YOLO-World](/derivatives/yoloworld/index) | **YOLO-Worldv2** 或 **YOLOE** |
| 🏆 **精度优先（学术评测）** | 读 [v11](/versions/v11/index) → [v12](/versions/v12/index) → [v13](/versions/v13/index) | **v13** / **v14** / **YOLO-Master** |
| 🔬 **研究创新点** | 先看 [系列家族](/global/family) → [演化事件](/global/evolutions) | 挑感兴趣的版本深挖 |

**读完这 5 页之后**：

- 想查某个术语 → [术语表](/glossary)（先看顶部的 **10 个核心术语卡片** 和 **8 个高频类比**）
- 想看完整时间线 → [时间线](/global/timeline)
- 想看全景 → [系列家族](/global/family)
- 想看**版本对比矩阵**（谁强在哪、谁弱在哪） → [对比矩阵](/compare-matrix)
- 想看**演化故事**（每个版本解决了什么问题） → [演化故事](/evolution-story)
- 想看**基准数据**（COCO AP、延迟、量化、TCO 六层数字） → [基准数据](/benchmarks)
- 想看**数据流全景**（一张图看懂 YOLO 家族的数据依赖） → [数据流](/data-flow)
- 想动手跑起来 → [实战案例](/cases/index)
- 想**按算力/精度/场景直接选**（比翻 5 页更快） → [选型决策树](/choose-a-model)

---

## 常见问答

**Q：我完全没入门，应该从哪儿开始？**
A：先看这一页的**第 1 页**（目标检测是什么），再看**第 2 页**（版本演化），然后**跳去 v8**，中间版本可以跳过。

**Q：v8、v11、v12、v13 我该选哪个？**
A：**学习选 v8**（文档最全、社区最大）；**生产选 v8 或 v11**（v11 更瘦，v8 最稳）；**要 SOTA 精度选 v13**（Apache-2.0 可商用）。

**Q：v14、YOLO26、YOLO-Master 是不是官方版？**
A：**v14 是第三方学术 fork**（iMoonLab），非 Ultralytics 官方；**YOLO26**（2026-01）是 Ultralytics 官方最新版；**YOLO-Master** 是腾讯学术线，未合并进 Ultralytics 主线。看 [系列家族](/global/family) 里的"官方线 vs 学术线"标签。

**Q：衍生分支我该看哪个？**
A：**只想换类别就选 YOLO-World**；**要"检测任何东西"选 YOLOE**；**追求多数据集 SOTA 选 YOLO-Master**。三个都是"闭集 YOLO 的扩展"，先看 [衍生分支总览](/derivatives/index) 再决定。
