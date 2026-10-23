# YOLO-Master 调研报告

> 调研日期：2026-10-23。全部结论均基于公开论文、GitHub API、官方站点交叉核验。

## 1. 论文与仓库

| 字段 | 值 |
|---|---|
| arXiv | **2512.23273**（v1: 2025-12-29，v2: 2025-12-30） |
| 标题 | YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection |
| 作者 | Xu Lin*、Jinlong Peng*、Zhenye Gan、Jiawen Zhu、Jun Liu（* 为共同一作） |
| 单位 | Tencent Youtu Lab、Singapore Management University |
| 发表 | CVPR 2026（论文页脚标注 accepted by CVPR 2026） |
| 官方仓库 | `https://github.com/Tencent/YOLO-Master`（主仓库；论文 arXiv 页脚另写 `isLinXu/YOLO-Master` 为个人 fork） |
| 主分支最新 commit（2026-10-22） | **`ca5a87ff6f11e2a3c1c7c4ebf282149bd2be1274`**（merge PR #319 codex/sparse-calibration-mps） |
| Release tag | `v26.08` → `1dc71f1da20424a10ebe186e16a7296528756643`；`YOLO-Master-v26.08` → `43d40117c30811204fb9347efeabddce15f11a62`；`YOLO-Master-v26.02` → `6bed010e3b0f67efbb735470b8c6e3cca65e4e33` |
| License | **AGPL-3.0**（`LICENSE` sha `be3f7b28…`） |
| 基座 | 论文 §3.1 明确 "builds upon recent YOLO architectures (e.g., YOLOv12)"；`pyproject.toml` 显示分发名仍为 `ultralytics`，目录树含完整 `ultralytics/`，属 **Ultralytics YOLOv12 fork**（未声明具体 v12 变体，从 N/S/M/L/X 尺寸矩阵看为全线覆盖） |

## 2. 架构核心创新

- **ES-MoE（Efficient Sparse MoE）Block**：插入 Backbone 与 Neck。专家为 depthwise separable conv，kernel 尺寸分别为 **3×3 / 5×5 / 7×7**，天然对应多尺度感受野；E ≫ K，输出经 Norm 后做加权聚合。
- **实例条件自适应计算（Instance-Conditional Adaptive Computation）**：不同样本按复杂度激活不同 expert 子集，打破静态 dense 计算。
- **动态路由网络（Dynamic Routing Network）**：轻量子网络从特征中提 routing 信号，经 Softmax Gating 产生专家权重。
- **Phased Routing（分阶段路由）**：训练用 **Soft Top-K** 保梯度流、推理用 **Hard Top-K** 保真实稀疏。
- **多样性增强目标 + 负载均衡**：Diversity Enhancing Objective 鼓励专家互补，Load Balancing Supervision 防止 expert collapse；论文 §3.5 描述完整 loss。
- **专家专用 Transformer（Specialized Transformers）**：标题层面的贡献，仓库另挂 `MoT_MoA_Ablation_Report.md`（Mixture-of-Transformers / Mixture-of-Attention）以及 `Mixture-of-Attention (MoA)` 模块。
- 附加能力：Cluster-Weighted NMS、Sparse SAHI（4K/8K 3–5× 加速）、LoRA 微调（~10% 可训练参数）。

## 3. 训练

- **数据**：MS COCO 主评测；扩展至 **PASCAL VOC、VisDrone、KITTI、SKU-110K** 五个大规模基准。
- **训练阶段**：论文以 "Phased Routing Strategy" 表述——**并非两阶段冻结专家→学路由**，而是**同一训练中路由 soft→hard 切换 + diversity/load-balancing 联合训练**（专家与 router 端到端一起学）。
- **关键超参**：`K ≪ E`（Top-K 稀疏）；kernel 3/5/7 三档专家；训练/推理 gating 语义分离。仓库 `configs/`、`experiments/`、`smoke/` 承载具体配置，`debug-router-nan-activations.md` 记录了 router NaN 激活的调试经验。
- **依赖**：`torch≥1.8.0`、`torchvision≥0.9.0`、`peft>=0.18.0,<0.20.0`（LoRA 路径，唯一带上界）。

## 4. 性能（论文 §4，仅摘要级数据）

| 指标 | YOLO-Master-N（MS COCO） |
|---|---|
| AP | **42.4** |
| Latency | **1.62 ms** |
| vs YOLOv13-N | +0.8 mAP、快 17.8% |
| vs YOLOv12 | +1.8 mAP |
| 参数量 / FLOPs | **论文未给出**（摘要与首页均无 params / GFLOPs 表） |

⚠️ 论文仅给出 headline 数字，未在正文披露 GPU/分辨率/batch/精度设置；仓库 wiki `MoE_Modules_Explanation_EN.md` 承载部署细节。

## 5. 与主线的关系

- **属 Ultralytics YOLOv12 fork**：`pyproject.toml` 中 `name="ultralytics"`、`description="Ultralytics YOLO 🚀 …"`，仓库根目录含完整 `ultralytics/` 源码树，YOLO-Master 独有资产位于 `agent/ configs/ benchmarks/ experiments/ tools/ model-zoo/ studio/ frontend/ web/` 等并行目录。
- **未被 Ultralytics 主线吸收**：`Ultralytics/ultralytics` 主干目前无 YOLO-Master 相关 PR/merge；本仓库的 `ultralytics/` 与主线独立演化。
- 分发风险：安装此仓库产出 pip 包名为 `ultralytics`，若与官方 ultralytics 共存于同一环境会冲突。

## 6. 限制与痛点

1. **推理稀疏收益依赖硬件**：Sparse routing 使理论 FLOPs 下降，但动态 top-k 在 CPU/普通 GPU 上因分支不可融合，实际 latency 需按硬件单独测。1.62 ms 未标注硬件。
2. **专家数膨胀带来部署复杂度**：多 kernel 专家组合 + router 需自定义算子，ONNX/TFLite/OpenVINO 导出路径需专门处理（仓库 `examples/` 22 个子目录含 ONNX Runtime / MNN / Triton / Rust ONNX / Axelera NPU）。
3. **训练稳定性问题**：仓库根目录提交 `debug-router-nan-activations.md`，说明 router 存在 NaN 激活风险，需 diversity + load balancing 双目标稳住。
4. **AGPL-3.0 传染性**：网络服务分发修改版需开源，商用受限于 copyleft。
5. **版本 tag 混乱**：`v26.08` 与 `YOLO-Master-v26.08` 是两个不同 commit，`v2026.02` 又是第四种命名。快照务必 pin SHA。

## 与 v13（A2C2f 全局注意力）路线对比建议

- **场景**：v13 的 A2C2f 走"全局注意力增强表征"，Master 走"输入条件化条件计算"。前者是通用性增强，后者在稠密小目标/复杂场景边际收益更大。
- **选择准则**：
  - 追求**通用精度 & 稳态部署** → 用 **YOLOv13**，主干与 Ultralytics 主线同步、AGPL 之外的分发路径清晰。
  - 追求**复杂/稠密场景 mAP 上限** + 可接受 MoE 部署复杂度 → 用 **YOLO-Master**。
  - 中长期看，MoE 在视觉检测侧尚未成主流，除非 v13 后续引入条件计算，否则两路线应并行维护，不建议把 Master 当主线迁移。

## 快照建议（Snapshots）

推荐 **两条线同时冻结**：

**主线快照（生产用）：`Tencent/YOLO-Master @ main`**
- Repo：`https://github.com/Tencent/YOLO-Master`
- Commit：`ca5a87ff6f11e2a3c1c7c4ebf282149bd2be1274`（2026-10-22，最新 push）
- 必拉文件：
  - `ultralytics/`（fork 后的检测主体）
  - `configs/`（模型结构 & 训练配置，含 MoE 变体）
  - `experiments/`（复现论文训练记录）
  - `benchmarks/`（性能测量脚本）
  - `wiki/MoE_Modules_Explanation_EN.md`（部署与路由机制细节）
  - `debug-router-nan-activations.md`（router 稳定性坑位备忘）
  - `README.md`、`README_CN.md`、`CITATION.cff`、`LICENSE`
  - `pyproject.toml`、`requirements.txt`、`requirements-a3-precision.txt`
  - `tests/`、`smoke/`（回归测试）
- 可跳过：`frontend/`、`web/`、`studio/`、`app.py`、`start_studio.py`、`docs.html`、`index.html`、`wiki-*.html`（Gradio / 站点构建产物，非运行时依赖）

**发布快照（可复现基线）：`YOLO-Master-v26.02` 或 `YOLO-Master-v26.08`**
- 用 `YOLO-Master-v26.02` → `6bed010e3b0f67efbb735470b8c6e3cca65e4e33`（论文对应的稳定 tag，2026-02-13）
- 或 `YOLO-Master-v26.08` → `43d40117c30811204fb9347efeabddce15f11a62`（2026-08-21，注意与 `v26.08` 是不同 commit）
- 若论文表格需要严格复现，建议先 pin `YOLO-Master-v26.02` 复跑，再对 main 分支做增量。

**验证命令**
```bash
git clone https://github.com/Tencent/YOLO-Master.git
cd YOLO-Master
git rev-parse ca5a87ff6f11e2a3c1c7c4ebf282149bd2be1274   # 应回显同 hash
git cat-file -t ca5a87ff6f11e2a3c1c7c4ebf282149bd2be1274 # → commit
```
