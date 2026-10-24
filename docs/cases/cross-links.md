# 案例 × 对比矩阵 一致性报告

> 由 `scripts/audit-cross-links.js` 自动生成，请勿手改。
> 最后更新：2026-10-25

## 总览

| 指标 | 数值 |
|------|------|
| 案例总数 | 12 |
| 被 cases/compare.md 引用 | 12 / 12 |
| 被 compare-matrix.md 引用 | 12 / 12 |
| 引用 /cases/compare 的案例 | 2 / 12 |
| 引用 /compare-matrix 的案例 | 0 / 12 |
| 横向引用其它案例的案例 | 11 / 12 |
| 关键版本在 compare-matrix 中出现 | 15 / 16 |
| 关键维度双方共现 | 8 / 11 |
| **告警（warn）** | **10** |
| 提示（info） | 1 |

---

## 案例 × 双向链接矩阵

| 案例 | 在 cases/compare.md | 在 compare-matrix.md | 引用 /cases/compare | 引用 /compare-matrix | 横向引用其它案例数 | 提到版本 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| [3d-bev](/cases/3d-bev) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v10 v11 |
| [agriculture](/cases/agriculture) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v8 |
| [autonomous-lane](/cases/autonomous-lane) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v8 v10 v11 v26 |
| [drone-obb](/cases/drone-obb) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v5 v8 v11 |
| [edge-deployment](/cases/edge-deployment) | ✅ | ✅ | — | — | 0 | v1 v5 v7 v8 v11 |
| [industrial-defect](/cases/industrial-defect) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v4 v8 v11 v13 |
| [medical-lesion](/cases/medical-lesion) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v4 v8 |
| [mot](/cases/mot) | ✅ | ✅ | ✅ | — | 5 | v8 |
| [multimodal](/cases/multimodal) | ✅ | ✅ | ✅ | — | 5 | v1 v2 v7 v8 |
| [retail-shelf](/cases/retail-shelf) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v4 v8 |
| [security-monitoring](/cases/security-monitoring) | ✅ | ✅ | — | — | 4 | v1 v2 v3 v8 |
| [video-stream](/cases/video-stream) | ✅ | ✅ | — | — | 5 | v8 v11 |

---

## 关键版本覆盖（compare-matrix.md）

| 版本 | 在 compare-matrix.md | 至少一个案例提到 |
|------|:---:|:---:|
| v1 | ✅ | ✅ |
| v2 | ✅ | ✅ |
| v3 | ✅ | ✅ |
| v4 | ✅ | ✅ |
| v5 | ✅ | ✅ |
| v6 | ✅ | — |
| v7 | ✅ | ✅ |
| v8 | ✅ | ✅ |
| v9 | ✅ | — |
| v10 | ✅ | ✅ |
| v11 | ✅ | ✅ |
| v12 | ✅ | — |
| v13 | ✅ | ✅ |
| v14 | ✅ | — |
| v26 | ✅ | ✅ |
| v27 | — | — |

---

## 关键维度覆盖（compare.md × compare-matrix.md）

| 维度 | 在 cases/compare.md | 在 compare-matrix.md |
|------|:---:|:---:|
| COCO AP | — | ✅ |
| COCO | — | ✅ |
| 参数量 | ✅ | ✅ |
| 延迟 | ✅ | ✅ |
| 量化 | ✅ | ✅ |
| INT8 | ✅ | ✅ |
| FP16 | ✅ | ✅ |
| 硬件 | ✅ | ✅ |
| 场景 | ✅ | ✅ |
| 精度 | ✅ | ✅ |
| 速度 | — | ✅ |

---

## 问题清单

- `⚠️` **3d-bev.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **agriculture.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **autonomous-lane.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **drone-obb.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **edge-deployment.md** · 未引用 /cases/compare 或 /compare-matrix
- `ℹ️` **edge-deployment.md** · 未横向引用其它案例
- `⚠️` **industrial-defect.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **medical-lesion.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **retail-shelf.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **security-monitoring.md** · 未引用 /cases/compare 或 /compare-matrix
- `⚠️` **video-stream.md** · 未引用 /cases/compare 或 /compare-matrix

---

## 修复建议

- **未被 cases/compare.md 引用**：在 compare.md 的选型速查表 / 场景分派表中补一行。
- **未被 compare-matrix.md 引用**：如案例不涉及版本对比，可忽略；否则在矩阵对应行补一句场景注。
- **未引用 /cases/compare 或 /compare-matrix**：在案例末尾「相关案例」或「深入阅读」中加横向链接。
- **提到版本 X 但 compare-matrix.md 无条目**：在 compare-matrix 中补该版本行，或从案例中移除版本提及。

---

**案例作者**：agent-yolo 文档组  
**最后更新**：2026-10-25  
