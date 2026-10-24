# 案例健康度报告 · 自动生成

> 运行 `node scripts/audit-cases.js` 重新生成。
> 最后更新：2026-10-25
> 扫描范围：`docs/cases/*.md`（排除 _template.md / index.md / compare.md） · 共 **12** 个案例页 · 平均得分 **100%**

## 总览

| 案例 | 得分 | 问题数 | 主要问题 |
|------|------|--------|----------|
| [3d-bev](/cases/3d-bev) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [agriculture](/cases/agriculture) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [autonomous-lane](/cases/autonomous-lane) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [drone-obb](/cases/drone-obb) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [edge-deployment](/cases/edge-deployment) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [industrial-defect](/cases/industrial-defect) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [medical-lesion](/cases/medical-lesion) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [mot](/cases/mot) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [multimodal](/cases/multimodal) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [retail-shelf](/cases/retail-shelf) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [security-monitoring](/cases/security-monitoring) | 🟢 7/7 | 0 | ✅ 全部通过 |
| [video-stream](/cases/video-stream) | 🟢 7/7 | 0 | ✅ 全部通过 |

## 问题清单

✅ 无问题，所有案例通过全部检查。

## 详情诊断

| 案例 | 最后更新 | 反哺信号 | 信号同步 | 横向对比 | sidebar | 死链 | 版本页 |
|------|---------|---------|---------|---------|---------|------|--------|
| [3d-bev](/cases/3d-bev) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [agriculture](/cases/agriculture) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [autonomous-lane](/cases/autonomous-lane) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [drone-obb](/cases/drone-obb) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [edge-deployment](/cases/edge-deployment) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [industrial-defect](/cases/industrial-defect) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [medical-lesion](/cases/medical-lesion) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [mot](/cases/mot) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [multimodal](/cases/multimodal) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [retail-shelf](/cases/retail-shelf) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [security-monitoring](/cases/security-monitoring) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [video-stream](/cases/video-stream) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 检查规则

1. **最后更新日期**：正文任意位置出现 `最后更新` + `YYYY-MM-DD` 视为通过。
2. **反哺信号章节**：存在 `## 反哺信号` 或包含"反哺信号"的 h2+ 标题。
3. **信号同步**：案例中的 `PainPoint / Trick / Metric` 信号名都出现在 [concepts-feedback.md](/concepts-feedback) 中（未同步视为漏跑 `sync-feedback.js`）。
4. **横向对比链接**：文件名在 [compare.md](/cases/compare) 或 [index.md](/cases) 中出现，或案例正文引用 `/cases/compare`。
5. **sidebar 注册**：`.vitepress/config.mjs` 的 sidebar 中注册了 `/cases/<name>`。
6. **内部链接**：Markdown 内部 `/xxx` 链接指向的 md 文件存在（外链跳过）。
7. **版本页存在**：文中提到的 YOLO 版本号（`YOLOvN` / `yoloN`）在 `/versions/vN/index.md` 存在对应页。

---

**得分**：总分 **84/84**（100%）
