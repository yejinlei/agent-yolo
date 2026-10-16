# YOLO 本体论研究站

基于「论文 → 理论 → 实现 → 实战」四层闭环的 YOLO 系列知识图谱。

## 快速开始

```bash
# 安装依赖
npm install

# 本地预览
npm run docs:dev

# 构建静态站
npm run docs:build
```

## 部署到 GitHub Pages

用 GitHub Actions 自动部署，见 `.github/workflows/deploy.yml`。

## 目录

- `ontology/` — 本体 Schema + 各版本实例（YAML）
- `papers/` — 论文精读
- `source/` — 源码快照与走读
- `experiments/` — 实验脚本骨架
- `docs/` — VitePress 站点
- `site/` — 备用（暂未启用）
- `.github/workflows/` — CI/CD
