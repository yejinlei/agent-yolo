import { defineConfig } from 'vitepress'
import markdownItKatex from '@traptitech/markdown-it-katex'

// @REASON: markdown-it-container 已被 VitePress 内部打包、无法直接 import，这里用零依赖的 mini block rule 实现
// 用法：::: figure 图注文字  ...（通常是 mermaid 块）...  :::
function figurePlugin(md) {
  const COLON = 0x3a
  function rule(state, startLine, endLine, silent) {
    const start = state.bMarks[startLine] + state.tShift[startLine]
    let pos = state.skipChars(start, COLON)
    if (pos - start < 3) return false
    const params = state.src.slice(pos, state.eMarks[startLine]).trim()
    if (!/^figure(\s|$)/.test(params)) return false
    if (silent) return true

    let nextLine = startLine + 1
    let haveEnd = false
    for (; nextLine < endLine; nextLine++) {
      const mem = state.bMarks[nextLine] + state.tShift[nextLine]
      if (state.src.charCodeAt(mem) !== COLON) continue
      const last = state.skipChars(mem, COLON)
      if (last - mem >= 3 && state.src.slice(last, state.eMarks[nextLine]).trim() === '') {
        haveEnd = true
        break
      }
    }
    if (!haveEnd) nextLine = endLine

    const title = params.replace(/^figure\s*/, '').trim()
    const tokOpen = state.push('diagram_figure_open', 'figure', 1)
    tokOpen.info = title
    tokOpen.block = true
    tokOpen.map = [startLine, nextLine]
    state.md.block.tokenize(state, startLine + 1, nextLine)
    const tokClose = state.push('diagram_figure_close', 'figure', -1)
    tokClose.block = true
    state.line = nextLine + (haveEnd ? 1 : 0)
    return true
  }
  md.block.ruler.before('fence', 'diagram_figure', rule, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.diagram_figure_open = (tokens, idx) => {
    const title = tokens[idx].info
    const cap = title ? `<figcaption>${md.utils.escapeHtml(title)}</figcaption>` : ''
    return `<figure class="diagram-figure">${cap}`
  }
  md.renderer.rules.diagram_figure_close = () => '</figure>'
}

export default defineConfig({
  title: 'agent-yolo',
  description: 'YOLO 系列本体论研究站 — 论文 → 理论 → 实现 → 实战',
  lang: 'zh-CN',
  base: '/agent-yolo/',

  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: '首页', link: '/' },
      { text: '小白入门', link: '/for-beginners' },
      { text: '全局视图', items: [
        { text: '系列家族', link: '/global/family' },
        { text: '演进故事', link: '/evolution-story' },
        { text: '对比矩阵', link: '/compare-matrix' },
        { text: '演化事件', link: '/global/evolutions' },
        { text: '时间线', link: '/global/timeline' },
        { text: '传统 CV 谱系', link: '/global/external-techniques' }
      ]},
      { text: 'YOLO 主线', items: [
        { text: 'YOLOv1', link: '/versions/v1/index' },
        { text: 'YOLOv3', link: '/versions/v3/index' },
        { text: 'YOLOv4', link: '/versions/v4/index' },
        { text: 'YOLOv5', link: '/versions/v5/index' },
        { text: 'YOLOv8', link: '/versions/v8/index' },
        { text: 'YOLOv11', link: '/versions/v11/index' },
        { text: 'YOLOv13', link: '/versions/v13/index' },
        { text: 'YOLOv14 (Omni)', link: '/versions/v14/index' },
        { text: 'YOLO26', link: '/versions/v26/index' },
        { text: 'YOLO27 (Preview)', link: '/models/v27-preview' }
      ]},
      { text: '衍生分支', items: [
        { text: '总览', link: '/derivatives/index' },
        { text: 'YOLO-World', link: '/derivatives/yoloworld/index' },
        { text: 'YOLOE', link: '/derivatives/yoloe/index' },
        { text: 'YOLO-Master', link: '/derivatives/master/index' },
        { text: '路线预告', link: '/derivatives/roadmap' }
      ]},
      { text: '任务', items: [
        { text: '实例分割', link: '/tasks/segment' },
        { text: '姿态估计', link: '/tasks/pose' },
        { text: '定向框 OBB', link: '/tasks/obb' },
        { text: '旋转检测', link: '/tasks/rotated' },
        { text: '单目深度估计', link: '/tasks/depth' }
      ]},
      { text: '实操', items: [
        { text: '训练与微调', link: '/recipes/train' },
        { text: 'ONNX / TensorRT 部署', link: '/recipes/export-onnx' },
        { text: '量化深度指南', link: '/recipes/quantization' },
        { text: '边缘设备实战', link: '/recipes/edge-deploy' },
        { text: '数据标注避坑', link: '/recipes/data-annotation' },
        { text: '高级训练技巧', link: '/recipes/advanced-train' },
        { text: '生产化落地', link: '/recipes/production' },
        { text: '交互式对比（浏览器 ONNX）', link: '/interactive/compare' }
      ]},
      { text: '问答', link: '/faq' },
      { text: '翻车案例', link: '/bad-cases' },
      { text: '进阶路线', link: '/mid-level-path' },
      { text: '横向使用', items: [
        { text: '选型决策树', link: '/choose-a-model' },
        { text: '基准数据', link: '/benchmarks' },
        { text: '数据流全景', link: '/data-flow' },
        { text: '故障排查', link: '/troubleshooting' },
        { text: '概念反向索引', link: '/concepts' },
        { text: '术语表', link: '/glossary' }
      ]},
      { text: '实战案例', link: '/cases' },
      { text: '即用配方', link: '/solutions/recipes' },
      { text: '本体', link: '/ontology/schema' },
      { text: '论文索引', link: '/references' },
      { text: '贡献指南', link: '/CONTRIBUTING' },
      { text: '反哺信号', link: '/concepts-feedback' },
      { text: '版本差异卡片', link: '/versions/deltas' },
      { text: '关于', link: '/about' },
      { text: 'GitHub', link: 'https://github.com/yejinlei/agent-yolo' }
    ],

    sidebar: {
      '/': [
        {
        text: '入口',
        items: [
          { text: '首页', link: '/' },
          { text: '小白入门', link: '/for-beginners' },
          { text: '进阶学习路线', link: '/mid-level-path' },
          { text: '实战案例总览', link: '/cases' },
          { text: '即用配方（Solutions）', link: '/solutions/recipes' },
          { text: '翻车案例集', link: '/bad-cases' }
        ]
      },
        {
          text: '全局视图',
          items: [
            { text: '系列家族', link: '/global/family' },
            { text: '演进故事', link: '/evolution-story' },
            { text: '对比矩阵', link: '/compare-matrix' },
            { text: '演化事件', link: '/global/evolutions' },
            { text: '时间线', link: '/global/timeline' },
            { text: '传统 CV 谱系', link: '/global/external-techniques' }
          ]
        },
        {
          text: '任务扩展',
          items: [
            { text: '实例分割 Segment', link: '/tasks/segment' },
            { text: '姿态估计 Pose', link: '/tasks/pose' },
            { text: '定向框 OBB', link: '/tasks/obb' },
            { text: '旋转检测 Rotated', link: '/tasks/rotated' }
          ]
        },
        {
          text: '实操手册',
          items: [
            { text: '训练与微调', link: '/recipes/train' },
            { text: 'ONNX / TensorRT 部署', link: '/recipes/export-onnx' }
          ]
        },
        {
          text: '横向使用页',
          items: [
            { text: '选型决策树', link: '/choose-a-model' },
            { text: '基准数据', link: '/benchmarks' },
            { text: '数据流全景', link: '/data-flow' },
            { text: '故障排查', link: '/troubleshooting' },
            { text: '概念反向索引', link: '/concepts' },
            { text: '反哺信号汇总', link: '/concepts-feedback' },
            { text: '常见问题 FAQ', link: '/faq' }
          ]
        },
        {
          text: '交互式',
          items: [
            { text: '交互式模型对比（浏览器 ONNX）', link: '/interactive/compare' }
          ]
        },
        {
        text: '参考',
        items: [
          { text: '术语表', link: '/glossary' },
          { text: '论文与仓库索引', link: '/references' },
          { text: '本体 Schema', link: '/ontology/schema' },
          { text: '编写新页指南', link: '/ontology/authoring' },
          { text: '贡献指南（CONTRIBUTING）', link: '/CONTRIBUTING' },
          { text: '关于本项目', link: '/about' }
        ]
      }
      ],
      '/global/': [
        {
          text: '全局视图',
          items: [
            { text: '系列家族', link: '/global/family' },
            { text: '演进故事', link: '/evolution-story' },
            { text: '对比矩阵', link: '/compare-matrix' },
            { text: '演化事件', link: '/global/evolutions' },
            { text: '时间线', link: '/global/timeline' },
            { text: '传统 CV 谱系', link: '/global/external-techniques' }
          ]
        },
        {
          text: '横向使用页',
          items: [
            { text: '选型决策树', link: '/choose-a-model' },
            { text: '基准数据', link: '/benchmarks' },
            { text: '数据流全景', link: '/data-flow' },
            { text: '故障排查', link: '/troubleshooting' },
            { text: '概念反向索引', link: '/concepts' }
          ]
        }
      ],
      '/versions/': [
        {
          text: 'YOLOv1 (2015)',
          items: [
            { text: '概览', link: '/versions/v1/index' },
            { text: '论文精读', link: '/versions/v1/paper' },
            { text: 'Architecture 逐层图解', link: '/versions/v1/architecture' },
            { text: '理论 (Concept)', link: '/versions/v1/concepts' },
            { text: '架构与层速查', link: '/versions/v1/layers' },
            { text: '损失函数', link: '/versions/v1/losses' },
            { text: '反向传播推导', link: '/versions/v1/loss-gradient' },
            { text: '训练技巧', link: '/versions/v1/tricks' },
            { text: '痛点追踪', link: '/versions/v1/painpoints' },
            { text: '源码走读', link: '/versions/v1/code-walkthrough' }
          ]
        },
        {
          text: '源码浏览（darknet 快照 · c7079e4）',
          collapsed: false,
          items: [
            { text: 'detection_layer.c', link: '/versions/v1/source/detection-layer' },
            { text: 'yolo.c', link: '/versions/v1/source/yolo' },
            { text: 'box.c', link: '/versions/v1/source/box' },
            { text: 'yolo.cfg', link: '/versions/v1/source/yolo-cfg' }
          ]
        },
        {
          text: 'YOLOv2 (2017-12)',
          items: [
            { text: '概览', link: '/versions/v2/index' },
            { text: '论文精读', link: '/versions/v2/paper' },
            { text: 'Architecture 逐层图解', link: '/versions/v2/architecture' },
            { text: '理论 (Concept)', link: '/versions/v2/concepts' },
            { text: '架构与层速查', link: '/versions/v2/layers' },
            { text: '损失函数', link: '/versions/v2/losses' },
            { text: '反向传播推导', link: '/versions/v2/loss-gradient' },
            { text: '训练技巧', link: '/versions/v2/tricks' },
            { text: '痛点追踪', link: '/versions/v2/painpoints' },
            { text: '源码走读', link: '/versions/v2/code-walkthrough' }
          ]
        },
        {
          text: '源码浏览（darknet 快照 · c6afc7f）',
          collapsed: true,
          items: [
            { text: 'region_layer.c', link: '/versions/v2/source/region-layer' },
            { text: 'yolo.c', link: '/versions/v2/source/yolo' },
            { text: 'box.c', link: '/versions/v2/source/box' },
            { text: 'yolo.cfg', link: '/versions/v2/source/yolo-cfg' }
        ]
      },
      {
        text: 'YOLOv3 (2018-04)',
        items: [
          { text: '概览', link: '/versions/v3/index' },
          { text: '论文精读', link: '/versions/v3/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v3/architecture' },
          { text: '理论 (Concept)', link: '/versions/v3/concepts' },
          { text: '架构与层速查', link: '/versions/v3/layers' },
          { text: '损失函数', link: '/versions/v3/losses' },
          { text: '反向传播推导', link: '/versions/v3/loss-gradient' },
          { text: '训练技巧', link: '/versions/v3/tricks' },
          { text: '痛点追踪', link: '/versions/v3/painpoints' },
          { text: '源码走读', link: '/versions/v3/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（darknet 快照 · e84933b）',
        collapsed: true,
        items: [
          { text: 'yolo_layer.c', link: '/versions/v3/source/yolo-layer' },
          { text: 'yolo.c', link: '/versions/v3/source/yolo' },
          { text: 'box.c', link: '/versions/v3/source/box' },
          { text: 'yolov3.cfg', link: '/versions/v3/source/yolo-cfg' }
        ]
      },
      {
        text: 'YOLOv4 (2020-04)',
        items: [
          { text: '概览', link: '/versions/v4/index' },
          { text: '论文精读', link: '/versions/v4/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v4/architecture' },
          { text: '理论 (Concept)', link: '/versions/v4/concepts' },
          { text: '架构与层速查', link: '/versions/v4/layers' },
          { text: '损失函数', link: '/versions/v4/losses' },
          { text: '反向传播推导', link: '/versions/v4/loss-gradient' },
          { text: '训练技巧', link: '/versions/v4/tricks' },
          { text: '痛点追踪', link: '/versions/v4/painpoints' },
          { text: '源码走读', link: '/versions/v4/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（darknet 快照 · 4131ae1）',
        collapsed: true,
        items: [
          { text: 'yolo_layer.c', link: '/versions/v4/source/yolo-layer' },
          { text: 'yolo.c', link: '/versions/v4/source/yolo' },
          { text: 'box.c', link: '/versions/v4/source/box' },
          { text: 'yolov4.cfg', link: '/versions/v4/source/yolo-cfg' }
        ]
      },
      {
        text: 'YOLOv5 (2021-04)',
        items: [
          { text: '概览', link: '/versions/v5/index' },
          { text: '论文精读', link: '/versions/v5/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v5/architecture' },
          { text: '理论 (Concept)', link: '/versions/v5/concepts' },
          { text: '架构与层速查', link: '/versions/v5/layers' },
          { text: '损失函数', link: '/versions/v5/losses' },
          { text: '反向传播推导', link: '/versions/v5/loss-gradient' },
          { text: '训练技巧', link: '/versions/v5/tricks' },
          { text: '痛点追踪', link: '/versions/v5/painpoints' },
          { text: '源码走读', link: '/versions/v5/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（PyTorch 快照 · f5b8f7d）',
        collapsed: true,
        items: [
          { text: 'models/yolo.py', link: '/versions/v5/source/yolo-py' },
          { text: 'models/common.py', link: '/versions/v5/source/common-py' },
          { text: 'utils/loss.py', link: '/versions/v5/source/loss-py' },
          { text: 'models/yolov5s.yaml', link: '/versions/v5/source/yolov5s-yaml' }
        ]
      },
      {
        text: 'YOLOv6 (2022-06)',
        items: [
          { text: '概览', link: '/versions/v6/index' },
          { text: '论文精读', link: '/versions/v6/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v6/architecture' },
          { text: '理论 (Concept)', link: '/versions/v6/concepts' },
          { text: '架构与层速查', link: '/versions/v6/layers' },
          { text: '损失函数', link: '/versions/v6/losses' },
          { text: '反向传播推导', link: '/versions/v6/loss-gradient' },
          { text: '训练技巧', link: '/versions/v6/tricks' },
          { text: '痛点追踪', link: '/versions/v6/painpoints' },
          { text: '源码走读', link: '/versions/v6/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（PyTorch 快照 · Meituan main）',
        collapsed: true,
        items: [
          { text: 'yolov6/layers/common.py', link: '/versions/v6/source/common-py' },
          { text: 'yolov6/models/losses/loss.py', link: '/versions/v6/source/loss-py' },
          { text: 'yolov6/assigners/tal_assigner.py', link: '/versions/v6/source/tal-assigner-py' },
          { text: 'configs/yolov6s.py', link: '/versions/v6/source/yolov6s-config' }
        ]
      },
      {
        text: 'YOLOv7 (2022-07)',
        items: [
          { text: '概览', link: '/versions/v7/index' },
          { text: '论文精读', link: '/versions/v7/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v7/architecture' },
          { text: '理论 (Concept)', link: '/versions/v7/concepts' },
          { text: '架构与层速查', link: '/versions/v7/layers' },
          { text: '损失函数', link: '/versions/v7/losses' },
          { text: '反向传播推导', link: '/versions/v7/loss-gradient' },
          { text: '训练技巧', link: '/versions/v7/tricks' },
          { text: '痛点追踪', link: '/versions/v7/painpoints' },
          { text: '源码走读', link: '/versions/v7/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（PyTorch 快照 · WongKinYiu main · a207844）',
        collapsed: true,
        items: [
          { text: 'models/common.py', link: '/versions/v7/source/common-py' },
          { text: 'models/yolo.py', link: '/versions/v7/source/yolo-py' },
          { text: 'utils/loss.py', link: '/versions/v7/source/loss-py' },
          { text: 'cfg/training/yolov7.yaml', link: '/versions/v7/source/yolov7-yaml' }
        ]
      },
      {
        text: 'YOLOv8 (2023-01)',
        items: [
          { text: '概览', link: '/versions/v8/index' },
          { text: '论文精读', link: '/versions/v8/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v8/architecture' },
          { text: '理论 (Concept)', link: '/versions/v8/concepts' },
          { text: '架构与层速查', link: '/versions/v8/layers' },
          { text: '损失函数', link: '/versions/v8/losses' },
          { text: '反向传播推导', link: '/versions/v8/loss-gradient' },
          { text: '训练技巧', link: '/versions/v8/tricks' },
          { text: '痛点追踪', link: '/versions/v8/painpoints' },
          { text: '源码走读', link: '/versions/v8/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（ultralytics 快照 · v8.0.100 · dce4efc）',
        collapsed: true,
        items: [
          { text: 'ultralytics/nn/tasks.py', link: '/versions/v8/source/tasks-py' },
          { text: 'ultralytics/nn/modules/block.py', link: '/versions/v8/source/common-py' },
          { text: 'ultralytics/yolo/utils/loss.py', link: '/versions/v8/source/loss-py' },
          { text: 'ultralytics/models/v8/yolov8.yaml', link: '/versions/v8/source/yolov8-yaml' }
        ]
      },
      {
        text: 'YOLOv9 (2024-02)',
        items: [
          { text: '概览', link: '/versions/v9/index' },
          { text: '论文精读', link: '/versions/v9/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v9/architecture' },
          { text: '理论 (Concept)', link: '/versions/v9/concepts' },
          { text: '架构与层速查', link: '/versions/v9/layers' },
          { text: '损失函数', link: '/versions/v9/losses' },
          { text: '反向传播推导', link: '/versions/v9/loss-gradient' },
          { text: '训练技巧', link: '/versions/v9/tricks' },
          { text: '痛点追踪', link: '/versions/v9/painpoints' },
          { text: '源码走读', link: '/versions/v9/code-walkthrough' }
        ]
      },
      {
        text: 'YOLOv10 (2024-05)',
        items: [
          { text: '概览', link: '/versions/v10/index' },
          { text: '论文精读', link: '/versions/v10/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v10/architecture' },
          { text: '理论 (Concept)', link: '/versions/v10/concepts' },
          { text: '架构与层速查', link: '/versions/v10/layers' },
          { text: '损失函数', link: '/versions/v10/losses' },
          { text: '反向传播推导', link: '/versions/v10/loss-gradient' },
          { text: '训练技巧', link: '/versions/v10/tricks' },
          { text: '痛点追踪', link: '/versions/v10/painpoints' },
          { text: '源码走读', link: '/versions/v10/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（ultralytics fork 快照 · 8d9f45a）',
        collapsed: true,
        items: [
          { text: 'nn/modules/head.py', link: '/versions/v10/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v10/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v10/source/tal-py' },
          { text: 'cfg/models/v10/yolov10n.yaml', link: '/versions/v10/source/yolov10-yaml' }
        ]
      },
      {
        text: 'YOLOv11 (2024-09)',
        items: [
          { text: '概览', link: '/versions/v11/index' },
          { text: '论文精读', link: '/versions/v11/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v11/architecture' },
          { text: '理论 (Concept)', link: '/versions/v11/concepts' },
          { text: '架构与层速查', link: '/versions/v11/layers' },
          { text: '损失函数', link: '/versions/v11/losses' },
          { text: '反向传播推导', link: '/versions/v11/loss-gradient' },
          { text: '训练技巧', link: '/versions/v11/tricks' },
          { text: '痛点追踪', link: '/versions/v11/painpoints' },
          { text: '源码走读', link: '/versions/v11/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（ultralytics 快照 · tag v8.3.0 @ 6e43d1e1）',
        collapsed: true,
        items: [
          { text: 'nn/modules/head.py', link: '/versions/v11/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v11/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v11/source/tal-py' },
          { text: 'cfg/models/11/yolo11.yaml', link: '/versions/v11/source/yolo11-yaml' }
        ]
      },
      {
        text: 'YOLOv12 (2025-02)',
        items: [
          { text: '概览', link: '/versions/v12/index' },
          { text: '论文精读', link: '/versions/v12/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v12/architecture' },
          { text: '理论 (Concept)', link: '/versions/v12/concepts' },
          { text: '架构与层速查', link: '/versions/v12/layers' },
          { text: '损失函数', link: '/versions/v12/losses' },
          { text: '反向传播推导', link: '/versions/v12/loss-gradient' },
          { text: '训练技巧', link: '/versions/v12/tricks' },
          { text: '痛点追踪', link: '/versions/v12/painpoints' },
          { text: '源码走读', link: '/versions/v12/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（sunsmarterjie/yolov12 快照 · tag v1.0 @ b1035f8）',
        collapsed: true,
        items: [
          { text: 'nn/modules/block.py', link: '/versions/v12/source/block-py' },
          { text: 'nn/modules/head.py', link: '/versions/v12/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v12/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v12/source/tal-py' },
          { text: 'cfg/models/v12/yolov12.yaml', link: '/versions/v12/source/yolov12-yaml' }
        ]
      },
      {
        text: 'YOLOv13 (2025-06)',
        items: [
          { text: '概览', link: '/versions/v13/index' },
          { text: '论文精读', link: '/versions/v13/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v13/architecture' },
          { text: '理论 (Concept)', link: '/versions/v13/concepts' },
          { text: '架构与层速查', link: '/versions/v13/layers' },
          { text: '损失函数', link: '/versions/v13/losses' },
          { text: '反向传播推导', link: '/versions/v13/loss-gradient' },
          { text: '训练技巧', link: '/versions/v13/tricks' },
          { text: '痛点追踪', link: '/versions/v13/painpoints' },
          { text: '源码走读', link: '/versions/v13/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（iMoonLab/yolov13 快照 · 73289949）',
        collapsed: true,
        items: [
          { text: 'nn/modules/block.py', link: '/versions/v13/source/block-py' },
          { text: 'nn/modules/conv.py', link: '/versions/v13/source/conv-py' },
          { text: 'nn/modules/head.py', link: '/versions/v13/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v13/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v13/source/tal-py' },
          { text: 'cfg/models/v13/yolov13.yaml', link: '/versions/v13/source/yolov13-yaml' }
        ]
      },
      {
        text: 'YOLOv14 / YOLO-Omni (2026-08)',
        items: [
          { text: '概览', link: '/versions/v14/index' },
          { text: '论文精读', link: '/versions/v14/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v14/architecture' },
          { text: '理论 (Concept)', link: '/versions/v14/concepts' },
          { text: '架构与层速查', link: '/versions/v14/layers' },
          { text: '损失函数', link: '/versions/v14/losses' },
          { text: '反向传播推导', link: '/versions/v14/loss-gradient' },
          { text: '训练技巧', link: '/versions/v14/tricks' },
          { text: '痛点追踪', link: '/versions/v14/painpoints' },
          { text: '源码走读', link: '/versions/v14/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（zhangcbb/yolov14 快照 · 0b8610d）',
        collapsed: true,
        items: [
          { text: 'nn/modules/block.py', link: '/versions/v14/source/block-py' },
          { text: 'nn/modules/head.py', link: '/versions/v14/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v14/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v14/source/tal-py' },
          { text: 'cfg/models/v14/yolov14.yaml', link: '/versions/v14/source/yolov14-yaml' },
          { text: 'cfg/models/v14/*variants.yaml', link: '/versions/v14/source/yolov14-variants' }
        ]
      },
      {
        text: 'YOLO26 (2026-06)',
        items: [
          { text: '概览', link: '/versions/v26/index' },
          { text: '论文精读', link: '/versions/v26/paper' },
          { text: 'Architecture 逐层图解', link: '/versions/v26/architecture' },
          { text: '理论 (Concept)', link: '/versions/v26/concepts' },
          { text: '架构与层速查', link: '/versions/v26/layers' },
          { text: '损失函数', link: '/versions/v26/losses' },
          { text: '反向传播推导', link: '/versions/v26/loss-gradient' },
          { text: '训练技巧', link: '/versions/v26/tricks' },
          { text: '痛点追踪', link: '/versions/v26/painpoints' },
          { text: '源码走读', link: '/versions/v26/code-walkthrough' }
        ]
      },
      {
        text: '源码浏览（ultralytics 官方主线 快照 · 2e241d0）',
        collapsed: true,
        items: [
          { text: 'nn/tasks.py', link: '/versions/v26/source/tasks-py' },
          { text: 'nn/modules/block.py', link: '/versions/v26/source/block-py' },
          { text: 'nn/modules/conv.py', link: '/versions/v26/source/conv-py' },
          { text: 'nn/modules/head.py', link: '/versions/v26/source/head-py' },
          { text: 'utils/loss.py', link: '/versions/v26/source/loss-py' },
          { text: 'utils/tal.py', link: '/versions/v26/source/tal-py' },
          { text: 'cfg/models/yolo26.yaml', link: '/versions/v26/source/yolo26-yaml' },
          { text: 'cfg/models/*variants.yaml', link: '/versions/v26/source/yolo26-variants' }
        ]
      },
        {
          text: '共享与横向',
          items: [
            { text: '术语表', link: '/glossary' },
            { divider: true },
            { text: '版本一句话差异卡片', link: '/versions/deltas' },
            { text: '版本改动清单', link: '/versions/changelog' },
            { text: 'v8 vs v11 架构 diff', link: '/versions/deep-dive-v8-v11' },
            { text: '逐版本 diff（v1→v26）', link: '/versions/version-diffs' }
          ]
        }
      ],
      '/ontology/': [
        {
          text: '本体',
          items: [
            { text: 'Schema', link: '/ontology/schema' },
            { text: '实体类型', link: '/ontology/entity-types' },
            { text: '关系类型', link: '/ontology/relations' },
            { text: '编写新页指南', link: '/ontology/authoring' }
          ]
        }
      ],
      '/cases/': [
        {
          text: '实战',
          items: [
            { text: '案例总览', link: '/cases' },
            { text: '横向对比（选型速查）', link: '/cases/compare' },
            { text: '案例模板（新案例写作）', link: '/cases/_template' },
            { divider: true },
            { text: '边缘部署', link: '/cases/edge-deployment' },
            { text: '工业质检', link: '/cases/industrial-defect' },
            { text: '自动驾驶', link: '/cases/autonomous-lane' },
            { text: '零售', link: '/cases/retail-shelf' },
            { text: '医疗', link: '/cases/medical-lesion' },
            { divider: true },
            { text: '安防监控', link: '/cases/security-monitoring' },
            { text: '智慧农业', link: '/cases/agriculture' },
            { text: '无人机 OBB', link: '/cases/drone-obb' },
            { text: '3D BEV', link: '/cases/3d-bev' },
            { divider: true },
            { text: '多目标跟踪（MOT）', link: '/cases/mot' },
            { text: '视频流检测优化', link: '/cases/video-stream' },
            { text: '多模态融合（YOLO + SAM + LLM）', link: '/cases/multimodal' }
          ]
        }
      ],
      '/derivatives/': [
        {
          text: '衍生分支总览',
          items: [
            { text: '总览', link: '/derivatives/index' },
            { text: '三衍生分支对比', link: '/derivatives/compare' },
            { text: '路线预告', link: '/derivatives/roadmap' }
          ]
        },
        {
          text: 'YOLO-World (2024-01 · Tencent)',
          items: [
            { text: '概览', link: '/derivatives/yoloworld/index' },
            { text: '论文精读', link: '/derivatives/yoloworld/paper' },
            { text: 'Architecture 逐层图解', link: '/derivatives/yoloworld/architecture' },
            { text: '理论 (Concept)', link: '/derivatives/yoloworld/concepts' },
            { text: '架构与层速查', link: '/derivatives/yoloworld/layers' },
            { text: '损失函数', link: '/derivatives/yoloworld/losses' },
            { text: '训练技巧', link: '/derivatives/yoloworld/tricks' },
            { text: '痛点追踪', link: '/derivatives/yoloworld/painpoints' },
            { text: '源码走读', link: '/derivatives/yoloworld/code-walkthrough' }
          ]
        },
        {
          text: 'YOLO-World 源码',
          collapsed: false,
          items: [
            { text: 'cfg/yolov8-world.yaml', link: '/derivatives/yoloworld/source/yolov8-world-yaml' },
            { text: 'cfg/yolov8-worldv2.yaml', link: '/derivatives/yoloworld/source/yolov8-worldv2-yaml' },
            { text: 'nn/tasks_worldmodel.py', link: '/derivatives/yoloworld/source/worldmodel-py' },
            { text: 'nn/modules/head.py (WorldDetect)', link: '/derivatives/yoloworld/source/head-py' },
            { text: 'models/yolo/world/train.py', link: '/derivatives/yoloworld/source/train-py' },
            { text: 'models/yolo/world/train_world.py', link: '/derivatives/yoloworld/source/train-world-py' },
            { text: 'AILab-CVC · detectors/yolo_world.py', link: '/derivatives/yoloworld/source/detector-py' },
            { text: 'AILab-CVC · dense_heads/yolo_world_head.py', link: '/derivatives/yoloworld/source/head-ailab-py' },
            { text: 'AILab-CVC · necks/yolo_world_pafpn.py', link: '/derivatives/yoloworld/source/pafpn-py' }
          ]
        },
        {
          text: 'YOLOE (2025-03 · Tsinghua)',
          items: [
            { text: '概览', link: '/derivatives/yoloe/index' },
            { text: '论文精读', link: '/derivatives/yoloe/paper' },
            { text: 'Architecture 逐层图解', link: '/derivatives/yoloe/architecture' },
            { text: '理论 (Concept)', link: '/derivatives/yoloe/concepts' },
            { text: '架构与层速查', link: '/derivatives/yoloe/layers' },
            { text: '损失函数', link: '/derivatives/yoloe/losses' },
            { text: '训练技巧', link: '/derivatives/yoloe/tricks' },
            { text: '痛点追踪', link: '/derivatives/yoloe/painpoints' },
            { text: '源码走读', link: '/derivatives/yoloe/code-walkthrough' }
          ]
        },
        {
          text: 'YOLOE 源码',
          collapsed: false,
          items: [
            { text: 'cfg/v8/yoloe-v8.yaml', link: '/derivatives/yoloe/source/yoloe-v8-yaml' },
            { text: 'cfg/v8/yoloe-v8-seg.yaml', link: '/derivatives/yoloe/source/yoloe-v8-seg-yaml' },
            { text: 'cfg/26/yoloe-26.yaml', link: '/derivatives/yoloe/source/yoloe-26-yaml' },
            { text: 'cfg/26/yoloe-26-seg.yaml', link: '/derivatives/yoloe/source/yoloe-26-seg-yaml' },
            { text: 'models/yolo/model.py', link: '/derivatives/yoloe/source/model-py' },
            { text: 'models/yolo/yoloe/train.py', link: '/derivatives/yoloe/source/train-py' },
            { text: 'models/yolo/yoloe/train_seg.py', link: '/derivatives/yoloe/source/train-seg-py' },
            { text: 'models/yolo/yoloe/val.py', link: '/derivatives/yoloe/source/val-py' },
            { text: 'models/yolo/yoloe/predict.py', link: '/derivatives/yoloe/source/predict-py' },
            { text: 'nn/head.py (YOLOEDetect/Segment)', link: '/derivatives/yoloe/source/head-py' },
            { text: 'nn/tasks.py', link: '/derivatives/yoloe/source/tasks-py' }
          ]
        },
        {
          text: 'YOLO-Master (2025-12 · Tencent)',
          items: [
            { text: '概览', link: '/derivatives/master/index' },
            { text: '论文精读', link: '/derivatives/master/paper' },
            { text: 'Architecture 逐层图解', link: '/derivatives/master/architecture' },
            { text: '理论 (Concept)', link: '/derivatives/master/concepts' },
            { text: '架构与层速查', link: '/derivatives/master/layers' },
            { text: '损失函数', link: '/derivatives/master/losses' },
            { text: '训练技巧', link: '/derivatives/master/tricks' },
            { text: '痛点追踪', link: '/derivatives/master/painpoints' },
            { text: '源码走读', link: '/derivatives/master/code-walkthrough' }
          ]
        },
        {
          text: 'YOLO-Master 源码',
          collapsed: false,
          items: [
            { text: 'cfg/master/yolo-master-n.yaml', link: '/derivatives/master/source/yolo-master-n-yaml' },
            { text: 'cfg/master/yolo-master-l.yaml', link: '/derivatives/master/source/yolo-master-l-yaml' },
            { text: 'cfg/master/yolo-master-esmoe-n.yaml', link: '/derivatives/master/source/yolo-master-esmoe-n-yaml' },
            { text: 'nn/modules/moe/modules.py (ES_MOE)', link: '/derivatives/master/source/moe-modules-py' },
            { text: 'nn/modules/moe/routers.py', link: '/derivatives/master/source/moe-routers-py' },
            { text: 'nn/modules/moe/experts.py', link: '/derivatives/master/source/moe-experts-py' },
            { text: 'nn/modules/moe/loss.py', link: '/derivatives/master/source/moe-loss-py' },
            { text: 'nn/modules/moe/topk_anneal.py', link: '/derivatives/master/source/moe-topk-anneal-py' },
            { text: 'nn/modules/mot/experts.py', link: '/derivatives/master/source/mot-experts-py' },
            { text: 'nn/modules/mot/router.py', link: '/derivatives/master/source/mot-router-py' },
            { text: 'nn/modules/transformer.py', link: '/derivatives/master/source/transformer-py' },
            { text: 'nn/modules/head.py', link: '/derivatives/master/source/head-py' },
            { text: 'models/yolo/detect/train.py', link: '/derivatives/master/source/train-py' }
          ]
        }
      ],
      '/interactive/': [
        {
          text: '交互式',
          items: [
            { text: '交互式模型对比（浏览器 ONNX）', link: '/interactive/compare' }
          ]
        }
      ],
      '/tasks/': [
        {
          text: '任务扩展',
          items: [
            { text: '实例分割 Segment', link: '/tasks/segment' },
            { text: '姿态估计 Pose', link: '/tasks/pose' },
            { text: '定向框 OBB', link: '/tasks/obb' },
            { text: '旋转检测 Rotated', link: '/tasks/rotated' },
            { text: '单目深度估计 Depth', link: '/tasks/depth' }
          ]
        },
        {
          text: '相关',
          items: [
            { text: '常见问题 FAQ', link: '/faq' },
            { text: '进阶学习路线', link: '/mid-level-path' }
          ]
        }
      ],
      '/solutions/': [
        {
          text: '即用配方',
          items: [
            { text: 'Solutions 配方总览', link: '/solutions/recipes' }
          ]
        },
        {
          text: '相关',
          items: [
            { text: '实战案例总览', link: '/cases' },
            { text: '安防监控案例', link: '/cases/security-monitoring' },
            { text: '训练与微调', link: '/recipes/train' }
          ]
        }
      ],
      '/recipes/': [
        {
          text: '实操手册',
          items: [
            { text: '训练与微调', link: '/recipes/train' },
            { text: 'ONNX / TensorRT 部署', link: '/recipes/export-onnx' },
            { divider: true },
            { text: '量化深度指南', link: '/recipes/quantization' },
            { text: '边缘设备实战', link: '/recipes/edge-deploy' },
            { text: '数据标注避坑', link: '/recipes/data-annotation' },
            { text: '高级训练技巧', link: '/recipes/advanced-train' },
            { text: '生产化落地', link: '/recipes/production' }
          ]
        },
        {
          text: '相关',
          items: [
            { text: '基准数据', link: '/benchmarks' },
            { text: '故障排查', link: '/troubleshooting' },
            { text: '翻车案例集', link: '/bad-cases' },
            { text: '术语表', link: '/glossary' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/yejinlei/agent-yolo' }
    ],

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    search: {
      provider: 'local'
    },

    footer: {
      message: 'YOLO 系列本体论研究 · 论文→理论→实现→实战',
      copyright: '© 2026 agent-yolo'
    }
  },

  markdown: {
    theme: {
      light: 'vitesse-light',
      dark: 'vitesse-dark'
    },
    config(md) {
      md.use(figurePlugin)
      md.use(markdownItKatex)
    }
  },

  // @REASON: /source/* 链接指向仓库内的 source/ 目录（源码快照），非 VitePress 站点内页面，需忽略死链检查
  // @REASON: /groups/* 是历史遗留分组链接，v7 已迁移到 /versions/v7/ 下
  // @REASON: 目录型链接（如 /cases）在部分 Markdown 严格模式下会误判为死链
  ignoreDeadLinks: [/^\/source\//, /^\/groups\//, /^\/cases$/, /^\/derivatives$/, /^\/ontology$/, /^\/versions\/v\d+$/, /^\/derivatives\/(yoloworld|yoloe|master)(\/[^/]+)?$/],

  // @REASON: 把 mermaid / KaTeX 这类重量级依赖切到独立 chunk，避免主 bundle 超 500KB
  // @REASON: 用函数形式而非字面量，因为 markdown-it 由 vitepress 内部打包，不能作为入口模块
  // @REASON: 主 bundle 仍会因 vitepress 自身 + vue 内联而超阈值，暂提高上限到 800 而非继续拆分
  vite: {
    build: {
      chunkSizeWarningLimit: 800,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules')) {
              if (id.includes('mermaid')) return 'vendor-mermaid'
              if (id.includes('katex')) return 'vendor-katex'
            }
          }
        }
      }
    }
  }
})
