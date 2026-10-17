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
      { text: '系列演化', link: '/global/family' },
      { text: '术语表', link: '/glossary' },
      { text: '时间线', link: '/global/timeline' },
      { text: '本体 Schema', link: '/ontology/schema' },
      { text: '版本', items: [
        { text: 'YOLOv1', link: '/versions/v1/index' },
        { text: 'YOLOv2', link: '/versions/v2/index' },
        { text: 'YOLOv3', link: '/versions/v3/index' }
      ]},
      { text: '实战案例', link: '/cases' },
      { text: 'GitHub', link: 'https://github.com/yejinlei/agent-yolo' }
    ],

    sidebar: {
      '/global/': [
        {
          text: '全局视图',
          items: [
            { text: '系列家族', link: '/global/family' },
            { text: '演化事件', link: '/global/evolutions' },
            { text: '时间线', link: '/global/timeline' },
            { text: '传统 CV 谱系', link: '/global/external-techniques' }
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
          text: '共享',
          items: [
            { text: '术语表', link: '/glossary' }
          ]
        }
      ],
      '/ontology/': [
        {
          text: '本体',
          items: [
            { text: 'Schema', link: '/ontology/schema' },
            { text: '实体类型', link: '/ontology/entity-types' },
            { text: '关系类型', link: '/ontology/relations' }
          ]
        }
      ],
      '/cases/': [
        {
          text: '实战',
          items: [
            { text: '案例总览', link: '/cases' },
            { text: '边缘部署', link: '/cases/edge-deployment' }
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
  }
})
