import { defineConfig } from 'vitepress'

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
      { text: '时间线', link: '/global/timeline' },
      { text: '本体 Schema', link: '/ontology/schema' },
      { text: '版本', items: [
        { text: 'YOLOv1', link: '/versions/v1/index' }
      ]},
      { text: '实战案例', link: '/cases' },
      { text: 'GitHub', link: 'https://github.com/your-user/agent-yolo' }
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
            { text: '训练技巧', link: '/versions/v1/tricks' },
            { text: '源码走读', link: '/versions/v1/code-walkthrough' },
            { text: '痛点追踪', link: '/versions/v1/painpoints' }
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
      { icon: 'github', link: 'https://github.com/your-user/agent-yolo' }
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
    }
  }
})
