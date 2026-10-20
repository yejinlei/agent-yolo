import DefaultTheme from 'vitepress/theme'
import { onMounted } from 'vue'
import './style.css'
import 'katex/dist/katex.min.css'

import SrcSnippet from './components/SrcSnippet.vue'
import SrcLink from './components/SrcLink.vue'
import SourceViewer from './components/SourceViewer.vue'
import DiagramGridCell from './components/DiagramGridCell.vue'
import DiagramTensor30 from './components/DiagramTensor30.vue'
import DiagramIou from './components/DiagramIou.vue'
import DiagramDecode from './components/DiagramDecode.vue'
import DiagramArch from './components/DiagramArch.vue'
import DiagramNms from './components/DiagramNms.vue'
import DiagramPipeline from './components/DiagramPipeline.vue'

// @AI_GUARD: Mermaid 主题变量必须与 style.css 的 d-* 调色板保持一致；明暗两套都要维护
const MERMAID_THEMES = {
  light: {
    primaryColor: '#ffffff',
    primaryBorderColor: '#059669',
    primaryTextColor: '#0f172a',
    lineColor: '#64748b',
    secondaryColor: '#f1f5f9',
    tertiaryColor: '#ffffff',
    edgeLabelBackground: '#ffffff',
    mainGroupBkg: '#f8fafc',
    clusterBkg: '#f8fafc',
    clusterBorder: '#94a3b8'
  },
  dark: {
    primaryColor: '#0f172a',
    primaryBorderColor: '#34d399',
    primaryTextColor: '#e2e8f0',
    lineColor: '#94a3b8',
    secondaryColor: '#1e293b',
    tertiaryColor: '#0f172a',
    edgeLabelBackground: '#0f172a',
    mainGroupBkg: '#0f172a',
    clusterBkg: '#0f172a',
    clusterBorder: '#475569'
  }
}

export default {
  extends: DefaultTheme,

  enhanceApp({ app, router }) {
    app.component('SrcSnippet', SrcSnippet)
    app.component('SrcLink', SrcLink)
    app.component('SourceViewer', SourceViewer)
    app.component('DiagramGridCell', DiagramGridCell)
    app.component('DiagramTensor30', DiagramTensor30)
    app.component('DiagramIou', DiagramIou)
    app.component('DiagramDecode', DiagramDecode)
    app.component('DiagramArch', DiagramArch)
    app.component('DiagramNms', DiagramNms)
    app.component('DiagramPipeline', DiagramPipeline)

    if (typeof window === 'undefined') return

    let mermaid = null
    let currentId = 0

    function isDark() {
      return document.documentElement.classList.contains('dark')
    }

    async function initMermaid() {
      if (!mermaid) {
        const mod = await import('mermaid')
        mermaid = mod.default
      }
      const vars = isDark() ? MERMAID_THEMES.dark : MERMAID_THEMES.light
      mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        securityLevel: 'loose',
        themeVariables: {
          // 原为修复「文字被裁」而临时降到 12px；根因已在 style.css 中修掉
          // （.mermaid-svg foreignObject p 的 line-height 泄漏），故恢复为可读的 14px
          fontSize: '14px',
          ...vars
        },
        flowchart: { htmlLabels: true, curve: 'basis', padding: 14 },
        pie: { useWidth: true }
      })
      return mermaid
    }

    async function renderInto(holder, code) {
      currentId += 1
      const id = `mermaid-${currentId}`
      const mm = await initMermaid()
      const { svg } = await mm.render(id, code)
      holder.innerHTML = svg
    }

    async function renderAll() {
      // 未渲染的代码块 -> 建 holder
      const blocks = Array.from(document.querySelectorAll('div.language-mermaid'))
      for (const block of blocks) {
        if (block.dataset.rendered === 'true') continue
        const codeEl = block.querySelector('pre > code')
        if (!codeEl) continue
        const code = codeEl.textContent || ''
        if (!code.trim()) continue
        const holder = document.createElement('div')
        holder.className = 'mermaid-svg'
        holder.dataset.rendered = 'true'
        holder.dataset.code = code
        block.replaceWith(holder)
        try {
          await renderInto(holder, code)
        } catch (err) {
          holder.classList.add('mermaid-failed')
          holder.dataset.error = String(err?.message || err)
          holder.textContent = '⚠ Mermaid 渲染失败'
        }
      }
    }

    // 明暗切换：用当前主题重新渲染所有已出图的 holder
    async function rerenderAll() {
      const holders = Array.from(document.querySelectorAll('.mermaid-svg'))
      if (!holders.length) return
      await initMermaid()
      for (const holder of holders) {
        const code = holder.dataset.code
        if (!code) continue
        holder.classList.remove('mermaid-failed')
        try {
          await renderInto(holder, code)
        } catch (err) {
          holder.classList.add('mermaid-failed')
          holder.dataset.error = String(err?.message || err)
        }
      }
    }

    let darkObserver = null
    onMounted(() => {
      renderAll()
      // VitePress 明暗切换只改 html.dark，观察 class 属性变化即可
      darkObserver = new MutationObserver((muts) => {
        for (const m of muts) {
          if (m.attributeName === 'class') {
            rerenderAll()
            break
          }
        }
      })
      darkObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    })

    router.onAfterRouteChanged = () => {
      renderAll()
    }
  }
}
