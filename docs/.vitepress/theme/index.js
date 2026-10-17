import DefaultTheme from 'vitepress/theme'
import { onMounted } from 'vue'
import './style.css'

export default {
  extends: DefaultTheme,

  enhanceApp({ router }) {
    if (typeof window === 'undefined') return

    let mermaid = null
    let currentId = 0

    async function initMermaid() {
      if (!mermaid) {
        const mod = await import('mermaid')
        mermaid = mod.default
        mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
        })
      }
      return mermaid
    }

    // 渲染页面上所有尚未渲染的 mermaid 代码块
    async function renderAll() {
      const blocks = Array.from(
        document.querySelectorAll('div.language-mermaid')
      )

      for (const block of blocks) {
        // 已渲染过（替换成 mermaid-svg 容器）跳过
        if (block.dataset.rendered === 'true') continue

        const codeEl = block.querySelector('pre > code')
        if (!codeEl) continue

        const code = codeEl.textContent || ''
        if (!code.trim()) continue

        currentId += 1
        const id = `mermaid-${currentId}`
        try {
          const mm = await initMermaid()
          const { svg } = await mm.render(id, code)
          const holder = document.createElement('div')
          holder.className = 'mermaid-svg'
          holder.dataset.rendered = 'true'
          // mermaid 生成的 svg 自带 <svg>，整段替换
          holder.innerHTML = svg
          block.replaceWith(holder)
        } catch (err) {
          // 渲染失败：标记为已尝试，避免死循环；保留原始代码便于排查
          block.dataset.rendered = 'true'
          block.dataset.error = String(err?.message || err)
          block.classList.add('mermaid-failed')
        }
      }
    }

    // 首次挂载渲染
    onMounted(renderAll)

    // 路由切换后 DOM 会重建，重新渲染（VitePress SPA）
    router.onAfterRouteChanged = (url) => {
      renderAll()
    }
  },
}
