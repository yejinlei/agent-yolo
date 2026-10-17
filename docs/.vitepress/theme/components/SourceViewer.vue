<script setup>
// @RELATED: components/SrcSnippet.vue（片段用同一套快照与配色）
// @REASON: 站内源码浏览页；支持 #L157 与 #L157-L161 锚点高亮并滚动定位，hashchange 时重新定位
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { loadSource, githubUrl, langOf } from '../composables/useSource.js'
import { highlightLines } from '../utils/highlight.js'

const props = defineProps({
  file: { type: String, required: true }
})

const raw = ref('')
const lines = ref([])
const hlSet = ref(new Set())
const anchorNo = ref(0)
const bodyEl = ref(null)

function parseHash() {
  const m = /#L(\d+)(?:-L?(\d+))?/.exec(window.location.hash || '')
  const set = new Set()
  let first = 0
  if (m) {
    first = parseInt(m[1], 10)
    const last = m[2] ? parseInt(m[2], 10) : first
    for (let i = first; i <= last; i++) set.add(i)
  }
  hlSet.value = set
  anchorNo.value = first
}

async function syncHashAndScroll() {
  parseHash()
  await nextTick()
  if (anchorNo.value && bodyEl.value) {
    const el = bodyEl.value.querySelector(`[data-no="${anchorNo.value}"]`)
    if (el) el.scrollIntoView({ block: 'center', behavior: 'auto' })
  }
}

onMounted(async () => {
  raw.value = await loadSource(props.file)
  lines.value = highlightLines(raw.value.replace(/\n$/, ''), langOf(props.file))
    .map((html, i) => ({ no: i + 1, html: html || ' ' }))
  await syncHashAndScroll()
  window.addEventListener('hashchange', syncHashAndScroll)
})
onBeforeUnmount(() => window.removeEventListener('hashchange', syncHashAndScroll))

const gh = computed(() => githubUrl(props.file))
</script>

<template>
  <div class="src-viewer">
    <div class="src-viewer-bar">
      <span class="src-name"><span class="src-ic">⌗</span>{{ file }} <span class="src-range">{{ lines.length }} 行</span></span>
      <a :href="gh" target="_blank" rel="noopener">GitHub 查看 ↗</a>
    </div>
    <div class="src-body src-viewer-body" ref="bodyEl">
      <pre class="src-pre"><code><div
        v-for="ln in lines"
        :key="ln.no"
        class="src-line"
        :class="{ 'is-hl': hlSet.has(ln.no) }"
        :data-no="ln.no"
      ><a class="src-ln" :href="`#L${ln.no}`">{{ ln.no }}</a><span class="src-code" v-html="ln.html"></span></div></code></pre>
    </div>
  </div>
</template>
