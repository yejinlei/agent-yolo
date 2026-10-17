<script setup>
// @RELATED: composables/useSource.js（快照加载与链接生成）
// @REASON: 页面里所有源码片段必须指向 source/yolov1/snapshot 真实文件与行号，禁止手抄
import { ref, computed, watch, onMounted } from 'vue'
import { loadSource, sourceUrl, githubUrl, langOf } from '../composables/useSource.js'
import { highlightLines } from '../utils/highlight.js'

const props = defineProps({
  file: { type: String, required: true },
  start: { type: Number, default: 0 },
  end: { type: Number, default: 0 },
  // 高亮真实行号，如 "154-161,172" 或 [157]
  hl: { type: [String, Array], default: '' },
  title: { type: String, default: '' },
  maxHeight: { type: [Number, String], default: 0 }
})

const raw = ref('')
const err = ref('')
const loading = ref(true)

async function load() {
  loading.value = true
  err.value = ''
  try {
    raw.value = await loadSource(props.file)
  } catch (e) {
    err.value = String(e.message || e)
  } finally {
    loading.value = false
  }
}
onMounted(load)
watch(() => props.file, load)

const hlSet = computed(() => {
  const set = new Set()
  const items = Array.isArray(props.hl) ? props.hl : String(props.hl).split(',')
  for (const item of items) {
    const t = String(item).trim()
    if (!t) continue
    const [a, b] = t.split('-').map((x) => parseInt(x, 10))
    if (!a) continue
    for (let i = a; i <= (b || a); i++) set.add(i)
  }
  return set
})

const lines = computed(() => {
  if (!raw.value) return []
  const all = highlightLines(raw.value.replace(/\n$/, ''), langOf(props.file))
  const s = props.start || 1
  const e = props.end || all.length
  const out = []
  for (let i = s; i <= e && i <= all.length; i++) {
    out.push({ no: i, html: all[i - 1] || ' ' })
  }
  return out
})

const rangeLabel = computed(() => {
  const s = props.start || 1
  const e = props.end || (raw.value ? raw.value.replace(/\n$/, '').split('\n').length : s)
  return s === e ? `L${s}` : `L${s}–L${e}`
})

const fullLink = computed(() => sourceUrl(props.file, props.start || 1, props.end || undefined))
const ghLink = computed(() => githubUrl(props.file, props.start || 1, props.end || undefined))
const wrapperStyle = computed(() =>
  props.maxHeight ? { maxHeight: typeof props.maxHeight === 'number' ? props.maxHeight + 'px' : props.maxHeight } : {}
)
</script>

<template>
  <figure class="src-snippet" v-if="!err">
    <figcaption class="src-head">
      <span class="src-name">
        <span class="src-ic">⌗</span>{{ title || file }}
        <span class="src-range">{{ rangeLabel }}</span>
      </span>
      <span class="src-links">
        <a :href="fullLink" :title="`站内查看 ${file} 完整文件`">完整文件 ↧</a>
        <a :href="ghLink" target="_blank" rel="noopener" title="在 GitHub 查看（带行号锚点）">GitHub ↗</a>
      </span>
    </figcaption>
    <div class="src-body" :style="wrapperStyle">
      <div v-if="loading" class="src-loading">加载源码中…</div>
      <pre v-else class="src-pre"><code><div
        v-for="ln in lines"
        :key="ln.no"
        class="src-line"
        :class="{ 'is-hl': hlSet.has(ln.no) }"
      ><a class="src-ln" :href="sourceUrl(file, ln.no)" :id="`src-${file.replace(/[\\/]/g,'-')}-${ln.no}`">{{ ln.no }}</a><span class="src-code" v-html="ln.html"></span></div></code></pre>
    </div>
  </figure>
  <figure v-else class="src-snippet src-error">
    <figcaption class="src-head"><span class="src-name">⌗ {{ file }}</span></figcaption>
    <div class="src-body"><pre class="src-pre">{{ err }}</pre></div>
  </figure>
</template>
