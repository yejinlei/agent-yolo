<script setup>
// @RELATED: composables/useSource.js#sourceUrl
// @REASON: 行内引用源码统一用本组件，保证所有 file:line 文本都可点击，白名单文件跳站内、其余跳 GitHub
import { computed } from 'vue'
import { sourceUrl, displayFile, hasPage } from '../composables/useSource.js'

const props = defineProps({
  file: { type: String, required: true },
  line: { type: Number, default: 0 },
  end: { type: Number, default: 0 }
})

const href = computed(() => sourceUrl(props.file, props.line || undefined, props.end || undefined))
const label = computed(() => {
  const base = displayFile(props.file)
  if (!props.line) return base
  return props.end && props.end !== props.line ? `${base}:${props.line}–${props.end}` : `${base}:${props.line}`
})
const external = computed(() => !hasPage(props.file))
</script>

<template>
  <a class="src-link" :href="href" :target="external ? '_blank' : null" :rel="external ? 'noopener' : null">
    <span class="src-link-ic">⌗</span><code>{{ label }}</code><span v-if="external" class="src-link-ext">↗</span>
  </a>
</template>
