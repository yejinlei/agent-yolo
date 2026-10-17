<script setup>
// 图：IoU 交并集几何 + v1 中两个用途（选 best box、rescore 时当 conf 目标）
defineProps({ caption: { type: String, default: '图：IoU = 交集面积 / 并集面积；v1 用它在 2 个候选 box 中选 best box，并在 rescore=1 时作为 confidence 的训练目标' } })
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 640 300" class="d-svg" role="img" aria-label="IoU geometry">
      <!-- 左：几何示意 -->
      <rect x="60" y="70" width="150" height="110" rx="6" fill="var(--d-sky-soft)" stroke="var(--d-sky)" stroke-width="2"/>
      <rect x="140" y="110" width="150" height="120" rx="6" fill="var(--d-amber-soft)" stroke="var(--d-amber)" stroke-width="2" fill-opacity="0.75"/>
      <text x="100" y="96" class="d-label d-label-sky">框 A（预测）</text>
      <text x="230" y="216" class="d-label d-label-amber">框 B（GT）</text>
      <text x="150" y="140" class="d-sub" text-anchor="middle">交集</text>
      <text x="160" y="252" class="d-sub" text-anchor="middle">重叠越多 → IoU 越接近 1</text>

      <!-- 中：公式 -->
      <g transform="translate(330,60)">
        <rect x="0" y="0" width="280" height="92" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="140" y="34" class="d-formula" text-anchor="middle">IoU = area(交集) / area(并集)</text>
        <text x="140" y="62" class="d-formula d-sub" text-anchor="middle">= inter / (areaA + areaB − inter)</text>
      </g>
      <!-- 数字例 -->
      <g transform="translate(330,168)">
        <rect x="0" y="0" width="280" height="72" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="16" y="28" class="d-sub">例：areaA=12，areaB=8，inter=4</text>
        <text x="16" y="54" class="d-mono">IoU = 4 / (12 + 8 − 4) = <tspan class="d-fw" fill="var(--d-emerald)">0.25</tspan></text>
      </g>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
