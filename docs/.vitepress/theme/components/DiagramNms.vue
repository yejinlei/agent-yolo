<script setup>
// 图：NMS 三帧 —— 按 class 分数排序、保留最高分、抑制与其 IoU>0.5 的框（box.c do_nms_sort）
defineProps({ caption: { type: String, default: '图：NMS（do_nms_sort，box.c L249）：按 类别分×objectness 排序 → 保留最高分 → 抑制 IoU>0.5 的重叠框 → 逐类重复' } })
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 680 260" class="d-svg" role="img" aria-label="nms steps">
      <!-- 面板 1 -->
      <g transform="translate(16,28)">
        <rect width="196" height="200" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="98" y="26" class="d-title" text-anchor="middle">① 98 个候选，按分数排序</text>
        <rect x="40" y="70" width="120" height="86" rx="4" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="2.5"/>
        <text x="100" y="62" class="d-tiny d-label-emerald" text-anchor="middle">0.92</text>
        <rect x="56" y="86" width="120" height="86" rx="4" fill="none" stroke="var(--d-amber)" stroke-width="2" stroke-dasharray="5 3"/>
        <text x="176" y="82" class="d-tiny d-label-amber" text-anchor="end">0.71</text>
        <rect x="30" y="96" width="118" height="80" rx="4" fill="none" stroke="var(--d-sky)" stroke-width="2" stroke-dasharray="5 3"/>
        <text x="36" y="112" class="d-tiny d-label-sky">0.63</text>
        <text x="98" y="184" class="d-sub" text-anchor="middle">分数 = P(class) × conf</text>
      </g>

      <!-- 箭头 -->
      <path d="M224 128 L250 128" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#nms-a)"/>
      <defs><marker id="nms-a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--d-ink-2)"/></marker></defs>

      <!-- 面板 2 -->
      <g transform="translate(258,28)">
        <rect width="196" height="200" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="98" y="26" class="d-title" text-anchor="middle">② 与最高分框算 IoU</text>
        <rect x="40" y="70" width="120" height="86" rx="4" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="2.5"/>
        <text x="100" y="62" class="d-tiny d-label-emerald" text-anchor="middle">保留 ✓</text>
        <rect x="56" y="86" width="120" height="86" rx="4" fill="var(--d-rose-soft)" stroke="var(--d-rose)" stroke-width="2" stroke-dasharray="5 3"/>
        <text x="176" y="82" class="d-tiny d-label-rose" text-anchor="end">IoU 0.72 ✕</text>
        <rect x="30" y="96" width="118" height="80" rx="4" fill="none" stroke="var(--d-sky)" stroke-width="2" stroke-dasharray="5 3"/>
        <text x="36" y="112" class="d-tiny d-label-sky">IoU 0.31</text>
        <text x="98" y="184" class="d-sub" text-anchor="middle">阈值 overlap = 0.5</text>
      </g>

      <path d="M466 128 L492 128" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#nms-a)"/>

      <!-- 面板 3 -->
      <g transform="translate(500,28)">
        <rect width="164" height="200" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="82" y="26" class="d-title" text-anchor="middle">③ 输出最终框</text>
        <rect x="30" y="76" width="112" height="80" rx="4" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="2.5"/>
        <text x="86" y="70" class="d-tiny d-label-emerald" text-anchor="middle">0.92 ✓</text>
        <rect x="22" y="98" width="106" height="72" rx="4" fill="var(--d-sky-soft)" stroke="var(--d-sky)" stroke-width="2"/>
        <text x="28" y="114" class="d-tiny d-label-sky">0.63 ✓</text>
        <text x="82" y="184" class="d-sub" text-anchor="middle">换下一类别重复</text>
      </g>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
