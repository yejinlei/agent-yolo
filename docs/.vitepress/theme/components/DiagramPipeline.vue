<script setup>
// 图：推理效果流水线 —— 原图经 4 步变成最终检测图（对应 test_yolo，yolo.c L311-361）
defineProps({ caption: { type: String, default: '图：推理效果流水线（test_yolo，yolo.c L311-361）——原图 ① resize 到 448×448 ② 单次 network_predict 出 98 候选框 ③ conf×class 阈值 + NMS 去重 ④ draw_detections 画最终框' } })

// 面板 3 的候选框：[x, y, w, h, 颜色变量, 是否误检]
const candidates = [
  [14, 46, 74, 48, '--d-emerald', 0], [20, 52, 66, 44, '--d-amber', 0],
  [10, 42, 80, 52, '--d-sky', 0], [26, 40, 58, 50, '--d-emerald', 0],
  [92, 88, 52, 40, '--d-emerald', 0], [98, 94, 46, 36, '--d-amber', 0],
  [88, 84, 56, 46, '--d-sky', 0], [104, 90, 44, 38, '--d-emerald', 0],
  [54, 10, 26, 20, '--d-rose', 1], [118, 22, 24, 18, '--d-rose', 1],
  [6, 108, 26, 20, '--d-rose', 1],
]
// 面板 2 的 7×7 网格线位置（150 / 7 ≈ 21.43）
const gridPos = [21.4, 42.9, 64.3, 85.7, 107.1, 128.6]
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 716 236" class="d-svg" role="img" aria-label="inference pipeline">
      <defs>
        <!-- 统一场景：一辆车 + 一只狗，四个面板复用 -->
        <g id="pp-scene">
          <rect width="150" height="96" fill="var(--d-sky-soft)"/>
          <rect y="96" width="150" height="54" fill="var(--d-panel)"/>
          <!-- 汽车 -->
          <rect x="30" y="52" width="34" height="14" rx="5" fill="var(--d-amber)" opacity=".8"/>
          <rect x="18" y="64" width="64" height="22" rx="6" fill="var(--d-amber)" opacity=".9"/>
          <circle cx="32" cy="88" r="6" fill="var(--d-ink-2)"/>
          <circle cx="68" cy="88" r="6" fill="var(--d-ink-2)"/>
          <!-- 狗 -->
          <ellipse cx="112" cy="112" rx="17" ry="9" fill="var(--d-ink-2)"/>
          <circle cx="130" cy="100" r="7" fill="var(--d-ink-2)"/>
          <path d="M96 108 L88 100" stroke="var(--d-ink-2)" stroke-width="3" stroke-linecap="round"/>
          <path d="M104 120 L104 132 M112 121 L112 133 M120 120 L120 132 M126 118 L126 130" stroke="var(--d-ink-2)" stroke-width="3" stroke-linecap="round"/>
        </g>
        <marker id="pp-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--d-ink-2)"/></marker>
      </defs>

      <!-- ① 原图 -->
      <g transform="translate(8,20)">
        <rect width="150" height="182" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="75" y="18" class="d-title" text-anchor="middle">① 输入原图</text>
        <g transform="translate(0,24)"><use href="#pp-scene"/></g>
        <text x="75" y="172" class="d-sub" text-anchor="middle">任意尺寸</text>
      </g>

      <path d="M162 104 L190 104" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#pp-arrow)"/>
      <text x="176" y="126" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">resize_image</text>
      <text x="176" y="138" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">yolo.c:340</text>

      <!-- ② resize + 网格 -->
      <g transform="translate(194,20)">
        <rect width="150" height="182" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="75" y="18" class="d-title" text-anchor="middle">② 448×448 + 7×7</text>
        <g transform="translate(0,24)">
          <use href="#pp-scene"/>
          <line v-for="p in gridPos" :key="'v'+p" :x1="p" y1="0" :x2="p" y2="150" stroke="var(--d-emerald)" stroke-width=".6" opacity=".55"/>
          <line v-for="p in gridPos" :key="'h'+p" x1="0" :y1="p" x2="150" :y2="p" stroke="var(--d-emerald)" stroke-width=".6" opacity=".55"/>
        </g>
        <text x="75" y="172" class="d-sub" text-anchor="middle">49 cell 各负责 2 box</text>
      </g>

      <path d="M348 104 L376 104" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#pp-arrow)"/>
      <text x="362" y="126" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">network_predict</text>
      <text x="362" y="138" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">yolo.c:343</text>

      <!-- ③ 98 候选框 -->
      <g transform="translate(380,20)">
        <rect width="150" height="182" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="75" y="18" class="d-title" text-anchor="middle">③ 98 个候选框</text>
        <g transform="translate(0,24)">
          <use href="#pp-scene"/>
          <rect v-for="(b, i) in candidates" :key="i" :x="b[0]" :y="b[1]" :width="b[2]" :height="b[3]"
                :fill="'var(' + b[4] + ')'" :fill-opacity="b[5] ? .05 : .12"
                :stroke="'var(' + b[4] + ')'" stroke-width="1.2" :stroke-dasharray="b[5] ? '3 2' : 'none'" :opacity="b[5] ? .7 : 1"/>
        </g>
        <text x="75" y="172" class="d-sub" text-anchor="middle">convert_yolo_detections :345</text>
      </g>

      <path d="M534 104 L562 104" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#pp-arrow)"/>
      <text x="548" y="126" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">do_nms_sort</text>
      <text x="548" y="138" class="d-tiny" text-anchor="middle" fill="var(--d-ink-2)">yolo.c:346</text>

      <!-- ④ 最终输出 -->
      <g transform="translate(566,20)">
        <rect width="150" height="182" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="75" y="18" class="d-title" text-anchor="middle">④ 最终检测</text>
        <g transform="translate(0,24)">
          <use href="#pp-scene"/>
          <rect x="14" y="46" width="74" height="48" rx="3" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="2.5"/>
          <rect x="14" y="34" width="52" height="12" rx="2" fill="var(--d-emerald)"/>
          <text x="18" y="43" class="d-tiny" fill="#fff">car 0.91</text>
          <rect x="92" y="88" width="52" height="40" rx="3" fill="var(--d-sky-soft)" stroke="var(--d-sky)" stroke-width="2.5"/>
          <rect x="92" y="76" width="54" height="12" rx="2" fill="var(--d-sky)"/>
          <text x="96" y="85" class="d-tiny" fill="#fff">dog 0.84</text>
        </g>
        <text x="75" y="172" class="d-sub" text-anchor="middle">draw_detections :348</text>
      </g>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
