<script setup>
// 图：7×7×30 输出张量解剖 —— 49 个 cell，每 cell 为 5+5+20 的 30 维条带
defineProps({ caption: { type: String, default: '图：FC 输出 1470 = 49×30，reshape 成 7×7×30；每格 30 维 = Box1(5) + Box2(5) + 类别(20)' } })
const cells = Array.from({ length: 49 }, (_, i) => i)
const cls20 = Array.from({ length: 20 }, (_, i) => i)
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 640 380" class="d-svg" role="img" aria-label="7x7x30 tensor anatomy">
      <!-- 7x7 平面 -->
      <text x="110" y="28" class="d-title" text-anchor="middle">7 × 7 = 49 个 cell</text>
      <g v-for="c in cells" :key="'c'+c">
        <rect
          :x="42 + (c % 7) * 20" :y="40 + Math.floor(c/7) * 20"
          width="19" height="19" rx="2"
          :fill="c === 24 ? 'var(--d-emerald-soft)' : 'var(--d-panel)'"
          :stroke="c === 24 ? 'var(--d-emerald)' : 'var(--d-grid)'"
          :stroke-width="c === 24 ? 2 : 1"
        />
      </g>
      <circle cx="42+3*20+9" cy="40+3*20+9" r="0"/>

      <!-- 引出线 -->
      <path d="M112 110 C 200 150, 250 160, 300 178" fill="none" stroke="var(--d-emerald)" stroke-width="2" marker-end="url(#t-arr)"/>
      <defs>
        <marker id="t-arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--d-emerald)"/>
        </marker>
      </defs>

      <!-- 放大 cell：30 维条带 -->
      <text x="400" y="160" class="d-title" text-anchor="middle">一个 cell 的 30 维</text>
      <!-- Box1: 5 格，宽 26 -->
      <g>
        <rect v-for="i in 5" :key="'b1'+i" :x="216+i*26" y="176" width="25" height="46" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="1.2"/>
        <text v-for="(t,i) in ['x','y','w','h','conf']" :key="'t1'+i" :x="216+i*26+12.5" y="204" class="d-tiny" text-anchor="middle">{{ t }}</text>
        <text x="281" y="240" class="d-label d-label-emerald" text-anchor="middle">Box 1（5 维）</text>
      </g>
      <!-- Box2 -->
      <g>
        <rect v-for="i in 5" :key="'b2'+i" :x="350+i*26" y="176" width="25" height="46" fill="var(--d-sky-soft)" stroke="var(--d-sky)" stroke-width="1.2"/>
        <text v-for="(t,i) in ['x','y','w','h','conf']" :key="'t2'+i" :x="350+i*26+12.5" y="204" class="d-tiny" text-anchor="middle">{{ t }}</text>
        <text x="415" y="240" class="d-label d-label-sky" text-anchor="middle">Box 2（5 维）</text>
      </g>
      <!-- 20 类：每格 12 -->
      <g>
        <rect v-for="i in cls20" :key="'cl'+i" :x="484+i*7" y="176" width="6.4" height="46" fill="var(--d-violet-soft)" stroke="var(--d-violet)" stroke-width="0.8"/>
        <text x="551" y="240" class="d-label d-label-violet" text-anchor="middle">20 类概率（两 box 共享）</text>
      </g>

      <!-- 真实内存布局提示 -->
      <rect x="180" y="270" width="420" height="80" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
      <text x="200" y="296" class="d-sub"><tspan class="d-fw">detection_layer.c 的实际排布顺序（每 cell 内）：</tspan></text>
      <text x="200" y="318" class="d-mono">class×20 → conf<tspan fill="var(--d-emerald)">▮</tspan>×2 → (x,y,w,h)×2</text>
      <text x="200" y="338" class="d-sub">偏移：p_index = cls 区之后；box_index = cls + n 区之后（见前向代码 L76 / L103）</text>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
