<script setup>
// 图：YOLOv1 的 7×7 网格责任划分 —— 物体中心落在哪个 cell，就由该 cell 的 2 个 box 负责
defineProps({ caption: { type: String, default: '图：7×7 grid 中，只有狗的中心点（●）所在 cell 被激活，输出 2 个 bbox 与 20 类共享概率' } })
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 600 340" class="d-svg" role="img" aria-label="7x7 grid responsibility">
      <!-- 左侧 7x7 网格 -->
      <!-- GT 框（跨多格，虚线） -->
      <rect x="112" y="72" width="128" height="96" rx="4" fill="var(--d-rose-soft)" stroke="var(--d-rose)" stroke-width="2" stroke-dasharray="6 4" opacity="0.85"/>
      <!-- 网格线 -->
      <g stroke="var(--d-grid)" stroke-width="1">
        <rect x="32" y="32" width="224" height="224" fill="var(--d-panel)" stroke="var(--d-ink-2)" stroke-width="1.5"/>
        <line v-for="i in 6" :key="'v'+i" :x1="32+i*32" y1="32" :x2="32+i*32" y2="256"/>
        <line v-for="i in 6" :key="'h'+i" x1="32" :y1="32+i*32" :x2="256" :y2="32+i*32"/>
      </g>
      <!-- 负责 cell：col=3,row=2（0-based）-->
      <rect x="128" y="96" width="32" height="32" fill="var(--d-emerald-soft)" stroke="var(--d-emerald)" stroke-width="2.5"/>
      <!-- GT 中心 -->
      <circle cx="144" cy="120" r="5.5" fill="var(--d-rose)" stroke="#fff" stroke-width="1.5"/>
      <text x="150" y="88" class="d-label d-label-rose" text-anchor="middle">GT 框（狗）</text>

      <!-- 连接曲线 -->
      <path d="M160 112 C 250 70, 300 70, 352 96" fill="none" stroke="var(--d-emerald)" stroke-width="2" marker-end="url(#arr-emerald)"/>
      <defs>
        <marker id="arr-emerald" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--d-emerald)"/>
        </marker>
      </defs>

      <!-- 右侧：放大的负责 cell -->
      <text x="420" y="40" class="d-title" text-anchor="middle">负责 cell 放大</text>
      <rect x="340" y="52" width="160" height="160" fill="var(--d-panel)" stroke="var(--d-ink-2)" stroke-width="1.5" rx="6"/>
      <g stroke="var(--d-grid)" stroke-width="1">
        <line x1="420" y1="52" x2="420" y2="212" stroke-dasharray="3 3"/>
        <line x1="340" y1="132" x2="500" y2="132" stroke-dasharray="3 3"/>
      </g>
      <!-- box A 宽扁（车形）-->
      <rect x="360" y="112" width="120" height="52" rx="4" fill="var(--d-sky-soft)" stroke="var(--d-sky)" stroke-width="2" opacity="0.9"/>
      <text x="420" y="143" class="d-label d-label-sky" text-anchor="middle">Box 1（宽扁）</text>
      <!-- box B 高瘦（人形）-->
      <rect x="392" y="76" width="56" height="120" rx="4" fill="var(--d-amber-soft)" stroke="var(--d-amber)" stroke-width="2" opacity="0.75"/>
      <text x="420" y="100" class="d-label d-label-amber" text-anchor="middle">Box 2（高瘦）</text>
      <!-- 中心 -->
      <circle cx="420" cy="132" r="5" fill="var(--d-rose)" stroke="#fff" stroke-width="1.5"/>

      <!-- 30 维标注 -->
      <rect x="340" y="232" width="160" height="40" rx="6" fill="var(--d-violet-soft)" stroke="var(--d-violet)" stroke-width="1.5"/>
      <text x="420" y="250" class="d-label d-label-violet" text-anchor="middle">5 + 5 + 20 = 30 维</text>
      <text x="420" y="266" class="d-sub" text-anchor="middle">两 box 各 5 维，类别 20 维共享</text>

      <!-- 左侧尺寸标注 -->
      <text x="144" y="276" class="d-sub" text-anchor="middle">448×448 输入，每格 64×64 像素</text>
      <text x="144" y="298" class="d-sub" text-anchor="middle">S = 7，共 49 个 cell</text>

      <!-- 底部规则 -->
      <text x="32" y="324" class="d-sub">
        <tspan class="d-dot d-dot-rose">●</tspan> 中心点：决定哪个 cell 负责（中心不在任何格内的物体无人预测 → 漏检根源）
      </text>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
