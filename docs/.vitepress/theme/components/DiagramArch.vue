<script setup>
// 图：YOLOv1 特征流金字塔 —— 6 次空间减半（448→7），通道 3→1024，尾部两层 FC 拍平成 1470
defineProps({ caption: { type: String, default: '图：特征流金字塔。空间经 6 次减半（2 次 stride=2 卷积 + 4 次 maxpool）到 7×7；FC4096 占全部参数约 79%，是 v1 又大又慢的主因' } })

const stages = [
  { res: 448, ch: 3,    h: 16, fill: 'var(--d-grid)' },
  { res: 224, ch: 64,   h: 30, fill: 'var(--d-sky)' },
  { res: 112, ch: 192,  h: 42, fill: 'var(--d-sky)' },
  { res: 56,  ch: 256,  h: 48, fill: 'var(--d-emerald)' },
  { res: 28,  ch: 512,  h: 60, fill: 'var(--d-emerald)' },
  { res: 14,  ch: 1024, h: 76, fill: 'var(--d-violet)' },
  { res: 7,   ch: 1024, h: 76, fill: 'var(--d-violet)' }
]
const X0 = 28, W = 60, STEP = 84, BASE = 210
const downs = ['Conv7 s2', 'Pool', 'Pool', 'Pool', 'Pool', 'Conv3 s2']
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 660 320" class="d-svg" role="img" aria-label="feature pyramid">
      <!-- 特征块 -->
      <g v-for="(s, i) in stages" :key="i">
        <rect :x="X0 + i*STEP" :y="BASE - s.h" :width="W" :height="s.h" rx="5"
          :fill="s.fill" :fill-opacity="i === 0 ? 0.25 : 0.35" :stroke="s.fill" stroke-width="2"/>
        <text :x="X0 + i*STEP + W/2" :y="BASE + 22" class="d-label" text-anchor="middle">{{ s.res }}×{{ s.res }}</text>
        <text :x="X0 + i*STEP + W/2" :y="BASE + 40" class="d-sub" text-anchor="middle">×{{ s.ch }}</text>
      </g>
      <!-- 下采样标注 -->
      <g v-for="(d, i) in downs" :key="'d'+i">
        <path :d="`M${X0 + i*STEP + W + 4} ${BASE - 24} L${X0 + (i+1)*STEP - 6} ${BASE - 24}`"
          stroke="var(--d-rose)" stroke-width="1.8" marker-end="url(#ar-rose)"/>
        <text :x="X0 + i*STEP + W/2 + 12" :y="BASE - 30" class="d-tiny d-label-rose" text-anchor="middle">{{ d }}</text>
      </g>
      <defs><marker id="ar-rose" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 z" fill="var(--d-rose)"/></marker></defs>

      <!-- 尾部 FC -->
      <path d="M596 150 L616 150" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#ar-rose)"/>
      <g transform="translate(616,118)">
        <rect width="34" height="64" rx="5" fill="var(--d-amber-soft)" stroke="var(--d-amber)" stroke-width="2"/>
        <text x="17" y="38" class="d-tiny" text-anchor="middle">FC</text>
      </g>

      <!-- 第二行：语义标注 -->
      <text x="112" y="286" class="d-sub" text-anchor="middle">Stem 快速降分辨率</text>
      <text x="364" y="286" class="d-sub" text-anchor="middle">Body 通道加深、语义提炼</text>
      <text x="560" y="286" class="d-sub" text-anchor="middle">Head</text>
      <text x="330" y="310" class="d-sub" text-anchor="middle">7×7×1024＝50,176 → FC 4096（≈2.06 亿参数）→ FC 1470 → reshape 7×7×30</text>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
