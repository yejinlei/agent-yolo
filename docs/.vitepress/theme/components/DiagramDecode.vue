<script setup>
// 图：bbox 编码/解码 —— 网络输出归一化 (x,y,√w,√h)，推理时加 cell 偏移、平方还原像素框（yolo.c:98-101）
defineProps({ caption: { type: String, default: '图：推理解码（yolo.c convert_yolo_detections L98–L101）：坐标加回 cell 偏移再乘全图尺寸，宽高由 √ 空间平方还原' } })
</script>

<template>
  <figure class="d-figure">
    <svg viewBox="0 0 680 330" class="d-svg" role="img" aria-label="bbox encode decode">
      <!-- ① 网络输出 -->
      <g transform="translate(20,60)">
        <text x="70" y="-16" class="d-title" text-anchor="middle">① 网络输出（归一化）</text>
        <rect v-for="(t,i) in ['x','y','√w','√h']" :key="i" :x="i*36" y="0" width="33" height="40" rx="4"
          fill="var(--d-emerald-soft)" stroke="var(--d-emerald)"/>
        <text v-for="(t,i) in ['x','y','√w','√h']" :key="'t'+i" :x="i*36+16.5" y="26" class="d-tiny" text-anchor="middle">{{ t }}</text>
        <text x="70" y="66" class="d-sub" text-anchor="middle">x,y 相对本 cell</text>
        <text x="70" y="84" class="d-sub" text-anchor="middle">√w,√h 相对全图</text>
      </g>

      <!-- 箭头 -->
      <path d="M170 80 L230 80" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#dc-a)"/>
      <defs><marker id="dc-a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--d-ink-2)"/></marker></defs>

      <!-- ② 解码公式 -->
      <g transform="translate(238,44)">
        <rect width="240" height="150" rx="8" fill="var(--d-panel)" stroke="var(--d-panel-border)"/>
        <text x="14" y="30" class="d-mono d-sm">x_img = (x + col) / 7 · W</text>
        <text x="14" y="58" class="d-mono d-sm">y_img = (y + row) / 7 · H</text>
        <text x="14" y="86" class="d-mono d-sm">w_img = <tspan class="d-fw" fill="var(--d-rose)">w²</tspan> · W</text>
        <text x="14" y="114" class="d-mono d-sm">h_img = <tspan class="d-fw" fill="var(--d-rose)">h²</tspan> · H</text>
        <text x="14" y="138" class="d-sub">col/row = cell 的列、行（0–6）</text>
      </g>

      <path d="M486 80 L540 80" stroke="var(--d-ink-2)" stroke-width="2" marker-end="url(#dc-a)"/>

      <!-- ③ 像素框 -->
      <g transform="translate(548,36)">
        <text x="60" y="8" class="d-title" text-anchor="middle">③ 像素坐标框</text>
        <rect x="0" y="20" width="120" height="120" fill="var(--d-panel)" stroke="var(--d-ink-2)" stroke-width="1.5"/>
        <g stroke="var(--d-grid)" stroke-width="0.8">
          <line v-for="i in 6" :key="'v'+i" :x1="i*120/7" y1="20" :x2="i*120/7" y2="140"/>
          <line v-for="i in 6" :key="'h'+i" x1="0" :y1="20+i*120/7" x2="120" :y2="20+i*120/7"/>
        </g>
        <rect x="30" y="46" width="64" height="52" rx="3" fill="var(--d-rose-soft)" stroke="var(--d-rose)" stroke-width="2"/>
        <circle cx="62" cy="72" r="3.5" fill="var(--d-rose)"/>
        <text x="60" y="160" class="d-sub" text-anchor="middle">中心 + 宽高 → 最终 bbox</text>
      </g>

      <!-- 底部：训练方向 -->
      <rect x="20" y="232" width="648" height="66" rx="8" fill="var(--d-amber-soft)" stroke="var(--d-amber)" stroke-width="1.2"/>
      <text x="36" y="258" class="d-sub"><tspan class="d-fw">训练时反向编码（detection_layer.c L98–L99 / L168）：</tspan></text>
      <text x="36" y="280" class="d-mono d-sm">truth.x /= side；truth.y /= side（GT 中心换算到 cell 归一坐标）　　delta_w = coord_scale · ( √truth_w − 输出w )</text>
    </svg>
    <figcaption v-if="caption" class="d-cap">{{ caption }}</figcaption>
  </figure>
</template>
