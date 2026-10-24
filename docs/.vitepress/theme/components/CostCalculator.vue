<script setup>
import { ref, computed, onMounted } from 'vue'

// ==================== 常量定义 ====================
// 参考基线：yolo26n + 640 + batch32 + 单 A100 → 1 epoch ≈ 1.8 h
const BASE_EPOCH_HOURS = 1.8
const BASE_PARAMS_M = 2.8
const BASE_RES = 640
const BASE_BATCH = 32
const COCO_TRAIN_SIZE = 118287
const AMP_FACTOR = 0.6

// 模型档位（参数量，单位：百万）
const MODELS = {
  yolo26n: { label: 'yolo26n', paramsM: 2.8 },
  yolo26s: { label: 'yolo26s', paramsM: 9.4 },
  yolo26m: { label: 'yolo26m', paramsM: 21.8 },
  yolo26l: { label: 'yolo26l', paramsM: 63.6 },
  yolo26x: { label: 'yolo26x', paramsM: 84.7 }
}

// GPU 性能因子（A100-80G = 1.0 基准）
const GPU_PERF = {
  'A100-80G': 1.0,
  'A100-40G': 0.95,
  'RTX4090': 0.65,
  'RTX3090': 0.5,
  'L40S': 0.85,
  'T4': 0.15
}

// GPU 显存（GB）
const GPU_VRAM = {
  'A100-80G': 80,
  'A100-40G': 40,
  'RTX4090': 24,
  'RTX3090': 24,
  'L40S': 48,
  'T4': 16
}

// GPU 单价（USD/小时）
const GPU_PRICE = {
  'A100-80G': 4.5,
  'A100-40G': 3.5,
  'RTX4090': 1.2,
  'RTX3090': 0.9,
  'L40S': 3.0,
  'T4': 0.5
}

// 多卡效率（非线性扩展）
const CARD_EFF = { 1: 1.0, 2: 0.88, 4: 0.78, 8: 0.65 }

// 分辨率选项
const RES_OPTIONS = [320, 416, 480, 640, 960, 1280]

// 数据集档位（含自定义）
const DATASET_PRESETS = [
  { label: '1k（原型验证）', value: 1000 },
  { label: '10k（小规模）', value: 10000 },
  { label: '100k（中规模）', value: 100000 }
]

// 平均目标数档位
const OBJECT_PRESETS = [
  { label: '10（稀疏）', value: 10 },
  { label: '50（常规）', value: 50 },
  { label: '200（密集）', value: 200 }
]

// ==================== 响应式状态 ====================
const imgCount = ref(118287)
const objectsPerImg = ref(50)
const resolution = ref(640)
const modelKey = ref('yolo26n')
const epochs = ref(300)
const batchSize = ref(32)
const gpuModel = ref('A100-80G')
const useAmp = ref(true)
const numCards = ref(1)

// 自定义数值输入
const imgCustomMode = ref(false)
const objCustomMode = ref(false)

// 保存/加载提示
const notice = ref('')
let noticeTimer = null

function flash(msg) {
  notice.value = msg
  if (noticeTimer) clearTimeout(noticeTimer)
  noticeTimer = setTimeout(() => { notice.value = '' }, 2400)
}

// ==================== 核心估算 ====================
// 返回 { totalHours, singleCardHours, vramGB, totalCost, recommendedBatch, recommendedCards, factors, warnings }
function calculate() {
  const model = MODELS[modelKey.value]
  const res = resolution.value
  const batch = batchSize.value
  const imgN = imgCount.value
  const objN = objectsPerImg.value
  const ep = epochs.value
  const gpu = gpuModel.value
  const amp = useAmp.value
  const cards = numCards.value

  const paramsM = model.paramsM
  const resFactor = (res / BASE_RES) ** 2
  const batchFactor = BASE_BATCH / batch
  const imgFactor = imgN / COCO_TRAIN_SIZE
  const paramsFactor = paramsM / BASE_PARAMS_M
  const ampFactor = amp ? AMP_FACTOR : 1.0
  const gpuFactor = GPU_PERF[gpu] || 1.0
  const cardEff = CARD_EFF[cards] || 1.0
  // 目标数轻微影响：约 (objN/50)^0.3
  const objFactor = Math.pow(Math.max(objN, 1) / 50, 0.3)

  // 单卡时长（总 epoch）
  const singleCardHours =
    BASE_EPOCH_HOURS *
    paramsFactor *
    resFactor *
    batchFactor *
    imgFactor *
    ampFactor *
    objFactor *
    ep /
    gpuFactor /
    cards /
    cardEff

  const totalHours = singleCardHours // 总 wall clock 时长已含并行加速

  // 显存：模型权重 + 激活
  // 权重 = paramsM * 4 bytes = paramsM / 256 GB（简化）
  const weightGB = (paramsM * 1e6 * 4) / (1024 ** 3)
  const activationGB = imgFactor * resFactor * paramsM * 4 * batch / 2048
  const vramGB = weightGB + activationGB

  // 显存不足判定
  const vramCap = GPU_VRAM[gpu]
  const vramSafe = vramCap * 0.9
  const oom = vramGB > vramSafe

  // 推荐 batch size（若 OOM，找最大可用）
  let recommendedBatch = batch
  let probeBatch = batch
  while (probeBatch > 1) {
    probeBatch = Math.max(1, Math.floor(probeBatch / 2))
    const probeGB = weightGB + imgFactor * resFactor * paramsM * 4 * probeBatch / 2048
    if (probeGB <= vramSafe) { recommendedBatch = probeBatch; break }
  }

  // 推荐卡数（若单卡 OOM 或过慢）
  let recommendedCards = cards
  if (oom && recommendedBatch < batch) {
    const nextCards = cards < 2 ? 2 : cards < 4 ? 4 : cards < 8 ? 8 : 8
    recommendedCards = nextCards
  }

  // 总成本
  const totalCost = totalHours * (GPU_PRICE[gpu] || 1.0) * cards

  // 各因子明细
  const factors = [
    { name: '模型参数因子', expr: `(${paramsM}M / 2.8M)`, value: paramsFactor },
    { name: '分辨率因子', expr: `(${res}/640)²`, value: resFactor },
    { name: 'Batch size 因子', expr: `32/${batch}`, value: batchFactor },
    { name: '数据量因子', expr: `${imgN}/118287`, value: imgFactor },
    { name: '目标数因子', expr: `(${objN}/50)^0.3`, value: objFactor },
    { name: 'AMP 混合精度', expr: amp ? '×0.6' : '×1.0', value: ampFactor },
    { name: 'GPU 性能', expr: `${gpu} → ×${gpuFactor}`, value: 1 / gpuFactor },
    { name: '卡数并行', expr: `${cards} × eff ${cardEff}`, value: 1 / (cards * cardEff) },
    { name: 'Epoch 数', expr: `×${ep}`, value: ep }
  ]

  return {
    totalHours,
    singleCardHours,
    vramGB,
    vramCap,
    vramSafe,
    totalCost,
    recommendedBatch,
    recommendedCards,
    oom,
    factors,
    warnings: buildWarnings(vramGB, vramCap, totalHours, cards, objN, imgN)
  }
}

function buildWarnings(vramGB, vramCap, totalHours, cards, objN, imgN) {
  const w = []
  if (vramGB > vramCap * 0.9) {
    w.push('显存不足，建议减小 batch size 或换更大显存 GPU')
  } else if (vramGB > vramCap * 0.7) {
    w.push('显存接近上限（>70%），建议 batch size -50% 或启用 AMP')
  }
  if (totalHours > 500) {
    w.push('训练时长 >500 小时，考虑减 epochs 或换更强 GPU')
  }
  if (cards >= 4 && CARD_EFF[cards] < 0.8) {
    w.push('多卡扩展效率 <80%，加卡边际收益递减')
  }
  if (objN > 150) {
    w.push('密集目标（>150/张）建议 P2 head 或提高分辨率')
  }
  if (imgN < 500) {
    w.push('数据量 <500 张，建议用 copy_paste / mixup 增强')
  }
  return w
}

// 计算结果（响应式）
const result = computed(calculate)

// ==================== 保存 / 加载 ====================
const STORAGE_KEY = 'cost-calc-config'

function saveConfig() {
  const payload = {
    imgCount: imgCount.value,
    objectsPerImg: objectsPerImg.value,
    resolution: resolution.value,
    modelKey: modelKey.value,
    epochs: epochs.value,
    batchSize: batchSize.value,
    gpuModel: gpuModel.value,
    useAmp: useAmp.value,
    numCards: numCards.value,
    imgCustomMode: imgCustomMode.value,
    objCustomMode: objCustomMode.value,
    ts: Date.now()
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
    flash('配置已保存到 localStorage')
  } catch (e) {
    flash('保存失败：' + (e.message || e))
  }
}

function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) { flash('未找到已保存配置'); return }
    const p = JSON.parse(raw)
    if (p.modelKey) modelKey.value = p.modelKey
    if (typeof p.epochs === 'number') epochs.value = p.epochs
    if (typeof p.batchSize === 'number') batchSize.value = p.batchSize
    if (p.gpuModel) gpuModel.value = p.gpuModel
    if (typeof p.useAmp === 'boolean') useAmp.value = p.useAmp
    if (typeof p.numCards === 'number') numCards.value = p.numCards
    if (typeof p.resolution === 'number') resolution.value = p.resolution
    if (typeof p.imgCount === 'number') {
      imgCount.value = p.imgCount
      imgCustomMode.value = ![1000, 10000, 100000].includes(p.imgCount)
    }
    if (typeof p.objectsPerImg === 'number') {
      objectsPerImg.value = p.objectsPerImg
      objCustomMode.value = ![10, 50, 200].includes(p.objectsPerImg)
    }
    flash('配置已恢复')
  } catch (e) {
    flash('加载失败：' + (e.message || e))
  }
}

onMounted(() => {
  // 尝试自动加载
  if (localStorage.getItem(STORAGE_KEY)) loadConfig()
})

// ==================== 辅助格式化 ====================
function fmtHours(h) {
  if (!isFinite(h)) return '—'
  if (h < 1) return (h * 60).toFixed(0) + ' 分钟'
  if (h < 100) return h.toFixed(1) + ' h'
  if (h < 1000) return h.toFixed(0) + ' h'
  return (h / 24).toFixed(1) + ' 天'
}
function fmtMoney(v) {
  if (!isFinite(v)) return '—'
  if (v >= 10000) return '$' + (v / 1000).toFixed(1) + 'k'
  if (v >= 100) return '$' + v.toFixed(0)
  return '$' + v.toFixed(2)
}
function fmtGB(v) {
  if (!isFinite(v)) return '—'
  if (v >= 100) return v.toFixed(0) + ' GB'
  return v.toFixed(1) + ' GB'
}
</script>

<template>
  <div class="cc-wrap">
    <div class="cc-grid">
      <!-- 左：输入参数 -->
      <div class="cc-panel">
        <div class="cc-title">输入参数</div>

        <div class="cc-field">
          <label>数据集图像数</label>
          <div class="cc-presets">
            <button v-for="p in DATASET_PRESETS" :key="p.value"
                    type="button" class="cc-chip" :class="{ on: !imgCustomMode && imgCount === p.value }"
                    @click="imgCustomMode = false; imgCount = p.value">{{ p.label }}</button>
            <button type="button" class="cc-chip" :class="{ on: imgCustomMode }"
                    @click="imgCustomMode = true">自定义</button>
          </div>
          <input v-if="imgCustomMode" v-model.number="imgCount" type="number" min="10" step="10"
                 class="cc-input" placeholder="图像数" />
        </div>

        <div class="cc-field">
          <label>平均目标数 / 张</label>
          <div class="cc-presets">
            <button v-for="p in OBJECT_PRESETS" :key="p.value"
                    type="button" class="cc-chip" :class="{ on: !objCustomMode && objectsPerImg === p.value }"
                    @click="objCustomMode = false; objectsPerImg = p.value">{{ p.label }}</button>
            <button type="button" class="cc-chip" :class="{ on: objCustomMode }"
                    @click="objCustomMode = true">自定义</button>
          </div>
          <input v-if="objCustomMode" v-model.number="objectsPerImg" type="number" min="1" max="500" step="1"
                 class="cc-input" placeholder="目标数" />
        </div>

        <div class="cc-field">
          <label>输入分辨率</label>
          <div class="cc-presets">
            <button v-for="r in RES_OPTIONS" :key="r" type="button"
                    class="cc-chip" :class="{ on: resolution === r }"
                    @click="resolution = r">{{ r }}</button>
          </div>
        </div>

        <div class="cc-field">
          <label>模型档位</label>
          <div class="cc-presets">
            <button v-for="(m, k) in MODELS" :key="k" type="button"
                    class="cc-chip" :class="{ on: modelKey === k }"
                    @click="modelKey = k">{{ m.label }} · {{ m.paramsM }}M</button>
          </div>
        </div>

        <div class="cc-row">
          <div class="cc-field">
            <label>Epochs</label>
            <select v-model.number="epochs" class="cc-select">
              <option v-for="e in [100, 200, 300, 500]" :key="e" :value="e">{{ e }}</option>
            </select>
          </div>
          <div class="cc-field">
            <label>Batch size</label>
            <select v-model.number="batchSize" class="cc-select">
              <option v-for="b in [16, 32, 64, 128]" :key="b" :value="b">{{ b }}</option>
            </select>
          </div>
        </div>

        <div class="cc-field">
          <label>GPU 型号</label>
          <select v-model="gpuModel" class="cc-select">
            <option v-for="g in Object.keys(GPU_PERF)" :key="g" :value="g">
              {{ g }}（{{ GPU_VRAM[g] }}GB · ${{ GPU_PRICE[g] }}/h）
            </option>
          </select>
        </div>

        <div class="cc-row">
          <div class="cc-field">
            <label>卡数</label>
            <select v-model.number="numCards" class="cc-select">
              <option v-for="c in [1, 2, 4, 8]" :key="c" :value="c">{{ c }}</option>
            </select>
          </div>
          <div class="cc-field">
            <label>AMP 混合精度</label>
            <label class="cc-toggle">
              <input v-model="useAmp" type="checkbox" />
              <span>{{ useAmp ? '开启' : '关闭' }}</span>
            </label>
          </div>
        </div>

        <div class="cc-actions">
          <button type="button" class="cc-btn" @click="saveConfig">💾 保存配置</button>
          <button type="button" class="cc-btn cc-btn-ghost" @click="loadConfig">📂 加载配置</button>
        </div>
        <div v-if="notice" class="cc-notice">{{ notice }}</div>
      </div>

      <!-- 中：结果 -->
      <div class="cc-panel">
        <div class="cc-title">估算结果</div>

        <div class="cc-hero">
          <div class="cc-hero-num">{{ fmtHours(result.totalHours) }}</div>
          <div class="cc-hero-sub">总训练时长（wall clock）</div>
        </div>

        <dl class="cc-stat">
          <dt>单卡时长</dt><dd>{{ fmtHours(result.singleCardHours) }}</dd>
          <dt>显存峰值</dt><dd :class="{ danger: result.oom }">{{ fmtGB(result.vramGB) }}</dd>
          <dt>GPU 显存</dt><dd>{{ result.vramCap }} GB（安全线 {{ result.vramSafe.toFixed(0) }} GB）</dd>
          <dt>总成本</dt><dd>{{ fmtMoney(result.totalCost) }}</dd>
          <dt>推荐 batch</dt><dd>{{ result.oom ? result.recommendedBatch : batchSize }}</dd>
          <dt>推荐卡数</dt><dd>{{ result.recommendedCards }}</dd>
        </dl>

        <div v-if="result.warnings.length" class="cc-warnings">
          <div class="cc-title">⚠ 警告</div>
          <ul>
            <li v-for="(w, i) in result.warnings" :key="i"
                class="cc-warn" :class="{ danger: i === 0 && result.oom }">{{ w }}</li>
          </ul>
        </div>
      </div>

      <!-- 右：明细 -->
      <div class="cc-panel">
        <div class="cc-title">估算因子明细</div>
        <table class="cc-table">
          <thead>
            <tr><th>因子</th><th>表达式</th><th>值</th></tr>
          </thead>
          <tbody>
            <tr v-for="(f, i) in result.factors" :key="i">
              <td>{{ f.name }}</td>
              <td class="cc-mono">{{ f.expr }}</td>
              <td class="cc-mono">{{ f.value.toFixed(3) }}</td>
            </tr>
          </tbody>
        </table>

        <div class="cc-note">
          <div class="cc-title">计算公式</div>
          <pre class="cc-formula">单卡时长 = 1.8h × (P/2.8) × (res/640)² × (32/batch)
         × (N/118287) × (obj/50)^0.3 × AMP × epochs
         ÷ GPU_factor ÷ (cards × card_eff)
显存 ≈ P×4bytes + N_factor × (res/640)² × P × 4 × batch / 2048 GB</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cc-wrap {
  --cc-primary: #059669;
  --cc-primary-dark: #047857;
  --cc-border: #e2e8f0;
  --cc-bg: #f8fafc;
  --cc-text: #0f172a;
  --cc-muted: #64748b;
  --cc-danger: #b91c1c;
  color: var(--cc-text);
  font-size: 13px;
  line-height: 1.5;
}
.cc-grid { display: grid; grid-template-columns: 300px 280px 1fr; gap: 14px; }
@media (max-width: 1080px) { .cc-grid { grid-template-columns: 1fr; } }
.cc-panel {
  background: var(--cc-bg); border: 1px solid var(--cc-border); border-radius: 8px;
  padding: 14px; display: flex; flex-direction: column; gap: 12px;
}
.cc-title {
  font-weight: 600; font-size: 12px; color: var(--cc-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.cc-field { display: flex; flex-direction: column; gap: 6px; }
.cc-field > label { font-size: 12px; color: var(--cc-muted); font-weight: 600; }
.cc-presets { display: flex; flex-wrap: wrap; gap: 4px; }
.cc-chip {
  padding: 4px 10px; border: 1px solid var(--cc-border); background: #fff;
  border-radius: 14px; cursor: pointer; font-size: 12px; color: var(--cc-text);
}
.cc-chip:hover { border-color: var(--cc-primary); color: var(--cc-primary); }
.cc-chip.on {
  background: var(--cc-primary); border-color: var(--cc-primary);
  color: #fff; font-weight: 600;
}
.cc-input, .cc-select {
  width: 100%; padding: 6px 8px; border: 1px solid var(--cc-border);
  border-radius: 6px; font-size: 12px; background: #fff; color: var(--cc-text);
}
.cc-input:focus, .cc-select:focus { outline: none; border-color: var(--cc-primary); }
.cc-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.cc-toggle {
  display: flex; align-items: center; gap: 6px; padding: 6px 8px;
  border: 1px solid var(--cc-border); border-radius: 6px; background: #fff;
  cursor: pointer; font-size: 12px;
}
.cc-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.cc-btn {
  padding: 6px 12px; border: 1px solid var(--cc-primary); background: var(--cc-primary);
  color: #fff; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;
}
.cc-btn:hover { background: var(--cc-primary-dark); border-color: var(--cc-primary-dark); }
.cc-btn-ghost { background: #fff; color: var(--cc-primary); }
.cc-btn-ghost:hover { background: #ecfdf5; }
.cc-notice {
  padding: 6px 10px; background: #ecfdf5; border: 1px solid #a7f3d0;
  color: var(--cc-primary-dark); border-radius: 6px; font-size: 12px;
}
.cc-hero {
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
  border: 1px solid #a7f3d0; border-radius: 8px; padding: 16px; text-align: center;
}
.cc-hero-num {
  font-size: 28px; font-weight: 700; color: var(--cc-primary-dark);
  font-variant-numeric: tabular-nums;
}
.cc-hero-sub { font-size: 12px; color: var(--cc-muted); margin-top: 2px; }
.cc-stat { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; margin: 0; font-size: 13px; }
.cc-stat dt { color: var(--cc-muted); }
.cc-stat dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
.cc-stat dd.danger { color: var(--cc-danger); }
.cc-warnings { display: flex; flex-direction: column; gap: 6px; }
.cc-warnings ul { margin: 6px 0 0 0; padding-left: 18px; }
.cc-warn {
  padding: 6px 10px; background: #fff7ed; border: 1px solid #fed7aa;
  color: #9a3412; border-radius: 6px; font-size: 12px; margin-bottom: 4px;
}
.cc-warn.danger { background: #fef2f2; border-color: #fecaca; color: var(--cc-danger); font-weight: 600; }
.cc-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.cc-table th, .cc-table td {
  border: 1px solid var(--cc-border); padding: 5px 8px; text-align: left;
}
.cc-table th {
  background: #fff; font-weight: 600; color: var(--cc-muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.03em;
}
.cc-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; }
.cc-note { margin-top: 6px; }
.cc-formula {
  margin: 6px 0 0 0; padding: 10px; background: #0f172a; color: #e2e8f0;
  border-radius: 6px; font-size: 11px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre; overflow-x: auto; line-height: 1.5;
}
</style>
