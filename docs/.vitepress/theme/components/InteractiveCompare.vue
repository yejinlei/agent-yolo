<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

// ==================== 模型定义 ====================
// 三个可选模型：本地打包在 docs/public/models/，避开 HuggingFace CDN 在国内访问慢/失败
// VitePress 会把 window.__VP_SITE_DATA__.base 注入全局，用于拼接站点 base 路径（如 /agent-yolo/）
function siteBase() {
  const w = window.__VP_SITE_DATA__
  if (!w) return '/'
  return (w.base || '/').replace(/\/$/, '') + '/'
}
const MODELS = {
  yolo26n: {
    url: () => siteBase() + 'models/yolo26n.onnx',
    label: 'yolo26n（2.4M · Preview）',
    note: '最新小模型，本地 ONNX 9.5MB。'
  },
  yolo11n: {
    url: () => siteBase() + 'models/yolo11n.onnx',
    label: 'yolo11n（2.6M · 主流）',
    note: '主流小模型，本地 ONNX 10.2MB。'
  },
  yolov8n: {
    url: () => siteBase() + 'models/yolov8n.onnx',
    label: 'yolov8n（3.2M · 基线）',
    note: '上一代基线，本地 ONNX 12.3MB。'
  }
}

// COCO 80 类中的前 20 类简写；其余归为 "other"
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane',
  'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
  'cat', 'dog', 'horse', 'sheep', 'cow'
]

// onnxruntime-web：主 CDN 是 npmmirror（国内可达），本地 /ort 目录作为兜底
// ort.min.js 会在加载时根据 window.ort.env.wasm.wasmFiles 拼出 wasm 文件路径
const ORT_CDN_LOCAL = () => siteBase() + 'ort/'
const ORT_CDN_FALLBACKS = [
  'https://registry.npmmirror.com/onnxruntime-web/1.20.0/files/dist/',
  'https://fastly.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/',
  'https://unpkg.com/onnxruntime-web@1.20.0/dist/'
]

// 内置示例图（本地打包，避开 Unsplash 外链）
const SAMPLE_IMG = () => siteBase() + 'models/sample.jpg'

// ==================== 响应式状态 ====================
const modelKey = ref('yolov8n')
const confThresh = ref(0.45)
const iouThresh = ref(0.5)

const status = ref('idle') // idle | loading-ort | loading-model | running | done | error
const errorMsg = ref('')
const log = ref([])
const canvasRef = ref(null)
const fileInputRef = ref(null)
const dragOver = ref(false)

const image = ref(null) // HTMLImageElement
const imageSize = ref({ w: 0, h: 0 })
const letterbox = ref({ padX: 0, padY: 0, scale: 1 })
const detections = ref([])
const perf = ref({
  modelSize: '',
  loadTime: '',
  inferTime: '',
  objectCount: 0
})

const currentModel = computed(() => MODELS[modelKey.value])

// 日志追加
function pushLog(msg, kind = 'info') {
  const t = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  log.value.push({ t, msg, kind })
  // 保留最近 50 条
  if (log.value.length > 50) log.value = log.value.slice(-50)
}

// ==================== onnxruntime-web 加载 ====================
let ortSession = null
let ortCache = null // key = modelKey

function injectScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script')
    s.src = src
    s.crossOrigin = 'anonymous'
    s.onload = () => {
      if (window.ort) resolve(src)
      else reject(new Error(`${src} 加载成功但未暴露 window.ort`))
    }
    s.onerror = () => reject(new Error(`无法加载 ${src}`))
    document.head.appendChild(s)
  })
}

async function ensureOrt() {
  if (typeof window !== 'undefined' && window.ort) return window.ort
  status.value = 'loading-ort'
  pushLog('加载 onnxruntime-web ...', 'loading')

  // 候选源：本地打包 → npmmirror → fastly.jsdelivr → unpkg
  const candidates = [ORT_CDN_LOCAL() + 'ort.min.js', ...ORT_CDN_FALLBACKS.map(u => u + 'ort.min.js')]
  let lastErr = null
  for (const url of candidates) {
    try {
      const loaded = await injectScript(url)
      // 关键：让 ort 从该 URL 所在目录加载 wasm 文件
      window.ort.env.wasm.wasmFiles = loaded.replace(/ort\.min\.js$/, '')
      window.ort.env.wasm.numThreads = 4
      window.ort.env.wasm.proxy = true
      window.ort.env.wasm.threads = true
      pushLog(`onnxruntime-web 就绪（源：${loaded} · WASM/SIMD · 4 线程）`, 'success')
      return window.ort
    } catch (e) {
      lastErr = e
      pushLog(`加载源失败：${url} · ${e.message}`, 'loading')
    }
  }
  status.value = 'error'
  throw lastErr || new Error('所有 onnxruntime-web 源均加载失败')
}

async function ensureSession() {
  if (ortCache && ortCache.key === modelKey.value) return ortCache.session
  const ort = await ensureOrt()
  const m = currentModel.value
  const url = typeof m.url === 'function' ? m.url() : m.url
  status.value = 'loading-model'
  pushLog(`加载模型：${m.label}（${url}）`, 'loading')
  const t0 = performance.now()
  // 本地 fetch，无需 CORS；失败时给可读提示
  const resp = await fetch(url, { cache: 'force-cache' })
  if (!resp.ok) throw new Error(`模型加载失败：HTTP ${resp.status} ${url}`)
  const buf = await resp.arrayBuffer()
  const sizeMB = (buf.byteLength / 1024 / 1024).toFixed(2)
  perf.value.modelSize = `${sizeMB} MB`
  pushLog(`模型已加载（${sizeMB} MB），开始构建 session...`, 'loading')
  const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] })
  const loadMs = Math.round(performance.now() - t0)
  perf.value.loadTime = `${loadMs} ms`
  pushLog(`Session 就绪，耗时 ${loadMs} ms`, 'success')
  ortCache = { key: modelKey.value, session }
  return session
}

// ==================== 图片加载 ====================
function loadImageFromFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    pushLog('请选择图片文件', 'error')
    return
  }
  const url = URL.createObjectURL(file)
  loadImage(url).then(() => URL.revokeObjectURL(url))
}

function isCrossOrigin(url) {
  try {
    const u = new URL(url, location.href)
    return u.origin !== location.origin
  } catch {
    return false
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    // 只有跨域时才设 crossOrigin=anonymous；同域设置会强制 CORS 预检，
    // 导致 VitePress dev/prod 未配 ACAO 头时 onerror 直接触发。
    if (isCrossOrigin(src)) img.crossOrigin = 'anonymous'
    img.onload = () => {
      image.value = img
      imageSize.value = { w: img.naturalWidth, h: img.naturalHeight }
      perf.value.objectCount = 0
      detections.value = []
      pushLog(`图片加载完成：${img.naturalWidth}×${img.naturalHeight}`, 'success')
      drawBase()
      resolve(img)
    }
    img.onerror = () => reject(new Error('图片加载失败：' + src))
    img.src = src
  })
}

async function loadSample() {
  status.value = 'loading-model'
  try {
    await loadImage(SAMPLE_IMG())
  } catch (e) {
    pushLog(e.message, 'error')
    status.value = 'error'
    errorMsg.value = e.message
  }
}

// ==================== 预处理：letterbox + normalize ====================
function letterboxTo640(img) {
  const size = 640
  const iw = img.naturalWidth
  const ih = img.naturalHeight
  const scale = Math.min(size / iw, size / ih)
  const nw = Math.round(iw * scale)
  const nh = Math.round(ih * scale)
  const padX = Math.floor((size - nw) / 2)
  const padY = Math.floor((size - nh) / 2)

  // 用 canvas 生成 640×640 letterboxed 图像
  const cv = document.createElement('canvas')
  cv.width = size
  cv.height = size
  const ctx = cv.getContext('2d')
  ctx.fillStyle = '#808080' // Ultralytics 默认灰底
  ctx.fillRect(0, 0, size, size)
  ctx.drawImage(img, padX, padY, nw, nh)
  const data = ctx.getImageData(0, 0, size, size)

  // HWC uint8 → CHW float32，除以 255
  const arr = new Float32Array(size * size * 3)
  const px = data.data
  for (let i = 0; i < size * size; i++) {
    arr[i] = px[i * 4] / 255
    arr[size * size + i] = px[i * 4 + 1] / 255
    arr[size * size * 2 + i] = px[i * 4 + 2] / 255
  }

  return {
    tensor: new window.ort.Tensor('float32', arr, [1, 3, size, size]),
    meta: { scale, padX, padY, nw, nh }
  }
}

// ==================== 后处理 ====================
function iou(a, b) {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x) * (a.y2 - a.y)
  const areaB = (b.x2 - b.x) * (b.y2 - b.y)
  const uni = areaA + areaB - inter
  return uni <= 0 ? 0 : inter / uni
}

function nms(boxes, iouThresh) {
  // boxes: [{x,y,x2,y2,score,label,idx}]
  // 按分数降序
  boxes.sort((a, b) => b.score - a.score)
  const keep = []
  const suppressed = new Set()
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue
    keep.push(boxes[i])
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed.has(j)) continue
      if (boxes[i].label !== boxes[j].label) continue
      if (iou(boxes[i], boxes[j]) > iouThresh) suppressed.add(j)
    }
  }
  return keep
}

function classLabel(cls) {
  return cls < COCO_CLASSES.length ? COCO_CLASSES[cls] : 'other'
}

function postProcess(output, meta) {
  // Ultralytics ONNX 输出：[1, 84, 8400]
  // 84 = 4 (cx,cy,w,h) + 80 class scores
  // 部分模型可能是 [1, 6, 8400]（含 objectness），此处按 [1, 84, 8400] 处理，其他形状给出提示
  const out = output.data // Float32Array
  let numAttr, numBoxes, objIdx = -1
  if (output.dims.length === 3) {
    const [_, a, b] = output.dims
    // 判断哪个是 attribute 维度（84 或 6）
    if (a === 84 && b === 8400) { numAttr = 84; numBoxes = 8400 }
    else if (b === 84 && a === 8400) { numAttr = 84; numBoxes = 8400 }
    else if (a === 6 && b === 8400) { numAttr = 6; numBoxes = 8400; objIdx = 4 }
    else if (b === 6 && a === 8400) { numAttr = 6; numBoxes = 8400; objIdx = 4 }
    else throw new Error(`不支持的输出形状：[1, ${a}, ${b}]（期望 [1,84,8400]）`)
  } else {
    throw new Error(`不支持的输出维度：${JSON.stringify(output.dims)}`)
  }

  const numClasses = numAttr - 4
  const confT = confThresh.value
  const size = 640
  const { scale, padX, padY } = meta
  const imgW = imageSize.value.w
  const imgH = imageSize.value.h

  // 反 letterbox：从 640 坐标 → 原图坐标
  const toImg = (cx, cy, w, h) => {
    const x1 = (cx - w / 2 - padX) / scale
    const y1 = (cy - h / 2 - padY) / scale
    const x2 = (cx + w / 2 - padX) / scale
    const y2 = (cy + h / 2 - padY) / scale
    return {
      x: Math.max(0, Math.min(imgW, x1)),
      y: Math.max(0, Math.min(imgH, y1)),
      x2: Math.max(0, Math.min(imgW, x2)),
      y2: Math.max(0, Math.min(imgH, y2))
    }
  }

  // 展平为 [numBoxes, numAttr]
  // 输出是 [numAttr, numBoxes] 布局：out[attr*numBoxes + box]
  const boxes = []
  for (let i = 0; i < numBoxes; i++) {
    let objScore = 1
    if (objIdx >= 0) {
      objScore = out[objIdx * numBoxes + i]
      if (objScore < confT) continue
    }
    // 找最大类分数
    let bestCls = -1
    let bestScore = 0
    for (let c = 0; c < numClasses; c++) {
      const sc = out[(objIdx >= 0 ? c + 5 : c + 4) * numBoxes + i] * (objIdx >= 0 ? objScore : 1)
      if (sc > bestScore) { bestScore = sc; bestCls = c }
    }
    if (bestScore < confT) continue
    // 4 个 box 属性
    const cx = out[0 * numBoxes + i]
    const cy = out[1 * numBoxes + i]
    const w = out[2 * numBoxes + i]
    const h = out[3 * numBoxes + i]
    const b = toImg(cx, cy, w, h)
    boxes.push({ ...b, score: bestScore, label: classLabel(bestCls), idx: i })
  }
  return nms(boxes, iouThresh.value)
}

// ==================== 推理 ====================
async function detect() {
  if (!image.value) {
    pushLog('请先上传图片', 'error')
    return
  }
  status.value = 'running'
  errorMsg.value = ''
  detections.value = []
  try {
    const session = await ensureSession()
    const { tensor, meta } = letterboxTo640(image.value)
    const t0 = performance.now()
    const results = await session.run({ input: tensor })
    const inferMs = Math.round(performance.now() - t0)
    const outputs = results.input ?? Object.values(results)[0]
    const preds = postProcess(outputs, meta)
    detections.value = preds
    perf.value.inferTime = `${inferMs} ms`
    perf.value.objectCount = preds.length
    status.value = 'done'
    pushLog(`推理完成：${inferMs} ms，检出 ${preds.length} 个目标`, 'success')
    drawWithBoxes()
  } catch (e) {
    status.value = 'error'
    errorMsg.value = e.message || String(e)
    pushLog('错误：' + errorMsg.value, 'error')
    drawBase()
  }
}

// ==================== 画布 ====================
function getCanvas() {
  return canvasRef.value
}

function drawBase() {
  const cv = getCanvas()
  if (!cv || !image.value) return
  const img = image.value
  // 显示区域限制到 max 720 宽，保持比例
  const maxW = 720
  const maxH = 480
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1)
  cv.width = Math.round(img.naturalWidth * scale)
  cv.height = Math.round(img.naturalHeight * scale)
  const ctx = cv.getContext('2d')
  ctx.fillStyle = '#0f172a'
  ctx.fillRect(0, 0, cv.width, cv.height)
  ctx.drawImage(img, 0, 0, cv.width, cv.height)
}

function drawWithBoxes() {
  drawBase()
  const cv = getCanvas()
  if (!cv) return
  const ctx = cv.getContext('2d')
  const scale = cv.width / image.value.naturalWidth
  // 颜色板
  const COLORS = [
    '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#a855f7', '#06b6d4'
  ]
  for (const d of detections.value) {
    const x = d.x * scale
    const y = d.y * scale
    const w = (d.x2 - d.x) * scale
    const h = (d.y2 - d.y) * scale
    const color = COLORS[Math.floor(d.score * 100) % COLORS.length]
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, w, h)
    // 标签
    const label = `${d.label} ${(d.score * 100).toFixed(0)}%`
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif'
    const tw = ctx.measureText(label).width + 8
    const th = 18
    const ly = y - th > 0 ? y - th : y
    ctx.fillStyle = color
    ctx.fillRect(x, ly, tw, th)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x + 4, ly + 13)
  }
}

// ==================== 事件处理 ====================
function onDragOver(e) {
  e.preventDefault()
  dragOver.value = true
}
function onDragLeave() { dragOver.value = false }
function onDrop(e) {
  e.preventDefault()
  dragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) loadImageFromFile(file)
}
function onFilePick(e) {
  const file = e.target?.files?.[0]
  if (file) loadImageFromFile(file)
  if (fileInputRef.value) fileInputRef.value.value = ''
}
function onModelChange() {
  // 模型切换：清掉缓存，重跑
  ortCache = null
  status.value = 'idle'
  pushLog(`切换到 ${currentModel.value.label}`, 'info')
}
function onParamChange() {
  if (status.value === 'done') {
    // 参数变了重跑
    detect()
  }
}

// ==================== 生命周期 ====================
onMounted(() => {
  pushLog('就绪 · 请上传图片或点击「加载示例图」', 'info')
})

onBeforeUnmount(() => {
  // session 由 ort 内部持有，无需手动释放；ortSession 未使用局部变量做管理
})
</script>

<template>
  <div class="ic-wrap">
    <div class="ic-grid">
      <!-- 左栏：模型选择 + 参数 + 图片输入 -->
      <div class="ic-panel ic-left">
        <div class="ic-section">
          <div class="ic-title">模型选择</div>
          <div class="ic-models">
            <label v-for="(m, k) in MODELS" :key="k" class="ic-radio">
              <input type="radio" :value="k" v-model="modelKey" @change="onModelChange" />
              <span class="ic-radio-label">{{ m.label }}</span>
              <span class="ic-radio-note">{{ m.note }}</span>
            </label>
          </div>
        </div>

        <div class="ic-section">
          <div class="ic-title">推理参数</div>
          <div class="ic-slider">
            <span class="ic-slider-label">置信度阈值</span>
            <input type="range" min="0.25" max="0.95" step="0.01" v-model.number="confThresh" @input="onParamChange" />
            <span class="ic-slider-val">{{ confThresh.toFixed(2) }}</span>
          </div>
          <div class="ic-slider">
            <span class="ic-slider-label">IoU 阈值</span>
            <input type="range" min="0.2" max="0.9" step="0.01" v-model.number="iouThresh" @input="onParamChange" />
            <span class="ic-slider-val">{{ iouThresh.toFixed(2) }}</span>
          </div>
        </div>

        <div class="ic-section">
          <div class="ic-title">图片输入</div>
          <div
            class="ic-drop"
            :class="{ 'is-dragover': dragOver }"
            @dragover="onDragOver"
            @dragleave="onDragLeave"
            @drop="onDrop"
          >
            <input ref="fileInputRef" type="file" accept="image/*" @change="onFilePick" class="ic-file" />
            <span class="ic-drop-text">拖拽图片到此处，或</span>
            <button type="button" class="ic-btn" @click.stop.prevent="fileInputRef?.click()">点击选择</button>
          </div>
          <button type="button" class="ic-btn ic-btn-ghost" @click="loadSample">加载示例图</button>
        </div>

        <button type="button" class="ic-btn ic-btn-primary" :disabled="status !== 'idle' && status !== 'done'" @click="detect">
          <span v-if="status === 'running'">推理中…</span>
          <span v-else-if="status === 'loading-model' || status === 'loading-ort'">加载模型…</span>
          <span v-else>▶ 运行推理</span>
        </button>

        <div v-if="errorMsg" class="ic-error">⚠ {{ errorMsg }}</div>
      </div>

      <!-- 中栏：画布 -->
      <div class="ic-panel ic-center">
        <div class="ic-canvas-wrap">
          <canvas ref="canvasRef" class="ic-canvas"></canvas>
          <div v-if="!image" class="ic-canvas-placeholder">
            <span>📷</span>
            <span>上传图片后此处显示结果</span>
          </div>
        </div>
      </div>

      <!-- 右栏：性能统计 + 日志 -->
      <div class="ic-panel ic-right">
        <div class="ic-section">
          <div class="ic-title">性能统计</div>
          <dl class="ic-stat">
            <dt>模型大小</dt>
            <dd>{{ perf.modelSize || '—' }}</dd>
            <dt>加载耗时</dt>
            <dd>{{ perf.loadTime || '—' }}</dd>
            <dt>推理耗时</dt>
            <dd>{{ perf.inferTime || '—' }}</dd>
            <dt>目标数</dt>
            <dd>{{ perf.objectCount }}</dd>
          </dl>
        </div>

        <div class="ic-section">
          <div class="ic-title">运行日志</div>
          <div class="ic-log">
            <div v-for="(l, i) in log" :key="i" class="ic-log-line" :class="'ic-log-' + l.kind">
              <span class="ic-log-t">{{ l.t }}</span>
              <span>{{ l.msg }}</span>
            </div>
            <div v-if="log.length === 0" class="ic-log-empty">暂无日志</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ic-wrap {
  --ic-primary: #059669;
  --ic-primary-dark: #047857;
  --ic-border: #e2e8f0;
  --ic-bg: #f8fafc;
  --ic-text: #0f172a;
  --ic-muted: #64748b;
  color: var(--ic-text);
  font-size: 13px;
  line-height: 1.5;
}

.ic-grid {
  display: grid;
  grid-template-columns: 260px 1fr 240px;
  gap: 14px;
}

@media (max-width: 960px) {
  .ic-grid { grid-template-columns: 1fr; }
}

.ic-panel {
  background: var(--ic-bg);
  border: 1px solid var(--ic-border);
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.ic-title {
  font-weight: 600;
  font-size: 12px;
  color: var(--ic-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
}

.ic-section { display: flex; flex-direction: column; gap: 6px; }

.ic-models { display: flex; flex-direction: column; gap: 4px; }
.ic-radio {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 8px;
  padding: 6px 8px;
  border: 1px solid var(--ic-border);
  border-radius: 6px;
  cursor: pointer;
  background: #fff;
}
.ic-radio:hover { border-color: var(--ic-primary); }
.ic-radio input { grid-row: 1 / span 2; align-self: center; }
.ic-radio-label { font-weight: 600; }
.ic-radio-note { font-size: 11px; color: var(--ic-muted); }

.ic-slider {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 8px;
  align-items: center;
}
.ic-slider-label { grid-column: 1 / span 2; font-size: 12px; color: var(--ic-muted); }
.ic-slider input { width: 100%; accent-color: var(--ic-primary); }
.ic-slider-val { font-variant-numeric: tabular-nums; font-weight: 600; }

.ic-drop {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: center;
  justify-content: center;
  padding: 14px;
  border: 2px dashed var(--ic-border);
  border-radius: 8px;
  background: #fff;
  min-height: 80px;
}
.ic-drop.is-dragover {
  border-color: var(--ic-primary);
  background: #ecfdf5;
}
.ic-drop-text { font-size: 12px; color: var(--ic-muted); }
.ic-file { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.ic-btn {
  padding: 6px 12px;
  border: 1px solid var(--ic-border);
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  color: var(--ic-text);
}
.ic-btn:hover { border-color: var(--ic-primary); color: var(--ic-primary); }
.ic-btn-ghost { background: transparent; }
.ic-btn-primary {
  background: var(--ic-primary);
  border-color: var(--ic-primary);
  color: #fff;
  font-weight: 600;
  padding: 8px 14px;
}
.ic-btn-primary:hover { background: var(--ic-primary-dark); color: #fff; }
.ic-btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.ic-error {
  padding: 8px 10px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #b91c1c;
  border-radius: 6px;
  font-size: 12px;
  word-break: break-all;
}

.ic-canvas-wrap {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fff;
  border: 1px solid var(--ic-border);
  border-radius: 8px;
  min-height: 320px;
  overflow: hidden;
}
.ic-canvas { max-width: 100%; height: auto; display: block; }
.ic-canvas-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--ic-muted);
  font-size: 13px;
}

.ic-stat {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 10px;
  margin: 0;
  font-size: 12px;
}
.ic-stat dt { color: var(--ic-muted); }
.ic-stat dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }

.ic-log {
  max-height: 240px;
  overflow-y: auto;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 6px;
  padding: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  line-height: 1.5;
}
.ic-log-line { display: block; word-break: break-all; }
.ic-log-t { color: #64748b; margin-right: 6px; }
.ic-log-info { color: #e2e8f0; }
.ic-log-loading { color: #fcd34d; }
.ic-log-success { color: #6ee7b7; }
.ic-log-error { color: #fca5a5; }
.ic-log-empty { color: #64748b; font-style: italic; }
</style>
