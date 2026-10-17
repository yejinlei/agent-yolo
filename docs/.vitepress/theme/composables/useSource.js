// @AI_GUARD: 源码快照只能来自 source/yolov*/snapshot/，组件通过 ?raw 在构建期内联，勿改成运行时 fetch
// @CONSTRAINT: glob 根路径相对本文件所在目录（docs/.vitepress/theme/composables/），到仓库根为 ../../../../
// @AI_GUARD: 每个快照独立 glob，避免不同 commit 的文件名冲突（如 v1 与 v2 都有 src/yolo.c）
import { withBase } from 'vitepress'

// 版本默认值：SrcLink 未显式指定 :version 时视为 v1（保持 v1 页面现有引用行为不变）
export const DEFAULT_VERSION = 'v1'

// @AI_GUARD: file 可写成 `src/yolo.c`（默认 v1）或 `v2:src/yolo.c`（显式版本）
// 组件层统一通过 splitKey 拆开，禁止在组件里手写 ':' 判断
export function splitKey(file) {
  const i = file.indexOf(':')
  if (i > 0) return [file.slice(0, i), file.slice(i + 1)]
  return [DEFAULT_VERSION, file]
}

const modules = {
  v1: import.meta.glob(
    '../../../../source/yolov1/snapshot/**/*.{c,h,cu,cfg}',
    { query: '?raw', import: 'default' }
  ),
  v2: import.meta.glob(
    '../../../../source/yolov2/snapshot/**/*.{c,h,cu,cfg}',
    { query: '?raw', import: 'default' }
  ),
  v3: import.meta.glob(
    '../../../../source/yolov3/snapshot/**/*.{c,h,cu,cfg}',
    { query: '?raw', import: 'default' }
  ),
  v4: import.meta.glob(
    '../../../../source/yolov4/snapshot/**/*.{c,h,cu,cfg}',
    { query: '?raw', import: 'default' }
  )
}

// @CONSTRAINT: 快照 → 站内浏览页 URL；白名单外的文件走 GitHub blob 外链
// @AI_GUARD: 每个快照只登记真正有浏览页的文件，SrcLink 会据此决定 target=_blank
export const SOURCE_PAGES = {
  // YOLOv1
  'v1:src/detection_layer.c': '/versions/v1/source/detection-layer',
  'v1:src/yolo.c':           '/versions/v1/source/yolo',
  'v1:src/box.c':            '/versions/v1/source/box',
  'v1:cfg/yolo.cfg':         '/versions/v1/source/yolo-cfg',
  // YOLOv2
  'v2:src/region_layer.c':   '/versions/v2/source/region-layer',
  'v2:src/yolo.c':           '/versions/v2/source/yolo',
  'v2:src/box.c':            '/versions/v2/source/box',
  'v2:cfg/yolo.cfg':         '/versions/v2/source/yolo-cfg',
  // YOLOv3
  'v3:src/yolo_layer.c':     '/versions/v3/source/yolo-layer',
  'v3:examples/yolo.c':      '/versions/v3/source/yolo', // v3 快照目录结构调整：yolo.c 移入 examples/
  'v3:src/box.c':            '/versions/v3/source/box',
  'v3:cfg/yolov3.cfg':       '/versions/v3/source/yolo-cfg',
  // YOLOv4
  'v4:src/yolo_layer.c':     '/versions/v4/source/yolo-layer',
  'v4:src/yolo.c':           '/versions/v4/source/yolo',
  'v4:src/box.c':            '/versions/v4/source/box',
  'v4:cfg/yolov4.cfg':       '/versions/v4/source/yolo-cfg'
}

const GH_BASES = {
  v1: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov1/snapshot/',
  v2: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov2/snapshot/',
  v3: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov3/snapshot/',
  v4: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov4/snapshot/'
}

function resolveKey(v, f) {
  if (!modules[v]) throw new Error(`未知快照版本: ${v}（可用: ${Object.keys(modules).join(', ')}）`)
  const suffix = `/snapshot/${f}`
  const key = Object.keys(modules[v]).find((k) => k.endsWith(suffix))
  if (!key) throw new Error(`源码快照中找不到文件: ${v}/${f}`)
  return key
}

/** 异步加载快照文件全文；file 支持 `src/x.c` 或 `v2:src/x.c`，version 可选显式覆盖 */
export async function loadSource(file, version) {
  const [v, f] = splitKey(file)
  const vv = version || v
  return modules[vv][resolveKey(vv, f)]()
}

/** GitHub 行号锚点链接 */
export function githubUrl(file, start, end, version) {
  const [v, f] = splitKey(file)
  const vv = version || v
  let hash = ''
  if (start) hash = end && end !== start ? `#L${start}-L${end}` : `#L${start}`
  return GH_BASES[vv] + f + hash
}

/** 站内源码页链接（带行号锚点）；白名单外的文件回退 GitHub */
// @AI_GUARD: 站内路径必须经 withBase 补站点 base（/agent-yolo/），否则部署到子路径时 SrcLink 全部 404
export function sourceUrl(file, start, end, version) {
  const [v, f] = splitKey(file)
  const vv = version || v
  const page = SOURCE_PAGES[`${vv}:${f}`]
  let hash = ''
  if (start) hash = end && end !== start ? `#L${start}-L${end}` : `#L${start}`
  return page ? withBase(page) + hash : githubUrl(file, start, end, version)
}

/** 判断某文件在当前版本快照下是否已在白名单内 */
export function hasPage(file, version) {
  const [v, f] = splitKey(file)
  return !!SOURCE_PAGES[`${version || v}:${f}`]
}

/** 去掉 `v2:` 前缀后的显示路径，供 UI 标签使用 */
export function displayFile(file) {
  const [, f] = splitKey(file)
  return f
}

/** 根据文件扩展名返回 shiki 语言标识 */
export function langOf(file) {
  if (file.endsWith('.cfg')) return 'ini'
  if (file.endsWith('.cu')) return 'cuda-cpp'
  if (file.endsWith('.h')) return 'c'
  return 'c'
}
