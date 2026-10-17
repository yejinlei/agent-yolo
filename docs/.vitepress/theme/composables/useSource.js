// @AI_GUARD: 源码快照只能来自 source/yolov1/snapshot/，组件通过 ?raw 在构建期内联，勿改成运行时 fetch
// @CONSTRAINT: glob 根路径相对本文件所在目录（docs/.vitepress/theme/composables/），到仓库根为 ../../../../
const modules = import.meta.glob(
  '../../../../source/yolov1/snapshot/**/*.{c,h,cu,cfg}',
  { query: '?raw', import: 'default' }
)

// 站内有完整浏览页的文件白名单（file 相对 snapshot 根）
export const SOURCE_PAGES = {
  'src/detection_layer.c': '/versions/v1/source/detection-layer',
  'src/yolo.c': '/versions/v1/source/yolo',
  'src/box.c': '/versions/v1/source/box',
  'cfg/yolo.cfg': '/versions/v1/source/yolo-cfg'
}

const GH_BASE = 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov1/snapshot/'

function resolveKey(file) {
  const suffix = `/snapshot/${file}`
  const key = Object.keys(modules).find((k) => k.endsWith(suffix))
  if (!key) throw new Error(`源码快照中找不到文件: ${file}`)
  return key
}

/** 异步加载快照文件全文 */
export async function loadSource(file) {
  return modules[resolveKey(file)]()
}

/** GitHub 行号锚点链接 */
export function githubUrl(file, start, end) {
  let hash = ''
  if (start) hash = end && end !== start ? `#L${start}-L${end}` : `#L${start}`
  return GH_BASE + file + hash
}

/** 站内源码页链接（带行号锚点）；白名单外的文件回退 GitHub */
export function sourceUrl(file, start, end) {
  const page = SOURCE_PAGES[file]
  let hash = ''
  if (start) hash = end && end !== start ? `#L${start}-L${end}` : `#L${start}`
  return page ? page + hash : githubUrl(file, start, end)
}

/** 根据文件扩展名返回 shiki 语言标识 */
export function langOf(file) {
  if (file.endsWith('.cfg')) return 'ini'
  if (file.endsWith('.cu')) return 'cuda-cpp'
  if (file.endsWith('.h')) return 'c'
  return 'c'
}
