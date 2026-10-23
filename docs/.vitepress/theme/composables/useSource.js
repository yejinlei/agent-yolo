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
  ),
  // v5 是 PyTorch 项目，源码以 .py / .yaml 为主
  v5: import.meta.glob(
    '../../../../source/yolov5/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v6 同为 PyTorch 项目（Meituan YOLOv6），源码以 .py 为主，配置是 Python dict（configs/*.py）
  v6: import.meta.glob(
    '../../../../source/yolov6/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v7 同为 PyTorch 项目（WongKinYiu/yolov7），源码以 .py / .yaml 为主
  v7: import.meta.glob(
    '../../../../source/yolov7/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v8 是 ultralytics 单一代码库（tag v8.0.100），源码以 .py / .yaml 为主
  v8: import.meta.glob(
    '../../../../source/yolov8/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v9 是 WongKinYiu/yolov9（v7 代码库 fork），源码以 .py / .yaml 为主
  v9: import.meta.glob(
    '../../../../source/yolov9/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v10 是 THU-MIG/yolov10（ultralytics 8.1.34 fork），目录同 v8 快照布局
  v10: import.meta.glob(
    '../../../../source/yolov10/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v11 是 ultralytics/ultralytics tag v8.3.0（YOLO11 首发），目录同 v8/v10 布局
  v11: import.meta.glob(
    '../../../../source/yolov11/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v12 是 sunsmarterjie/yolov12 tag v1.0（论文前一日），目录同 v8/v10/v11 布局；注意 cfg 目录带 v 前缀 v12/
  v12: import.meta.glob(
    '../../../../source/yolov12/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v13 是 iMoonLab/yolov13（学术论文 2506.17733 官方实现），目录同 v11/v12 布局；cfg 目录带 v 前缀 v13/
  v13: import.meta.glob(
    '../../../../source/yolov13/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v14 是 zhangcbb/yolov14（学术论文 2608.04720v3 YOLO-Omni 官方实现），cfg 目录扁平为 cfg_models_v14/
  v14: import.meta.glob(
    '../../../../source/yolov14/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // v26 是 ultralytics/ultralytics 官方主线（arXiv 2606.03748），cfg 目录扁平为 cfg_models_26/
  v26: import.meta.glob(
    '../../../../source/yolo26/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // YOLO-World（衍生分支）：AILab-CVC 原始仓库（v2.1）+ Ultralytics 8.1.39 集成
  // AILab-CVC 配置为 mmengine Python dict，需一并纳入 .py glob
  yoloworld: import.meta.glob(
    '../../../../source/yoloworld/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // YOLOE（衍生分支）：Ultralytics 8.3.99 官方集成（THU-MIG/yoloe 论文实现仅落 README+LICENSE）
  yoloe: import.meta.glob(
    '../../../../source/yoloe/snapshot/**/*.{py,yaml}',
    { query: '?raw', import: 'default' }
  ),
  // YOLO-Master（衍生分支）：Tencent/YOLO-Master（arXiv 2512.23273 CVPR 2026），YOLOv12 fork
  master: import.meta.glob(
    '../../../../source/yolomaster/snapshot/**/*.{py,yaml}',
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
  'v4:cfg/yolov4.cfg':       '/versions/v4/source/yolo-cfg',
  // YOLOv5（PyTorch 目录结构）
  'v5:models/yolo.py':       '/versions/v5/source/yolo-py',
  'v5:models/common.py':     '/versions/v5/source/common-py',
  'v5:utils/loss.py':        '/versions/v5/source/loss-py',
  'v5:models/yolov5s.yaml':  '/versions/v5/source/yolov5s-yaml',
  // YOLOv6（Meituan，Python dict 配置）
  'v6:yolov6/layers/common.py':       '/versions/v6/source/common-py',
  'v6:yolov6/models/losses/loss.py':  '/versions/v6/source/loss-py',
  'v6:yolov6/assigners/tal_assigner.py': '/versions/v6/source/tal-assigner-py',
  'v6:configs/yolov6s.py':            '/versions/v6/source/yolov6s-config',
  // YOLOv7（WongKinYiu/yolov7，PyTorch 目录结构）
  'v7:models/common.py':        '/versions/v7/source/common-py',
  'v7:models/yolo.py':          '/versions/v7/source/yolo-py',
  'v7:utils/loss.py':           '/versions/v7/source/loss-py',
  'v7:cfg/training/yolov7.yaml': '/versions/v7/source/yolov7-yaml',
  // YOLOv8（ultralytics 单一代码库，tag v8.0.100 旧布局）；快照根是仓库根，源码在 ultralytics/ 包目录下
  'v8:ultralytics/nn/tasks.py':           '/versions/v8/source/tasks-py',
  'v8:ultralytics/nn/modules/block.py':   '/versions/v8/source/common-py',
  'v8:ultralytics/yolo/utils/loss.py':    '/versions/v8/source/loss-py',
  'v8:ultralytics/models/v8/yolov8.yaml': '/versions/v8/source/yolov8-yaml',
  // YOLOv9（WongKinYiu/yolov9，v7 代码库 fork 的目录结构）
  'v9:models/common.py':             '/versions/v9/source/common-py',
  'v9:models/yolo.py':               '/versions/v9/source/yolo-py',
  'v9:utils/loss_tal_dual.py':       '/versions/v9/source/loss-tal-dual-py',
  'v9:models/detect/yolov9-c.yaml':  '/versions/v9/source/yolov9-c-yaml',
  // YOLOv10（THU-MIG/yolov10，ultralytics 8.1.34 fork，目录同 v8 布局）
  'v10:ultralytics/nn/modules/head.py':         '/versions/v10/source/head-py',
  'v10:ultralytics/utils/loss.py':              '/versions/v10/source/loss-py',
  'v10:ultralytics/utils/tal.py':               '/versions/v10/source/tal-py',
  'v10:ultralytics/cfg/models/v10/yolov10n.yaml': '/versions/v10/source/yolov10-yaml',
  // YOLOv11（ultralytics/ultralytics tag v8.3.0 首发；目录同 v8/v10 布局；注意 cfg 目录是 11/ 不带 v 前缀）
  'v11:ultralytics/nn/modules/head.py':         '/versions/v11/source/head-py',
  'v11:ultralytics/utils/loss.py':              '/versions/v11/source/loss-py',
  'v11:ultralytics/utils/tal.py':               '/versions/v11/source/tal-py',
  'v11:ultralytics/cfg/models/11/yolo11.yaml':  '/versions/v11/source/yolo11-yaml',
  // YOLOv12（sunsmarterjie/yolov12 tag v1.0；目录同 v8/v10 布局；注意 cfg 目录是 v12/ 带 v 前缀）
  'v12:ultralytics/nn/modules/block.py':        '/versions/v12/source/block-py',
  'v12:ultralytics/nn/modules/head.py':         '/versions/v12/source/head-py',
  'v12:ultralytics/utils/loss.py':              '/versions/v12/source/loss-py',
  'v12:ultralytics/utils/tal.py':               '/versions/v12/source/tal-py',
  'v12:ultralytics/cfg/models/v12/yolov12.yaml': '/versions/v12/source/yolov12-yaml',
  // YOLOv13（iMoonLab/yolov13 @ 73289949；目录同 v11/v12 布局；注意 cfg 目录是 v13/ 带 v 前缀）
  'v13:ultralytics/nn/modules/block.py':        '/versions/v13/source/block-py',
  'v13:ultralytics/nn/modules/conv.py':         '/versions/v13/source/conv-py',
  'v13:ultralytics/nn/modules/head.py':         '/versions/v13/source/head-py',
  'v13:ultralytics/utils/loss.py':              '/versions/v13/source/loss-py',
  'v13:ultralytics/utils/tal.py':               '/versions/v13/source/tal-py',
  'v13:ultralytics/cfg/models/v13/yolov13.yaml': '/versions/v13/source/yolov13-yaml',
  // YOLOv14 / YOLO-Omni（zhangcbb/yolov14 @ 0b8610d；cfg 目录扁平为 cfg_models_v14/）
  'v14:nn/modules/block.py':                    '/versions/v14/source/block-py',
  'v14:nn/modules/head.py':                     '/versions/v14/source/head-py',
  'v14:utils/loss.py':                          '/versions/v14/source/loss-py',
  'v14:utils/tal.py':                           '/versions/v14/source/tal-py',
  'v14:cfg_models_v14/yolov14.yaml':            '/versions/v14/source/yolov14-yaml',
  'v14:cfg_models_v14/yolov14-panorama.yaml':   '/versions/v14/source/yolov14-variants',
  // YOLO26（ultralytics/ultralytics @ 2e241d0；cfg 目录扁平为 cfg_models_26/）
  'v26:nn/tasks.py':                            '/versions/v26/source/tasks-py',
  'v26:nn/modules/block.py':                    '/versions/v26/source/block-py',
  'v26:nn/modules/conv.py':                     '/versions/v26/source/conv-py',
  'v26:nn/modules/head.py':                     '/versions/v26/source/head-py',
  'v26:utils/loss.py':                          '/versions/v26/source/loss-py',
  'v26:utils/tal.py':                           '/versions/v26/source/tal-py',
  'v26:cfg_models_26/yolo26.yaml':              '/versions/v26/source/yolo26-yaml',
  // YOLO-World（衍生）— Ultralytics 集成侧
  'yoloworld:ultralytics/cfg/models/v8/yolov8-world.yaml':   '/derivatives/yoloworld/source/yolov8-world-yaml',
  'yoloworld:ultralytics/cfg/models/v8/yolov8-worldv2.yaml': '/derivatives/yoloworld/source/yolov8-worldv2-yaml',
  'yoloworld:ultralytics/nn/tasks_worldmodel.py':            '/derivatives/yoloworld/source/worldmodel-py',
  'yoloworld:ultralytics/nn/modules/head.py':                '/derivatives/yoloworld/source/head-py',
  'yoloworld:ultralytics/models/yolo/world/train.py':        '/derivatives/yoloworld/source/train-py',
  'yoloworld:ultralytics/models/yolo/world/train_world.py':  '/derivatives/yoloworld/source/train-world-py',
  // YOLO-World（衍生）— AILab-CVC 原始仓库侧
  'yoloworld:ailab-cvc/yolo_world/models/detectors/yolo_world.py':             '/derivatives/yoloworld/source/detector-py',
  'yoloworld:ailab-cvc/yolo_world/models/dense_heads/yolo_world_head.py':       '/derivatives/yoloworld/source/head-ailab-py',
  'yoloworld:ailab-cvc/yolo_world/models/necks/yolo_world_pafpn.py':           '/derivatives/yoloworld/source/pafpn-py',
  // YOLOE（衍生）— Ultralytics 8.3.99 集成
  'yoloe:ultralytics/cfg/models/v8/yoloe-v8.yaml':   '/derivatives/yoloe/source/yoloe-v8-yaml',
  'yoloe:ultralytics/cfg/models/v8/yoloe-v8-seg.yaml': '/derivatives/yoloe/source/yoloe-v8-seg-yaml',
  'yoloe:ultralytics/cfg/models/26/yoloe-26.yaml':   '/derivatives/yoloe/source/yoloe-26-yaml',
  'yoloe:ultralytics/cfg/models/26/yoloe-26-seg.yaml': '/derivatives/yoloe/source/yoloe-26-seg-yaml',
  'yoloe:ultralytics/models/yolo/model.py':           '/derivatives/yoloe/source/model-py',
  'yoloe:ultralytics/models/yolo/yoloe/train.py':     '/derivatives/yoloe/source/train-py',
  'yoloe:ultralytics/models/yolo/yoloe/train_seg.py': '/derivatives/yoloe/source/train-seg-py',
  'yoloe:ultralytics/models/yolo/yoloe/val.py':       '/derivatives/yoloe/source/val-py',
  'yoloe:ultralytics/models/yolo/yoloe/predict.py':   '/derivatives/yoloe/source/predict-py',
  'yoloe:ultralytics/nn/head.py':                     '/derivatives/yoloe/source/head-py',
  'yoloe:ultralytics/nn/tasks.py':                    '/derivatives/yoloe/source/tasks-py',
  // YOLO-Master（衍生）— ES-MoE + MoT + cfg
  'master:ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml':        '/derivatives/master/source/yolo-master-n-yaml',
  'master:ultralytics/cfg/models/master/v0/det/yolo-master-l.yaml':        '/derivatives/master/source/yolo-master-l-yaml',
  'master:ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-coco.yaml': '/derivatives/master/source/yolo-master-esmoe-n-yaml',
  'master:ultralytics/nn/modules/moe/modules.py':     '/derivatives/master/source/moe-modules-py',
  'master:ultralytics/nn/modules/moe/routers.py':     '/derivatives/master/source/moe-routers-py',
  'master:ultralytics/nn/modules/moe/experts.py':     '/derivatives/master/source/moe-experts-py',
  'master:ultralytics/nn/modules/moe/loss.py':        '/derivatives/master/source/moe-loss-py',
  'master:ultralytics/nn/modules/moe/topk_anneal.py': '/derivatives/master/source/moe-topk-anneal-py',
  'master:ultralytics/nn/modules/mot/experts.py':     '/derivatives/master/source/mot-experts-py',
  'master:ultralytics/nn/modules/mot/router.py':      '/derivatives/master/source/mot-router-py',
  'master:ultralytics/nn/modules/transformer.py':     '/derivatives/master/source/transformer-py',
  'master:ultralytics/nn/modules/head.py':            '/derivatives/master/source/head-py',
  'master:ultralytics/models/yolo/detect/train.py':   '/derivatives/master/source/train-py'
}

const GH_BASES = {
  v1: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov1/snapshot/',
  v2: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov2/snapshot/',
  v3: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov3/snapshot/',
  v4: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov4/snapshot/',
  v5: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov5/snapshot/',
  v6: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov6/snapshot/',
  v7: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov7/snapshot/',
  v8: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov8/snapshot/',
  v9: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov9/snapshot/',
  v10: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov10/snapshot/',
  v11: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov11/snapshot/',
  v12: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov12/snapshot/',
  v13: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov13/snapshot/',
  v14: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolov14/snapshot/',
  v26: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yolo26/snapshot/',
  yoloworld: 'https://github.com/yejinlei/agent-yolo/blob/main/source/yoloworld/snapshot/',
  yoloe:     'https://github.com/yejinlei/agent-yolo/blob/main/source/yoloe/snapshot/',
  master:    'https://github.com/yejinlei/agent-yolo/blob/main/source/yolomaster/snapshot/'
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
  if (file.endsWith('.yaml') || file.endsWith('.yml')) return 'yaml'
  if (file.endsWith('.py')) return 'python'
  if (file.endsWith('.cu')) return 'cuda-cpp'
  if (file.endsWith('.h')) return 'c'
  return 'c'
}
