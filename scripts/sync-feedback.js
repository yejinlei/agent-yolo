#!/usr/bin/env node
// 用法：node scripts/sync-feedback.js
//
// 功能：扫描 docs/**/*.md 中所有「反哺信号」章节，抽取 PainPoint / Trick / Metric
//      三类标记，生成 docs/concepts-feedback.md 汇总页。
//
// 支持两种格式（并存，用于兼容新旧案例页）：
//   新格式：- **PainPoint 新增**：`pain-xxx`（说明）
//   旧格式：1. **new PainPoint**: `pain-xxx` — 说明
//
import fs from 'node:fs'
import path from 'node:path'

const ROOT = process.cwd()
const DOCS = path.join(ROOT, 'docs')
const OUT = path.join(DOCS, 'concepts-feedback.md')

// 递归遍历 docs/，跳过 .vitepress 目录
function* walk(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.name === '.vitepress' || ent.name.startsWith('.')) continue
    const p = path.join(dir, ent.name)
    if (ent.isDirectory()) yield* walk(p)
    else if (ent.name.endsWith('.md')) yield p
  }
}

// 主正则：匹配「类型 + 动作」+ 反引号包的名字 + 可选括号/破折号说明
// 分组：1=类型 2=名称 3=括号说明 4=破折号说明
const SIG_RE = /^[-\d.]*\s+\*\*(?:new\s+)?(PainPoint|Trick|Metric)\s*(?:新增|强化|更新|关联|new)?\*\*\s*[:：]\s*`([a-z0-9][a-z0-9-]*)`\s*(?:（([^）]+)）|(?:—\s*(.+))?)?/

// 剥离 markdown 代码块（``` 或 ~~~ 包裹的内容全部去掉）
// 遵循 CommonMark：闭合 fence 的标记字符数必须 >= 开启时的字符数
// 逐行 in/out 状态判断，避免嵌套代码块的正则错位
function stripCodeBlocks(text) {
  const lines = text.split('\n')
  let inFence = false
  let fenceCh = '' // '`' 或 '~'
  let fenceLen = 0 // 开启 fence 的字符数（例如 ```` 是 4）
  const out = lines.map((line) => {
    const m = line.match(/^(\s*)(`{3,}|~{3,})/)
    if (m) {
      const mark = m[2]
      const ch = mark[0]
      const len = mark.length
      if (!inFence) {
        inFence = true
        fenceCh = ch
        fenceLen = len
        return line.replace(/[^\n]/g, ' ')
      }
      // 已在 fence 内：只有同字符且长度 >= fenceLen 才闭合
      if (ch === fenceCh && len >= fenceLen) {
        inFence = false
        fenceCh = ''
        fenceLen = 0
        return line.replace(/[^\n]/g, ' ')
      }
    }
    if (inFence) return line.replace(/[^\n]/g, ' ')
    return line
  })
  return out.join('\n')
}

const signals = new Map() // key: type:name → {type, name, desc, sources}

let scanned = 0
let matchedFiles = 0

for (const file of walk(DOCS)) {
  const rel = path.relative(DOCS, file).replace(/\\/g, '/')
  if (rel === 'concepts-feedback.md') continue

  const raw = fs.readFileSync(file, 'utf8')
  scanned++
  // 去掉代码块内容，防止示例代码里的 pain-xxx 被误抓
  const text = stripCodeBlocks(raw)

  // 只扫「反哺信号」标题行下的章节
  const allMatches = []
  let from = 0
  while (true) {
    const idx = text.indexOf('反哺信号', from)
    if (idx < 0) break
    // 定位到该标题所在行
    const start = text.lastIndexOf('\n', idx)
    const lineStart = start >= 0 ? start + 1 : 0
    const lineEnd = text.indexOf('\n', lineStart)
    const line = text.slice(lineStart, lineEnd)
    // 必须是被 markdown 标题包裹的「反哺信号」，才当作章节起点
    if (/^#{2,4}\s+/.test(line)) {
      const section = text.slice(lineStart)
      // 找下一个同级别或更高级别的标题
      const next = section.indexOf('\n## ', 5)
      const chunk = next > 0 ? section.slice(0, next) : section
      allMatches.push(chunk)
    }
    from = idx + '反哺信号'.length
  }
  if (allMatches.length === 0) continue
  const body = allMatches.join('\n')
  matchedFiles++

  for (const line of body.split('\n')) {
    const m = line.match(SIG_RE)
    if (!m) continue
    const [, type, name, desc1, desc2] = m
    const desc = (desc1 || desc2 || '').trim()
    const key = `${type}:${name}`
    if (!signals.has(key)) {
      signals.set(key, { type, name, desc: desc || '', sources: [] })
    }
    const entry = signals.get(key)
    if (!entry.desc && desc) entry.desc = desc
    if (!entry.sources.includes(rel)) entry.sources.push(rel)
  }
}

// 生成 markdown 输出
const byType = { PainPoint: [], Trick: [], Metric: [] }
for (const s of signals.values()) byType[s.type].push(s)

// 用本地日期（避免 UTC 与 Asia/Shanghai 时区差导致日期偏移）
function localDate() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

const lines = [
  '# 反哺信号汇总 · 全库扫描结果',
  '',
  '> 自动生成，请勿手改。运行 `node scripts/sync-feedback.js` 重新生成。',
  `> 最后更新：${localDate()}`,
  `> 扫描范围：\`docs/**/*.md\` · 共扫描 **${scanned}** 个 md 文件，检出 **${signals.size}** 条信号（分布在 **${matchedFiles}** 个文件）`,
  '',
  '## 分类速览',
  '',
  '| 类型 | 数量 |',
  '|-----|-----|',
  `| PainPoint | ${byType.PainPoint.length} |`,
  `| Trick | ${byType.Trick.length} |`,
  `| Metric | ${byType.Metric.length} |`,
  `| **合计** | **${signals.size}** |`,
  ''
]

for (const [type, list] of Object.entries(byType)) {
  if (list.length === 0) continue
  lines.push(`## ${type}（${list.length} 条）`, '')
  lines.push('| 名称 | 说明 | 来源 |', '|------|------|------|')
  for (const s of list.sort((a, b) => a.name.localeCompare(b.name))) {
    const srcs = [...new Set(s.sources)]
      .map(p => `\`[${p}](/${p.replace(/\.md$/, '')})\``)
      .join('<br>')
    lines.push(`|\`${s.name}\`| ${s.desc || '—'} | ${srcs} |`)
  }
  lines.push('')
}

lines.push(
  '---',
  '',
  '## 使用方式',
  '',
  '- 想查某个 trick 在哪些案例用过 → 直接搜索本页',
  '- 想追踪某个痛点的演进 → 先看 [cases/compare.md](/cases/compare) 的横向对比',
  '- 想新增信号 → 在案例页 / 任务页 / 配方页末尾的「反哺信号」章节加一行，再运行 `node scripts/sync-feedback.js`',
  ''
)

fs.writeFileSync(OUT, lines.join('\n') + '\n', 'utf8')
console.log(`✅ 已生成 ${OUT}`)
console.log(`   · 扫描 ${scanned} 个 md 文件，命中 ${matchedFiles} 个含「反哺信号」章节的文件`)
console.log(`   · 共 ${signals.size} 条信号（PainPoint ${byType.PainPoint.length} / Trick ${byType.Trick.length} / Metric ${byType.Metric.length}）`)
