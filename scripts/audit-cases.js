#!/usr/bin/env node
// 用法：node scripts/audit-cases.js
//
// 功能：对 docs/cases/*.md 中每个「案例页」跑 7 项健康检查，
//      生成 docs/cases/health.md 审计报告。
//
// 7 项检查：
//   a) 最后更新日期    （末尾是否有 "最后更新" 字段）
//   b) 反哺信号章节    （是否有 ## 反哺信号 或 反哺信号汇总 章节）
//   c) 信号同步        （该案例的信号是否已进入 concepts-feedback.md）
//   d) 横向对比链接    （是否被 compare.md 索引，或引用 /cases/compare）
//   e) Sidebar 注册    （是否在 .vitepress/config.mjs 的 /cases/ 分组里）
//   f) 内部链接死链    （Markdown 内 /xxx 链接是否指向存在的 .md 文件）
//   g) 版本页存在      （文中提到的 YOLO 版本号是否有对应 /versions/vN/index.md）
//
import fs from 'node:fs'
import path from 'node:path'

const ROOT = process.cwd()
const DOCS = path.join(ROOT, 'docs')
const CASES = path.join(DOCS, 'cases')
const OUT = path.join(CASES, 'health.md')

// 已知版本页（config.mjs 中的 YOLO 主线）
const KNOWN_VERSIONS = new Set([
  'v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
  'v11','v12','v13','v14','v26','v27'
])

// 剥离 markdown 代码块（``` 或 ~~~ 围栏）
function stripCodeBlocks(text) {
  const lines = text.split('\n')
  let inFence = false, fenceCh = '', fenceLen = 0
  const out = lines.map((line) => {
    const m = line.match(/^(\s*)(`{3,}|~{3,})/)
    if (m) {
      const ch = m[2][0], len = m[2].length
      if (!inFence) { inFence = true; fenceCh = ch; fenceLen = len; return ' ' }
      if (ch === fenceCh && len >= fenceLen) { inFence = false; fenceCh = ''; fenceLen = 0; return ' ' }
    }
    return inFence ? ' ' : line
  })
  return out.join('\n')
}

// 收集所有 docs 下的 .md 文件
function collectPages() {
  const pages = new Set()
  function walk(dir) {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ent.name.startsWith('.')) continue
      const p = path.join(dir, ent.name)
      if (ent.isDirectory()) walk(p)
      else if (ent.name.endsWith('.md')) pages.add(p)
    }
  }
  walk(DOCS)
  return pages
}

// 从 /path/to/page 或 /path/to/page.md 解析出真实 md 文件路径
function resolveLink(link, pages) {
  if (/^[a-z]+:\/\//.test(link) || link.startsWith('#') || link.startsWith('mailto:')) return null
  if (link.includes('/agent-yolo/')) link = link.replace(/^.*?\/agent-yolo\//, '/')
  if (!link.startsWith('/')) return null
  let noAnchor = link.split('#')[0].split('?')[0]
  if (!noAnchor) return null
  let file
  if (noAnchor.endsWith('.md')) file = path.join(DOCS, noAnchor)
  else {
    if (noAnchor.endsWith('/')) noAnchor += 'index'
    file = path.join(DOCS, noAnchor + '.md')
  }
  return pages.has(file) ? file : null
}

// 从 config.mjs 提取 sidebar 里出现过的 /cases/xxx 链接
function parseSidebarLinks() {
  const cfg = fs.readFileSync(path.join(DOCS, '.vitepress', 'config.mjs'), 'utf8')
  const stripped = stripCodeBlocks(cfg)
  const links = new Set()
  for (const m of stripped.matchAll(/\/cases\/[a-z0-9-]+/g)) links.add(m[0])
  return links
}

// ---- 单文件检查 ----
function getSignals(text) {
  const body = stripCodeBlocks(text)
  const lines = body.split('\n')
  let inSec = false
  const out = []
  const re = /^\s*(?:[-*]\s*|\d+\.\s*)\*\*(?:new\s+)?(PainPoint|Trick|Metric)\s*(?:新增|强化|更新|关联|new)?\*\*\s*[:：]\s*`([a-z0-9-]+)`/
  for (const line of lines) {
    if (/^#{1,4}\s+/.test(line)) {
      if (/反哺信号/.test(line)) inSec = true
      else if (inSec) break
      continue
    }
    if (!inSec) continue
    const m = line.match(re)
    if (m) out.push({ type: m[1], name: m[2] })
  }
  return out
}

function runChecks(file, basename, sidebarLinks, pages, feedbackText) {
  const raw = fs.readFileSync(file, 'utf8')
  const text = stripCodeBlocks(raw)
  const problems = []
  const results = []
  const push = (name, pass, problem) => {
    results.push({ name, pass, problem: pass ? '' : (problem || '') })
    if (!pass && problem) problems.push(problem)
  }

  // a) 最后更新日期
  const hasUpdated = /最后更新[^\n]*\d{4}-\d{2}-\d{2}/.test(text)
  push('最后更新日期', hasUpdated, hasUpdated ? '' : '缺少 "最后更新" 日期字段')

  // b) 反哺信号章节
  const hasSigSection = /^#{2,4}\s+.*反哺信号/m.test(text)
  push('反哺信号章节', hasSigSection, hasSigSection ? '' : '缺少 "## 反哺信号" 章节')

  // c) 信号同步
  const sigs = getSignals(text)
  const missingSync = sigs.filter(s => !feedbackText.includes('`' + s.name + '`'))
  push('信号同步', missingSync.length === 0, missingSync.length === 0
    ? (sigs.length === 0 ? '' : `已同步 ${sigs.length} 条`)
    : `未同步：${missingSync.map(s => s.name).join(', ')}`)

  // d) 横向对比链接
  let inCompareMd = false, inIndexMd = false, hasCompareLink = false
  try { inCompareMd = fs.readFileSync(path.join(CASES, 'compare.md'), 'utf8').includes(basename) } catch {}
  try { inIndexMd = fs.readFileSync(path.join(CASES, 'index.md'), 'utf8').includes(basename) } catch {}
  hasCompareLink = /\/cases\/compare/.test(text)
  const dPass = inCompareMd || inIndexMd || hasCompareLink
  const dInfo = inCompareMd ? '在 compare.md' : (inIndexMd ? '在 index.md' : (hasCompareLink ? '引用 /cases/compare' : ''))
  push('横向对比链接', dPass, dPass ? dInfo : '未被 compare.md / index.md 索引，也未引用 /cases/compare')

  // e) sidebar 注册
  const sidebarPath = `/cases/${basename}`
  const inSidebar = sidebarLinks.has(sidebarPath)
  push('sidebar 注册', inSidebar, inSidebar ? '' : `未在 .vitepress/config.mjs 的 /cases/ 分组里注册 ${sidebarPath}`)

  // f) 内部链接死链
  const brokenLinks = []
  const stripped2 = stripCodeBlocks(raw)
  for (const m of stripped2.matchAll(/\[[^\]]+\]\(([^)]+)\)/g)) {
    const link = m[1].trim()
    if (/^[a-z]+:\/\//.test(link) || link.startsWith('#') || link.startsWith('mailto:') || link.startsWith('tel:')) continue
    if (!link.startsWith('/')) continue
    if (!resolveLink(link, pages)) brokenLinks.push(link)
  }
  push('内部链接', brokenLinks.length === 0, brokenLinks.length === 0 ? '' : `死链：${[...new Set(brokenLinks)].join(', ')}`)

  // g) 版本页存在
  const mentioned = new Set()
  for (const m of text.matchAll(/(?:yolo|YOLO|Yolo)\s*(v?\d{1,2})/gi)) {
    const v = m[1].toLowerCase()
    const key = v[0] === 'v' ? v : 'v' + v
    if (KNOWN_VERSIONS.has(key)) mentioned.add(key)
  }
  const missing = [...mentioned].filter(v => !fs.existsSync(path.join(DOCS, 'versions', v, 'index.md')))
  push('版本页', missing.length === 0, missing.length === 0
    ? (mentioned.size === 0 ? '' : `提到 ${[...mentioned].join(',')}`)
    : `缺版本页：${missing.join(', ')}`)

  return { results, problems, sigs, mentioned: [...mentioned] }
}

// ---- 主流程 ----
const pages = collectPages()
const sidebarLinks = parseSidebarLinks()
const feedbackText = fs.existsSync(path.join(DOCS, 'concepts-feedback.md'))
  ? stripCodeBlocks(fs.readFileSync(path.join(DOCS, 'concepts-feedback.md'), 'utf8'))
  : ''

const SKIP_CASES = new Set(['_template.md', 'index.md', 'compare.md', 'health.md'])
const files = fs.readdirSync(CASES).filter(f =>
  f.endsWith('.md') && !SKIP_CASES.has(f)
).sort()

const reports = files.map(f => {
  const basename = f.slice(0, -3)
  const file = path.join(CASES, f)
  const r = runChecks(file, basename, sidebarLinks, pages, feedbackText)
  const score = r.results.filter(x => x.pass).length
  return { file: f, basename, score, total: r.results.length, ...r }
})

const withIssues = reports.filter(r => r.problems.length > 0)
const totalScore = reports.reduce((s, r) => s + r.score, 0)
const total = reports.reduce((s, r) => s + r.total, 0)
const pct = total === 0 ? 0 : Math.round(totalScore / total * 100)

// 生成 health.md
const lines = []
lines.push('# 案例健康度报告 · 自动生成')
lines.push('')
lines.push('> 运行 `node scripts/audit-cases.js` 重新生成。')
lines.push(`> 最后更新：${localDate()}`)
lines.push(`> 扫描范围：\`docs/cases/*.md\`（排除 _template.md / index.md / compare.md） · 共 **${reports.length}** 个案例页 · 平均得分 **${pct}%**`)
lines.push('')
lines.push('## 总览')
lines.push('')
lines.push('| 案例 | 得分 | 问题数 | 主要问题 |')
lines.push('|------|------|--------|----------|')
for (const r of reports) {
  const main = r.problems.length > 0
    ? r.problems.map(p => p.replace(/\|/g, '\\|')).join('<br>')
    : '✅ 全部通过'
  const badge = r.score === r.total ? '🟢' : (r.score >= 5 ? '🟡' : '🔴')
  lines.push(`| [${r.basename}](/cases/${r.basename}) | ${badge} ${r.score}/${r.total} | ${r.problems.length} | ${main} |`)
}
lines.push('')
if (withIssues.length === 0) {
  lines.push('## 问题清单')
  lines.push('')
  lines.push('✅ 无问题，所有案例通过全部检查。')
} else {
  lines.push('## 问题清单')
  lines.push('')
  lines.push('| 案例 | 检查项 | 问题描述 |')
  lines.push('|------|--------|----------|')
  for (const r of withIssues) {
    for (let i = 0; i < r.results.length; i++) {
      if (!r.results[i].pass && r.results[i].problem) {
        lines.push(`| [${r.basename}](/cases/${r.basename}) | ${r.results[i].name} | ${r.results[i].problem.replace(/\|/g, '\\|')} |`)
      }
    }
  }
}
lines.push('')
lines.push('## 详情诊断')
lines.push('')
lines.push('| 案例 | 最后更新 | 反哺信号 | 信号同步 | 横向对比 | sidebar | 死链 | 版本页 |')
lines.push('|------|---------|---------|---------|---------|---------|------|--------|')
for (const r of reports) {
  const cells = r.results.map(res => {
    if (res.pass) {
      const info = res.problem
      return info ? `✅ ${info.replace(/\|/g, '\\|')}` : '✅'
    }
    return `❌ ${res.problem.replace(/\|/g, '\\|')}`
  })
  lines.push(`| [${r.basename}](/cases/${r.basename}) | ${cells.join(' | ')} |`)
}
lines.push('')
lines.push('## 检查规则')
lines.push('')
lines.push('1. **最后更新日期**：正文任意位置出现 `最后更新` + `YYYY-MM-DD` 视为通过。')
lines.push('2. **反哺信号章节**：存在 `## 反哺信号` 或包含"反哺信号"的 h2+ 标题。')
lines.push('3. **信号同步**：案例中的 `PainPoint / Trick / Metric` 信号名都出现在 [concepts-feedback.md](/concepts-feedback) 中（未同步视为漏跑 `sync-feedback.js`）。')
lines.push('4. **横向对比链接**：文件名在 [compare.md](/cases/compare) 或 [index.md](/cases) 中出现，或案例正文引用 `/cases/compare`。')
lines.push('5. **sidebar 注册**：`.vitepress/config.mjs` 的 sidebar 中注册了 `/cases/<name>`。')
lines.push('6. **内部链接**：Markdown 内部 `/xxx` 链接指向的 md 文件存在（外链跳过）。')
lines.push('7. **版本页存在**：文中提到的 YOLO 版本号（`YOLOvN` / `yoloN`）在 `/versions/vN/index.md` 存在对应页。')
lines.push('')
lines.push('---')
lines.push('')
lines.push('**得分**：总分 **' + totalScore + '/' + total + '**（' + pct + '%）')
fs.writeFileSync(OUT, lines.join('\n') + '\n', 'utf8')

function localDate() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

// ---- 命令行摘要 ----
console.log('📋 案例健康度审计')
console.log(`   案例数：${reports.length}   平均得分：${pct}%   问题案例：${withIssues.length}`)
for (const r of reports) {
  const tag = r.score === r.total ? '✅' : (r.score >= 5 ? '⚠️' : '❌')
  console.log(`  ${tag} ${r.basename.padEnd(24)} ${r.score}/${r.total}${r.problems.length ? '  ' + r.problems.slice(0, 2).join('; ') : ''}`)
}
console.log(`   报告已写入：${path.relative(ROOT, OUT)}`)
