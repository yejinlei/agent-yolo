#!/usr/bin/env node
// 用法：node scripts/audit-cross-links.js
//
// 功能：校验 docs/cases/*.md ↔ docs/cases/compare.md ↔ docs/compare-matrix.md
//      三方一致性。生成 docs/cases/health.md 的姊妹报告：
//      docs/cases/cross-links.md
//
// 校验项：
//   A) 案例 ↔ cases/compare.md：每个案例是否被 compare.md 引用；是否引用了 compare.md
//   B) 案例 ↔ compare-matrix.md：案例里提到的关键版本（v8/v11/v13/v26 等）是否在 compare-matrix 的对比行中出现
//   C) compare.md ↔ compare-matrix.md：compare.md 里的对比维度关键词是否被 compare-matrix.md 覆盖
//   D) 案例横向链接：每个案例至少引用 1 个其它案例；每个案例至少引用 1 个 compare/matrix 入口
//
import fs from 'node:fs'
import path from 'node:path'

const ROOT = process.cwd()
const DOCS = path.join(ROOT, 'docs')
const CASES = path.join(DOCS, 'cases')
const OUT = path.join(CASES, 'cross-links.md')

// 剥离 markdown 代码块
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

// 抓 /xxx 或 /xxx.md 形式内部链接
function extractLinks(text) {
  const re = /\[[^\]]+\]\((\/[^)\s]+?)\)/g
  const out = new Set()
  let m
  while ((m = re.exec(text)) !== null) out.add(m[1])
  return out
}

// 收集案例（排除 _template / compare / health / index / cross-links / derivatives）
const caseFiles = fs.readdirSync(CASES)
  .filter(f => f.endsWith('.md'))
  .filter(f => !['compare.md', 'health.md', 'index.md', 'cross-links.md', '_template.md'].includes(f))

// 关键版本（compare-matrix 应该覆盖的对比维度）
const KEY_VERSIONS = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v26','v27']

// 关键对比维度（compare.md 与 compare-matrix.md 应互相覆盖的字段）
const KEY_DIMS = ['COCO AP', 'COCO', '参数量', '延迟', '量化', 'INT8', 'FP16', '硬件', '场景', '精度', '速度']

// 读文件
function read(rel) {
  const p = path.join(DOCS, rel)
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : ''
}

const compareCasesText = stripCodeBlocks(read('cases/compare.md'))
const compareMatrixText = stripCodeBlocks(read('compare-matrix.md'))

// 每个案例的解析结果
const caseResults = caseFiles.map(f => {
  const rel = `cases/${f}`
  const text = stripCodeBlocks(read(rel))
  const links = extractLinks(text)
  const mentionedVersions = KEY_VERSIONS.filter(v => new RegExp(`\\b(?:YOLO)?${v}\\b`, 'i').test(text))
  const mentionedCases = Array.from(links)
    .filter(l => /^\/cases\/[\w-]+/.test(l) && l !== '/cases/compare' && l !== '/cases/health' && l !== '/cases/index')
  const mentionsCompare = links.has('/cases/compare')
  const mentionsMatrix = links.has('/compare-matrix')
  const mentionsDerivativesCompare = links.has('/derivatives/compare')
  return { file: f, name: f.replace(/\.md$/, ''), links: Array.from(links), mentionedVersions, mentionedCases, mentionsCompare, mentionsMatrix, mentionsDerivativesCompare, text }
})

// A) 案例在 compare.md 中被引用
const caseInCompareMd = caseResults.filter(c => compareCasesText.includes(c.name))
// B) 案例在 compare-matrix.md 中被引用（宽松：名字子串匹配）
const caseInMatrixMd = caseResults.filter(c => compareMatrixText.includes(c.name) || compareMatrixText.includes(c.name.replace(/-/g, ' ')))

// C) 案例在 compare.md 中被点名的关键维度（用于交叉校验）
const compareDimHits = KEY_DIMS.map(d => ({ dim: d, inCompare: compareCasesText.includes(d), inMatrix: compareMatrixText.includes(d) }))

// D) 版本覆盖情况
const versionCoverage = KEY_VERSIONS.map(v => {
  const inMatrix = compareMatrixText.includes(v)
  const inCases = caseResults.some(c => c.mentionedVersions.includes(v))
  return { version: v, inMatrix, inCases }
})

// 生成 markdown 报告
function localDate() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

// 结果统计
const issues = []
caseResults.forEach(c => {
  if (!compareCasesText.includes(c.name)) issues.push({ level: 'warn', item: c.file, msg: `未被 cases/compare.md 引用` })
  if (!c.mentionsCompare && !c.mentionsMatrix) issues.push({ level: 'warn', item: c.file, msg: `未引用 /cases/compare 或 /compare-matrix` })
  if (c.mentionedCases.length === 0) issues.push({ level: 'info', item: c.file, msg: `未横向引用其它案例` })
  c.mentionedVersions.forEach(v => {
    if (!compareMatrixText.includes(v)) issues.push({ level: 'warn', item: c.file, msg: `提到 ${v} 但 compare-matrix.md 无对应条目` })
  })
})

const lines = [
  '# 案例 × 对比矩阵 一致性报告',
  '',
  `> 由 \`scripts/audit-cross-links.js\` 自动生成，请勿手改。`,
  `> 最后更新：${localDate()}`,
  '',
  '## 总览',
  '',
  `| 指标 | 数值 |`,
  `|------|------|`,
  `| 案例总数 | ${caseResults.length} |`,
  `| 被 cases/compare.md 引用 | ${caseInCompareMd.length} / ${caseResults.length} |`,
  `| 被 compare-matrix.md 引用 | ${caseInMatrixMd.length} / ${caseResults.length} |`,
  `| 引用 /cases/compare 的案例 | ${caseResults.filter(c => c.mentionsCompare).length} / ${caseResults.length} |`,
  `| 引用 /compare-matrix 的案例 | ${caseResults.filter(c => c.mentionsMatrix).length} / ${caseResults.length} |`,
  `| 横向引用其它案例的案例 | ${caseResults.filter(c => c.mentionedCases.length > 0).length} / ${caseResults.length} |`,
  `| 关键版本在 compare-matrix 中出现 | ${versionCoverage.filter(v => v.inMatrix).length} / ${KEY_VERSIONS.length} |`,
  `| 关键维度双方共现 | ${compareDimHits.filter(d => d.inCompare && d.inMatrix).length} / ${KEY_DIMS.length} |`,
  `| **告警（warn）** | **${issues.filter(i => i.level === 'warn').length}** |`,
  `| 提示（info） | ${issues.filter(i => i.level === 'info').length} |`,
  '',
  '---',
  '',
  '## 案例 × 双向链接矩阵',
  '',
  '| 案例 | 在 cases/compare.md | 在 compare-matrix.md | 引用 /cases/compare | 引用 /compare-matrix | 横向引用其它案例数 | 提到版本 |',
  '|------|:---:|:---:|:---:|:---:|:---:|------|',
  ...caseResults.map(c => `| [${c.name}](/cases/${c.name}) | ${compareCasesText.includes(c.name) ? '✅' : '—'} | ${compareMatrixText.includes(c.name) || compareMatrixText.includes(c.name.replace(/-/g, ' ')) ? '✅' : '—'} | ${c.mentionsCompare ? '✅' : '—'} | ${c.mentionsMatrix ? '✅' : '—'} | ${c.mentionedCases.length} | ${c.mentionedVersions.join(' ') || '—'} |`),
  '',
  '---',
  '',
  '## 关键版本覆盖（compare-matrix.md）',
  '',
  '| 版本 | 在 compare-matrix.md | 至少一个案例提到 |',
  '|------|:---:|:---:|',
  ...versionCoverage.map(v => `| ${v.version} | ${v.inMatrix ? '✅' : '—'} | ${v.inCases ? '✅' : '—'} |`),
  '',
  '---',
  '',
  '## 关键维度覆盖（compare.md × compare-matrix.md）',
  '',
  '| 维度 | 在 cases/compare.md | 在 compare-matrix.md |',
  '|------|:---:|:---:|',
  ...compareDimHits.map(d => `| ${d.dim} | ${d.inCompare ? '✅' : '—'} | ${d.inMatrix ? '✅' : '—'} |`),
  '',
  '---',
  '',
  '## 问题清单',
  '',
  issues.length === 0
    ? '✅ 无告警，三方一致性良好。'
    : issues.map(i => `- \`${i.level === 'warn' ? '⚠️' : 'ℹ️'}\` **${i.item}** · ${i.msg}`).join('\n'),
  '',
  '---',
  '',
  '## 修复建议',
  '',
  '- **未被 cases/compare.md 引用**：在 compare.md 的选型速查表 / 场景分派表中补一行。',
  '- **未被 compare-matrix.md 引用**：如案例不涉及版本对比，可忽略；否则在矩阵对应行补一句场景注。',
  '- **未引用 /cases/compare 或 /compare-matrix**：在案例末尾「相关案例」或「深入阅读」中加横向链接。',
  '- **提到版本 X 但 compare-matrix.md 无条目**：在 compare-matrix 中补该版本行，或从案例中移除版本提及。',
  '',
  '---',
  '',
  '**案例作者**：agent-yolo 文档组  ',
  `**最后更新**：${localDate()}  `
]

fs.writeFileSync(OUT, lines.join('\n') + '\n', 'utf8')
console.log(`✅ 已生成 ${OUT}`)
console.log(`   · 扫描 ${caseResults.length} 个案例 × 2 个对比页`)
console.log(`   · warn ${issues.filter(i => i.level === 'warn').length} / info ${issues.filter(i => i.level === 'info').length}`)
