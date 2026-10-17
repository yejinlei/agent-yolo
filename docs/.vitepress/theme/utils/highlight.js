// @AI_GUARD: 轻量语法着色器，仅用于 ?raw 动态加载的源码；先 escape 再替换，任何 token 都不允许直接拼接原始文本
const C_KEYWORDS = new Set([
  'if', 'else', 'for', 'while', 'do', 'return', 'switch', 'case', 'default',
  'break', 'continue', 'goto', 'int', 'float', 'double', 'char', 'void',
  'const', 'static', 'struct', 'typedef', 'unsigned', 'long', 'short',
  'sizeof', 'ifdef', 'ifndef', 'if', 'endif', 'define', 'include', 'pragma'
])

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/**
 * 逐 token 着色 C 代码（支持块注释跨行状态）。
 * @param {string} src 全文
 * @returns {{html: string, block: boolean}} 按行返回 html 数组与结束时的块注释状态
 */
function tokenizeC(src) {
  const re = /(?<comment_line>\/\/[^\n]*)|(?<comment_block>\/\*[\s\S]*?\*\/)|(?<string>"(?:\\.|[^"\\])*")|(?<char>'(?:\\.|[^'\\])*')|(?<pp>#[a-z]+)|(?<num>\b\d[\d.fFuUlLxX]*\b)|(?<ident>[A-Za-z_]\w*)|(?<op>[{}()[\];,.])/g
  const out = []
  let m
  let last = 0
  function push(plain, cls) {
    const text = escapeHtml(plain)
    out.push(cls ? `<span class="tk-${cls}">${text}</span>` : text)
  }
  while ((m = re.exec(src))) {
    if (m.index > last) push(src.slice(last, m.index), '')
    const g = m.groups
    if (g.comment_line) push(m[0], 'com')
    else if (g.comment_block) push(m[0], 'com')
    else if (g.string) push(m[0], 'str')
    else if (g.char) push(m[0], 'str')
    else if (g.pp) push(m[0], 'pp')
    else if (g.num) push(m[0], 'num')
    else if (g.ident) {
      const word = m[0]
      const after = src[m.index + word.length]
      if (C_KEYWORDS.has(word)) push(word, 'kw')
      else if (after === '(') push(word, 'fn')
      else push(word, '')
    } else if (g.op) push(m[0], '')
    last = m.index + m[0].length
  }
  push(src.slice(last), '')
  return out.join('')
}

/** 对整份源码着色，返回按行切分的 html 数组（行号 1-based 对齐） */
export function highlightLines(src, lang) {
  let html
  if (lang === 'ini') {
    html = highlightIni(src)
  } else {
    html = tokenizeC(src)
  }
  return html.split('\n')
}

/** darknet cfg（INI 风格）：[section] / key=value / # 注释 */
function highlightIni(src) {
  return escapeHtml(src)
    .replace(/(^|\n)(\[[^\]]*\])/g, '$1<span class="tk-kw">$2</span>')
    .replace(/(^|\n)(\s*#.*)$/g, '$1<span class="tk-com">$2</span>')
    .replace(/(^|\n)(\s*)([A-Za-z_]\w*)(=)/g, '$1$2<span class="tk-pp">$3</span>$4')
}
