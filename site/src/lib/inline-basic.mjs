// Inline markdown without maths, for code that runs in the browser (keeps KaTeX out of the client bundle).
export const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
export const inlBasic = (s, math = m => esc(`$${m}$`)) => {
  const ms = [];
  // \$ is a literal dollar (escaped by the importer for courses that never used $ for maths)
  s = String(s).replace(/\\\$/g, '\u0001').replace(/\$([^$\n]+?)\$/g, (_, m) => { ms.push(m); return `\u0000${ms.length - 1}\u0000`; });
  s = esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*/g, '$1<em>$2</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => math(ms[+i])).replace(/\u0001/g, '$');
};
export const inl = s => inlBasic(s);
