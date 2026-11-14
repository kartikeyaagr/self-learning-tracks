"""Derive engine/app.html (the template for ml-toolkit, machine-learning, deep-learning) from System Design 101's.

    uv run --no-project engine/derive.py [path/to/system-design-101/src/app.html]

Re-run after updating System Design 101, then rebuild the courses with engine/build_course.py.
Every course-specific literal becomes a COURSE value or a build-time placeholder; then the site features are added
(KaTeX, "Practice this" on lessons, study-plan links on modules). Each edit asserts its anchor, so a template change
that moves one fails loudly instead of producing a half-converted engine.
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def derive(s):
    def rep(a, b, n=1):
        nonlocal s
        assert s.count(a) == n, f'anchor found {s.count(a)}x (expected {n}): {a[:90]!r}'
        s = s.replace(a, b)

    def sub(pat, b, flags=0):
        nonlocal s
        s, k = re.subn(pat, b, s, count=1, flags=flags)
        assert k == 1, f'pattern not found: {pat[:90]!r}'

    # ---- course-specific literals -> COURSE / placeholders
    rep('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
    rep('<title>System Design Course</title>', '__HEAD__')
    for a, b in [('--accent:#2E80D9', '--accent:__ACC_L__'), ('--accent-soft:#EEF8FE', '--accent-soft:__ACC_SOFT_L__'),
                 ('--accent-line:#CBE2F7', '--accent-line:__ACC_LINE_L__')]:
        rep(a, b)
    for a, b in [('--accent:#6AA8EC', '--accent:__ACC_D__'), ('--accent-ink:#0A1522', '--accent-ink:__ACC_INK_D__'),
                 ('--accent-soft:#15212E', '--accent-soft:__ACC_SOFT_D__'), ('--accent-line:#24384F', '--accent-line:__ACC_LINE_D__')]:
        rep(a, b, 2)
    i = s.index('<a class="brand" href="#">'); j = s.index('</a>', i)
    s = s[:i] + '<a class="brand" href="#">__LOGO__<span class="wd">__BRAND__</span></a>' + s[j + 4:]
    sub(r'placeholder="Search \d+ lessons…[^"]*"', 'placeholder="__SEARCH__"')
    sub(r"const PARTS = \[\{t:'Fundamentals'[^;]*;",
        "const COURSE = __COURSE__;                   // course.json: title, key, parts, hero, footnote, plans\n"
        "const PARTS = COURSE.parts;                   // [{t, a, b}]: module number ranges\n"
        "const PRACTICE = __PRACTICE__;               // lesson id -> [[problem slug, title, difficulty]] from practice/links.json")
    rep("localStorage.getItem('sdsd')", "localStorage.getItem(COURSE.key)")
    rep("localStorage.setItem('sdsd'", "localStorage.setItem(COURSE.key")
    sub(r'const LV = \[[^\]]*\];', 'const LV = __LV__;')
    sub(r'const LVN = \[[^\]]*\];', 'const LVN = __LVN__;')
    sub(r'const GOAL = \d+;', 'const GOAL = __GOAL__;')
    sub(r"\['∞', '[^']*', 'Finish all \d+ lessons', n >= ALL\.length\],",
        "['∞', COURSE.finalBadge || 'Graduate', `Finish all ${ALL.length} lessons`, n >= ALL.length],")
    a = s[s.index('function covColor(n){'): s.index('function coverHTML(m, big){')]
    s = s.replace(a, """function covColor(n){
  // One hue family per part, stepping through the part's modules.
  const pi = Math.max(0, PARTS.findIndex(P => n >= P.a && n <= P.b)), P = PARTS[pi], k = P.b > P.a ? (n - P.a) / (P.b - P.a) : 0.5;
  const base = (COURSE.hues || [150, 20, 280, 200, 45, 330])[pi % 6];
  return [`hsl(${Math.round(base + k * 40) % 360} ${Math.round(34 + k * 12)}% ${Math.round(48 + Math.sin(k * Math.PI) * 4)}%)`, '#fff'];
}
""")
    rep('${ILL[m.n]}</g></svg>', "${ILL[m.n] || ''}</g></svg>")
    i = s.index('const ILL = {'); j = s.index('\n', i)
    s = s[:i] + 'const ILL = COURSE.ill || {};                 // optional per-module cover line art (SVG inner markup)' + s[j:]
    sub(r"const mini = flowSVG\(\[[^\]]*\], true\);", "const mini = COURSE.heroFig ? flowSVG(COURSE.heroFig, true) : '';")
    sub(r'<div class="lab">\$\{greet\} · \d+ modules · \d+ lessons</div>',
        '<div class="lab">${greet} · ${DATA.length} modules · ${ALL.length} lessons</div>')
    sub(r'<h1>How large systems are built, <span class="px">[^<]*</span></h1>', '<h1>${COURSE.hero}</h1>')
    sub(r'<h3>All \d+ lessons done\.</h3>', '<h3>All ${ALL.length} lessons done.</h3>')
    k = s.index('<p class="foot-note">'); e = s.index('</p>', k)
    s = s[:k] + "<p class=\"foot-note\">${COURSE.footnote || ''}</p>" + s[e + 4:]
    sub(r'<div class="lab">Module \$\{m\.n\} of \d+ · Part', '<div class="lab">Module ${m.n} of ${DATA.length} · Part')

    # ---- site features
    i = s.find('<p style="color:var(--muted);margin:0 0 16px;font-size:14.5px">Original lesson on')   # older templates only
    if i >= 0:
        j = s.index('</p>', i) + 4
        s = s[:i] + "${l.f ? `" + s[i:j] + "` : ''}" + s[j:]
    sub(r"""<div class="float f1" aria-hidden="true"><div class="lab">[^<]*</div>\$\{svgOf\(mini, '', false\)\}</div>""",
        "${mini ? `<div class=\"float f1\" aria-hidden=\"true\"><div class=\"lab\">${esc(COURSE.heroFigLabel || 'Fig')}</div>${svgOf(mini, '', false)}</div>` : ''}")
    # "Practice this" goes before the sources card (older templates) or before the completion zone (newer ones).
    anchor = ('      <section class="card blk"><div class="lab">Go deeper</div>' if 'Go deeper</div>' in s else '      <div class="done-zone">')
    rep(anchor,
        """      ${(PRACTICE[l.id] || []).length ? `<section class="card blk" id="practice"><div class="lab">Practice this</div><h3>Problems that use this lesson</h3>
        <div class="plinks">${PRACTICE[l.id].map(([slug, t, d]) => `<a href="../practice/index.html#l${slug}"><span>${esc(t)}</span><span class="pd ${d}">${d}</span></a>`).join('')}</div></section>` : ''}
""" + anchor)
    rep('  <div class="mnav">${prev ?', """  ${(COURSE.plans?.[m.n] || []).length ? `<section class="card blk" style="max-width:760px;margin:30px auto 0"><div class="lab">Practise this module</div><h3>Study plan sections</h3>
    <div class="plinks">${COURSE.plans[m.n].map(([id, title, sec]) => `<a href="../practice/index.html#plan-${id}"><span>${esc(title)}${sec ? ` · ${esc(sec)}` : ''}</span><span class="pd">Plan →</span></a>`).join('')}</div></section>` : ''}
  <div class="mnav">${prev ?""")
    rep('</style>\n</head>', """.plinks{display:flex;flex-direction:column;gap:8px;margin-top:12px}
.plinks a{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 14px;border-radius:12px;border:1px solid var(--line);background:var(--bg);text-decoration:none;font-size:15px}
.plinks a:hover{border-color:var(--accent)}
.plinks .pd{font:500 11.5px var(--mono);color:var(--muted);white-space:nowrap}
.plinks .pd.Easy{color:var(--go)} .plinks .pd.Medium{color:var(--gold)} .plinks .pd.Hard{color:var(--bad)}
</style>
</head>""")
    rep('&display=swap" rel="stylesheet">', '''&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>''')
    rep("const inl = s => esc(s).replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/`(.+?)`/g,'<code>$1</code>').replace(/(^|[^*])\\*([^*]+?)\\*/g,'$1<em>$2</em>');",
        """// Inline markdown; $math$ is kept verbatim for KaTeX (rendered after the page is drawn, see mathify()).
function inl(s){
  const math = []; s = String(s).replace(/\\$\\$[\\s\\S]+?\\$\\$|\\$[^$\\n]+?\\$/g, m => { math.push(m); return `\\u0000${math.length - 1}\\u0000`; });
  s = esc(s).replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/`(.+?)`/g,'<code>$1</code>').replace(/(^|[^*\\w])\\*([^*\\s][^*]*?)\\*/g,'$1<em>$2</em>');
  return s.replace(/\\u0000(\\d+)\\u0000/g, (_, i) => esc(math[+i]));
}
function mathify(el){ if (el && window.renderMathInElement) renderMathInElement(el, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], throwOnError: false}); }""")
    rep("  const hs = [...document.querySelectorAll('#prose h2')]; hs.forEach((h, i) => h.id = 's' + (i + 1));",
        "  mathify($('#view'));\n  const hs = [...document.querySelectorAll('#prose h2')]; hs.forEach((h, i) => h.id = 's' + (i + 1));")
    rep("</script>\n</body>", "addEventListener('load', () => mathify($('#view')));\n</script>\n</body>")
    # Design lessons can carry a rubric: shown with the model answer as the points a strong answer covers.
    rep('<div class="sample"><div class="lab">Model answer</div><p>${inl(l.s)}</p></div>',
        '<div class="sample"><div class="lab">Model answer</div><p>${inl(l.s)}</p></div>'
        '${(l.rubric || []).length ? `<div class="sample" style="background:var(--card);border-color:var(--line)"><div class="lab">A strong answer covers</div>'
        '<ul style="margin:6px 0 0;padding-left:20px">${l.rubric.map(x => `<li>${inl(x)}</li>`).join(\'\')}</ul></div>` : \'\'}')
    assert 'sdsd' not in s and 'System Design' not in s.split('<script>')[0], 'System Design specifics left in the head'
    return s


if __name__ == '__main__':
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'system-design-101' / 'src' / 'app.html'
    (ROOT / 'engine' / 'app.html').write_text(derive(src.read_text(encoding='utf-8')), encoding='utf-8')
    print(f'engine/app.html derived from {src}')
