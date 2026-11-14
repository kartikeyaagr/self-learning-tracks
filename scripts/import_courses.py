"""Import courses into the Astro site as content files.

    uv run --no-project scripts/import_courses.py            # every course
    uv run --no-project scripts/import_courses.py ml-toolkit # just these

Writes, per course:
  site/src/content/courses/<dir>/course.json             config + modules + parts (as module-code lists)
  site/src/content/courses/<dir>/lessons/<id>-<slug>.md  YAML frontmatter + Markdown notes
  site/src/content/courses/<dir>/math.json               exports only: ⟦n⟧ → pre-rendered MathML, for frontmatter text

Sources:
  engine courses   <dir>/src/course.json + src/modules/NN.json (ml-toolkit, machine-learning, deep-learning, ml-system-design)
  claude.ai exports <dir>/index.html (or assets/data.js) + add-on modules from addons/<dir>/*.json

The house lesson dialect becomes standard Markdown that remark parses:
  %%flow caption … %%   → ~~~dg-flow caption … ~~~   (house diagrams, rendered to SVG at build time)
  course-only figures   → ~~~dg-raw … ~~~            (original source, rendered in the browser by that course's own code)
  ! Example: title …    → <div class="ex"> worked-example box
  | a | b | tables       → GFM tables (the |---| row is added after the header)
  > Takeaway: …         → <div class="take"> key-takeaway box
  ⟦n⟧ maths references   → the pre-rendered MathML they point to (ML Maths and Inference store maths this way)
Once imported, the Markdown files are the source of truth (decided 2026-11-14); re-import only to pull in a new export.
"""
import ast, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'site' / 'src' / 'content' / 'courses'
STD = ('flow', 'seq', 'layers', 'state', 'tree', 'timeline', 'compare', 'matrix', 'bars', 'widget')   # house renderer
slug = lambda s: re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:60].strip('-') or 'lesson'

ENGINE = ['ml-toolkit', 'machine-learning', 'deep-learning', 'ml-system-design']
EXPORTS = {   # raw = figure kinds that stay course-specific; single = those written as one line, no closing %%
    'ml-maths-course':       dict(name='ML Maths', data='index.html', engine='index.html', key='mlmaths', layer=1, raw={'viz'}, single={'viz'}),
    'ai-research-track-web': dict(name='AI Research', data='index.html', engine='index.html', key='air-fanout-v1', layer=3, raw=set(), single=set()),
    'ai-engineering-course': dict(name='AI Engineering', data='assets/data.js', engine='assets/app.js', key='aieng', layer=3, raw={'widget'}, single={'widget'}),
    'inference-eng-course':  dict(name='Inference Engineering', data='index.html', engine='index.html', key='ieng', layer=4,
                                  raw={'widget', 'chart', 'grid', 'bars', 'timeline'}, single=set()),
    'system-design-101':     dict(name='System Design', data='index.html', engine='index.html', key='sdsd', layer=4, raw=set(), single=set(), house=True),
}
LAYER = {'ml-toolkit': 1, 'machine-learning': 2, 'deep-learning': 2, 'ml-system-design': 4}
# Where add-on modules sit in an export's parts (mirrors build_addons.py).
ADDON_PART = {'ml-maths-course': {'13': 'Probability and statistics', '14': 'Optimization'},
              'inference-eng-course': {'31': 'Runtime & kernel engineering'}}


def js_value(src, name):
    i = src.index(f'const {name} = ') + len(f'const {name} = ')
    return json.JSONDecoder().raw_decode(src, i)[0]


def literal_dollars(t):
    """Escape $ outside `code`: the exported courses never used $ for maths, so a $ there is a price, not TeX."""
    return ''.join(seg if k % 2 else seg.replace('$', '\\$') for k, seg in enumerate(re.split(r'(`+[^`]*`+)', t)))


def convert(a, raw=frozenset(), single=frozenset(), math=None, dollars=False):
    """House dialect → Markdown. `math` resolves ⟦n⟧ references to inline HTML; `dollars` escapes literal $."""
    if math:
        a = re.sub('\u27e6(\\d+)\u27e7', lambda m: math[int(m.group(1))], a)
    lines, out, i = a.split('\n'), [], 0
    if dollars:   # escape outside code fences and outside course-specific figure blocks (rendered by their own code)
        fixed, fence, rawb = [], False, False
        for L in lines:
            if L.startswith('```'):
                fence = not fence
            elif not fence and re.match(r'^%%(\w+)', L):          # figure blocks: labels are escaped by their renderer
                rawb = re.match(r'^%%(\w+)', L).group(1) not in single
            elif rawb and L.strip() == '%%':
                rawb = False
            elif not fence and not rawb and not re.match(r'^%%(\w+)', L):
                L = literal_dollars(L)
            fixed.append(L)
        lines = fixed
    block = lambda xs: out.extend(['', *xs, ''])
    while i < len(lines):
        L = lines[i]
        m = re.match(r'^%%(\w+)\b\s*(.*)$', L)
        if m and m.group(1) in single:
            block(['~~~dg-raw', L, '~~~']); i += 1; continue
        if m:
            body = []; i += 1
            while i < len(lines) and lines[i].strip() != '%%':
                body.append(lines[i]); i += 1
            i += 1
            if m.group(1) in raw or m.group(1) not in STD:
                block(['~~~dg-raw', L, *body, '%%', '~~~'])
            else:
                block([f'~~~dg-{m.group(1)} {m.group(2)}'.rstrip(), *body, '~~~'])   # ~~~: captions can contain `code`
            continue
        if L.startswith('```'):
            body = [L]; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                body.append(lines[i]); i += 1
            body.append(lines[i] if i < len(lines) else '```'); i += 1
            block(body); continue
        if L.startswith('! '):
            ex = []
            while i < len(lines) and lines[i].startswith('! '):
                ex.append(lines[i][2:]); i += 1
            title = re.sub(r'^Example:\s*', '', ex[0])
            block(['<div class="ex">', '<div class="lab">Worked example</div>', '', f'**{title}**', '',
                   *[x for p in ex[1:] for x in (p, '')], '</div>']); continue
        if L.startswith('| '):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]); i += 1
            if len(rows) < 2 or not re.match(r'^\|[\s:|-]+\|?\s*$', rows[1]):
                ncol = rows[0].strip().strip('|').count(' | ') + 1
                rows.insert(1, '|' + ' --- |' * ncol)
            block(rows); continue
        if L.startswith('> '):
            block(['<div class="take">', '<div class="lab">Key takeaway</div>', '', re.sub(r'^> (Takeaway: )?', '', L), '', '</div>'])
            i += 1; continue
        if L.startswith(('## ', '### ')):
            block([L]); i += 1; continue
        if re.match(r'^(- |\d+\. )', L) and out and out[-1].strip() and not re.match(r'^(\s*- |\s*\d+\. |\s{2,})', out[-1]):
            out.append('')
        out.append(L); i += 1
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip() + '\n'


def front(d):
    return '---\n' + '\n'.join(f'{k}: {json.dumps(v, ensure_ascii=False)}' for k, v in d.items() if v not in (None, [], '', {})) + '\n---\n'


def sources(r):
    return [{'title': x[0], 'url': x[1], 'note': x[2] if len(x) > 2 else ''} for x in r if len(x) > 1 and str(x[1]).startswith('http')]


def write(dirname, cfg, mods, render_lesson, math=None):
    dest = OUT / dirname
    (dest / 'lessons').mkdir(parents=True, exist_ok=True)
    for f in (dest / 'lessons').glob('*.md'):
        f.unlink()
    n = 0
    for m in mods:
        used = set()
        for j, L in enumerate(m['lessons'], 1):
            s = slug(L['t'])
            if s in used:
                s = f'{s}-{j}'
            used.add(s)
            fm, body = render_lesson(L, m)
            fm = {'id': L['id'], 'course': dirname, 'module': m['n'], 'slug': s, 'title': L['t'], **fm}
            (dest / 'lessons' / f"{L['id'].replace('.', '-')}-{s[:40]}.md").write_text(front(fm) + '\n' + body, encoding='utf-8')
            n += 1
    cfg['modules'] = [{'n': m['n'], 'name': m['name'], 'slug': slug(m['name']), 'desc': m.get('d') or m.get('desc') or ''} for m in mods]
    (dest / 'course.json').write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding='utf-8')
    if math is not None:
        (dest / 'math.json').write_text(json.dumps(math, ensure_ascii=False), encoding='utf-8')
    elif (dest / 'math.json').exists():
        (dest / 'math.json').unlink()
    print(f'{dirname:24} {len(mods):>2} modules {n:>4} lessons')


def engine_lesson(L, m):
    return ({'keyIdeas': L['n'], 'question': L['q'], 'answer': L['s'], 'rubric': L.get('rubric', []), 'sources': sources(L['r']),
             'practice': L.get('p', []), 'tags': L.get('k', []), 'tools': L.get('x', [])}, convert(L['a']))


def import_engine(dirname):
    src = ROOT / dirname / 'src'
    cfg = json.loads((src / 'course.json').read_text(encoding='utf-8'))
    mods = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((src / 'modules').glob('*.json'))]
    if not mods:
        print(f'{dirname:24} skipped: no modules yet'); return
    cfg['parts'] = [{'t': P['t'], 'm': [m['n'] for m in mods if P['a'] <= int(m['n']) <= P['b']]} for P in cfg['parts']]
    cfg['layer'] = LAYER[dirname]; cfg['css'] = 'house'
    write(dirname, cfg, mods, engine_lesson)


def import_export(dirname, ex):
    d = ROOT / dirname
    data_src = (d / ex['data']).read_text(encoding='utf-8')
    code = (d / ex['engine']).read_text(encoding='utf-8')
    page = (d / 'index.html').read_text(encoding='utf-8')
    mods = js_value(data_src, 'RAW')['modules'] if 'const RAW = ' in data_src else js_value(data_src, 'DATA')
    mods = [m for m in mods if not m.get('addon')]                  # injected by build_addons.py; re-added below
    math = js_value(data_src, 'MATH') if 'const MATH = ' in data_src else None
    addons = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((ROOT / 'addons' / dirname).glob('*.json'))]
    num = lambda pat: ast.literal_eval(re.search(pat, code).group(1))
    parts_src = re.search(r'const PARTS = (\[.*?\]);', code).group(1)
    if 'm:[' in parts_src.replace(' ', ''):
        parts = [{'t': t, 'm': re.findall(r"'([^']+)'", ms)} for t, ms in re.findall(r"\{t:'([^']*)', m:\[([^\]]*)\]\}", parts_src)]
    else:
        parts = [{'t': t, 'm': [m['n'] for m in mods if int(a) <= int(m['n']) <= int(b)]}
                 for t, a, b in re.findall(r"\{t:'([^']*)', a:(\d+), b:(\d+)\}", parts_src)]
    for am in addons:
        title = ADDON_PART.get(dirname, {}).get(am['n']) or am.get('part')
        P = next((p for p in parts if p['t'] == title), None) if title else parts[-1]
        if P is None:
            P = {'t': title, 'm': []}; parts.append(P)
        P['m'].append(am['n'])
    brand = re.search(r'<a class="brand" href="#">(<svg.*?</svg>)\s*<span class="wd">(.*?)</span>', page, re.S)
    hero = re.search(r'<h1>(.*?)</h1>', code[code.find('function home('):])
    title = re.search(r'<title>(.*?)</title>', page).group(1)
    desc = re.search(r'<meta name="description" content="([^"]*)"', page)
    cfg = {'key': ex['key'], 'title': ex['name'], 'brand': brand.group(2), 'logo': brand.group(1),
           'description': desc.group(1) if desc else title, 'hero': hero.group(1) if hero else title,
           'parts': parts, 'LV': num(r'const LV = (\[[^\]]*\])'), 'LVN': num(r'const LVN = (\[[^\]]*\])'),
           'GOAL': num(r'const GOAL = (\d+)'), 'layer': ex['layer'], 'css': 'house' if ex.get('house') else dirname}

    def lesson(L, m):
        if m.get('addon_src'):
            return engine_lesson(L, m)
        lit = literal_dollars
        fm = {'keyIdeas': [lit(x) for x in L.get('n', [])], 'question': lit(L.get('q', '')), 'answer': lit(L.get('s', '')),
              'sources': [{**x, 'note': lit(x['note'])} for x in sources(L.get('r', []))], 'tags': L.get('k', []), 'tools': L.get('x', [])}
        if L.get('tp'):
            fm['topic'] = {'id': L['tp'], 'title': L.get('tt', '')}
        if L.get('h'):                                             # AI Research: pre-rendered HTML lessons
            fm.update(html=L['h'], lead=L.get('lead', ''), mins=L.get('mins'), blocks=L.get('blocks', []))
            return fm, ''
        return fm, convert(L.get('a') or '\n'.join(f'- {x}' for x in L.get('n', [])), ex['raw'], ex['single'], math, dollars=True)

    for am in addons:
        am['addon_src'] = True
    write(dirname, cfg, mods + addons, lesson, math)
    sheets = d / 'sheets'
    if sheets.exists():                                            # notebooks linked from AI Research's blocks
        out = ROOT / 'site/public/courses' / dirname / 'sheets'
        out.mkdir(parents=True, exist_ok=True)
        for f in sheets.glob('*.ipynb'):
            (out / f.name).write_bytes(f.read_bytes())


if __name__ == '__main__':
    for dname in sys.argv[1:] or ENGINE + list(EXPORTS):
        if dname in EXPORTS:
            import_export(dname, EXPORTS[dname])
        else:
            import_engine(dname)
