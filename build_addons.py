"""Inject add-on modules into the claude.ai course exports.

    uv run --no-project build_addons.py        (then: uv run --no-project build_hub.py)

addons/<course-dir>/NN.json are modules in the engine lesson format (see engine/build_course.py), written for
topics the exported course doesn't cover. This script appends them to that course's inline data, patches the
hard-coded totals, places them in the course's parts, and adds KaTeX so $math$ renders.

Safe to re-run, and it MUST be re-run after re-exporting a course: modules marked "addon": true are dropped and
re-added, and the original totals are remembered in an <!--addons:...--> marker on first run.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent
KATEX = ('<!--addons-katex--><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">'
         '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>'
         '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>'
         '<script>addEventListener("load",()=>{const v=document.getElementById("view");if(!v||!window.renderMathInElement)return;'
         'const go=()=>renderMathInElement(v,{delimiters:[{left:"$$",right:"$$",display:true},{left:"$",right:"$",display:false}],'
         'throwOnError:false,ignoredClasses:["katex"]});go();new MutationObserver(go).observe(v,{childList:true});});</script>')

# Per export: where the data lives, the lesson shape it expects, module description field, and which part gets the module.
COURSES = {
    'ml-maths-course': dict(var='DATA', desc='d', parts='range',
                            descs={'13': 'Describe data, test hypotheses and run experiments: the statistics behind evaluation and A/B tests.',
                                   '14': 'How the optimisers that train models actually work: gradient descent to Adam, schedules and second-order methods.'},
                            new_parts={'14': 'Optimization'}),
    'inference-eng-course': dict(var='DATA', desc=None, parts='list', part_of={'31': 'Runtime & kernel engineering'}),
    'ai-research-track-web': dict(var='RAW', desc='desc', parts=None,
                                  descs={'13': 'What it takes to train a large language model across many GPUs: memory, parallelism, scaling laws, data and post-training.'}),
}


def blob(src, var):
    """(start, end, value) of the JSON literal after `const <var> = `."""
    i = src.index(f'const {var} = ') + len(f'const {var} = ')
    val, n = json.JSONDecoder().raw_decode(src, i)
    return i, n, val


def lesson_for(dirname, L):
    base = {k: L[k] for k in ('id', 't', 'n', 'a', 'q', 's', 'r')}
    base.update(k=L.get('k', []), x=L.get('x', []), ak='', f='')
    if dirname == 'ai-research-track-web':           # this export renders l.h; the injected patch falls back to render(l.a)
        words = len(L['a'].split())
        lead = next((p for p in L['a'].split('\n') if p.strip() and not p.startswith(('#', '-', '%', '>', '!', '|', '`'))), '')
        base.update(h='', free=True, w=words, mins=max(2, round(words / 180)), nx=0, lead=lead, blocks=[])
    return base


def inject(dirname, cfg):
    page = ROOT / dirname / 'index.html'
    src = page.read_text(encoding='utf-8')
    mods = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((ROOT / 'addons' / dirname).glob('*.json'))]
    i, j, val = blob(src, cfg['var'])
    modules = val['modules'] if cfg['var'] == 'RAW' else val
    modules[:] = [m for m in modules if not m.get('addon')]
    base_mods, base_lessons = len(modules), sum(len(m['lessons']) for m in modules)
    have = {m['n'] for m in modules}
    for m in mods:
        assert m['n'] not in have, f'{dirname}: add-on module {m["n"]} clashes with an existing module'
        mm = {'n': m['n'], 'name': m['name'], 'addon': True, 'lessons': [lesson_for(dirname, L) for L in m['lessons']]}
        if cfg['desc']:
            mm[cfg['desc']] = cfg.get('descs', {}).get(m['n']) or m.get('desc', '')
        if dirname == 'ml-maths-course':
            mm['diag'] = ''
        modules.append(mm)
    data = json.dumps(val, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    src = src[:i] + data + src[j:]

    # Totals: remember the export's own numbers once, then rewrite them from those every run.
    mk = re.search(r'<!--addons:(\d+),(\d+)-->', src)
    om, ol = (int(mk.group(1)), int(mk.group(2))) if mk else (base_mods, base_lessons)
    if not mk:
        src = src.replace('</body>', f'<!--addons:{om},{ol}--></body>', 1)
    nm, nl = om + len(mods), ol + sum(len(m['lessons']) for m in mods)
    cur_l = re.search(r'Search (\d+) lessons', src)
    cur_m = re.search(r'· (\d+) modules · ', src)
    for old in {ol, int(cur_l.group(1)) if cur_l else ol}:
        src = re.sub(rf'\b{old} lessons\b', f'{nl} lessons', src)
    for old in {om, int(cur_m.group(1)) if cur_m else om}:
        src = re.sub(rf'\b{old} modules\b', f'{nm} modules', src)
        src = src.replace(f'Module ${{m.n}} of {old} ', f'Module ${{m.n}} of {nm} ')

    # Parts
    if cfg['parts'] == 'range':                          # [{t, a, b}] by module number
        pm = re.search(r'const PARTS = (\[.*?\]);', src)
        parts = re.findall(r"\{t:'([^']*)', a:(\d+), b:(\d+)\}", pm.group(1))
        parts = [[t, int(a), int(b)] for t, a, b in parts]
        for m in mods:
            n = int(m['n'])
            title = cfg.get('new_parts', {}).get(m['n']) or m.get('part')   # a module file may name its part
            if title:
                if not any(p[0] == title for p in parts):
                    parts.append([title, n, n])
                else:
                    next(p for p in parts if p[0] == title)[2] = n
            elif not any(p[1] <= n <= p[2] for p in parts):
                parts[-1][2] = max(parts[-1][2], n)     # extend the last part
        lit = '[' + ', '.join(f"{{t:'{t}', a:{a}, b:{b}}}" for t, a, b in parts) + ']'
        src = src[:pm.start(1)] + lit + src[pm.end(1):]
    elif cfg['parts'] == 'list':                         # [{t, m:[codes]}]
        for code, title in cfg['part_of'].items():
            pat = re.compile(r"(\{t:'" + re.escape(title) + r"', m:\[)([^\]]*)(\])")
            mm = pat.search(src); assert mm, f'{dirname}: part {title} not found'
            codes = [c.strip().strip("'") for c in mm.group(2).split(',') if c.strip()]
            if code not in codes:
                codes.append(code)
            src = src[:mm.start(2)] + ', '.join(f"'{c}'" for c in codes) + src[mm.end(2):]

    if dirname == 'ai-research-track-web':               # Markdown lessons for add-ons
        src = src.replace('const body = l.h;', 'const body = l.h || render(l.a);')
        assert 'l.h || render(l.a)' in src
    if '<!--addons-katex-->' not in src:
        src = src.replace('</head>', KATEX + '\n</head>', 1)
    page.write_text(src, encoding='utf-8')
    print(f'{dirname:<24} +{len(mods)} modules  +{nl - ol} lessons  ({om}→{nm} modules, {ol}→{nl} lessons)')
    return nl


if __name__ == '__main__':
    for d, c in COURSES.items():
        if list((ROOT / 'addons' / d).glob('*.json')):
            inject(d, c)
