"""Build a course page from engine/app.html.

    uv run --no-project engine/build_course.py ml-toolkit [machine-learning ...]

A course lives in <dir>/src/:
  course.json      title, key (localStorage + hub key), brand, logo, accent, parts, hero, levels, plans
  modules/NN.json  {"n": "NN", "name": ..., "lessons": [lesson, ...]}

Lesson fields (the System Design 101 format):
  id "NN.MM", t title, n [key ideas], a notes (markdown dialect + %%flow/%%seq/%%layers diagrams),
  q self-check question, s model answer, r [[source title, url, what to read]], k [tags], x [tools]

Writes <dir>/index.html and <dir>/favicon.svg. Lessons get "Practice this" links from practice/links.json,
and modules get study-plan links from course.json "plans". Run build_hub.py afterwards.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = (ROOT / 'engine' / 'app.html').read_text(encoding='utf-8')
REQ = {'id': str, 't': str, 'n': list, 'a': str, 'q': str, 's': str, 'r': list}
js = lambda x: json.dumps(x, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')


def check_lesson(L, where):
    for k, typ in REQ.items():
        assert isinstance(L.get(k), typ) and L[k], f'{where}: missing or empty {k}'
    assert len(L['a'].split()) >= 350, f'{where}: notes are only {len(L["a"].split())} words (min 350)'
    assert len(L['n']) >= 3, f'{where}: give at least 3 key ideas'
    for r in L['r']:
        assert isinstance(r, list) and len(r) == 3 and r[1].startswith('https://'), f'{where}: bad source {r}'
    L.setdefault('k', []); L.setdefault('x', []); L.setdefault('ak', '')
    return L


def build(dirname):
    src = ROOT / dirname / 'src'
    cfg = json.loads((src / 'course.json').read_text(encoding='utf-8'))
    mods = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((src / 'modules').glob('*.json'))]
    assert mods, f'{dirname}: no modules'
    for i, m in enumerate(mods, 1):
        assert m['n'] == f'{i:02d}', f'{dirname}: module numbers must run 01, 02, … (got {m["n"]} at position {i})'
        for j, L in enumerate(m['lessons'], 1):
            assert L.get('id') == f'{m["n"]}.{j:02d}', f'{dirname}/{m["n"]}: lesson {j} id should be {m["n"]}.{j:02d}'
            check_lesson(L, f'{dirname} {L["id"]}')
    last = max(P['b'] for P in cfg['parts'])
    assert last == len(mods) and cfg['parts'][0]['a'] == 1, f'{dirname}: parts must cover modules 1..{len(mods)}'

    links = json.loads((ROOT / 'practice' / 'links.json').read_text(encoding='utf-8'))
    pre = cfg['key'] + ':'
    practice = {k[len(pre):]: v for k, v in links.items() if k.startswith(pre)}
    # Lessons can also name their practice problems directly ("p": [slugs]); merged with problems that cite the lesson.
    probs = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in (ROOT / 'practice' / 'data').glob('*.json')}
    for m in mods:
        for L in m['lessons']:
            have = {x[0] for x in practice.get(L['id'], [])}
            for slug in L.get('p', []):
                assert slug in probs, f'{dirname} {L["id"]}: unknown practice problem {slug}'
                if slug not in have:
                    practice.setdefault(L['id'], []).append([slug, probs[slug]['t'], probs[slug]['d']])
    known = {p.stem for p in (ROOT / 'practice' / 'src' / 'plans').glob('*.json')}
    for n, refs in cfg.get('plans', {}).items():
        assert all(r[0] in known for r in refs), f'{dirname}: unknown study plan in plans[{n}]'

    a = cfg['accent']
    head = (f'<meta name="description" content="{cfg["description"]}">\n<meta name="theme-color" content="#F7F7F8">\n'
            f'<link rel="icon" href="favicon.svg" type="image/svg+xml">\n<title>{cfg["title"]}</title>')
    course = {k: cfg[k] for k in ('key', 'title', 'parts', 'hero', 'footnote', 'finalBadge', 'plans', 'heroFig', 'heroFigLabel', 'hues', 'ill') if k in cfg}
    out = ENGINE
    for k, v in {'__HEAD__': head, '__LOGO__': cfg['logo'], '__BRAND__': cfg['brand'],
                 '__SEARCH__': cfg.get('search', f'Search {sum(len(m["lessons"]) for m in mods)} lessons…'),
                 '__ACC_L__': a['l'], '__ACC_SOFT_L__': a['soft_l'], '__ACC_LINE_L__': a['line_l'],
                 '__ACC_D__': a['d'], '__ACC_INK_D__': a['ink_d'], '__ACC_SOFT_D__': a['soft_d'], '__ACC_LINE_D__': a['line_d'],
                 '__COURSE__': js(course), '__PRACTICE__': js(practice), '__DATA__': js(mods),
                 '__LV__': json.dumps(cfg['LV']), '__LVN__': json.dumps(cfg['LVN']), '__GOAL__': str(cfg.get('GOAL', 2))}.items():
        assert k in out, k
        out = out.replace(k, v)
    (ROOT / dirname / 'index.html').write_text(out, encoding='utf-8')
    fav = cfg['logo'].replace('var(--ink)', '#272727').replace('var(--bg)', '#F7F7F8').replace('var(--accent)', a['l'])
    (ROOT / dirname / 'favicon.svg').write_text(fav.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1), encoding='utf-8')
    n = sum(len(m['lessons']) for m in mods)
    print(f'{cfg["title"]:<20} {len(mods):>2} modules {n:>4} lessons · {sum(len(v) for v in practice.values())} practice links')


def check(paths):
    """Validate module files only (for parallel authors): structure, ids, lengths, sources, practice slugs."""
    probs = {f.stem for f in (ROOT / 'practice' / 'src' / 'problems').glob('*.py')}
    bad = 0
    for raw in paths:
        f = Path(raw).resolve()
        try:
            m = json.loads(f.read_text(encoding='utf-8'))
            assert re.fullmatch(r'\d\d', m.get('n', '')) and m.get('name') and m.get('lessons'), 'module needs n ("NN"), name, lessons'
            for j, L in enumerate(m['lessons'], 1):
                assert L.get('id') == f'{m["n"]}.{j:02d}', f'lesson {j} id should be {m["n"]}.{j:02d}'
                check_lesson(L, L['id'])
                unknown = [p for p in L.get('p', []) if p not in probs]
                assert not unknown, f'{L["id"]}: unknown practice problems {unknown}'
                for line in L['a'].split('\n'):
                    assert not re.match(r'^\|[\s:|-]+\|?$', line.strip()), f'{L["id"]}: drop the |---| table separator row'
            print(f'OK    {f.name}  ({len(m["lessons"])} lessons, {sum(len(L["a"].split()) for L in m["lessons"])} words)')
        except Exception as e:
            bad += 1
            print(f'FAIL  {f.name}: {type(e).__name__}: {e}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--check']:
        check(sys.argv[2:])
    else:
        for d in sys.argv[1:]:
            build(d)
