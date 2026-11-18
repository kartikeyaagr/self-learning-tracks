# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.*", "pandas==2.3.*", "torch>=2.6", "jax[cpu]>=0.6", "numba>=0.61", "numba-cuda>=0.10", "scipy==1.14.*"]
# ///
# Versions match the browser runtime (Pyodide 0.29.5: Python 3.13, numpy 2.2, pandas 2.3) so expected outputs agree.
"""Build the Practice track: verify every problem, then write the site files.

    uv run practice/src/build.py          (then rebuild the site: cd site && npm run build)

Each problems/<slug>.py is a module with META, STATEMENT, STARTER, SOLUTION, HINTS, TESTS, EXPLANATION.
Expected outputs are computed from SOLUTION here, so a problem only ships if:
  - SOLUTION passes every test after a JSON round trip, twice with the same result (deterministic);
  - STARTER fails at least one test (the problem isn't pre-solved);
  - tags, related slugs and lesson links all resolve.
Writes site/public/practice/: index.json, data/<slug>.json, quiz/<plan>.json, nb/<slug>.ipynb (Colab), links.json, runner,
and routes.json (Unroll: concepts and destinations from site/src/content/, Arena sets from plan sections).
"""
import copy, importlib.util, json, os, re, sys
# CUDA problems are Numba kernels: verified on the CPU simulator here, run on a real GPU in Colab.
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1"); os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent
ROOT = OUT.parent
sys.path.insert(0, str(OUT))
import harness  # noqa: E402

REPO = 'kartikeyaagr/self-learning-tracks'   # public GitHub repo; Colab opens notebooks straight from it

# The 46 topic tags (TensorTonic's topic list), grouped into parts for the sidebar. Order = sidebar order.
PARTS = [
    ('Math', ['Linear Algebra', 'Calculus', 'Probability and Statistics', 'Probability', 'Optimization']),
    ('Data', ['Data Processing', 'Feature Engineering', 'Time Series']),
    ('Classical ML', ['Classic ML', 'Regression', 'Classification', 'Clustering', 'Metrics & Evaluation',
                      'Evaluation Metrics', 'Recommender Systems']),
    ('Deep learning', ['Neural Networks', 'Activation Functions', 'Loss Functions', 'Training',
                       'Convolutional Networks', 'Computer Vision', 'Object Detection', '3D Geometry', 'U-Net']),
    ('Language', ['NLP', 'Embeddings', 'Attention', 'Transformers', 'Language Models', 'Scaling Laws']),
    ('Generative', ['Generative Models', 'Diffusion Models', 'Noise Schedules', 'Score Matching', 'Flow Matching',
                    'ODE Solvers', 'Sampling', 'Conditional Generation', 'Conditioning']),
    ('Reinforcement learning', ['Reinforcement Learning']),
    ('Systems', ['GPU Programming', 'Inference', 'Quantization', 'Distributed Computing', 'JAX', 'MLOps']),
]
TAGS = [t for _, ts in PARTS for t in ts]
DIFFS = {'Easy': 10, 'Medium': 20, 'Hard': 30}            # minutes, shown like a lesson's reading time
BROWSER = {'python': [], 'numpy': [], 'pandas': ['pandas'], 'sql': ['sqlite3']}   # runtime -> extra Pyodide packages
COLAB = {'torch', 'jax', 'cuda', 'triton'}                 # run in Colab (notebook with the same tests)
# Triton needs an NVIDIA GPU, which the build machine doesn't have: expected outputs come from the problem's
# REFERENCE (a NumPy/PyTorch-CPU implementation of the same spec) and SOLUTION is only syntax-checked here.
GPU_ONLY = {'triton'}
SETUP = {'cuda': '!pip -q install "numba-cuda[cu12]"', 'jax': '', 'torch': '', 'triton': ''}
NEEDS_GPU = {'cuda', 'triton'}
slugify = lambda s: re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compile_fn(code, name):
    ns = {}
    exec(compile(code, name, 'exec'), ns)
    return ns


SITE = ROOT / 'site'
WEB = SITE / 'public' / 'practice'                      # what the site serves: data/, quiz/, nb/, index.json, runner
BASE = '/self-learning-tracks'                          # GitHub Pages base path (site/astro.config.mjs)


def site_lessons():
    """Every lesson on the site: (course key, course title, lesson frontmatter, href). Frontmatter values are JSON."""
    for cj in sorted((SITE / 'src/content/courses').glob('*/course.json')):
        cfg = json.loads(cj.read_text(encoding='utf-8'))
        mods = {m['n']: m['slug'] for m in cfg['modules']}
        for f in sorted((cj.parent / 'lessons').glob('*.md')):
            head = f.read_text(encoding='utf-8').split('\n---\n', 1)[0]
            fm = {k: json.loads(v) for k, v in re.findall(r'^(\w+): (.*)$', head, re.M)}
            yield cfg['key'], cfg['title'], fm, f"{BASE}/library/{cj.parent.name}/{mods[fm['module']]}/{fm['slug']}/"


def lesson_index():
    """'trackkey:lessonid' -> {href, t (title), c (course)} for every lesson on the site."""
    return {f"{k}:{fm['id']}": {'href': href, 't': fm['title'], 'c': title} for k, title, fm, href in site_lessons()}


def build_one(path, lessons):
    m, slug = load(path), path.stem
    meta = m.META
    where = f'{path.name}:'
    assert re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', slug), f'{where} slug must be kebab-case'
    tags = meta['tags']
    assert tags and all(t in TAGS for t in tags), f'{where} unknown tag in {tags}'
    assert meta['diff'] in DIFFS, f'{where} diff'
    rt = meta['runtime']
    assert rt in BROWSER or rt in COLAB, f'{where} runtime {rt}'
    assert len(m.HINTS) >= 2, f'{where} needs at least 2 hints'
    assert len(m.TESTS) >= 5, f'{where} needs at least 5 tests'
    for k in ('STATEMENT', 'EXPLANATION'):
        assert len(getattr(m, k).strip()) > 80, f'{where} {k} too short'
    links = []
    for ref in meta.get('lessons', []):
        assert ref in lessons, f'{where} unknown lesson {ref}'
        links.append(lessons[ref])

    if rt == 'sql':
        fn_name, target = None, m.SOLUTION
        starter_target = m.STARTER
        args_list = [(setup,) for setup in m.TESTS]
        call = lambda f, a: harness.run_sql(f, a[0])
    else:
        fn_name = re.search(r'^def (\w+)\(', m.STARTER, re.M).group(1)
        if rt in GPU_ONLY:                                  # can't import or run here: syntax-check, grade REFERENCE instead
            import ast
            ast.parse(m.STARTER); ast.parse(m.SOLUTION)
            assert re.search(rf'^def {fn_name}\(', m.SOLUTION, re.M), f'{where} SOLUTION must define {fn_name}()'
            assert hasattr(m, 'REFERENCE'), f'{where} {rt} problems need REFERENCE (code defining {fn_name}() without a GPU)'
            ref = compile_fn(m.REFERENCE, f'{slug}:REFERENCE')
            assert fn_name in ref, f'{where} REFERENCE must define {fn_name}()'
            target, starter_target = ref[fn_name], None
        else:
            ns = compile_fn(m.SOLUTION, f'{slug}:SOLUTION')
            assert fn_name in ns, f'{where} SOLUTION must define {fn_name}()'
            target, starter_target = ns[fn_name], compile_fn(m.STARTER, f'{slug}:STARTER').get(fn_name)
        assert all(isinstance(t, tuple) for t in m.TESTS), f'{where} each test is a tuple of args'
        args_list = m.TESTS
        prt = 'torch' if rt in GPU_ONLY else rt                 # reference runs on CPU tensors
        call = lambda f, a: harness.plain(f(*copy.deepcopy(harness.prep(list(a), prt))))

    tol = meta.get('tol', [1e-6, 1e-8])
    tests = [{'a': harness.enc(list(a)), 'e': harness.enc(call(target, a))} for a in args_list]
    tests = json.loads(json.dumps(tests))                  # exactly what the browser will receive
    again = [{'a': t['a'], 'e': harness.enc(call(target, harness.dec(t['a'])))} for t in tests]
    assert json.loads(json.dumps(again)) == tests, f'{where} SOLUTION is not deterministic'
    res = harness.grade(target, tests, 'torch' if rt in GPU_ONLY else rt, *tol)
    assert all(r['ok'] for r in res), f'{where} SOLUTION fails its own tests: {[r for r in res if not r["ok"]][:1]}'
    if starter_target is not None:
        res = harness.grade(starter_target, tests, rt, *tol)
        assert not all(r['ok'] for r in res), f'{where} STARTER already passes every test'

    nex = meta.get('examples', 2)
    p = {'id': slug, 't': meta['title'], 'tags': tags, 'd': meta['diff'], 'rt': rt, 'fn': fn_name, 'tol': tol,
         'pkgs': BROWSER.get(rt, []), 'st': m.STATEMENT.strip(), 'starter': m.STARTER.strip() + '\n',
         'hints': m.HINTS, 'sol': m.SOLUTION.strip() + '\n', 'ex_md': m.EXPLANATION.strip(),
         'tests': tests, 'nex': nex,
         'shown': [{'in': harness.show_args(harness.dec(t['a']), rt), 'out': harness.short(harness.dec(t['e']), 600)}
                   for t in tests[:nex]],
         'lessons': links, 'lesson_refs': meta.get('lessons', []), 'related': meta.get('related', []),
         'theory': meta.get('theory', [])}
    if rt in GPU_ONLY:
        p['gpu_only'] = True                                # UI: "solution checked on a GPU only"
    if rt in NEEDS_GPU:
        p['gpu'] = True
    if rt in COLAB:
        p['colab'] = f'https://colab.research.google.com/github/{REPO}/blob/main/site/public/practice/nb/{slug}.ipynb'
    return p


def notebook(p):
    cell = lambda kind, src: {'cell_type': kind, 'metadata': {}, 'source': src,
                              **({'outputs': [], 'execution_count': None} if kind == 'code' else {})}
    harness_src = (OUT / 'harness.py').read_text(encoding='utf-8')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
            'cells': [
                cell('markdown', f"# {p['t']}\n\n{p['st']}\n\n"
                     + ("**Runtime → Change runtime type → GPU** before running.\n\n" if p.get('gpu') else '')
                     + "Write your solution in the next cell, then run the test cell."),
                *([cell('code', SETUP[p['rt']])] if SETUP.get(p['rt']) else []),
                cell('code', p['starter']),
                cell('code', '# Tests (same harness and cases as the website)\n'
                     + f"_HARNESS = {harness_src!r}\n_TESTS = {json.dumps(p['tests'])!r}\n"
                     + "import json, types\nharness = types.ModuleType('harness'); exec(_HARNESS, harness.__dict__)\n"
                     + f"res = harness.grade({p['fn']}, json.loads(_TESTS), {p['rt']!r}, *{p['tol']!r})\n"
                     + "for i, r in enumerate(res): print(f\"test {i+1}: {'pass' if r['ok'] else 'FAIL  ' + r['why']}\")\n"
                     + "print(f\"{sum(r['ok'] for r in res)}/{len(res)} passed\")"),
            ]}


CACHE = SRC / '.cache'                                     # verified problems, keyed by source + harness + VERIFY
VERIFY = b'verify-v1'   # bump when build_one's checks or output fields change, to force re-verification of every problem


def build_cached(f, lessons):
    """build_one, skipped when the problem, the harness and this build script are unchanged since it last passed.
    CUDA problems run on a slow CPU simulator, so a full rebuild without this takes over an hour."""
    import hashlib, time
    h = hashlib.sha256(f.read_bytes() + (OUT / 'harness.py').read_bytes() + VERIFY).hexdigest()[:16]
    c = CACHE / f'{f.stem}.json'
    if c.exists():
        hit = json.loads(c.read_text(encoding='utf-8'))
        if hit.get('hash') == h:
            p = hit['p']
            p['lessons'] = [lessons[r] for r in p.get('lesson_refs', []) if r in lessons]   # lesson titles may change
            return p
    t0 = time.time()
    p = build_one(f, lessons)
    if time.time() - t0 > 20:
        print(f'  slow: {f.stem} took {time.time() - t0:.0f}s')
    CACHE.mkdir(exist_ok=True)
    c.write_text(json.dumps({'hash': h, 'p': p}, ensure_ascii=False), encoding='utf-8')
    return p


def main():
    lessons = lesson_index()
    probs = [build_cached(f, lessons) for f in sorted((SRC / 'problems').glob('*.py'))]
    ids = {p['id'] for p in probs}
    for p in probs:
        bad = [r for r in p['related'] if r not in ids]
        assert not bad, f"{p['id']}: unknown related {bad}"

    # Course lessons can list practice problems ("p"); show those lessons on the problem page too ("Learn the idea").
    byid = {p['id']: p for p in probs}
    curated = {}                                            # "trackkey:lessonid" -> slugs the lesson picked, best first
    for ckey, ctitle, fm, href in site_lessons():
        picks = [x for x in fm.get('practice', []) if x in byid]
        curated[f"{ckey}:{fm['id']}"] = picks
        for slug in picks:
            if all(x['href'] != href for x in byid[slug]['lessons']):
                byid[slug]['lessons'].append({'href': href, 't': fm['title'], 'c': ctitle})

    # Problem -> theory (Unroll): META['theory'] when the author wrote one, else every lesson the problem is linked to.
    picked_by = {}
    for ref, slugs in curated.items():
        for slug in slugs:
            picked_by.setdefault(slug, []).append(ref)
    for p in probs:
        p['theory'] = theory_links(p, picked_by.get(p['id'], []), lessons)

    data_dir, nb_dir = WEB / 'data', WEB / 'nb'
    data_dir.mkdir(parents=True, exist_ok=True); nb_dir.mkdir(parents=True, exist_ok=True)
    for f in [*data_dir.glob('*.json'), *nb_dir.glob('*.ipynb')]:
        f.unlink()                                          # drop files of renamed/removed problems
    for p in probs:
        if 'colab' in p:
            (nb_dir / f"{p['id']}.ipynb").write_text(json.dumps(notebook(p), indent=1), encoding='utf-8')
        # The page gets tests only as JSON text (Python must see 3.0 as a float), and Colab problems get none:
        # their tests live in the notebook. This keeps every problem file small.
        page = {k: v for k, v in p.items() if k not in ('tests', 'lesson_refs')}
        page['ntests'] = len(p['tests'])
        if 'colab' not in p:
            page['tj'] = json.dumps(p['tests']); page['tj_ex'] = json.dumps(p['tests'][:p['nex']])
        (data_dir / f"{p['id']}.json").write_text(json.dumps(page, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

    # index.json follows the course contract (topics -> lessons) for the site's Practice pages.
    # A problem lives in the module of its first tag; its other tags show as filters.
    by_tag = {t: [] for t in TAGS}
    for p in sorted(probs, key=lambda p: (list(DIFFS).index(p['d']), p['t'])):
        by_tag[p['tags'][0]].append({'id': p['id'], 't': p['t'], 'mins': DIFFS[p['d']], 'd': p['d'],
                                     'rt': p['rt'], 'tags': p['tags']})
    data = [{'n': f'{i+1:02d}', 'name': t, 'slug': slugify(t), 'part': next(pn for pn, ts in PARTS if t in ts),
             'lessons': by_tag[t]} for i, t in enumerate(TAGS)]
    data = [m for m in data if m['lessons']]               # topics appear as soon as they have a problem
    plans = [load_plan(f, ids) for f in sorted((SRC / 'plans').glob('*.json'))]
    builds, quizzes, rounds = extras(ids, lessons)
    rt = routes(lessons, ids, {r['id']: {'title': r['title'], 'minutes': r['minutes'], 'role': r['role']} for r in rounds},
                {pl['id']: pl for pl in plans}, probs)
    (WEB / 'routes.json').write_text(json.dumps(rt, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    qdir = WEB / 'quiz'; qdir.mkdir(exist_ok=True)
    for f in qdir.glob('*.json'):
        f.unlink()
    for pid, q in quizzes.items():                         # fetched by the plan page when opened
        (qdir / f'{pid}.json').write_text(json.dumps(q, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    for pl in plans:
        pl['quiz'] = [s['section'] for s in quizzes.get(pl['id'], {}).get('sections', [])]
    # index.json: everything the Practice pages list (problem bodies stay in data/<slug>.json)
    (WEB / 'index.json').write_text(json.dumps({'topics': data, 'parts': [pn for pn, _ in PARTS], 'plans': plans, 'builds': builds,
                                                'rounds': rounds}, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    pt, pg = placement()
    for f in ('runner.js', 'harness.py', 'shapetrace.py', 'placement.py'):   # the in-browser runner and the graders it shares with this build
        (WEB / f).write_text((OUT / f).read_text(encoding='utf-8'), encoding='utf-8')

    # links.json: "trackkey:lessonid" -> [[slug, title, difficulty]], for "Practice this" on lesson pages
    refs = {k: [[x, byid[x]['t'], byid[x]['d']] for x in v] for k, v in curated.items() if v}
    for p in sorted(probs, key=lambda p: (list(DIFFS).index(p['d']), p['t'])):
        for ref in p['lesson_refs']:
            if all(x[0] != p['id'] for x in refs.get(ref, [])):
                refs.setdefault(ref, []).append([p['id'], p['t'], p['d']])
    (WEB / 'links.json').write_text(json.dumps(refs, ensure_ascii=False, indent=0), encoding='utf-8')

    by_rt = {}
    for p in probs:
        by_rt[p['rt']] = by_rt.get(p['rt'], 0) + 1
    print(f'{len(probs)} problems verified · {len(data)}/{len(TAGS)} topics · {len(plans)} plans · '
          f'{sum(b["kind"] == "paper" for b in builds)} papers · {sum(b["kind"] == "project" for b in builds)} projects · '
          f'{len(quizzes)} quizzes · {len(rounds)} rounds · runtimes {by_rt}')
    print(f"routes: {len(rt['concepts'])} concepts · {len(rt['destinations'])} destinations · {len(rt['sets'])} sets "
          f"({sum(1 for x in rt['sets'] if x['boss'])} with a boss) · "
          f"{sum(1 for p in probs if p['theory'])}/{len(probs)} problems link theory")
    print(f"placement: {len(pt)} tasks · {sum(len(g['questions']) for g in pg)} module questions")


def load_plan(path, ids):
    """plans/<id>.json: a study plan (ordered sections of problem slugs). Every slug must exist."""
    plan = json.loads(path.read_text(encoding='utf-8'))
    where = f'plans/{path.name}:'
    assert plan.get('id') == path.stem, f'{where} id must equal the file name'
    for k in ('group', 'title', 'blurb', 'sections'):
        assert plan.get(k), f'{where} missing {k}'
    for sec in plan['sections']:
        assert sec.get('name') and sec.get('problems'), f'{where} each section needs a name and problems'
        bad = [x for x in sec['problems'] if x not in ids]
        assert not bad, f'{where} unknown problems {bad}'
    return plan


ROLES = ['Data Scientist', 'ML Engineer', 'Research Engineer', 'GPU Engineer', 'AI Engineer']


def load_build(path, ids, lessons):
    """builds/<id>.json: a paper implementation or a project, as ordered steps that are problems, plus a capstone."""
    b = json.loads(path.read_text(encoding='utf-8'))
    where = f'builds/{path.name}:'
    assert b.get('id') == path.stem, f'{where} id must equal the file name'
    assert b.get('kind') in ('paper', 'project'), f'{where} kind must be "paper" or "project"'
    for k in ('group', 'title', 'blurb', 'steps', 'capstone'):
        assert b.get(k), f'{where} missing {k}'
    if b['kind'] == 'paper':
        for k in ('authors', 'year', 'url'):
            assert b.get(k), f'{where} papers need {k}'
        assert str(b['url']).startswith('https://'), f'{where} url must be https'
    assert len(b['steps']) >= 3, f'{where} give at least 3 steps'
    for st in b['steps']:
        assert st.get('name') and st.get('problem'), f'{where} each step needs name and problem'
    slugs = [st['problem'] for st in b['steps']] + [b['capstone']]
    bad = [x for x in slugs if x not in ids]
    assert not bad, f'{where} unknown problems {bad}'
    assert b['capstone'] not in slugs[:-1], f'{where} capstone must be its own problem, not a step'
    badl = [x for x in b.get('lessons', []) if x not in lessons]
    assert not badl, f'{where} unknown lessons {badl}'
    return b


def load_quiz(path, plans):
    """quizzes/<plan id>.json: multiple-choice questions for each section of a study plan."""
    q = json.loads(path.read_text(encoding='utf-8'))
    where = f'quizzes/{path.name}:'
    plan = plans.get(path.stem)
    assert plan, f'{where} no study plan named {path.stem}'
    names = [s['name'] for s in plan['sections']]
    for sec in q['sections']:
        assert sec.get('section') in names, f'{where} section {sec.get("section")!r} is not in the plan ({names})'
        assert len(sec.get('questions', [])) >= 5, f'{where} {sec["section"]}: give at least 5 questions'
        for i, x in enumerate(sec['questions'], 1):
            assert x.get('q') and x.get('why'), f'{where} {sec["section"]} q{i}: needs q and why'
            o = x.get('options', [])
            assert 3 <= len(o) <= 5 and len(set(o)) == len(o), f'{where} {sec["section"]} q{i}: 3-5 distinct options'
            assert isinstance(x.get('answer'), int) and 0 <= x['answer'] < len(o), f'{where} {sec["section"]} q{i}: answer is the 0-based index of the right option'
    return q


def load_round(path, ids, lessons):
    """rounds/<id>.json: a timed interview round mixing coding problems and theory questions from course lessons."""
    r = json.loads(path.read_text(encoding='utf-8'))
    where = f'rounds/{path.name}:'
    assert r.get('id') == path.stem, f'{where} id must equal the file name'
    assert r.get('role') in ROLES, f'{where} role must be one of {ROLES}'
    for k in ('title', 'blurb', 'minutes'):
        assert r.get(k), f'{where} missing {k}'
    assert isinstance(r['minutes'], int) and 10 <= r['minutes'] <= 240, f'{where} minutes must be 10-240'
    assert r.get('coding') or r.get('theory'), f'{where} needs coding problems, theory questions or both'
    bad = [x for x in r.get('coding', []) if x not in ids]
    assert not bad, f'{where} unknown problems {bad}'
    badl = [x for x in r.get('theory', []) if x not in lessons]
    assert not badl, f'{where} unknown lessons {badl}'
    return r


def extras(ids, lessons):
    """All builds, quizzes and rounds, validated. Quizzes are keyed by plan id."""
    plans = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in (SRC / 'plans').glob('*.json')}
    builds = [load_build(f, ids, lessons) for f in sorted((SRC / 'builds').glob('*.json'))]
    quizzes = {f.stem: load_quiz(f, plans) for f in sorted((SRC / 'quizzes').glob('*.json'))}
    rounds = [load_round(f, ids, lessons) for f in sorted((SRC / 'rounds').glob('*.json'))]
    for x in builds:
        x['lesson_links'] = [lessons[r] for r in x.get('lessons', [])]
    for x in rounds:
        x['theory_links'] = {r: lessons[r] for r in x.get('theory', [])}
    return builds, quizzes, rounds


def theory_links(p, picked, lessons):
    """[{lesson, step, label, href}]: the lessons that help solve problem p. step is a lesson step (1-based) or None."""
    where = f"problems/{p['id']}.py:"
    if p.get('theory'):
        out = []
        for t in p['theory']:
            assert isinstance(t, dict) and t.get('lesson') in lessons, f"{where} theory link to unknown lesson {t}"
            assert t.get('step') is None or (isinstance(t['step'], int) and t['step'] >= 1), f"{where} theory step must be null or >= 1"
            out.append({'lesson': t['lesson'], 'step': t.get('step'), 'label': t.get('label') or lessons[t['lesson']]['t'],
                        'href': lessons[t['lesson']]['href']})
        return out
    refs = list(dict.fromkeys([*p['lesson_refs'], *picked]))
    return [{'lesson': r, 'step': None, 'label': lessons[r]['t'], 'href': lessons[r]['href']} for r in refs if r in lessons]


def lesson_minutes():
    """'trackkey:lessonid' -> reading minutes, the same rule as the course pages
    (site/src/pages/courses/[course]/index.astro): frontmatter mins, else words / 180, at least 2."""
    out = {}
    for cj in (SITE / 'src/content/courses').glob('*/course.json'):
        key = json.loads(cj.read_text(encoding='utf-8'))['key']
        for f in (cj.parent / 'lessons').glob('*.md'):
            head, body = f.read_text(encoding='utf-8')[4:].split('\n---\n', 1)
            fm = {k: json.loads(v) for k, v in re.findall(r'^(\w+): (.*)$', head, re.M)}
            words = len((body + ' ' + re.sub(r'<[^>]+>', ' ', fm.get('html') or '')).split())
            out[f"{key}:{fm['id']}"] = fm.get('mins') or max(2, -(-words // 180))
    return out


STAGES = ['foundations', 'core', 'systems', 'serving']
CONTENT = SITE / 'src' / 'content'


def routes(lessons, ids, round_ids, plans, probs=None):
    """Unroll route data, validated: concepts (site/src/content/concepts.json), destinations (destinations.json) and
    Arena sets (one per study-plan section, optional "boss" round). Concepts get derived minutes, unlocks and lessons."""
    cfgs = {cj.parent.name: json.loads(cj.read_text(encoding='utf-8')) for cj in sorted((CONTENT / 'courses').glob('*/course.json'))}
    courses = {k: c['key'] for k, c in cfgs.items()}
    mins = lesson_minutes()
    slugs = {f"{k}:{fm['id']}": fm['slug'] for k, _, fm, _ in site_lessons()}
    meta = {p['id']: {'id': p['id'], 't': p['t'], 'd': p['d'], 'rt': p['rt']} for p in probs or []}
    concepts = json.loads((CONTENT / 'concepts.json').read_text(encoding='utf-8'))['concepts']
    by, owner = {}, {}
    for c in concepts:
        where = f"concepts.json: {c.get('id')!r}:"
        assert re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', c.get('id') or ''), f'{where} id must be kebab-case'
        assert c['id'] not in by, f'{where} duplicate id'
        for k in ('title', 'summary'):
            assert isinstance(c.get(k), str) and c[k].strip(), f'{where} missing {k}'
        assert c.get('course') in courses, f'{where} course must be a folder in site/src/content/courses/'
        assert c.get('stage') in STAGES, f'{where} stage must be one of {STAGES}'
        assert c.get('lessons'), f'{where} needs at least one lesson'
        for ref in c['lessons']:
            assert ref in lessons, f'{where} unknown lesson {ref}'
            assert ref.split(':')[0] == courses[c['course']], f"{where} lesson {ref} is not in {c['course']}"
            assert ref not in owner, f"{where} lesson {ref} already belongs to {owner[ref]!r}"
            owner[ref] = c['id']
        bad = [x for x in c.get('problems', []) if x not in ids]
        assert not bad, f'{where} unknown problems {bad}'
        by[c['id']] = c
    for c in concepts:
        bad = [x for x in c.get('needs', []) if x not in by or x == c['id']]
        assert not bad, f"concepts.json: {c['id']!r}: unknown needs {bad}"
    state = {}
    def visit(cid, path=()):                                # needs must form a DAG
        assert state.get(cid) != 1, f"concepts.json: needs cycle {' -> '.join([*path, cid])}"
        if state.get(cid) != 2:
            state[cid] = 1
            for n in by[cid].get('needs', []):
                visit(n, (*path, cid))
            state[cid] = 2
    for cid in by:
        visit(cid)
    out_c = [{**c, 'needs': c.get('needs', []), 'problems': c.get('problems', []),
              'unlocks': [d['id'] for d in concepts if c['id'] in d.get('needs', [])],
              'minutes': sum(mins[r] for r in c['lessons']),
              'lesson_links': [{'lesson': r, 'slug': slugs[r], 'href': lessons[r]['href'], 't': lessons[r]['t'], 'mins': mins[r]} for r in c['lessons']],
              'problem_links': [meta.get(x, {'id': x}) for x in c.get('problems', [])]}
             for c in concepts]

    dests = json.loads((CONTENT / 'destinations.json').read_text(encoding='utf-8'))['destinations']
    seen = set()
    for d in dests:
        where = f"destinations.json: {d.get('id')!r}:"
        assert re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', d.get('id') or '') and d['id'] not in seen, f'{where} id must be unique kebab-case'
        seen.add(d['id'])
        for k in ('name', 'description'):
            assert isinstance(d.get(k), str) and d[k].strip(), f'{where} missing {k}'
        route = d.get('concepts', [])
        assert len(set(route)) == len(route), f'{where} a concept appears twice'
        bad = [x for x in route if x not in by]
        assert not bad, f'{where} unknown concepts {bad}'
        assert set(d.get('optional', [])) <= set(route), f'{where} optional concepts must be on the route'
        if not route:                                       # Free roam: no route, no final round
            assert not d.get('finalRound'), f'{where} a destination without concepts has no final round'
            continue
        assert d.get('finalRound') in round_ids, f"{where} finalRound must be a round in practice/src/rounds/"
        for i, cid in enumerate(route):
            missing = [n for n in by[cid].get('needs', []) if n not in route[:i]]
            assert not missing, f'{where} {cid!r} needs {missing} earlier on the route'
            if i:
                assert STAGES.index(by[cid]['stage']) >= STAGES.index(by[route[i - 1]]['stage']), \
                    f"{where} {cid!r} ({by[cid]['stage']}) comes after a later-stage concept"

    sets = []
    for pid, pl in sorted(plans.items()):
        for sec in pl['sections']:
            boss = sec.get('boss')
            assert boss is None or boss in round_ids, f"plans/{pid}.json: {sec['name']!r}: unknown boss round {boss}"
            sets.append({'id': f"{pid}/{slugify(sec['name'])}", 'plan': pid, 'title': sec['name'],
                         'problems': sec['problems'], 'boss': boss})
    info = {k: {'key': c['key'], 'title': c['title'], 'layer': c.get('layer')} for k, c in cfgs.items()}
    rounds = {k: v for k, v in round_ids.items()} if isinstance(round_ids, dict) else {}   # titles for final rounds and bosses
    return {'stages': STAGES, 'courses': info, 'rounds': rounds, 'concepts': out_c, 'destinations': dests, 'sets': sets}


PLACEMENT_MODULES = [f'{i:02d}' for i in range(1, 17)]   # ML Maths modules 01-16


def placement():
    """ML Maths placement (placement/mlmaths/): T*.py coding tasks and mcq.json question groups, graded in the browser by
    practice/placement.py. Ships only if every SOLUTION passes its CHECK, every STARTER fails it, and every module is tested."""
    import tempfile
    import placement as P
    d, tasks = SRC / 'placement' / 'mlmaths', []
    def mcq_ok(q, where):
        assert isinstance(q.get('q'), str) and q['q'].strip(), f'{where}: missing q'
        assert isinstance(q.get('options'), list) and 3 <= len(q['options']) <= 5, f'{where}: 3-5 options'
        assert isinstance(q.get('answer'), int) and 0 <= q['answer'] < len(q['options']), f'{where}: answer must index options'
        assert isinstance(q.get('why'), str) and q['why'].strip(), f'{where}: missing why'
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(d.glob('T*.py')):
            m, where = load(f), f'placement/mlmaths/{f.name}'
            for k in ('TITLE', 'MODULES', 'STATEMENT', 'STARTER', 'CHECK', 'SOLUTION', 'MCQ'):
                assert hasattr(m, k), f'{where}: missing {k}'
            assert m.MODULES and set(m.MODULES) <= set(PLACEMENT_MODULES), f'{where}: MODULES must be ML Maths module numbers'
            r = P.grade(m.SOLUTION, m.CHECK, f'{tmp}/user.py')
            assert r['ok'], f"{where}: SOLUTION fails its check: {r['msg']}"
            assert not P.grade(m.STARTER, m.CHECK, f'{tmp}/user.py')['ok'], f'{where}: STARTER passes its check'
            assert len(m.MCQ) == 2, f'{where}: two follow-up questions'
            for i, q in enumerate(m.MCQ):
                mcq_ok(q, f'{where} MCQ {i + 1}')
            tasks.append({'id': f.stem, 'title': m.TITLE, 'modules': m.MODULES, 'statement': m.STATEMENT.strip(),
                          'starter': m.STARTER, 'check': m.CHECK, 'mcq': m.MCQ})
    groups = json.loads((d / 'mcq.json').read_text(encoding='utf-8'))['groups']
    for g in groups:
        where = f"placement/mlmaths/mcq.json {g.get('module')}"
        assert g.get('module') in PLACEMENT_MODULES and len(g.get('questions', [])) == 3, f'{where}: module + three questions'
        for i, q in enumerate(g['questions']):
            mcq_ok(q, f'{where} Q{i + 1}')
    missing = set(PLACEMENT_MODULES) - {n for t in tasks for n in t['modules']} - {g['module'] for g in groups}
    assert not missing, f'placement: modules with no task or questions: {sorted(missing)}'
    (WEB / 'placement-mlmaths.json').write_text(json.dumps({'tasks': tasks, 'groups': groups}, ensure_ascii=False,
                                                          separators=(',', ':')), encoding='utf-8')
    return tasks, groups


def check(paths):
    """Verify only the given problem/plan files and write nothing, so parallel authors can't clash.
    Related slugs and plan entries may point at any problem file that exists on disk."""
    lessons, ids = lesson_index(), {f.stem for f in (SRC / 'problems').glob('*.py')}
    bad = 0
    for raw in paths:
        f = Path(raw).resolve()
        try:
            if f.name in ('concepts.json', 'destinations.json'):
                plans = {g.stem: load_plan(g, ids) for g in (SRC / 'plans').glob('*.json')}
                rids = {g.stem for g in (SRC / 'rounds').glob('*.json')}
                rt = routes(lessons, ids, rids, plans)
                print(f"OK    {f.name}  ({len(rt['concepts'])} concepts, {len(rt['destinations'])} destinations)")
                continue
            if f.suffix == '.json':
                kind = f.parent.name
                if kind == 'builds':
                    b = load_build(f, ids, lessons); print(f'OK    {f.name}  ({b["kind"]}, {len(b["steps"])} steps + capstone)')
                elif kind == 'quizzes':
                    plans = {g.stem: json.loads(g.read_text(encoding='utf-8')) for g in (SRC / 'plans').glob('*.json')}
                    q = load_quiz(f, plans); print(f'OK    {f.name}  ({sum(len(x["questions"]) for x in q["sections"])} questions)')
                elif kind == 'rounds':
                    r = load_round(f, ids, lessons); print(f'OK    {f.name}  ({len(r.get("coding", []))} coding, {len(r.get("theory", []))} theory)')
                else:
                    plan = load_plan(f, ids)
                    print(f'OK    {f.name}  ({sum(len(x["problems"]) for x in plan["sections"])} problems)')
                continue
            p = build_one(f, lessons)
            unknown = [r for r in p['related'] if r not in ids]
            assert not unknown, f'{f.name}: unknown related {unknown}'
            print(f'OK    {f.name}')
        except Exception as e:  # report every file, not just the first failure
            bad += 1
            print(f'FAIL  {f.name}: {type(e).__name__}: {e}')
    print(f'{len(paths) - bad}/{len(paths)} passed')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--placement']:              # verify the placement test only (writes placement-mlmaths.json)
        tasks, groups = placement(); print(f'placement OK: {len(tasks)} tasks, {len(groups)} question groups')
    elif sys.argv[1:2] == ['--check']:
        check(sys.argv[2:])
    else:
        main()
