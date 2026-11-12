# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.*", "pandas==2.3.*", "torch>=2.6"]
# ///
# Versions match the browser runtime (Pyodide 0.29.5: Python 3.13, numpy 2.2, pandas 2.3) so expected outputs agree.
"""Build the Practice track: verify every problem, then write the site files.

    uv run practice/src/build.py          (then: uv run --no-project build_hub.py)

Each problems/<slug>.py is a module with META, STATEMENT, STARTER, SOLUTION, HINTS, TESTS, EXPLANATION.
Expected outputs are computed from SOLUTION here, so a problem only ships if:
  - SOLUTION passes every test after a JSON round trip, twice with the same result (deterministic);
  - STARTER fails at least one test (the problem isn't pre-solved);
  - tags, related slugs and lesson links all resolve.
Writes ../index.html (the app, data inlined), ../data/<slug>.json (one per problem), ../nb/<slug>.ipynb (Colab runtimes).
"""
import copy, importlib.util, json, re, sys
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
COLAB = {'torch'}                                          # run in Colab; jax/cuda/triton join when they have problems
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


def lesson_index():
    """'trackkey:lessonid' -> link, from hub-data.js (the hub's own export of every course)."""
    s = (ROOT / 'hub-data.js').read_text(encoding='utf-8')
    out = {}
    for t in json.loads(s[s.index('['):s.rindex(';')]):
        if t['key'] == 'practice':
            continue
        for m in t['modules']:
            for lid, title, _ in m['lessons']:
                out[f"{t['key']}:{lid}"] = {'href': f"../{t['dir']}/index.html#l{lid.replace('.', '-', 1)}",
                                            't': title, 'c': t['name']}
    return out


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
        ns = compile_fn(m.SOLUTION, f'{slug}:SOLUTION')
        assert fn_name in ns, f'{where} SOLUTION must define {fn_name}()'
        target, starter_target = ns[fn_name], compile_fn(m.STARTER, f'{slug}:STARTER').get(fn_name)
        assert all(isinstance(t, tuple) for t in m.TESTS), f'{where} each test is a tuple of args'
        args_list = m.TESTS
        call = lambda f, a: f(*copy.deepcopy(harness.prep(list(a), rt)))

    tol = meta.get('tol', [1e-6, 1e-8])
    tests = [{'a': harness.enc(list(a)), 'e': harness.enc(call(target, a))} for a in args_list]
    tests = json.loads(json.dumps(tests))                  # exactly what the browser will receive
    again = [{'a': t['a'], 'e': harness.enc(call(target, harness.dec(t['a'])))} for t in tests]
    assert json.loads(json.dumps(again)) == tests, f'{where} SOLUTION is not deterministic'
    res = harness.grade(target, tests, rt, *tol)
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
         'lessons': links, 'related': meta.get('related', [])}
    if rt in COLAB:
        p['colab'] = f'https://colab.research.google.com/github/{REPO}/blob/main/practice/nb/{slug}.ipynb'
    return p


def notebook(p):
    cell = lambda kind, src: {'cell_type': kind, 'metadata': {}, 'source': src,
                              **({'outputs': [], 'execution_count': None} if kind == 'code' else {})}
    harness_src = (OUT / 'harness.py').read_text(encoding='utf-8')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
            'cells': [
                cell('markdown', f"# {p['t']}\n\n{p['st']}\n\nWrite your solution in the next cell, then run the test cell."),
                cell('code', p['starter']),
                cell('code', '# Tests (same harness and cases as the website)\n'
                     + f"_HARNESS = {harness_src!r}\n_TESTS = {json.dumps(p['tests'])!r}\n"
                     + "import json, types\nharness = types.ModuleType('harness'); exec(_HARNESS, harness.__dict__)\n"
                     + f"res = harness.grade({p['fn']}, json.loads(_TESTS), {p['rt']!r}, *{p['tol']!r})\n"
                     + "for i, r in enumerate(res): print(f\"test {i+1}: {'pass' if r['ok'] else 'FAIL  ' + r['why']}\")\n"
                     + "print(f\"{sum(r['ok'] for r in res)}/{len(res)} passed\")"),
            ]}


def main():
    lessons = lesson_index()
    probs = [build_one(f, lessons) for f in sorted((SRC / 'problems').glob('*.py'))]
    ids = {p['id'] for p in probs}
    for p in probs:
        bad = [r for r in p['related'] if r not in ids]
        assert not bad, f"{p['id']}: unknown related {bad}"

    data_dir, nb_dir = OUT / 'data', OUT / 'nb'
    data_dir.mkdir(exist_ok=True); nb_dir.mkdir(exist_ok=True)
    for f in [*data_dir.glob('*.json'), *nb_dir.glob('*.ipynb')]:
        f.unlink()                                          # drop files of renamed/removed problems
    for p in probs:
        (data_dir / f"{p['id']}.json").write_text(json.dumps(p, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        if 'colab' in p:
            (nb_dir / f"{p['id']}.ipynb").write_text(json.dumps(notebook(p), indent=1), encoding='utf-8')

    # DATA follows the course contract (modules -> lessons) so build_hub.py reads it like any course.
    # A problem lives in the module of its first tag; its other tags show as filters.
    by_tag = {t: [] for t in TAGS}
    for p in sorted(probs, key=lambda p: (list(DIFFS).index(p['d']), p['t'])):
        by_tag[p['tags'][0]].append({'id': p['id'], 't': p['t'], 'mins': DIFFS[p['d']], 'd': p['d'],
                                     'rt': p['rt'], 'tags': p['tags']})
    data = [{'n': f'{i+1:02d}', 'name': t, 'slug': slugify(t), 'part': next(pn for pn, ts in PARTS if t in ts),
             'lessons': by_tag[t]} for i, t in enumerate(TAGS)]
    data = [m for m in data if m['lessons']]               # topics appear as soon as they have a problem
    plans = [load_plan(f, ids) for f in sorted((SRC / 'plans').glob('*.json'))]
    js = lambda x: json.dumps(x, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    app = (SRC / 'app.html').read_text(encoding='utf-8')
    app = app.replace('__DATA__', js(data)).replace('__PARTS__', js([pn for pn, _ in PARTS])).replace('__PLANS__', js(plans))
    app = app.replace('__TOTAL__', str(len(probs)))
    (OUT / 'index.html').write_text(app, encoding='utf-8')

    by_rt = {}
    for p in probs:
        by_rt[p['rt']] = by_rt.get(p['rt'], 0) + 1
    print(f'{len(probs)} problems verified · {len(data)}/{len(TAGS)} topics · {len(plans)} plans · runtimes {by_rt}')


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


def check(paths):
    """Verify only the given problem/plan files and write nothing, so parallel authors can't clash.
    Related slugs and plan entries may point at any problem file that exists on disk."""
    lessons, ids = lesson_index(), {f.stem for f in (SRC / 'problems').glob('*.py')}
    bad = 0
    for raw in paths:
        f = Path(raw).resolve()
        try:
            if f.suffix == '.json':
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
    if sys.argv[1:2] == ['--check']:
        check(sys.argv[2:])
    else:
        main()
