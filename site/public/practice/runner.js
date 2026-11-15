// Web Worker: runs learner code in Pyodide and grades it with harness.py (the same file build.py uses).
// The page kills and respawns this worker on timeout: GitHub Pages can't send COOP/COEP headers,
// so Pyodide's SharedArrayBuffer interrupt isn't available.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/';
importScripts(PYODIDE + 'pyodide.js');

let py = null, out = [];
const loaded = new Set();
const ready = (async () => {
  py = await loadPyodide({indexURL: PYODIDE});
  py.setStdout({batched: s => out.push(s)});
  py.setStderr({batched: s => out.push(s)});
  await py.loadPackage('numpy');
  py.FS.writeFile('harness.py', await (await fetch('harness.py')).text());
  py.FS.writeFile('shapetrace.py', await (await fetch('shapetrace.py')).text());   // the problem page's shape trace
  py.runPython('import sys; sys.path.insert(0, ".")');
  postMessage({type: 'ready'});
})();

onmessage = async ({data: m}) => {
  try {
    await ready;
    const need = (m.pkgs || []).filter(p => !loaded.has(p));
    if (need.length) { postMessage({type: 'status', text: `Loading ${need.join(', ')}…`}); await py.loadPackage(need); need.forEach(p => loaded.add(p)); }
    out = [];
    const g = py.globals.get('dict')();
    g.set('_code', m.code); g.set('_fn', m.fn); g.set('_tests', typeof m.tests === 'string' ? m.tests : JSON.stringify(m.tests)); g.set('_rt', m.runtime);
    g.set('_tol', py.toPy(m.tol || [1e-6, 1e-8])); g.set('_trace', !!m.trace);
    const res = py.runPython(`
import json, harness
_ns = {}
if _rt == 'sql':
    _target = _code
else:
    exec(compile(_code, 'solution.py', 'exec'), _ns)
    if _fn not in _ns:
        raise NameError(f"define a function named {_fn}()")
    _target = _ns[_fn]
_res = harness.grade(_target, json.loads(_tests), _rt, *_tol)
_tr = None
_bad = next((i for i, r in enumerate(_res) if not r['ok']), None)
if _trace and _rt != 'sql' and _bad is not None:   # replay the first failing test, line by line
    import shapetrace
    _t = json.loads(_tests)[_bad]
    _tr = shapetrace.run(_target, harness._copy(harness.prep(harness.dec(_t['a']), _rt)))
    _tr['test'], _tr['want'] = _bad, shapetrace.desc(harness.dec(_t['e']))
json.dumps({'results': _res, 'trace': _tr})`, {globals: g});
    g.destroy();
    const r = JSON.parse(res);
    postMessage({type: 'done', id: m.id, results: r.results, trace: r.trace, stdout: out.join('\n')});
  } catch (e) {
    // Python tracebacks carry Pyodide's own frames first; keep the learner's part.
    const msg = String(e.message || e), i = msg.indexOf('File "solution.py"');
    postMessage({type: 'done', id: m.id, error: i >= 0 ? msg.slice(i) : msg.split('\n').slice(-4).join('\n'), stdout: out.join('\n')});
  }
};
