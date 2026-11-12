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
    g.set('_code', m.code); g.set('_fn', m.fn); g.set('_tests', JSON.stringify(m.tests)); g.set('_rt', m.runtime);
    g.set('_tol', py.toPy(m.tol || [1e-6, 1e-8]));
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
json.dumps(harness.grade(_target, json.loads(_tests), _rt, *_tol))`, {globals: g});
    g.destroy();
    postMessage({type: 'done', id: m.id, results: JSON.parse(res), stdout: out.join('\n')});
  } catch (e) {
    // Python tracebacks carry Pyodide's own frames first; keep the learner's part.
    const msg = String(e.message || e), i = msg.indexOf('File "solution.py"');
    postMessage({type: 'done', id: m.id, error: i >= 0 ? msg.slice(i) : msg.split('\n').slice(-4).join('\n'), stdout: out.join('\n')});
  }
};
