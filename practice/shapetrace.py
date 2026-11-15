"""Shape trace: what each line of the learner's code did to their arrays, for one failing test (the problem page).

The browser runner (runner.js) calls run() after grading, on the first failing test only. It follows frames whose code
comes from solution.py (the learner's function and their helpers) with sys.settrace, and after each line records every
local whose shape or dtype changed: ndarrays, torch tensors, jax arrays, pandas DataFrames/Series. Kept out of
harness.py on purpose: build.py hashes harness.py into its verify cache, so editing it would re-verify every problem.

    python3 practice/shapetrace.py      (self-check)
"""
import sys

LIMIT = 200          # rows; a loop over a big array would otherwise record thousands


def desc(v):
    """'(2, 8, 64) float64' for array-likes, else None."""
    shape = getattr(v, 'shape', None)
    if shape is None or isinstance(v, type):
        return None
    try:
        dims = tuple(int(s) for s in shape)
    except (TypeError, ValueError):
        return None
    dt = getattr(v, 'dtype', None)
    if dt is None and hasattr(v, 'dtypes'):                 # DataFrame: one dtype per column
        dt = 'mixed' if len(set(map(str, v.dtypes))) > 1 else (str(v.dtypes.iloc[0]) if len(v.dtypes) else '')
    s = '(' + ', '.join(map(str, dims)) + (',' if len(dims) == 1 else '') + ')'
    return f"{s} {str(dt).replace('torch.', '')}".strip()


def run(fn, args, filename='solution.py'):
    """Call fn(*args) under the tracer. {'rows': [{line, name, shape}], 'error': {line, msg} | None,
    'ret': shape of the result | None, 'cut': True if LIMIT was hit}."""
    rows, state, err = [], {}, None                        # state[frame id] = [previous line, {name: shape}]

    def record(frame):
        prev, seen = state.setdefault(id(frame), [frame.f_lineno, {}])
        for name, v in list(frame.f_locals.items()):
            d = desc(v)
            if d and not d.startswith('()') and seen.get(name) != d:   # 0-d values (loop scalars) are noise
                seen[name] = d
                if len(rows) < LIMIT:
                    rows.append({'line': prev, 'name': name, 'shape': d})

    def local(frame, event, arg):
        nonlocal err
        if event == 'line':
            record(frame); state[id(frame)][0] = frame.f_lineno
        elif event == 'return':
            record(frame); state.pop(id(frame), None)
        elif event == 'exception' and err is None:
            record(frame); err = {'line': frame.f_lineno, 'msg': f'{arg[0].__name__}: {arg[1]}'}
        return local

    def call(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        state[id(frame)] = [frame.f_lineno, {n: desc(v) for n, v in frame.f_locals.items()}]   # arguments: known already
        return local

    ret, old = None, sys.gettrace()
    sys.settrace(call)
    try:
        ret = fn(*args)
    except Exception:                                      # recorded by the tracer; grading already reported it
        pass
    finally:
        sys.settrace(old)
    return {'rows': rows, 'error': err, 'ret': desc(ret), 'cut': len(rows) >= LIMIT}


if __name__ == '__main__':
    import numpy as np
    src = (
        "import numpy as np\n"
        "def attn(q, k):\n"
        "    s = q @ k.T\n"                                  # line 3: (4, 4)
        "    s = s / 2\n"                                    # line 4: shape unchanged, not recorded again
        "    w = s.astype(np.float32)\n"                     # line 5: dtype change
        "    return w.sum(axis=1)\n"
        "def bad(q):\n"
        "    t = q.reshape(2, 8)\n"                          # line 8: (2, 8)
        "    return t @ q\n")                                # line 9: matmul shape error
    ns = {}
    exec(compile(src, 'solution.py', 'exec'), ns)
    q = np.zeros((4, 8))
    r = run(ns['attn'], [q, q])
    assert [(x['line'], x['name'], x['shape']) for x in r['rows']] == [(3, 's', '(4, 4) float64'), (5, 'w', '(4, 4) float32')], r
    assert r['ret'] == '(4,) float32' and r['error'] is None, r
    r = run(ns['bad'], [np.zeros(16)])
    assert r['rows'] == [{'line': 8, 'name': 't', 'shape': '(2, 8) float64'}], r
    assert r['error']['line'] == 9 and r['error']['msg'].startswith('ValueError'), r
    assert desc(3) is None and desc([1, 2]) is None and desc(np.float64(2.0)) == '() float64'
    print('shapetrace self-check passed')
