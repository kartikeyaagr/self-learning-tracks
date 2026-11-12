"""Test harness shared by build.py (CPython) and the browser runner (Pyodide).

One file, so a test that passes at build time is graded the same way in the browser.
Values cross JSON with small tags: ndarray, tuple, DataFrame/Series, non-finite floats.
"""
import json, math

try:
    import numpy as np
except ImportError:  # pure-python problems never need it
    np = None


def enc(x):
    if np is not None and isinstance(x, np.ndarray):
        return {'__nd__': enc(x.tolist()), 'dtype': str(x.dtype)}
    if np is not None and isinstance(x, np.generic):
        return enc(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return {'__f__': repr(x)}
    if isinstance(x, tuple):
        return {'__tup__': [enc(v) for v in x]}
    if isinstance(x, list):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {'__dict__': [[enc(k), enc(v)] for k, v in x.items()]}
    t = type(x).__name__
    if t == 'Tensor':  # torch, in Colab and at build time
        return enc(x.detach().cpu().numpy())
    if t == 'DataFrame':
        return {'__df__': json.loads(x.to_json(orient='split', index=False, double_precision=15)),
                'dtypes': [str(d) for d in x.dtypes]}
    if t == 'Series':
        return {'__sr__': enc(x.tolist()), 'name': x.name}
    return x


def dec(x):
    if isinstance(x, list):
        return [dec(v) for v in x]
    if not isinstance(x, dict):
        return x
    if '__nd__' in x:
        return np.array(dec(x['__nd__']), dtype=x['dtype'])
    if '__f__' in x:
        return float(x['__f__'])
    if '__tup__' in x:
        return tuple(dec(v) for v in x['__tup__'])
    if '__dict__' in x:
        return {_hashable(dec(k)): dec(v) for k, v in x['__dict__']}
    if '__df__' in x:
        import pandas as pd
        s = x['__df__']
        df = pd.DataFrame(s['data'], columns=s['columns'])
        for c, d in zip(df.columns, x['dtypes']):
            if d.startswith(('int', 'float', 'bool')):
                df[c] = df[c].astype(d)
        return df
    if '__sr__' in x:
        import pandas as pd
        return pd.Series(dec(x['__sr__']), name=x['name'])
    return x


def _hashable(k):
    return tuple(k) if isinstance(k, list) else k


def same(got, exp, rtol=1e-6, atol=1e-8, path='output'):
    """(ok, why). Floats compare with tolerance, everything else exactly. Shapes must match."""
    if type(exp).__name__ == 'DataFrame':
        if type(got).__name__ != 'DataFrame':
            return False, f'{path}: expected a DataFrame, got {type(got).__name__}'
        if list(got.columns) != list(exp.columns):
            return False, f'{path}: columns {list(got.columns)} != expected {list(exp.columns)}'
        if got.shape != exp.shape:
            return False, f'{path}: shape {got.shape} != expected {exp.shape}'
        for c in exp.columns:
            ok, why = same(got[c].tolist(), exp[c].tolist(), rtol, atol, f'{path}[{c!r}]')
            if not ok:
                return ok, why
        return True, ''
    if type(exp).__name__ == 'Series':
        exp = exp.tolist()
        got = got.tolist() if type(got).__name__ == 'Series' else got
    if np is not None and isinstance(exp, np.ndarray):
        try:
            g = np.asarray(got)
        except Exception:
            return False, f'{path}: expected an array, got {type(got).__name__}'
        if g.shape != exp.shape:
            return False, f'{path}: shape {g.shape} != expected {exp.shape}'
        if exp.dtype.kind in 'fc' or g.dtype.kind in 'fc':
            if g.dtype.kind not in 'fciub':
                return False, f'{path}: expected numbers, got dtype {g.dtype}'
            bad = ~np.isclose(g.astype(complex if exp.dtype.kind == 'c' else float), exp, rtol=rtol, atol=atol, equal_nan=True)
            if bad.any():
                i = tuple(int(v) for v in np.argwhere(bad)[0])
                return False, f'{path}{list(i)}: {g[i].tolist()!r} != expected {exp[i].tolist()!r}'
            return True, ''
        if not np.array_equal(g, exp):
            i = tuple(int(v) for v in np.argwhere(g != exp)[0]) if g.shape else ()
            return False, f'{path}{list(i)}: {g[i].tolist()!r} != expected {exp[i].tolist()!r}'
        return True, ''
    if isinstance(exp, float) or (isinstance(exp, int) and isinstance(got, float) and not isinstance(exp, bool)):
        if np is not None and isinstance(got, np.generic):
            got = got.item()
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return False, f'{path}: expected a number, got {type(got).__name__}'
        if math.isnan(exp) and math.isnan(got):
            return True, ''
        return (True, '') if math.isclose(got, exp, rel_tol=rtol, abs_tol=atol) else (False, f'{path}: {got!r} != expected {exp!r}')
    if isinstance(exp, (list, tuple)):
        if np is not None and isinstance(got, np.ndarray):
            got = got.tolist()
        if not isinstance(got, (list, tuple)):
            return False, f'{path}: expected a {type(exp).__name__}, got {type(got).__name__}'
        if len(got) != len(exp):
            return False, f'{path}: length {len(got)} != expected {len(exp)}'
        for i, (g, e) in enumerate(zip(got, exp)):
            ok, why = same(g, e, rtol, atol, f'{path}[{i}]')
            if not ok:
                return ok, why
        return True, ''
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            return False, f'{path}: expected a dict, got {type(got).__name__}'
        if set(got) != set(exp):
            return False, f'{path}: keys {sorted(map(str, got))} != expected {sorted(map(str, exp))}'
        for k in exp:
            ok, why = same(got[k], exp[k], rtol, atol, f'{path}[{k!r}]')
            if not ok:
                return ok, why
        return True, ''
    if np is not None and isinstance(got, np.generic):
        got = got.item()
    return (True, '') if got == exp else (False, f'{path}: {got!r} != expected {exp!r}')


def run_sql(query, setup):
    """Run `setup` (CREATE/INSERT script) in a fresh in-memory db, then `query`; rows as lists."""
    import sqlite3  # a separate package in Pyodide: only SQL problems load it
    con = sqlite3.connect(':memory:')
    try:
        con.executescript(setup)
        cur = con.execute(query)
        return [[c[0] for c in cur.description], [list(r) for r in cur.fetchall()]]
    finally:
        con.close()


def show_args(args, runtime, n=600):
    """Arguments as the learner would write them: one per line, SQL setup as-is."""
    if runtime == 'sql':
        return args[0] if len(args[0]) <= n else args[0][:n] + ' …'
    return '\n'.join(short(a, n) for a in args)


def short(x, n=300):
    s = repr(x)
    return s if len(s) <= n else s[:n] + ' …'


def grade(fn, tests, runtime, rtol=1e-6, atol=1e-8):
    """Run `fn` (a callable, or SQL text for runtime 'sql') on each test; list of result dicts."""
    out = []
    for t in tests:
        args, exp = prep(dec(t['a']), runtime), dec(t['e'])
        try:
            got = run_sql(fn, args[0]) if runtime == 'sql' else fn(*_copy(args))
            ok, why = same(got, exp, rtol, atol)
        except Exception as e:  # learner code raised: report it like a failed test
            got, ok, why = None, False, f'{type(e).__name__}: {e}'
        out.append({'ok': ok, 'why': why, 'got': short(got) if not ok else '',
                    'inp': show_args(args, runtime, 400) if not ok else ''})
    return out


def prep(args, runtime):
    """Arrays cross JSON as ndarrays; torch problems take tensors."""
    if runtime == 'torch':
        import torch
        return [torch.as_tensor(a) if np is not None and isinstance(a, np.ndarray) else a for a in args]
    return args


def _copy(args):
    # Tests reuse inputs across Run and Submit, so a function that mutates its input mustn't poison later cases.
    import copy
    return copy.deepcopy(args)
