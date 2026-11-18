"""Placement checks, shared by build.py (CPython) and the browser runner (Pyodide), like harness.py.

A task is the learner's code plus a CHECK that defines _check(): it raises CheckFail (or NotImplementedError) with a
reason, or returns when every check passes. The helpers are the ML Maths diagnostic notebook's, unchanged, so the
browser grades exactly what the notebook did. grade() writes the code to a real file so inspect.getsource works.
"""
import ast, base64, contextlib, importlib, inspect, json, linecache, sys, textwrap
import numpy as np, scipy, scipy.linalg, scipy.special, scipy.stats

if sys.platform == 'emscripten':   # Pyodide's numpy.linalg.cholesky returns NaNs where CPython raises (no FP error flags in wasm)
    def _cholesky(a, *args, _chol=np.linalg.cholesky, **kw):
        L = _chol(a, *args, **kw)
        if np.isnan(L).any() and not np.isnan(np.asarray(a)).any():
            raise np.linalg.LinAlgError('Matrix is not positive definite')
        return L
    np.linalg.cholesky = _cholesky


class CheckFail(Exception):
    pass


def expect(cond, msg):
    if not cond:
        raise CheckFail(msg)


def _tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def loop_depth(fn):
    def depth(node, d=0):
        best = d
        for ch in ast.iter_child_nodes(node):
            nd = d + isinstance(ch, (ast.For, ast.While, ast.comprehension))
            best = max(best, depth(ch, nd))
        return best
    return depth(_tree(fn))


def names_used(fn):
    out = set()
    for n in ast.walk(_tree(fn)):
        if isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.alias):
            out.add(n.name.split(".")[-1])
    return out


def static_rules(fns, banned, max_loops):
    for fn in fns:
        bad = names_used(fn) & set(banned)
        expect(not bad, f"{fn.__name__} uses forbidden name(s): {sorted(bad)}")
        if max_loops is not None:
            ld = loop_depth(fn)
            expect(ld <= max_loops, f"{fn.__name__} has loop nesting depth {ld}; at most {max_loops} allowed (vectorise it)")


@contextlib.contextmanager
def blocked(*paths):
    saved = []
    for p in paths:
        mod, attr = p.rsplit(".", 1)
        m = importlib.import_module(mod)
        if not hasattr(m, attr):
            continue
        saved.append((m, attr, getattr(m, attr)))
        def boom(*a, _p=p, **k):
            raise CheckFail(f"your code called forbidden {_p}")
        setattr(m, attr, boom)
    try:
        yield
    finally:
        for m, a, f in saved:
            setattr(m, a, f)


def grade(code, check, path='user.py'):
    """{'ok': bool, 'msg': str} for the learner's code against one task's CHECK."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)
    linecache.checkcache(path)
    ns = {'__name__': 'user'}
    exec('from placement import *', ns)
    try:
        exec(compile(code, path, 'exec'), ns)
        exec(compile(check, 'check.py', 'exec'), ns)
        ns['_check']()
    except NotImplementedError:
        return {'ok': False, 'msg': 'not implemented yet'}
    except CheckFail as e:
        return {'ok': False, 'msg': str(e)}
    except Exception as e:
        return {'ok': False, 'msg': f'your code raised {type(e).__name__}: {e}'}
    return {'ok': True, 'msg': 'every check passed'}
