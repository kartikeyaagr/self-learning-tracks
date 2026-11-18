import hashlib as _hl
import json as _js

META = dict(title='Pipeline cache keys and invalidation over a DAG', tags=['MLOps', 'Data Processing'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:14.04'], related=['topologically-sort-computation-dag', 'content-hash-data-manifest', 'retry-backoff-schedule'])

STATEMENT = r"""
An ML pipeline (prepare data, tokenize, train, evaluate, export) is a DAG of steps. An orchestrator can skip a step whose output already exists **for exactly the same inputs**. The usual trick is a **cache key**: a hash of the step's own code version and parameters together with the cache keys of the steps it depends on. A change anywhere upstream then changes every downstream key automatically.

Write `plan_run(steps, cache)`.

- `steps`: a dict `name -> {"deps": [names], "code": str, "params": dict}`. The graph is acyclic and every dependency is a key of `steps`.
- `cache`: a list of cache keys whose outputs already exist in the artifact store.

## 1. Order
Return the steps in **topological order**: a step comes after all its dependencies. Among the steps that are ready at the same time, take the alphabetically smallest name first.

## 2. Cache keys
Going through that order, the key of step `s` is the first 12 characters of the SHA-256 hex digest of the UTF-8 bytes of
```python
json.dumps({"name": s, "code": code, "params": params, "deps": dep_keys}, sort_keys=True, separators=(",", ":"))
```
where `dep_keys` is the list of the dependencies' keys, ordered by **dependency name** (so the order in `deps` does not matter).

## 3. Plan
A step must run if its key is not in `cache`. Return
`{"order": [...], "run": [steps that must run, in topological order], "keys": {name: key}}`.
"""

STARTER = '''import hashlib
import json

def plan_run(steps, cache):
    """Return the topological order, the steps to run and every step's cache key."""
    order = sorted(steps)
    keys = {}
    for s in order:
        blob = json.dumps({"name": s, "code": steps[s]["code"], "params": steps[s]["params"]}, sort_keys=True)
        keys[s] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    run = [s for s in order if keys[s] not in cache]
    return {"order": order, "run": run, "keys": keys}
'''

SOLUTION = '''import hashlib
import heapq
import json

def plan_run(steps, cache):
    # Kahn's algorithm; a heap makes ties deterministic (smallest name first)
    indeg = {s: len(set(spec["deps"])) for s, spec in steps.items()}
    children = {s: [] for s in steps}
    for s, spec in steps.items():
        for d in set(spec["deps"]):
            children[d].append(s)
    ready = [s for s, k in indeg.items() if k == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        s = heapq.heappop(ready)
        order.append(s)
        for c in children[s]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, c)

    keys, have = {}, set(cache)
    for s in order:                                   # deps are keyed before s, by construction
        spec = steps[s]
        dep_keys = [keys[d] for d in sorted(set(spec["deps"]))]
        blob = json.dumps({"name": s, "code": spec["code"], "params": spec["params"], "deps": dep_keys},
                          sort_keys=True, separators=(",", ":"))
        keys[s] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    run = [s for s in order if keys[s] not in have]
    return {"order": order, "run": run, "keys": keys}
'''

HINTS = [
    'Compute keys in topological order, so every dependency already has its key when you reach a step.',
    'If a step\'s key ignores its dependencies, changing the tokenizer will not rerun training, and the "cached" model was trained on the old tokens.',
    'For the order, keep the ready steps in a heap (`heapq`) and always pop the smallest name.',
]


def _keys(steps):
    out = {}
    pending = dict(steps)
    while pending:
        s = min(n for n, sp in pending.items() if all(d in out for d in sp['deps']))
        sp = pending.pop(s)
        blob = _js.dumps({"name": s, "code": sp["code"], "params": sp["params"],
                          "deps": [out[d] for d in sorted(set(sp["deps"]))]}, sort_keys=True, separators=(",", ":"))
        out[s] = _hl.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return list(out.values())


def _pipe(tok_code="tok-v1", gsm_shots=8, lr=3e-4, eval_deps_order=("train",)):
    return {
        "data": {"deps": [], "code": "data-v3", "params": {"snapshot": "mix-v12"}},
        "tokenize": {"deps": ["data"], "code": tok_code, "params": {"vocab": 65536}},
        "train": {"deps": ["tokenize"], "code": "train-v7", "params": {"lr": lr, "steps": 20000}},
        "eval_gsm": {"deps": list(eval_deps_order), "code": "eval-v2", "params": {"shots": gsm_shots}},
        "eval_mmlu": {"deps": ["train"], "code": "eval-v2", "params": {"shots": 5}},
        "export": {"deps": ["train"], "code": "export-v1", "params": {"dtype": "bf16"}},
    }


_BASE = _pipe()
_C0 = _keys(_BASE)
_C_TOK = _keys(_pipe(tok_code="tok-v2"))
_DIAMOND = {
    "b": {"deps": ["a"], "code": "1", "params": {}},
    "c": {"deps": ["a"], "code": "1", "params": {"x": 1}},
    "a": {"deps": [], "code": "1", "params": {}},
    "e": {"deps": [], "code": "1", "params": {}},
    "d": {"deps": ["c", "b", "e"], "code": "1", "params": {}},
}

TESTS = [
    (_BASE, []),
    (_pipe(gsm_shots=4), _C0),
    (_pipe(tok_code="tok-v2"), _C0),
    (_BASE, _C0 + _C_TOK),
    (_pipe(eval_deps_order=("train", "tokenize")), _C0),
    (_pipe(lr=1e-4), _C0 + _C_TOK),
    (_DIAMOND, []),
]

EXPLANATION = r"""
## Why the key must include upstream keys
A cache key is a promise: "this output was produced from exactly these inputs." The inputs of `train` include the tokenized data, so the key of `train` must change when anything that produced that data changes. Hashing the upstream **keys** (not their outputs, and not just their names) does this cheaply and transitively: test 3 changes only the tokenizer code and every step from `tokenize` down reruns, while `data` is reused.

## Content addressing makes reverts free
The cache is a set of keys, not a "latest output" per step. Test 4 goes back to the original configuration after the tokenizer experiment, and nothing reruns, because the old outputs are still stored under their old keys. Test 2 changes one eval setting: only `eval_gsm` reruns, which is what makes cheap re-evaluation of a trained model possible.

## Determinism
Both the order and the keys must be identical on every machine. Sorting the dependency keys by name removes dependence on how `deps` was written; `sort_keys=True` and fixed separators fix the JSON bytes; a heap fixes the tie order among ready steps. Adding a dependency (test 5) is a real change and does change the key.

## Complexity
Topological sort is $O(V + E)$ plus $O(V \log V)$ for the heap; hashing is linear in the serialised size.

## Where this breaks
The key covers only what you put in it. A step that reads "the latest" table, the current time or an unpinned package is not a pure function of its key, and caching it silently serves stale or wrong results. Pin those inputs (a snapshot ID, a lockfile hash) and include them in `params`, or disable caching for that step.
"""
