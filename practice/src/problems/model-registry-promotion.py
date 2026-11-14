META = dict(title='Model registry promotion with gates and rollback', tags=['MLOps'], diff='Medium', runtime='python')

STATEMENT = r"""
Simulate a model registry for one model. Write `run_registry(events, min_gain)`. Stages are `"None"`, `"Staging"`, `"Production"`, `"Archived"`. Process `events` in order:

- `("register", v, metric)`: add version `v` in stage `"None"` with its offline metric (higher is better). Rejected if `v` already exists.
- `("stage", v)`: move `v` from `"None"` to `"Staging"`. Rejected otherwise.
- `("promote", v)`: move `v` from `"Staging"` to `"Production"`. Rejected unless `v` is in Staging and, when a production version exists, `metric(v) >= metric(prod) + min_gain`. On success the old production version becomes `"Archived"`.
- `("rollback",)`: the current production version becomes `"Archived"` and the version it **replaced by promotion** returns to `"Production"`. Repeated rollbacks keep walking back: a version removed by a rollback is never restored by a later rollback. Rejected if there is no production version or nothing left to roll back to.

Return `{"stages": [[v, stage], ...] sorted by v, "production": v or None, "rejected": [event indices]}`.
"""

STARTER = '''def run_registry(events, min_gain):
    """Return {"stages", "production", "rejected"}."""
    stages, prod = {}, None
    for e in events:
        if e[0] == "register":
            stages[e[1]] = "None"
        elif e[0] == "promote":
            prod = e[1]
            stages[e[1]] = "Production"
    return {"stages": sorted([v, s] for v, s in stages.items()), "production": prod, "rejected": []}
'''

SOLUTION = '''def run_registry(events, min_gain):
    stage, metric, prod, history, rejected = {}, {}, None, [], []
    for k, e in enumerate(events):
        op = e[0]
        if op == "register":
            v, m = e[1], e[2]
            if v in stage:
                rejected.append(k); continue
            stage[v], metric[v] = "None", m
        elif op == "stage":
            v = e[1]
            if stage.get(v) != "None":
                rejected.append(k); continue
            stage[v] = "Staging"
        elif op == "promote":
            v = e[1]
            if stage.get(v) != "Staging" or (prod is not None and metric[v] < metric[prod] + min_gain):
                rejected.append(k); continue
            if prod is not None:
                stage[prod] = "Archived"
                history.append(prod)
            stage[v], prod = "Production", v
        elif op == "rollback":
            if prod is None or not history:
                rejected.append(k); continue
            prev = history.pop()
            stage[prod] = "Archived"
            stage[prev], prod = "Production", prev
        else:
            rejected.append(k)
    return {"stages": sorted([v, s] for v, s in stage.items()), "production": prod, "rejected": rejected}
'''

HINTS = [
    'Keep a stack of versions that were replaced by a promotion, most recent last.',
    'A rollback pops that stack; the version being rolled back is archived but not pushed, so rollbacks walk backwards.',
    'Compare against the current production metric only when one exists; the first promotion has no gate.',
]

TESTS = [
    ([("register", 1, 0.80), ("stage", 1), ("promote", 1)], 0.01),
    ([("register", 1, 0.80), ("stage", 1), ("promote", 1), ("register", 2, 0.805), ("stage", 2), ("promote", 2)], 0.01),
    ([("register", 1, 0.80), ("stage", 1), ("promote", 1), ("register", 2, 0.82), ("stage", 2), ("promote", 2), ("rollback",)], 0.01),
    ([("rollback",), ("promote", 3), ("register", 1, 0.7), ("register", 1, 0.9), ("stage", 1), ("stage", 1)], 0.0),
    ([("register", 1, 0.7), ("stage", 1), ("promote", 1), ("register", 2, 0.8), ("stage", 2), ("promote", 2),
      ("register", 3, 0.9), ("stage", 3), ("promote", 3), ("rollback",), ("rollback",), ("rollback",)], 0.0),
    ([("register", 1, 0.7), ("stage", 1), ("promote", 1), ("rollback",)], 0.0),
]

EXPLANATION = r"""
## Why a registry has stages
Training produces many versions; only one should serve traffic. Stages make the path explicit: a version must be registered with its metrics, pass validation in staging, and clear a gate before production. Every transition is an auditable event.

## The promotion gate
Requiring a minimum gain stops noise-level "improvements" from churning production. Offline metrics are only a gate; a real rollout still goes through shadow traffic or a canary before full exposure.

## Rollback
Rollback is the most important operation in practice. It must be one step, must not need retraining, and must restore the last version that actually served, not merely the newest archived one, which may never have passed production checks. A rolled-back version is known bad, so it should not come back on the next rollback.

## Interview follow-ups
- What metadata besides a metric would you require at registration (data snapshot, code commit, feature schema)?
"""
