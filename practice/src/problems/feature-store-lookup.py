META = dict(title='Online feature store lookup with TTLs and defaults', tags=['MLOps', 'Feature Engineering'], diff='Easy', runtime='python')

STATEMENT = r"""
Write `lookup(store, entity, features, now, ttl, defaults)` for an online feature store.

- `store[entity][feature]` is a `(value, timestamp)` pair. Entities or features may be missing.
- `ttl[feature]` is a maximum age in seconds, or `None` for no expiry. Features missing from `ttl` never expire.
- A stored value is **usable** if `timestamp <= now` (a value from the future is a bug in a backfill and must not leak) and, when a TTL applies, `now - timestamp <= ttl`.
- Unusable or missing features take `defaults[feature]`, or `None` if there is no default.

Return `{"values": {feature: value}, "defaulted": sorted list of features that used the fallback}`.
"""

STARTER = '''def lookup(store, entity, features, now, ttl, defaults):
    """Return {"values": ..., "defaulted": [...]}"""
    row = store.get(entity, {})
    return {"values": {f: row[f][0] for f in features if f in row}, "defaulted": []}
'''

SOLUTION = '''def lookup(store, entity, features, now, ttl, defaults):
    row = store.get(entity, {})
    values, defaulted = {}, []
    for f in features:
        ok = False
        if f in row:
            v, ts = row[f]
            limit = ttl.get(f)
            ok = ts <= now and (limit is None or now - ts <= limit)
        if ok:
            values[f] = v
        else:
            values[f] = defaults.get(f)
            defaulted.append(f)
    return {"values": values, "defaulted": sorted(defaulted)}
'''

HINTS = [
    'Treat "missing", "expired" and "from the future" the same way: fall back to the default.',
    'Use dict.get for both ttl and defaults so absent keys behave as documented.',
]

_store = {"u1": {"clicks_7d": (12, 1000), "avg_order": (35.5, 400), "country": ("DE", 10), "score": (0.8, 2000)},
          "u2": {"clicks_7d": (3, 990)}}
_ttl = {"clicks_7d": 600, "avg_order": 300, "score": 60}
_def = {"clicks_7d": 0, "avg_order": 20.0}
TESTS = [
    (_store, "u1", ["clicks_7d", "avg_order", "country"], 1100, _ttl, _def),
    (_store, "u1", ["score", "clicks_7d"], 1500, _ttl, _def),
    (_store, "u3", ["clicks_7d", "country"], 1000, _ttl, _def),
    (_store, "u2", ["clicks_7d"], 1590, _ttl, _def),
    (_store, "u2", ["clicks_7d"], 1591, _ttl, _def),
    (_store, "u1", ["country"], 10 ** 9, _ttl, {}),
]

EXPLANATION = r"""
## Why TTLs
A seven-day click count that stopped updating a week ago is not a seven-day count any more. Serving it silently gives the model a value from a different distribution than training. A TTL turns staleness into an explicit default the model was trained to expect.

## Why reject future timestamps
If a backfill writes values stamped after `now`, serving or training on them leaks future information. Point-in-time correctness is the main reason feature stores exist: training joins and online lookups must both see only values that existed at the decision time.

## Defaults and monitoring
Log the defaulted list. A spike in defaults is often the first sign that an upstream pipeline broke, well before model metrics move.

## Interview follow-ups
- How do you make the training-time join use the same TTL rules as serving?
"""
