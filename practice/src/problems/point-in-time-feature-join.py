import numpy as np
import pandas as pd

META = dict(title='Point-in-time correct feature join with late arrivals', tags=['MLOps', 'Feature Engineering', 'Time Series'], diff='Hard', runtime='pandas',
            lessons=[], related=['training-serving-skew', 'lag-and-lead'])

STATEMENT = r"""
To build a training set, every label row needs the feature values the model **would have seen** at prediction time. Joining on the latest feature value overall leaks the future. Joining on the latest value by event time still leaks if that value was only written to the feature store later (a late-arriving batch job, a backfill, a correction).

Write `pit_join(labels, features, ttl)` that attaches point-in-time correct features to each label row.

## Inputs
- `labels`: columns `entity` (str), `ts` (int, prediction time) and `label` (int). Rows may repeat.
- `features`: columns `entity` (str), `event_ts` (int, when the value describes), `available_ts` (int, when it was written to the store, `>= event_ts`), followed by one or more float feature columns. The triple `(entity, event_ts, available_ts)` is unique.
- `ttl`: int `>= 0`, the maximum feature age.

## Which feature row a label row gets
A feature row is **usable** for a label row when all three hold:
1. same `entity`;
2. `available_ts <= ts` (it was in the store at prediction time);
3. `ts - event_ts <= ttl` (not stale). Note `event_ts > ts` can't be usable, because then `available_ts > ts`.

Among the usable rows, take the one with the largest `event_ts`; break ties by the largest `available_ts` (the most recent correction). If no row is usable, every feature value is `NaN`.

## Output
A new DataFrame with the columns of `labels` (`entity, ts, label`), then the feature columns in the order they appear in `features`, then `age` (float: `ts - event_ts` of the chosen row, `NaN` if none). Feature columns and `age` are float. Rows are in the same order as `labels`; the index is ignored. Don't modify the inputs.
"""

STARTER = '''import pandas as pd

def pit_join(labels, features, ttl):
    """Attach point-in-time correct features (see statement) to each label row."""
    f = features.sort_values('event_ts').drop(columns='available_ts')
    out = pd.merge_asof(labels.sort_values('ts'), f, left_on='ts', right_on='event_ts', by='entity')
    out['age'] = (out['ts'] - out['event_ts']).astype(float)
    return out.drop(columns='event_ts')
'''

SOLUTION = '''import numpy as np
import pandas as pd

KEYS = ['entity', 'event_ts', 'available_ts']

def pit_join(labels, features, ttl):
    fcols = [c for c in features.columns if c not in KEYS]
    lab = labels.reset_index(drop=True)
    lab['_row'] = np.arange(len(lab))                       # remembers the original order

    # Every (label row, candidate feature row) pair for the same entity, then keep the usable ones.
    pairs = lab[['_row', 'entity', 'ts']].merge(features, on='entity')
    usable = pairs[(pairs['available_ts'] <= pairs['ts']) & (pairs['ts'] - pairs['event_ts'] <= ttl)]

    # Newest event first, newest correction first, then the first row per label row wins.
    best = (usable.sort_values(['_row', 'event_ts', 'available_ts'], ascending=[True, False, False])
                  .drop_duplicates('_row'))
    best = best.assign(age=(best['ts'] - best['event_ts']).astype(float))[['_row'] + fcols + ['age']]

    out = lab.merge(best, on='_row', how='left').sort_values('_row')
    out = out.drop(columns='_row').reset_index(drop=True)
    return out.astype({c: 'float64' for c in fcols + ['age']})
'''

HINTS = [
    "`pd.merge_asof` matches on a single time column, but usability depends on two (`available_ts` for the store, `event_ts` for freshness), so it can't express this join directly. Generate candidates instead: merge each label row with every feature row of its entity.",
    "Give each label row an id first (`np.arange(len(labels))`), because label rows may repeat. Filter the candidate pairs with `available_ts <= ts` and `ts - event_ts <= ttl`.",
    "Sort the survivors by `[row_id, event_ts, available_ts]` with `ascending=[True, False, False]` and `drop_duplicates(row_id)`. Left-merge the winners back onto the labels so rows with no usable feature get NaN, then restore the original order.",
]

def _f(rows, cols=('clicks_1h', 'spend_1d')):
    return pd.DataFrame(rows, columns=['entity', 'event_ts', 'available_ts', *cols])

def _l(rows):
    return pd.DataFrame(rows, columns=['entity', 'ts', 'label'])

_rng = np.random.default_rng(23)
_ents = ['u1', 'u2', 'u3', 'u4']
_fr = _f([[e, int(t), int(t + d), float(v), float(w)]
          for e in _ents for t, d, v, w in zip(np.sort(_rng.choice(1000, 25, replace=False)), _rng.integers(0, 60, 25),
                                                _rng.integers(0, 50, 25), _rng.integers(0, 200, 25) / 4)])
_lr = _l([[str(_rng.choice(_ents + ['u9'])), int(_rng.integers(0, 1100)), int(_rng.integers(0, 2))] for _ in range(40)])

TESTS = [
    (_l([['a', 100, 1], ['a', 200, 0]]),
     _f([['a', 90, 95, 1.0, 10.0], ['a', 150, 210, 2.0, 20.0]]), 1000),
    (_l([['a', 100, 1], ['b', 100, 0]]),
     _f([['a', 50, 50, 1.0, 5.0], ['a', 80, 80, 2.0, 6.0]]), 30),
    (_l([['a', 100, 1]]),
     _f([['a', 90, 92, 1.0, 1.0], ['a', 90, 99, 1.5, 1.25], ['a', 90, 101, 9.0, 9.0]]), 50),
    (_l([['a', 100, 1], ['a', 100, 0], ['a', 5, 1]]),
     _f([['a', 100, 100, 3.0, 4.0]]), 0),
    (_l([['x', 10, 0], ['y', 10, 1]]),
     _f([['z', 0, 0, 1.0, 1.0]]), 100),
    (_l([['b', 300, 1], ['a', 300, 0], ['b', 100, 1]]),
     _f([['a', 250, 260, 0.5], ['b', 90, 95, 0.25], ['b', 280, 290, 0.75], ['b', 299, 301, 9.0]], cols=('score',)), 100),
    (_lr, _fr, 120),
    (_lr, _fr, 45),
]

EXPLANATION = r"""
## Why it works
The join reproduces what an online lookup would have returned at time `ts`: the store only contained rows with `available_ts <= ts`, the serving code ignores values older than the TTL, and among what's left it reads the newest value. Filtering candidate pairs on exactly those conditions and then taking the top row per label row encodes that literally. Every label row is matched independently, so repeated label rows (test 4) and unsorted labels (test 6) need no special cases.

Test 1 is the classic leak: the value for `event_ts = 150` describes the past at `ts = 200` but was only written at 210, so the model must see the older value. A join on `event_ts` alone would hand the model a value it couldn't have had.

## Complexity
The candidate merge is $O(\sum_e L_e F_e)$ (labels times feature rows per entity), followed by a sort. That's fine for moderate data. At scale you'd sort both sides by time and sweep (an as-of join on `available_ts`, keeping the best `event_ts` seen so far), or let the feature store's offline engine do it.

## Common bugs
- `merge_asof` on `event_ts` (the starter): ignores `available_ts`, so late-arriving values leak (tests 1 and 3).
- Forgetting the TTL, so a value from months ago silently fills in for a missing feature (test 2).
- Using `ts - event_ts < ttl` instead of `<=` (test 4 has `ttl = 0` and an exact-time row).
- Losing the original row order or the duplicate label rows by merging on `(entity, ts)` instead of a row id.

## Interview follow-ups
- Training data built this way still differs from online serving. Name one more source of skew a point-in-time join can't fix. (Online lookup latency: the serving system may read a cache that's a few seconds staler than the store.)
- How would you compute this for 1B label rows and 10B feature rows? (Partition by entity, sort each partition by time and do a single merge sweep, as Spark-based feature stores do.)
"""
