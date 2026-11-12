import numpy as np
import pandas as pd

META = dict(title='Sessionize a clickstream', tags=['Feature Engineering', 'Data Processing', 'Time Series'], diff='Hard',
            runtime='pandas', related=['groupby-basics', 'top-n-per-group', 'sql-running-total'])

STATEMENT = r"""
Web analytics groups a user's clicks into **sessions**: a new session starts whenever the user has been idle for longer than a timeout.

`df` has columns `user` (str), `ts` (int, seconds) and `page` (str). Write `sessionize(df, gap)` that:

1. orders the events by `user` ascending, then `ts` ascending; events of the same user with equal `ts` keep their order from `df`;
2. numbers each user's sessions `1, 2, 3, ...`: a user's first event opens session 1, and each later event opens a new session if `ts - previous_ts > gap` (strictly; the previous event is the same user's preceding event in this order), otherwise it joins the current session;
3. returns a DataFrame with columns `user, ts, page, session, event_no, span`, in that row order, where
   - `session` is the session number within the user,
   - `event_no` is the event's 1-based position within its session,
   - `span` is the session's last `ts` minus its first `ts` (0 for a one-event session), repeated on every row of the session.

## Input
- `gap`: an int, `gap >= 0`. `df` has at least one row.

## Example
With `gap = 30`, user `u`'s events at `ts = 0, 10, 100, 125` form two sessions: `(0, 10)` with span 10 and `(100, 125)` with span 25.
"""

STARTER = '''import pandas as pd

def sessionize(df, gap):
    """Sort by user, ts (stable); new session when idle > gap. Add session, event_no, span."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def _session_ids(ts, gap):
    # Custom per-group transform: a new session starts where the idle time exceeds gap.
    # diff() is NaN for the first event, and NaN > gap is False, so it stays in session 1.
    return (ts.diff() > gap).cumsum() + 1

def sessionize(df, gap):
    out = df.sort_values(['user', 'ts'], kind='stable').reset_index(drop=True)
    out['session'] = out.groupby('user')['ts'].transform(_session_ids, gap)
    sess = out.groupby(['user', 'session'])
    out['event_no'] = sess.cumcount() + 1
    out['span'] = sess['ts'].transform('max') - sess['ts'].transform('min')
    return out[['user', 'ts', 'page', 'session', 'event_no', 'span']]
'''

HINTS = [
    'After sorting, the idle time before each event is `ts.diff()` within the user. Where it exceeds `gap`, a session boundary starts.',
    'A boolean "new session starts here" Series turns into session numbers with `cumsum()`. Run it per user with `groupby("user")["ts"].transform(your_function)`, so the count restarts for each user.',
    'Then group by `["user", "session"]`: `cumcount() + 1` gives `event_no`, and `transform("max") - transform("min")` on `ts` gives `span` on every row.',
]

_r = np.random.default_rng(1234)
_ev = []
for _u in ['zed', 'amy', 'kim']:
    _t = 0
    for _ in range(int(_r.integers(6, 14))):
        _t += int(_r.choice([5, 20, 45, 300, 1800]))
        _ev.append([_u, _t, str(_r.choice(['home', 'search', 'item', 'cart']))])
_ev = [_ev[i] for i in _r.permutation(len(_ev))]


def _df(rows):
    return pd.DataFrame(rows, columns=['user', 'ts', 'page'])


TESTS = [
    (_df([['u', 0, 'home'], ['u', 10, 'item'], ['u', 100, 'home'], ['u', 125, 'cart']]), 30),
    (_df([['b', 50, 'x'], ['a', 5, 'y'], ['b', 10, 'z'], ['a', 6, 'w']]), 20),
    (_df([['solo', 42, 'home']]), 0),
    (_df([['u', 0, 'a'], ['u', 30, 'b'], ['u', 61, 'c'], ['u', 91, 'd']]), 30),
    (_df([['u', 7, 'first'], ['u', 7, 'second'], ['u', 7, 'third'], ['u', 8, 'fourth']]), 0),
    (_df([['p', -100, 'a'], ['q', 10**9, 'b'], ['p', -50, 'c'], ['q', 10**9 + 3600, 'd']]), 60),
    (_df(_ev), 60),
]

EXPLANATION = r"""
## Why it works
Sessionization is a run-length problem: once events are in time order per user, the boundaries are the places where the gap to the previous event exceeds the timeout. A boolean "boundary here" Series turned into integers by `cumsum` labels every run, because the counter goes up by one exactly at each boundary. Doing this inside `groupby("user").transform(...)` restarts the counter for each user and returns a result aligned with the original rows.

That custom function is the general pattern for `transform`: it receives one group's Series and must return a Series of the same length. Once `session` exists, everything else is a built-in group operation: `cumcount` for the position in the session, and `transform("max")`/`"min"` to broadcast each session's extent back to its rows.

The same thing fully vectorised, without a Python function per user: `new = (out.ts.diff() > gap) | (out.user != out.user.shift())`, then `session = new.groupby(out.user).cumsum()`. On millions of users that avoids one function call per group.

## Complexity
$O(N \log N)$ for the sort, $O(N)$ for the rest.

## Common bugs
- `ts.diff()` on the whole column compares the first event of user `b` with the last event of user `a` (test 2).
- `>=` instead of `>`: an idle time exactly equal to `gap` must stay in the same session (test 4).
- `groupby(...).apply(f)` instead of `transform(f)`: `apply` may add the group key to the index and needs extra work to line up with the rows.
- An unstable sort: equal timestamps (test 5) could swap pages.
- Computing `span` from the first and last rows **of the user** instead of the session.

## Interview follow-ups
- Add a `session_id` that is unique across users, such as `user + "-" + session`, and count sessions per user per day.
- Events arrive as a stream, out of order by up to a minute. How do you sessionize incrementally, and when can a session be declared closed?
"""
