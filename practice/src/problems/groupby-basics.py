import numpy as np
import pandas as pd

META = dict(title='Team summary from game logs', tags=['Data Processing'], diff='Medium', runtime='pandas',
            related=['top-n-per-group', 'multi-level-groupby', 'pivot-tables'])

STATEMENT = r"""
`df` is a game log with one row per player per game: columns `team` (str), `player` (str) and `points` (int). A player belongs to exactly one team.

Write `team_summary(df)` that returns one row per team with these columns, in this order:

| column | value |
|---|---|
| `team` | the team |
| `players` | number of **distinct** players on the team |
| `games` | number of rows for the team |
| `total` | sum of `points` |
| `avg` | mean `points` per row (float) |
| `top_player` | the player with the highest **total** points for the team; ties go to the alphabetically first name |

Sort the rows by `total` descending, ties by `team` ascending. `df` has at least one row.

## Example
```
  team player  points
0    A    ann      10
1    A    bob       4
2    B    cat       7
3    A    bob       6
```
gives

| team | players | games | total | avg | top_player |
|---|---|---|---|---|---|
| A | 2 | 3 | 20 | 6.667 | ann |
| B | 1 | 1 | 7 | 7.0 | cat |

ann and bob both total 10 for team A, so the alphabetical tie rule picks ann.
"""

STARTER = '''import pandas as pd

def team_summary(df):
    """One row per team: team, players, games, total, avg, top_player; sorted by total desc, team asc."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def team_summary(df):
    # Named aggregation: output_column=(input_column, function).
    out = df.groupby('team').agg(players=('player', 'nunique'), games=('points', 'size'),
                                 total=('points', 'sum'), avg=('points', 'mean'))
    per_player = df.groupby(['team', 'player'], as_index=False)['points'].sum()
    best = (per_player.sort_values(['team', 'points', 'player'], ascending=[True, False, True])
                      .drop_duplicates('team'))                 # first row per team = the winner
    out['top_player'] = best.set_index('team')['player']        # aligns on the team index
    out = out.reset_index()
    return out.sort_values(['total', 'team'], ascending=[False, True]).reset_index(drop=True)
'''

HINTS = [
    '`df.groupby("team").agg(players=("player", "nunique"), total=("points", "sum"), ...)` builds several named summary columns in one call.',
    'The top player needs a second grouping: sum points per `(team, player)` first, then pick one row per team.',
    'Sort the per-player totals by `team` asc, `points` desc, `player` asc and keep the first row per team (`drop_duplicates("team")`). Assign it by team, then sort the summary by `["total", "team"]` with `ascending=[False, True]`.',
]

_r = np.random.default_rng(23)
_roster = {'Hawks': ['ali', 'bea', 'cho'], 'Owls': ['dan', 'eve'], 'Foxes': ['fay', 'gus', 'hal', 'ivy']}
_rows = [[t, p, int(_r.integers(0, 30))] for _ in range(6) for t, ps in _roster.items() for p in ps if _r.random() < 0.8]


def _df(rows):
    return pd.DataFrame(rows, columns=['team', 'player', 'points'])


TESTS = [
    (_df([['A', 'ann', 10], ['A', 'bob', 4], ['B', 'cat', 7], ['A', 'bob', 6]]),),
    (_df([['X', 'p', 5], ['Y', 'q', 5], ['X', 'p', 0]]),),
    (_df([['solo', 'me', 0]]),),
    (_df([['T', 'zed', 3], ['T', 'amy', 3], ['T', 'max', 1], ['T', 'max', 2]]),),
    (_df([['b', 'u', 100], ['a', 'v', 100], ['c', 'w', 99], ['c', 'x', 1]]),),
    (_df(_rows),),
]

EXPLANATION = r"""
## Why it works
`groupby` splits the rows by key, applies a reduction to each piece, and combines the results into one row per key. Named aggregation (`name=(column, func)`) lets one call produce several differently-named statistics, and the result is indexed by `team`, which makes it easy to attach more per-team columns later by index alignment.

`top_player` is a reduction over a **different** grouping: points must first be summed per `(team, player)`, otherwise a player's best single game would win. Sorting those totals with the tie rule built into the sort keys, then keeping the first row per team, gives a deterministic winner.

## Complexity
Two hash groupings, $O(N)$ each, plus $O(P \log P)$ to sort the $P$ per-player totals and $O(T \log T)$ for the final sort.

## Common bugs
- `games=("player", "count")` is fine here, but `count` skips NaN while `size` does not. Know which one you mean.
- `nunique` vs `count` for `players`: a player appears once per game, so `count` overcounts (test 1).
- `idxmax` on per-row points picks the best **game**, not the best total; on ties it picks the first row in data order, not alphabetical order (test 4).
- Sorting by `total` alone leaves equal totals in an unspecified order (test 5).
- Integer division: `avg` must be a float mean, which `mean` already returns.

## Interview follow-ups
- Add each player's share of their team's points as a new column on the **original** rows. (`groupby("team")["points"].transform("sum")`.)
- Do the same summary in SQL. How does `GROUP BY` plus a window function (`ROW_NUMBER() OVER (PARTITION BY team ORDER BY total DESC, player)`) mirror the two-step pandas version?
"""
