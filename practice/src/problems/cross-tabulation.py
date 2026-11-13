import numpy as np
import pandas as pd

META = dict(title='Confusion matrix with crosstab', tags=['Metrics & Evaluation', 'Data Processing', 'Classification'],
            diff='Medium', runtime='pandas', lessons=['mlmaths:11.01'], related=['pivot-tables', 'precision-recall-at-k'])

STATEMENT = r"""
A classifier's predictions sit next to the true labels in a DataFrame with columns `actual` and `predicted` (both str). Summarise them as a confusion matrix.

Write `confusion_table(df, normalize)` that returns a DataFrame with:

- a first column `actual`, then one column per label;
- one row per label;
- the label set is the **sorted union** of every value in `actual` and in `predicted`, used for both rows and columns, so the matrix is square even when a class is never predicted or never occurs;
- cell (row $a$, column $p$) = the number of rows with `actual == a` and `predicted == p`.

If `normalize` is True, divide every row by its row sum so each row gives $P(\text{predicted} \mid \text{actual})$; a row whose sum is 0 (a label that only ever appears as a prediction) becomes all `0.0`. If False, return the integer counts.

## Example
```
  actual predicted                actual  cat  dog
0    cat       cat     ->    0       cat    1    1
1    cat       dog           1       dog    0    1
2    dog       dog
```
With `normalize=True` the `cat` row becomes `0.5, 0.5`.

## Input
`df` has at least one row; no label is named `actual`.
"""

STARTER = '''import pandas as pd

def confusion_table(df, normalize):
    """Square label x label counts (rows actual, columns predicted); row-normalised if normalize."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def confusion_table(df, normalize):
    labels = sorted(set(df['actual']) | set(df['predicted']))
    counts = (pd.crosstab(df['actual'], df['predicted'])
                .reindex(index=labels, columns=labels, fill_value=0))   # make it square
    if normalize:
        sums = counts.sum(axis=1)
        counts = counts.div(sums.where(sums > 0, 1), axis=0)        # empty rows stay 0
    counts.columns.name = None
    return counts.rename_axis('actual').reset_index()
'''

HINTS = [
    '`pd.crosstab(df["actual"], df["predicted"])` counts every (actual, predicted) pair, but only for labels that actually occur on each side.',
    'Build the full label list as the sorted union of both columns and `reindex(index=labels, columns=labels, fill_value=0)`.',
    'For the normalised version divide row-wise with `counts.div(row_sums, axis=0)`, replacing zero sums by 1 first so empty rows stay 0. Then `rename_axis("actual").reset_index()`.',
]

_r = np.random.default_rng(101)
_n = 60
_y = _r.choice(['bird', 'cat', 'dog'], _n)
_noise = _r.random(_n) < 0.35
_pred = np.where(_noise, _r.choice(['bird', 'cat', 'dog', 'fox'], _n), _y)
_eval = pd.DataFrame({'actual': _y, 'predicted': _pred})


def _df(rows):
    return pd.DataFrame(rows, columns=['actual', 'predicted'])


TESTS = [
    (_df([['cat', 'cat'], ['cat', 'dog'], ['dog', 'dog']]), False),
    (_df([['cat', 'cat'], ['cat', 'dog'], ['dog', 'dog']]), True),
    (_df([['a', 'a']]), False),
    (_df([['spam', 'ham'], ['spam', 'ham'], ['spam', 'spam']]), True),
    (_df([['x', 'y'], ['y', 'z'], ['z', 'x']]), False),
    (_eval, False),
    (_eval, True),
]

EXPLANATION = r"""
## Why it works
`crosstab` is a group-by-two-keys count laid out as a grid, which is exactly a confusion matrix. It only knows about labels that occur on each axis, so a class that is never predicted has no column and a class that never occurs has no row. Reindexing both axes with the same sorted label list makes the matrix square and aligned, which is what every downstream metric (per-class recall, precision, the diagonal) assumes.

Row normalisation turns counts into an empirical conditional distribution $\hat P(\text{pred} = p \mid \text{actual} = a)$: the diagonal is each class's recall.

## Complexity
$O(N)$ to count, $O(K^2)$ for the $K \times K$ grid.

## Common bugs
- Skipping the reindex: with `ham` never actual, test 4 would be $1 \times 2$ instead of $2 \times 2$.
- Reindexing rows and columns with **different** label lists, so the diagonal no longer means "correct".
- Normalising with `pd.crosstab(..., normalize="index")` and then reindexing without `fill_value=0`: the added rows and columns come back NaN, and the statement wants 0.0.
- Normalising by the grand total (`normalize="all"`) or by columns (`"columns"`, which gives precision-style conditionals) instead of rows.

## Interview follow-ups
- Compute per-class precision, recall and macro-F1 from this table without going back to the raw rows.
- With 1,000 classes the matrix is mostly zeros. How would you store and report it? (Keep the long `groupby(["actual", "predicted"]).size()` form, and list the top confused pairs.)
"""
