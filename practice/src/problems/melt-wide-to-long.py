import numpy as np
import pandas as pd

META = dict(title='Gradebook from wide to long', tags=['Data Processing'], diff='Easy', runtime='pandas',
            related=['unstack-long-to-wide', 'pivot-tables'])

STATEMENT = r"""
A gradebook has one row per student and one column per subject. Plotting and grouping tools want it **long**: one row per (student, subject) score.

`df` has a column `student` (str, unique) followed by one float column per subject; NaN means the student did not take that subject. Write `scores_long(df)` that returns a DataFrame with columns `student, subject, score`:

- one row per student and subject **with a score** (NaN cells are dropped);
- rows sorted by `student` ascending; within a student, subjects in the order their columns appear in `df`.

## Example
```
  student  math  art            student  subject  score
0     zoe  90.0  NaN      ->    ali      math     70.0
1     ali  70.0  85.0           ali      art      85.0
                                zoe      math     90.0
```
"""

STARTER = '''import pandas as pd

def scores_long(df):
    """Wide gradebook -> long (student, subject, score), NaN dropped, sorted by student then column order."""
    raise NotImplementedError
'''

SOLUTION = '''import pandas as pd

def scores_long(df):
    long = df.melt(id_vars='student', var_name='subject', value_name='score')
    long = long.dropna(subset=['score'])
    # melt emits all rows of the first subject, then the second, ...; a stable sort by student
    # keeps that column order within each student.
    return long.sort_values('student', kind='stable').reset_index(drop=True)
'''

HINTS = [
    '`df.melt(id_vars=...)` keeps the id column and turns every other column into (variable, value) pairs.',
    'Name the two new columns with `var_name="subject"` and `value_name="score"`, then drop the rows where `score` is NaN.',
    '`melt` lists all students for the first subject, then all for the second. A **stable** sort by `student` keeps the subjects in column order within each student.',
]

_r = np.random.default_rng(4)
_subjects = ['math', 'physics', 'history', 'art', 'music']
_book = pd.DataFrame(_r.integers(40, 101, size=(8, 5)).astype(float), columns=_subjects)
_book = _book.mask(_r.random(_book.shape) < 0.25)
_book.insert(0, 'student', _r.permutation([f's{i}' for i in range(8)]))
TESTS = [
    (pd.DataFrame({'student': ['zoe', 'ali'], 'math': [90.0, 70.0], 'art': [np.nan, 85.0]}),),
    (pd.DataFrame({'student': ['b', 'a', 'c'], 'z_last': [1.0, 2.0, 3.0], 'a_first': [4.0, 5.0, 6.0]}),),
    (pd.DataFrame({'student': ['solo'], 'chem': [np.nan]}),),
    (pd.DataFrame({'student': ['x'], 'q1': [0.0], 'q2': [-5.5], 'q3': [100.0]}),),
    (pd.DataFrame({'student': ['m', 'n'], 'bio': [np.nan, np.nan], 'geo': [1.0, np.nan]}),),
    (_book,),
]

EXPLANATION = r"""
## Why it works
`melt` is the inverse of a pivot: it keeps the `id_vars` columns and stacks every remaining column, emitting one row per original cell with the column's name in `var_name` and the cell in `value_name`. Missing scores become rows with NaN, which `dropna` removes.

`melt` produces rows column by column (all students for `math`, then all for `art`). Sorting by `student` with a **stable** algorithm reorders only by student and keeps the column order inside each student, which is the required secondary order. Test 2 has subjects whose names sort the opposite way to their column order.

## Complexity
$O(S \cdot K)$ for $S$ students and $K$ subjects, plus $O(n \log n)$ for the sort of the $n$ kept rows.

## Common bugs
- Sorting by `["student", "subject"]` orders subjects alphabetically, not by column (test 2).
- The default sort is not stable, so ties on `student` may come out in any order.
- Forgetting `id_vars`: `student` is melted too, and its strings mix with the scores.
- `set_index("student").stack()` also produces student-major pairs, but its NaN handling changed with pandas' new `stack` implementation (`future_stack=True`), so check your version before relying on it.

## Interview follow-ups
- Columns are named like `math_2023`, `math_2024`. Melt them and split `subject` and `year` into two columns. (`str.split("_", expand=True)`, or `pd.wide_to_long`.)
- Why do plotting libraries and `groupby` prefer the long format, and when is wide better (model input matrices, human reading)?
"""
