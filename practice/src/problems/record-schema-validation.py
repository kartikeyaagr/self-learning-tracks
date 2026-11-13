import math

META = dict(title='Validate incoming records against a data contract', tags=['MLOps', 'Data Processing'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:11.12'], related=['concatenate-dataframes', 'training-serving-skew'])

STATEMENT = r"""
Before a batch reaches feature computation, a validation step checks it against a schema (a "data contract") and reports every violation, so bad upstream data is caught at the door instead of as a silent accuracy drop a week later.

Write `validate_records(records, schema)` that returns a list of violations, each a tuple `(row, column, code)`.

## The schema
`schema` maps column name to a spec dict. Keys of a spec:

| key | meaning | default |
|---|---|---|
| `type` | `'int'`, `'float'`, `'str'` or `'bool'` | required |
| `required` | the key must be present in the record | `True` |
| `nullable` | `None` is allowed | `False` |
| `min`, `max` | inclusive numeric bounds (only used for `int` / `float`) | none |
| `allowed` | list of permitted values | none |

## Checks
For each record (row index from 0) and each schema column, **in schema order**, run these checks and report only the **first** that fails:

1. Key absent: `'missing'` if `required`, otherwise skip the column.
2. Null: the value is `None`, or a float NaN in a `float` column. If `nullable` it's fine (stop checking this column), otherwise `'null'`.
3. Type, else `'type'`:
   - `int`: an `int` that is not a `bool`.
   - `float`: an `int` or `float`, not a `bool`.
   - `str`: a `str`. `bool`: a `bool`.
4. Range: below `min` or above `max` → `'range'`.
5. Category: not in `allowed` → `'category'`.

Then, for keys in the record that are not in the schema, report `(row, key, 'unexpected')`, keys in sorted order.

Violations are ordered by row, then within a row in the order above (schema columns, then unexpected keys). A clean batch returns `[]`.
"""

STARTER = '''import math

def validate_records(records, schema):
    """Return [(row, column, code), ...]."""
    out = []
    for r, rec in enumerate(records):
        for col, spec in schema.items():
            if col not in rec:
                out.append((r, col, 'missing'))
            elif spec['type'] == 'int' and not isinstance(rec[col], int):
                out.append((r, col, 'type'))
    return out
'''

SOLUTION = '''import math

TYPE_OK = {
    'int': lambda v: isinstance(v, int) and not isinstance(v, bool),        # bool is a subclass of int
    'float': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    'str': lambda v: isinstance(v, str),
    'bool': lambda v: isinstance(v, bool),
}

def check_value(v, spec):
    """First failing code for one present value, or None."""
    t = spec['type']
    if v is None or (t == 'float' and isinstance(v, float) and math.isnan(v)):
        return None if spec.get('nullable', False) else 'null'
    if not TYPE_OK[t](v):
        return 'type'
    if t in ('int', 'float'):
        if ('min' in spec and v < spec['min']) or ('max' in spec and v > spec['max']):
            return 'range'
    if 'allowed' in spec and v not in spec['allowed']:
        return 'category'
    return None

def validate_records(records, schema):
    out = []
    for r, rec in enumerate(records):
        for col, spec in schema.items():
            if col not in rec:
                if spec.get('required', True):
                    out.append((r, col, 'missing'))
                continue
            code = check_value(rec[col], spec)
            if code:
                out.append((r, col, code))
        out.extend((r, k, 'unexpected') for k in sorted(set(rec) - set(schema)))
    return out
'''

HINTS = [
    'Write a helper that returns the first failing code for one value (or `None`) and stops at the first failure with early returns. The main loop then only deals with missing keys and unexpected keys.',
    "`isinstance(True, int)` is `True` in Python, so the `int` and `float` checks must explicitly exclude `bool`. NaN only counts as null for `float` columns; test it with `math.isnan` after checking it's a float.",
    'A nullable `None` passes and skips the remaining checks. Unexpected keys are `sorted(set(record) - set(schema))`, appended after the schema columns of the same row.',
]

_schema = {
    'user_id': {'type': 'int', 'min': 1},
    'age': {'type': 'int', 'min': 0, 'max': 120, 'nullable': True},
    'country': {'type': 'str', 'allowed': ['DE', 'IN', 'US']},
    'spend': {'type': 'float', 'min': 0.0},
    'is_new': {'type': 'bool', 'required': False},
}

TESTS = [
    ([{'user_id': 1, 'age': 30, 'country': 'US', 'spend': 12.5}], _schema),
    ([{'user_id': 0, 'age': 30, 'country': 'FR', 'spend': 3},
      {'user_id': 2, 'country': 'IN', 'spend': -1.0, 'note': 'x'}], _schema),
    ([{'user_id': True, 'age': None, 'country': 'DE', 'spend': float('nan'), 'is_new': 1}], _schema),
    ([{'user_id': 5, 'age': 121, 'country': None, 'spend': 1.0, 'is_new': False, 'z': 1, 'a': 2}], _schema),
    ([], _schema),
    ([{'user_id': 3.0, 'age': '40', 'country': 'US', 'spend': '9.99'}], _schema),
    ([{'x': 1}, {}, {'x': None}],
     {'x': {'type': 'float', 'nullable': True, 'max': 0.5}}),
    ([{'label': 2}, {'label': 1}, {'label': True}, {'label': -1}],
     {'label': {'type': 'int', 'allowed': [0, 1]}}),
    ([{'user_id': 7, 'age': 0, 'country': 'IN', 'spend': 0.0, 'is_new': True},
      {'user_id': 8, 'age': 120, 'country': 'IN', 'spend': 1e9}], _schema),
]

EXPLANATION = r"""
## Why it works
Each column runs through a short-circuit chain, so a value gets at most one code and the most basic problem wins: a string where an int was expected is reported as `'type'`, not as a range violation that couldn't even be evaluated. Reporting violations as data, rather than raising on the first one, lets the pipeline decide per code what to do: quarantine the row, fail the batch, or just count it on a dashboard.

## Complexity
$O(R \cdot C)$ for $R$ records and $C$ schema columns, plus sorting each row's unexpected keys. `allowed` is a list here; convert it to a set once if it's large.

## Common bugs
- `isinstance(True, int)` is `True`, so `bool` slips through `int` columns (tests 3 and 8) and `1` passes as a `bool` unless the check is `isinstance(v, bool)`.
- Rejecting ints in float columns: JSON has one number type, and `3` is a perfectly good spend (test 2).
- Treating NaN as a valid float. In pandas pipelines a missing number usually *is* NaN, so it must go through the null rule (test 3).
- Reporting a missing optional column, or forgetting that a nullable `None` must skip the range and category checks (test 7).

## Interview follow-ups
- The upstream team adds a column every few weeks. Should `'unexpected'` fail the batch? (Usually warn, not fail; failing on unexpected columns blocks additive changes that are backwards-compatible.)
- Per-record checks can't catch "30% of `country` values are suddenly null". What would you add? (Batch-level expectations: null-rate and distinct-count bounds, row-count ranges, distribution checks such as PSI.)
"""
