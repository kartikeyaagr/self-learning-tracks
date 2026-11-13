META = dict(title='Reproducible train/val/test split by hashing ids', tags=['MLOps', 'Data Processing'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:11.04'], related=['k-fold-cross-validation'])

STATEMENT = r"""
`random.shuffle` gives a different split every time the dataset grows, so yesterday's test users can leak into today's training set. Production pipelines instead assign each example to a split by **hashing a stable id**: the same id always lands in the same split, on every machine, forever.

Write `assign_splits(ids, val_pct, test_pct, salt='')` that returns a list of split names, one per id, in the same order as `ids`.

## The bucket of an id
1. Build the key string `salt + ':' + id` (so with the default salt the key for `'u42'` is `':u42'`).
2. Hash its UTF-8 bytes with SHA-256 (`hashlib.sha256`) and take the hex digest.
3. Read the **first 8 hex characters** as a base-16 integer and take it modulo 100. That's the bucket, an integer in `0..99`.

## From bucket to split
- bucket `< test_pct` → `'test'`
- `test_pct <= ` bucket `< test_pct + val_pct` → `'val'`
- otherwise → `'train'`

`ids` is a list of strings (duplicates allowed; equal ids always get the same split). `val_pct` and `test_pct` are integers with `0 <= val_pct + test_pct <= 100`.
"""

STARTER = '''import hashlib

def assign_splits(ids, val_pct, test_pct, salt=''):
    """Return ['train' | 'val' | 'test', ...], one per id."""
    out = []
    for i in ids:
        bucket = hash(salt + ':' + i) % 100      # Python's hash() is not what you want here
        out.append('test' if bucket < test_pct else 'val' if bucket < val_pct else 'train')
    return out
'''

SOLUTION = '''import hashlib

def assign_splits(ids, val_pct, test_pct, salt=''):
    out = []
    for i in ids:
        digest = hashlib.sha256((salt + ':' + i).encode('utf-8')).hexdigest()
        bucket = int(digest[:8], 16) % 100        # stable across runs, machines and Python versions
        if bucket < test_pct:
            out.append('test')
        elif bucket < test_pct + val_pct:          # val sits right after test on the 0..99 line
            out.append('val')
        else:
            out.append('train')
    return out
'''

HINTS = [
    "Python's built-in `hash()` of a string is randomised per process (PYTHONHASHSEED), so it can't give a reproducible split. Use `hashlib.sha256(key.encode('utf-8')).hexdigest()`.",
    '`int(digest[:8], 16) % 100` turns the first 8 hex characters into a bucket in 0..99.',
    'Lay the splits out on the 0..99 line: test first, then val from `test_pct` up to `test_pct + val_pct`, then train. Compare against the cumulative boundary, not `val_pct` alone.',
]

_many = [f'user-{k}' for k in range(60)]

TESTS = [
    (['u1', 'u2', 'u3', 'u4', 'u5', 'u6'], 20, 20),
    (['u1', 'u2', 'u3', 'u4', 'u5', 'u6'], 20, 20, 'v2'),
    (['a', 'b', 'a', 'c', 'b'], 30, 30),
    (['x1', 'x2', 'x3'], 0, 0),
    (['x1', 'x2', 'x3'], 0, 100),
    (['x1', 'x2', 'x3'], 100, 0),
    (['café', 'naïve', '日本'], 40, 40),
    (_many, 10, 10, 'exp-2026'),
    (_many, 25, 10, 'exp-2026'),
]

EXPLANATION = r"""
## Why it works
A cryptographic hash behaves like a uniform random number that is a fixed function of its input. So each id gets a bucket that looks random (splits come out close to the requested percentages) but is fully reproducible. Adding new ids never moves old ones, which is the property `train_test_split(random_state=...)` lacks once the data changes.

Because test is laid out first on the 0..99 line, growing the validation share from 10 to 25 percent (tests 8 and 9) moves some train ids into val but never touches the test set. Changing the `salt` (test 2) is how you deliberately draw a fresh, independent split.

## Complexity
$O(n)$ hashes, each $O(\text{len(id)})$. It parallelises trivially: any worker can compute any id's split with no coordination.

## Common bugs
- Using `hash()`: different every process, so the split silently changes between the training job and the evaluation job.
- Comparing the bucket with `val_pct` instead of `test_pct + val_pct`.
- Hashing the row index, or a timestamp, instead of the entity id: then one user's rows spread over train and test, which leaks.
- Encoding with something other than UTF-8, so non-ASCII ids (test 7) bucket differently in another service.

## Interview follow-ups
- Your data has many rows per user. Which id do you hash, and why? (The grouping unit, here the user, so no user appears in two splits.)
- How would you get a stratified split (same label balance in every split) while keeping it hash-based? (Hash within each stratum, or hash to a uniform number and take per-stratum quantiles.)
"""
