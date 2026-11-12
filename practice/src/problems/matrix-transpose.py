META = dict(title='Transpose a list-of-lists matrix', tags=['Linear Algebra'], diff='Easy', runtime='python',
            related=['matrix-multiply', 'matrix-vector-multiply'])

STATEMENT = r"""
The transpose of an $m \times n$ matrix $A$ is the $n \times m$ matrix with rows and columns swapped:

$$(A^\top)_{ij} = A_{ji}$$

NumPy hides this behind `.T` (which doesn't even copy data). Here you'll do it by hand on plain Python lists, which is how you'd write it in an interview without NumPy.

Write `transpose(A)`.

## Input
- `A`: a list of $m$ rows, each a list of $n$ numbers. All rows have the same length.
- $m$ may be 0 (`A == []`), and $n$ may be 0 (e.g. `[[], []]`).

## Output
- A new list of $n$ rows, each a list of $m$ numbers, with `out[j][i] == A[i][j]`.
- If $m = 0$ or $n = 0$, return `[]`.
- Do not modify `A`.
"""

STARTER = '''def transpose(A):
    """A: list of m rows of length n. Return the n x m transpose as a new list of lists."""
    raise NotImplementedError
'''

SOLUTION = '''def transpose(A):
    if not A or not A[0]:
        return []                       # no rows, or rows with no columns
    # zip(*A) yields the columns of A as tuples; each column becomes a row of the result.
    return [list(col) for col in zip(*A)]
'''

HINTS = [
    'Row $j$ of the result is column $j$ of `A`. How do you collect column $j$ from a list of rows?',
    'A nested comprehension works: `[[A[i][j] for i in range(m)] for j in range(n)]`. Mind the empty cases before reading `len(A[0])`.',
    '`zip(*A)` unpacks the rows as separate arguments and yields the columns as tuples: `[list(c) for c in zip(*A)]`, with `[]` for empty input.',
]

TESTS = [
    ([[1, 2, 3], [4, 5, 6]],),
    ([[1, 2], [3, 4]],),
    ([[7]],),
    ([[1, 2, 3, 4]],),
    ([[1], [2], [3]],),
    ([],),
    ([[], []],),
    ([[0.5, -1.0], [2.5, 3.0], [-4.0, 1e9]],),
]

EXPLANATION = r"""
## Why `zip(*A)` works
`zip(*A)` is `zip(A[0], A[1], ..., A[m-1])`: it walks all rows in lockstep and yields the $j$-th element of each, which is column $j$. Converting each tuple to a list gives the rows of $A^\top$. The guard handles the shapes where the generic code would give `[]` anyway or crash on `A[0]`.

## Properties worth knowing
- $(A^\top)^\top = A$ and $(AB)^\top = B^\top A^\top$ (note the order swap).
- $x^\top y$ is the dot product; $A^\top A$ is always symmetric and positive semi-definite.
- In NumPy, `A.T` is a **view**: no data moves, only the strides are swapped. That's why it's $O(1)$, and why writing into `A.T` changes `A`.

## Complexity
$O(mn)$ time and memory: every entry is copied once.

## Common bugs
- `[[0] * m] * n` to preallocate: every row is the **same** list object, so writing one row writes all of them.
- Returning `[[]]` or `[[], []]` for empty inputs instead of `[]`.
- Swapping in place, which only works for square matrices.

## Interview follow-ups
- How would you transpose a square matrix in place? (Swap `A[i][j]` with `A[j][i]` for `j > i` only.)
- Why can transposing a large NumPy array and then iterating over rows be slow? (The view is not contiguous: row access jumps across memory, so caches miss; `np.ascontiguousarray` fixes the layout.)
"""
