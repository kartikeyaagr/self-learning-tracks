META = dict(title='Matrix-vector product by hand', tags=['Linear Algebra'], diff='Easy', runtime='python',
            lessons=['mlmaths:04.02'], related=['dot-product', 'linear-combination', 'matrix-multiply'])

STATEMENT = r"""
For $A \in \mathbb{R}^{m\times n}$ and $x \in \mathbb{R}^n$, the product $y = Ax \in \mathbb{R}^m$ can be read two ways:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j \quad\text{(row view: dot products)} \qquad y = \sum_{j=1}^{n} x_j\, A_{:,j} \quad\text{(column view: combination of columns)}$$

Write `matvec(A, x)` in plain Python, without NumPy.

## Inputs
- `A`: list of $m \ge 1$ rows, each a list of $n \ge 1$ numbers.
- `x`: list of numbers.

## Output
- If `len(x) != n`, return `None` (the shapes don't conform).
- Otherwise a list of $m$ numbers, `y[i] = sum(A[i][j] * x[j] for j in range(n))`.
"""

STARTER = '''def matvec(A, x):
    """A: m x n list of lists, x: list of length n. Return A x as a list, or None on a shape mismatch."""
    raise NotImplementedError
'''

SOLUTION = '''def matvec(A, x):
    if len(x) != len(A[0]):
        return None
    # Row view: entry i is the dot product of row i with x.
    return [sum(a * b for a, b in zip(row, x)) for row in A]
'''

HINTS = [
    'Check the shapes first: the number of columns of `A` must equal the length of `x`.',
    'Each output entry pairs up one row of `A` with `x` and adds the products. `zip(row, x)` gives you the pairs.',
    '`[sum(a * b for a, b in zip(row, x)) for row in A]`, after returning `None` when `len(x) != len(A[0])`.',
]

TESTS = [
    ([[1, 2], [3, 4], [5, 6]], [1, -1]),
    ([[2, 0, 0], [0, 3, 0], [0, 0, 4]], [1, 1, 1]),
    ([[1, 2, 3]], [4, 5, 6]),
    ([[1], [2], [3]], [10]),
    ([[1, 2], [3, 4]], [1, 2, 3]),
    ([[0.5, -1.5], [2.0, 0.25]], [4.0, 2.0]),
    ([[1, 0, 0], [0, 1, 0]], [7, 8, 9]),
    ([[1e10, 1], [1, 1e-10]], [1e-10, 1e10]),
]

EXPLANATION = r"""
## Two views, one answer
The row view computes $m$ dot products, one per row. The column view scales column $j$ by $x_j$ and adds the columns. They are the same sums in a different order. The column view is the one that explains linear algebra: $Ax$ always lies in the **column space** of $A$, and $Ax = b$ is solvable exactly when $b$ is a combination of the columns.

Test 7 shows a $2\times 3$ matrix that drops the last coordinate: a projection from $\mathbb{R}^3$ onto a plane.

## Complexity
$O(mn)$ time, $O(m)$ for the output. In NumPy, `A @ x` calls BLAS `gemv`, which is memory-bound: it reads every entry of $A$ once and does just 2 flops per entry.

## Common bugs
- Iterating `for j in range(len(A))` for the inner loop, which mixes up $m$ and $n$ on non-square matrices.
- Forgetting the shape check, so `zip` silently truncates the longer list (test 5).

## Interview follow-ups
- Which loop order is faster in C for a row-major matrix, row view or column view? (Row view: it walks memory contiguously.)
- How would you compute $A^\top x$ without building $A^\top$? (Column view on the rows: $\sum_i x_i A_{i,:}$.)
"""
