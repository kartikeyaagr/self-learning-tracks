META = dict(title='Determinant by cofactor expansion', tags=['Linear Algebra'], diff='Easy', runtime='python',
            lessons=['mlmaths:03.10'], related=['matrix-rank', 'lu-decomposition', 'solve-linear-system'])

STATEMENT = r"""
The determinant of a square matrix measures how the linear map scales volume (and whether it flips orientation). It is zero exactly when the matrix is singular.

Laplace (cofactor) expansion along the first row defines it recursively:

$$\det(A) = \sum_{j=0}^{n-1} (-1)^{j}\, A_{0j}\, \det(M_{0j})$$

where $M_{0j}$ is $A$ with row 0 and column $j$ removed. The base case is the $0 \times 0$ matrix, whose determinant is 1 (the empty product).

Write `det(A)` in plain Python using this recursion.

## Input
- `A`: a square list of lists of size $n \times n$, $0 \le n \le 6$. `[]` is the $0\times 0$ matrix.

## Output
- The determinant as a number. Integer input should give an exact integer.
"""

STARTER = '''def det(A):
    """A: n x n list of lists (n may be 0). Return det(A) by cofactor expansion along the first row."""
    raise NotImplementedError
'''

SOLUTION = '''def det(A):
    if not A:
        return 1                                   # det of the 0x0 matrix is the empty product
    total = 0
    for j, a in enumerate(A[0]):
        if a == 0:
            continue                               # zero entries contribute nothing; skip their subtree
        minor = [row[:j] + row[j + 1:] for row in A[1:]]
        total += (-1) ** j * a * det(minor)
    return total
'''

HINTS = [
    'Start with the base case. Then for each entry of the first row you need its minor: drop row 0 and column $j$.',
    'The minor for column $j$ is `[row[:j] + row[j+1:] for row in A[1:]]`. Signs alternate `+, -, +, ...` across the first row.',
    'Loop over `enumerate(A[0])` and accumulate `(-1)**j * a * det(minor)`. Return `1` for `[]`. Skipping zero entries prunes whole branches.',
]

TESTS = [
    ([[1, 2], [3, 4]],),
    ([[2, 0, 0], [0, 3, 0], [0, 0, 4]],),
    ([[5]],),
    ([],),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],),
    ([[0, 1], [1, 0]],),
    ([[0, 2, 1], [3, -1, 2], [4, 0, 1]],),
    ([[2, -1, 0, 3], [1, 0, 4, -2], [0, 5, -1, 1], [3, 2, 2, 0]],),
    ([[0.5, 1.5], [-2.0, 4.0]],),
    ([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 2, 1, 0], [0, 0, 1, 2, 0], [0, 0, 0, 0, -3]],),
]

EXPLANATION = r"""
## Why it works
The determinant is the unique function that is linear in each row, flips sign when two rows swap, and equals 1 on the identity. Expanding along a row is a direct consequence of linearity in that row: split row 0 into $\sum_j A_{0j} e_j$, and each piece reduces to a smaller determinant with sign $(-1)^j$. The first-row expansion uses columns $0..n-1$, so the sign is $(-1)^{0+j}$.

Test 5 is singular (row 3 = 2 * row 2 - row 1), test 6 is a row swap (determinant $-1$), and test 10 is block diagonal, so its determinant is the product of the blocks' determinants: $(-2)(3)(-3) = 18$.

## Complexity
$O(n!)$ time: $n$ subproblems of size $n-1$. Fine for $n \le 6$, hopeless for $n = 20$. Real libraries compute an LU factorisation in $O(n^3)$ and multiply the diagonal of $U$, with a sign for the row swaps.

## Common bugs
- Wrong base case: returning 0 for `[]`, or only handling $1\times 1$ and crashing on the empty matrix.
- Removing column $j$ from row 0 too (it is already gone with the row) or from the wrong rows.
- Forgetting the alternating sign.

## Interview follow-ups
- How do you get the determinant from an LU factorisation $PA = LU$? ($\det A = \det(P)^{-1}\prod_i U_{ii}$, where $\det P = \pm 1$ from the number of swaps.)
- Why is `det` a bad singularity test in floating point? ($\det(0.1\,I_{100}) = 10^{-100}$ for a perfectly well-conditioned matrix; use the condition number or singular values instead.)
"""
