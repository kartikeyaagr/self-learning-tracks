META = dict(title='Matrix multiplication from scratch', tags=['Linear Algebra'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:03.02', 'mlmaths:03.05'], related=['matrix-vector-multiply', 'matrix-transpose', 'matrix-trace'])

STATEMENT = r"""
For $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$, the product $C = AB \in \mathbb{R}^{m\times n}$ has entries

$$C_{ij} = \sum_{t=1}^{k} A_{it} B_{tj}$$

Entry $(i, j)$ is the dot product of row $i$ of $A$ with column $j$ of $B$. Composing two linear maps multiplies their matrices.

Write `matmul(A, B)` in plain Python, without NumPy.

## Inputs
- `A`: list of $m \ge 1$ rows of length $k \ge 1$. `B`: list of rows, each of equal length $n \ge 1$.

## Output
- If the number of columns of `A` differs from the number of rows of `B`, return `None`.
- Otherwise the $m \times n$ product as a list of lists.
"""

STARTER = '''def matmul(A, B):
    """A: m x k, B: k x n (lists of lists). Return the m x n product, or None if shapes don't conform."""
    raise NotImplementedError
'''

SOLUTION = '''def matmul(A, B):
    if len(A[0]) != len(B):
        return None
    cols = list(zip(*B))                      # columns of B, extracted once instead of per entry
    return [[sum(a * b for a, b in zip(row, col)) for col in cols] for row in A]
'''

HINTS = [
    'The shapes conform when the inner dimensions match: `len(A[0]) == len(B)`.',
    'Three nested loops: over rows of `A`, over columns of `B`, and over the shared index. Pre-extracting the columns of `B` makes the inner step a dot product.',
    '`cols = list(zip(*B))`; then `[[sum(a*b for a, b in zip(row, col)) for col in cols] for row in A]`.',
]

TESTS = [
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[1, 2, 3]], [[1], [0], [-1]]),
    ([[1], [0], [-1]], [[1, 2, 3]]),
    ([[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]),
    ([[2, 0], [0, 3]], [[1, 1, 1], [1, 1, 1]]),
    ([[0, -1], [1, 0]], [[0, -1], [1, 0]]),
    ([[0.5, 1.5, -2.0], [1.0, 0.0, 4.0]], [[2.0, 1.0], [0.0, -1.0], [0.25, 0.5]]),
    ([[7]], [[6]]),
]

EXPLANATION = r"""
## Why it works
Each output entry is one dot product, and there are $mn$ of them, each of length $k$. Tests 2 and 3 show why order matters: a $1\times 3$ times $3\times 1$ is a scalar (an inner product), while $3\times 1$ times $1\times 3$ is a $3\times 3$ outer product. Test 6 is a 90 degree rotation applied twice, giving $-I$: multiplying matrices composes the maps.

## Complexity
$O(mkn)$ time, $O(mn)$ memory. Strassen-style algorithms beat $n^3$ asymptotically, but BLAS libraries win in practice with cache blocking and SIMD, which is why NumPy's `@` is thousands of times faster than this code.

## Common bugs
- Checking `len(A) == len(B)` (rows against rows) instead of the inner dimensions.
- Building the result with `[[0] * n] * m`, which aliases one row $m$ times.
- Looking up column $j$ of `B` inside the innermost loop, which turns an $O(k)$ step into $O(k)$ list building every time.

## Interview follow-ups
- Is matrix multiplication commutative? Associative? (No; yes. Associativity lets you pick the cheapest parenthesisation, e.g. $(AB)x$ vs $A(Bx)$.)
- How many flops is a forward pass of a linear layer with batch $b$, input $d$, output $h$? (About $2bdh$.)
"""
