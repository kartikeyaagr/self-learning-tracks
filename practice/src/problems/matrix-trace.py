import numpy as np

META = dict(title='Trace of a product without the product', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:05.02'], related=['matrix-multiply', 'hadamard-product', 'eigendecomposition'])

STATEMENT = r"""
The trace of a square matrix is the sum of its diagonal, $\operatorname{tr}(M) = \sum_i M_{ii}$. It shows up in Frobenius norms ($\lVert A\rVert_F^2 = \operatorname{tr}(A^\top A)$), in matrix calculus, and it equals the sum of the eigenvalues.

A common need is $\operatorname{tr}(AB)$ for $A \in \mathbb{R}^{n\times m}$, $B \in \mathbb{R}^{m\times n}$. Forming $AB$ costs $O(n^2 m)$ and then throws away everything except $n$ numbers. Write `trace_of_product(A, B)` that gets the same answer in $O(nm)$.

## Inputs
- `A`: float array `(n, m)`, `B`: float array `(m, n)`, with $n, m \ge 1$.

## Output
- $\operatorname{tr}(AB)$ as a Python float.

## Requirements
- Do not compute `A @ B` (or `np.trace` of it). Use only elementwise operations and sums.
"""

STARTER = '''import numpy as np

def trace_of_product(A, B):
    """A: (n, m), B: (m, n). Return tr(A @ B) as a float, without forming A @ B."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def trace_of_product(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # tr(AB) = sum_i sum_k A[i, k] * B[k, i] = sum of the elementwise product of A and B^T.
    return float((A * B.T).sum())
'''

HINTS = [
    'Write the $i$-th diagonal entry of $AB$ as a sum over $k$. Which entries of $A$ and $B$ does it touch?',
    '$(AB)_{ii} = \\sum_k A_{ik} B_{ki}$. Summing over $i$ too, you multiply $A_{ik}$ by $B^\\top_{ik}$ and add everything up.',
    '`float((A * B.T).sum())`, or `np.einsum("ik,ki->", A, B)`.',
]

_r = np.random.default_rng(17)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])),
    (np.array([[1.0, 2.0, 3.0]]), np.array([[4.0], [5.0], [6.0]])),
    (np.array([[1.0], [2.0]]), np.array([[3.0, 4.0]])),
    (np.eye(3), np.array([[2.0, 9.0, 9.0], [9.0, -1.0, 9.0], [9.0, 9.0, 4.0]])),
    (np.array([[3.5]]), np.array([[-2.0]])),
    (np.zeros((2, 3)), _r.normal(size=(3, 2))),
    (_r.normal(size=(6, 4)), _r.normal(size=(4, 6))),
    (_r.normal(size=(3, 50)) * 1e3, _r.normal(size=(50, 3))),
]

EXPLANATION = r"""
## Why it works
$$\operatorname{tr}(AB) = \sum_i (AB)_{ii} = \sum_i \sum_k A_{ik} B_{ki} = \sum_{i,k} A_{ik}\,(B^\top)_{ik}$$

So the trace of a product is the sum of the Hadamard product $A \odot B^\top$, an inner product between two matrices. In particular $\operatorname{tr}(A^\top A) = \sum_{ik} A_{ik}^2 = \lVert A\rVert_F^2$.

## Complexity
$O(nm)$ time versus $O(n^2 m)$ for forming $AB$. For $n = 10^4$, $m = 100$ that's $10^6$ multiplications instead of $10^{10}$.

## Common bugs
- `(A * B).sum()`: the shapes are `(n, m)` and `(m, n)`, so this only broadcasts when $n = m$, and then computes $\operatorname{tr}(AB^\top)$, the wrong quantity.
- Assuming $\operatorname{tr}(AB) = \operatorname{tr}(A)\operatorname{tr}(B)$. It isn't; test 4 shows it.

## Interview follow-ups
- Prove $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ even though $AB$ is $n\times n$ and $BA$ is $m\times m$. (Swap the order of the double sum.) Is $\operatorname{tr}(ABC) = \operatorname{tr}(BAC)$? (Not in general: only cyclic shifts are allowed.)
- How do you estimate $\operatorname{tr}(M)$ when you can only compute $Mv$? (Hutchinson: average $z^\top M z$ over random $z$ with $\pm 1$ entries.)
"""
