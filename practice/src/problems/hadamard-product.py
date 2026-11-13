import numpy as np

META = dict(title='Strict Hadamard product', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            related=['matrix-trace', 'matrix-multiply', 'outer-product'])

STATEMENT = r"""
The Hadamard (elementwise) product of two matrices of the **same shape** is

$$(A \odot B)_{ij} = A_{ij}\, B_{ij}$$

Gating in LSTMs, dropout masks and attention masks are all Hadamard products. In NumPy it's just `A * B`, but `*` also broadcasts: a `(3, 1)` times a `(1, 3)` silently becomes a `(3, 3)` outer product. In a training loop that kind of shape mistake produces wrong numbers instead of an error.

Write `hadamard(A, B)` that refuses to broadcast.

## Inputs
- `A`, `B`: array-likes (NumPy arrays or nested lists) of any shape.

## Output
- If `A` and `B` have exactly the same shape: a float array `A * B` of that shape.
- Otherwise: `None`. That includes cases NumPy would happily broadcast, such as `(3,)` with `(1, 3)`, `(2, 1)` with `(2, 3)`, or a scalar with a matrix.
"""

STARTER = '''import numpy as np

def hadamard(A, B):
    """Elementwise product if shapes match exactly, else None (no broadcasting)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def hadamard(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        return None                     # broadcasting would succeed for some of these; we refuse on purpose
    return A * B
'''

HINTS = [
    'The multiplication is the easy part. What exactly must be true about the two inputs?',
    'Convert both inputs with `np.asarray` so lists have a `.shape`, then compare the shape tuples.',
    '`if A.shape != B.shape: return None`, else `return A * B` (as floats).',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[10.0, 0.0], [-1.0, 0.5]])),
    (np.array([[1.0], [2.0], [3.0]]), np.array([[1.0, 2.0, 3.0]])),
    (np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0, 3.0]])),
    (np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), np.array([[3.0], [4.0]])),
    ([[1, 2, 3]], [[4, 5, 6]]),
    (2.0, np.array([1.0, 2.0])),
    (np.array([1e200, -3.0]), np.array([1e-200, -3.0])),
    (_r.normal(size=(2, 3, 4)), _r.normal(size=(2, 3, 4))),
]

EXPLANATION = r"""
## Why check shapes explicitly
NumPy's broadcasting rule aligns shapes from the right and stretches any axis of size 1. That's what makes `X - X.mean(0)` work, but it also means `A * B` almost never fails, even when the two operands mean different things. A shape assertion is the cheapest bug-catcher in numerical code.

## Identities that use it
- $\operatorname{tr}(A^\top B) = \sum_{ij} (A \odot B)_{ij}$: the Frobenius inner product.
- $\operatorname{diag}(x)\,A\,\operatorname{diag}(y) = A \odot (x y^\top)$: scaling rows and columns is a Hadamard product with an outer product.
- The Hadamard product of two positive semi-definite matrices is PSD (Schur product theorem), which is why products of kernels are kernels.

## Complexity
$O(N)$ for $N$ entries, plus an $O(1)$ shape check.

## Common bugs
- Comparing `A.size == B.size`: `(2, 3)` and `(3, 2)` have the same size but are not compatible.
- Calling `.shape` on a Python list or float.
- Using `np.multiply(A, B)` or `A * B` and trusting it to raise on mismatch.

## Interview follow-ups
- Is the Hadamard product commutative and associative? Does it distribute over addition? (Yes to all three, unlike the matrix product, which isn't commutative.)
- What is the identity element, and when does $A$ have a Hadamard inverse? (The all-ones matrix; when no entry is zero.)
"""
