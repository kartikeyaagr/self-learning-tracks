import numpy as np

META = dict(title='Jacobian of a polynomial map', tags=['Calculus'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:09.03', 'air-fanout-v1:01.09'],
            related=['second-order-partial-derivatives', 'jacobian-of-neural-network-layer'])

STATEMENT = r"""
A polynomial map $F: \mathbb R^n \to \mathbb R^m$ can be stored as a shared list of $K$ monomials plus a coefficient matrix. Monomial $k$ is $\prod_j x_j^{E_{kj}}$, and output $i$ is a weighted sum of monomials:

$$F_i(x) = \sum_{k=1}^{K} C_{ik} \prod_{j=1}^{n} x_j^{E_{kj}}$$

For example, with $E = \begin{bmatrix}2 & 0\\ 1 & 1\end{bmatrix}$ (monomials $x_1^2$ and $x_1x_2$) and $C = \begin{bmatrix}1 & 0\\ 0 & 3\end{bmatrix}$, $F(x) = (x_1^2,\ 3x_1x_2)$.

Write `poly_map_jacobian(E, C, x)` that returns the tuple `(F, J)`: the value $F(x)$ with shape `(m,)` and the Jacobian $J_{ij} = \partial F_i/\partial x_j$ with shape `(m, n)`.

## Inputs
- `E`: non-negative int array `(K, n)`; `C`: float array `(m, K)`; `x`: float array `(n,)`.
- `x` may contain zeros. Use $0^0 = 1$.
- Vectorise: no Python loops.
"""

STARTER = '''import numpy as np

def poly_map_jacobian(E, C, x):
    """F_i(x) = sum_k C[i, k] * prod_j x[j] ** E[k, j]. Return (F (m,), J (m, n))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def poly_map_jacobian(E, C, x):
    E, C, x = np.asarray(E), np.asarray(C, dtype=float), np.asarray(x, dtype=float)
    n = len(x)
    F = C @ np.prod(x ** E, axis=1)
    # Differentiating monomial k by x_j lowers exponent j by one and multiplies by E[k, j].
    lowered = E[:, None, :] - np.eye(n, dtype=E.dtype)          # (K, n, n): row j = exponents after d/dx_j
    D = E * np.prod(x ** np.maximum(lowered, 0), axis=2)       # (K, n); E[k, j] = 0 zeroes the clipped case
    return F, C @ D
'''

HINTS = [
    'Work on monomials first: build the matrix $D_{kj} = \\partial(\\text{monomial } k)/\\partial x_j$. Then $J = C\\,D$, since $F = C \\cdot \\text{monomials}$ is linear in them.',
    '$\\partial/\\partial x_j$ of $\\prod_l x_l^{E_{kl}}$ is $E_{kj}\\, x_j^{E_{kj}-1}\\prod_{l\\ne j} x_l^{E_{kl}}$: the same monomial with exponent $j$ lowered by one, times $E_{kj}$.',
    'Make a `(K, n, n)` exponent array `E[:, None, :] - np.eye(n, dtype=int)` (slice `[:, j, :]` is "lowered in $j$"), clip it at 0 to avoid `0.0 ** -1`, take the product over the last axis and multiply by `E`.',
]

_r = np.random.default_rng(19)
TESTS = [
    (np.array([[2, 0], [1, 1]]), np.array([[1.0, 0.0], [0.0, 3.0]]), np.array([2.0, -1.0])),
    (np.array([[0, 0], [1, 0], [0, 1]]), np.array([[5.0, 1.0, 1.0]]), np.array([3.0, 4.0])),
    (np.array([[1, 2, 3]]), np.array([[1.0], [2.0]]), np.array([0.0, 1.0, 2.0])),
    (np.array([[1, 1, 0], [0, 2, 1], [3, 0, 0]]), np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]), np.array([0.0, 0.0, 0.0])),
    (np.array([[0, 0, 0]]), np.array([[7.0]]), np.array([1.5, -2.0, 0.5])),
    (_r.integers(0, 4, size=(6, 3)), _r.normal(size=(4, 6)), _r.normal(size=3)),
    (_r.integers(0, 3, size=(5, 4)), _r.integers(-2, 3, size=(2, 5)).astype(float), np.array([1.0, -1.0, 0.0, 2.0])),
]

EXPLANATION = r"""
## Structure first
$F = C\,\phi(x)$ where $\phi$ maps $x$ to the vector of monomials. By the chain rule, $J_F = C\,J_\phi$, so all the work is in the $K\times n$ monomial Jacobian, and $C$ just mixes its rows. That's the same shape of argument as "a linear layer after a feature map".

## Zeros
At $x_j = 0$ the formula $E_{kj} x_j^{E_{kj}-1}$ hits $0^{-1}$ when $E_{kj} = 0$. That term is multiplied by $E_{kj} = 0$ anyway, so clipping the lowered exponent at 0 sidesteps the division without changing any value. For $E_{kj} = 1$, the lowered exponent is 0 and $0^0 = 1$ gives the correct derivative of $x_j$.

## Complexity
$O(Kn^2)$ for the lowered-exponent products, plus $O(mKn)$ for $C D$.

## Common bugs
- Computing the derivative as `E * monomial / x`, which divides by zero when any $x_j = 0$ (tests 3 and 4).
- Transposing $J$: rows are outputs, columns are inputs.
- Integer overflow or integer division if `x` stays an int array.

## Interview follow-ups
- Check your $J$ against central finite differences. What step size makes the agreement best?
- How would you compute the Hessian tensor of this map with the same exponent-lowering trick?
"""
