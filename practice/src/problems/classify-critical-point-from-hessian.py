import numpy as np

META = dict(title='Classify a critical point of a polynomial', tags=['Optimization', 'Calculus'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:09.52', 'mlmaths:07.01'],
            related=['convexity-certificate-via-hessian', 'newton-s-method'])

STATEMENT = r"""
A polynomial in two variables is given as a list of terms `(i, j, c)`, each meaning $c\,x^i y^j$:

$$f(x, y) = \sum_{(i, j, c)} c\, x^i y^j.$$

Write `classify_point(terms, point, tol=1e-8)` that evaluates the gradient and Hessian of $f$ **analytically** at `point = (x0, y0)` and returns one of these strings:

1. `'not critical'` if $\lVert \nabla f(x_0, y_0) \rVert_2 > $ `tol`. Otherwise, with $\lambda_{\min} \le \lambda_{\max}$ the eigenvalues of the Hessian:
2. `'saddle'` if $\lambda_{\min} < -$`tol` and $\lambda_{\max} > $ `tol`.
3. `'minimum'` if $\lambda_{\min} > $ `tol`.
4. `'maximum'` if $\lambda_{\max} < -$`tol`.
5. `'degenerate'` otherwise (some eigenvalue is within `tol` of 0 and the test is inconclusive).

## Inputs
- `terms`: list of tuples `(i, j, c)` with integers `i, j >= 0` and float `c`. The same `(i, j)` may appear more than once (the coefficients add). The list may be empty ($f \equiv 0$).
- `point`: tuple of two floats.

Don't use finite differences: differentiate each monomial exactly. Watch out for $0 \cdot x^{-1}$ when $x = 0$.
"""

STARTER = '''import numpy as np

def classify_point(terms, point, tol=1e-8):
    """terms: list of (i, j, c) meaning c * x**i * y**j. Return one of
    'not critical', 'saddle', 'minimum', 'maximum', 'degenerate'."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _mono(x, k, order):
    """d^order/dx^order of x**k, evaluated at x. Zero once the derivative kills the term."""
    if order > k:
        return 0.0
    coef = 1.0
    for r in range(order):
        coef *= k - r                         # k (k-1) ... falling factorial
    return coef * x ** (k - order)

def classify_point(terms, point, tol=1e-8):
    x, y = map(float, point)
    g = np.zeros(2)
    H = np.zeros((2, 2))
    for i, j, c in terms:
        g[0] += c * _mono(x, i, 1) * _mono(y, j, 0)
        g[1] += c * _mono(x, i, 0) * _mono(y, j, 1)
        H[0, 0] += c * _mono(x, i, 2) * _mono(y, j, 0)
        H[1, 1] += c * _mono(x, i, 0) * _mono(y, j, 2)
        H[0, 1] += c * _mono(x, i, 1) * _mono(y, j, 1)
    H[1, 0] = H[0, 1]                         # mixed partials agree for polynomials
    if np.linalg.norm(g) > tol:
        return 'not critical'
    lo, hi = np.linalg.eigvalsh(H)            # ascending
    if lo < -tol and hi > tol:
        return 'saddle'
    if lo > tol:
        return 'minimum'
    if hi < -tol:
        return 'maximum'
    return 'degenerate'
'''

HINTS = [
    'The derivative of $c\\,x^i y^j$ with respect to $x$ is $c\\,i\\,x^{i-1} y^j$. Write one helper that returns the $k$-th derivative of a power, and returns 0 when $k$ exceeds the exponent.',
    'You need five numbers: $f_x, f_y, f_{xx}, f_{yy}, f_{xy}$. Accumulate each over all terms, then build the symmetric 2x2 Hessian.',
    'Check the gradient norm first. Then `np.linalg.eigvalsh(H)` gives sorted eigenvalues; compare them against `tol` in the order the statement lists.',
]

TESTS = [
    ([(2, 0, 1.0), (0, 2, 1.0)], (0.0, 0.0)),
    ([(2, 0, 1.0), (0, 2, -1.0)], (0.0, 0.0)),
    ([(2, 0, 1.0), (0, 2, 1.0)], (1.0, 0.0)),
    ([(4, 0, 1.0), (0, 2, 1.0)], (0.0, 0.0)),
    ([(2, 0, -2.0), (1, 1, 1.0), (0, 2, -3.0), (1, 0, 0.0)], (0.0, 0.0)),
    ([(3, 0, 1.0), (1, 0, -3.0), (0, 2, 1.0)], (1.0, 0.0)),
    ([(3, 0, 1.0), (1, 0, -3.0), (0, 2, 1.0)], (-1.0, 0.0)),
    ([(1, 1, 1.0), (1, 1, 1.0), (0, 0, 5.0)], (0.0, 0.0)),
    ([], (2.0, -3.0)),
    ([(2, 2, 1.0), (2, 0, -2.0), (0, 2, -2.0)], (1.4142135623730951, 1.4142135623730951)),
]

EXPLANATION = r"""
## The second-derivative test
Near a critical point $p$ (where $\nabla f(p) = 0$), Taylor's theorem gives $f(p + h) \approx f(p) + \tfrac12 h^\top H h$. Diagonalising $H = Q \Lambda Q^\top$ turns the quadratic form into $\sum_k \lambda_k z_k^2$:
- all $\lambda_k > 0$: $f$ rises in every direction, a strict local **minimum**;
- all $\lambda_k < 0$: a strict local **maximum**;
- mixed signs: up along some directions, down along others, a **saddle**;
- a zero eigenvalue: the quadratic term is flat along that direction, so higher-order terms decide. $x^4 + y^2$ (test 4) is a minimum and $x^3 + y^2$ would not be, yet both have the same Hessian at the origin.

Test 10, $x^2y^2 - 2x^2 - 2y^2$ at $(\sqrt2, \sqrt2)$, is a saddle despite looking symmetric: $f_{xx} = f_{yy} = 0$ and $f_{xy} = 8$, giving eigenvalues $\pm 8$.

## Complexity
$O(T)$ for $T$ terms, plus a constant-size eigenvalue solve.

## Common bugs
- Evaluating `i * x**(i-1)` with `i = 0` and `x = 0.0`: Python raises `ZeroDivisionError` for `0.0 ** -1`. Skip terms whose derivative is identically zero.
- Classifying before checking that the point is critical. A non-critical point is none of these.
- Using the determinant test $f_{xx}f_{yy} - f_{xy}^2$ without a tolerance; it hides which eigenvalue is near zero.

## Interview follow-ups
- In high dimensions, why are saddle points far more common than local minima for random critical points? (Each of $d$ eigenvalues must be positive for a minimum; with random signs that has probability $2^{-d}$.)
- How do gradient descent and Newton's method behave near a saddle? (GD escapes slowly along the negative-curvature direction; pure Newton is attracted to the saddle because it jumps to any stationary point of the local quadratic.)
"""
