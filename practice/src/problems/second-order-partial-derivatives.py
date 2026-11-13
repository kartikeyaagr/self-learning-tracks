import numpy as np

META = dict(title='Hessian of a two-variable polynomial', tags=['Calculus'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:09.01', 'mlmaths:09.02'],
            related=['jacobian-of-vector-valued-polynomial', 'hessian-tensor-of-vector-valued-function'])

STATEMENT = r"""
A polynomial in two variables can be stored as a coefficient matrix $C$:

$$f(x, y) = \sum_{i}\sum_{j} C_{ij}\, x^i\, y^j$$

so `C[i, j]` multiplies $x^i y^j$. For example `C = [[0, 0, 1], [0, 3, 0], [2, 0, 0]]` is $y^2 + 3xy + 2x^2$.

Write `hessian_2d(C, x, y)` that returns the $2\times 2$ matrix of second partial derivatives at the point $(x, y)$:

$$\begin{bmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{bmatrix}$$

## Requirements
- `C`: float array of shape `(p, q)` with $p, q \ge 1$; `x`, `y`: floats.
- Return a float array of shape `(2, 2)`.
- Differentiate the coefficients exactly (no finite differences).
"""

STARTER = '''import numpy as np

def hessian_2d(C, x, y):
    """C[i, j] multiplies x**i * y**j. Return [[f_xx, f_xy], [f_yx, f_yy]] at (x, y)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.polynomial import polynomial as P

def hessian_2d(C, x, y):
    C = np.asarray(C, dtype=float)
    fxx = P.polyval2d(x, y, P.polyder(C, 2, axis=0))         # axis 0 holds the powers of x
    fyy = P.polyval2d(x, y, P.polyder(C, 2, axis=1))
    fxy = P.polyval2d(x, y, P.polyder(P.polyder(C, axis=0), axis=1))
    return np.array([[fxx, fxy], [fxy, fyy]])                # mixed partials are equal for polynomials
'''

HINTS = [
    'Differentiating with respect to $x$ maps the term $C_{ij}x^iy^j$ to $i\\,C_{ij}x^{i-1}y^j$: multiply row $i$ by $i$ and shift the rows up by one.',
    'NumPy does this for you: `numpy.polynomial.polynomial.polyder(C, m, axis=0)` differentiates $m$ times in $x$ (rows) and `axis=1` in $y$ (columns). `polyval2d(x, y, C)` evaluates the result.',
    'For $f_{xy}$, differentiate once along each axis. For polynomials $f_{xy} = f_{yx}$, so the matrix is symmetric.',
]

_r = np.random.default_rng(27)
TESTS = [
    (np.array([[0.0, 0.0, 1.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]]), 1.0, 1.0),
    (np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), 2.0, -1.0),
    (np.array([[5.0]]), 0.3, 0.7),
    (np.array([[1.0, 2.0, 3.0]]), 4.0, 2.0),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), 0.0, 0.0),
    (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), -1.5, 2.0),
    (_r.integers(-3, 4, size=(4, 4)).astype(float), 0.5, -1.25),
    (_r.normal(size=(3, 5)), -0.8, 1.1),
]

EXPLANATION = r"""
## Differentiating coefficient arrays
$\partial_x (x^i y^j) = i x^{i-1}y^j$, so differentiation in $x$ scales row $i$ by $i$ and drops row 0. In $y$ it does the same to columns. Two derivatives in $x$ scale row $i$ by $i(i-1)$ and drop two rows. Because each axis is handled independently, $\partial_x\partial_y$ and $\partial_y\partial_x$ give the same array: Schwarz's theorem, visible in the bookkeeping.

## What the Hessian tells you
At a critical point ($\nabla f = 0$), a positive definite Hessian means a local minimum, negative definite a maximum, and mixed signs a saddle. For $f = xy$ (test 5) the Hessian is $\begin{bmatrix}0&1\\1&0\end{bmatrix}$ with eigenvalues $\pm1$: the origin is a saddle, the shape that dominates high-dimensional loss landscapes.

## Complexity
$O(pq)$ per derivative array and per evaluation.

## Common bugs
- Swapping axes: `C[i, j]` is $x^i y^j$, so $x$ is axis 0.
- Returning the gradient's derivatives in the wrong order, e.g. `[[fxx, fyy], ...]`.
- Expecting `polyder` to fail when the degree is too low. It returns a zero polynomial, which is the right answer (test 3).

## Interview follow-ups
- Classify every critical point of $f(x,y) = x^3 - 3x + y^2$ using the Hessian.
- When can $f_{xy} \ne f_{yx}$? (Only if the mixed partials aren't continuous; the standard counterexample is $xy(x^2-y^2)/(x^2+y^2)$ at the origin.)
"""
