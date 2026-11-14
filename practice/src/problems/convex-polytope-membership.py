import numpy as np

META = dict(title='Points inside a convex polytope', tags=['Optimization', 'Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:03.01', 'mlmaths:09.52'],
            related=['convexity-certificate-via-hessian', 'lasso-regression-using-ista'])

STATEMENT = r"""
A convex polytope can be written as the intersection of $m$ half-spaces:

$$P = \{x \in \mathbb{R}^d : G x \le h\}, \qquad G \in \mathbb{R}^{m \times d},\; h \in \mathbb{R}^m.$$

Row $j$ of $G$ is the outward normal $g_j$ of face $j$. The **signed distance** of a point to the hyperplane $g_j^\top x = h_j$ is $(g_j^\top x - h_j) / \lVert g_j \rVert$: negative on the inside of that face, positive outside.

Write `polytope_membership(G, h, X, tol=1e-9)` that returns the tuple `(inside, margin)` for a batch of points:

- `margin`: float array `(n,)`, $\text{margin}_i = \max_j \dfrac{g_j^\top x_i - h_j}{\lVert g_j\rVert}$, the worst signed distance over all faces.
- `inside`: bool array `(n,)`, `True` where `margin <= tol` (points on the boundary count as inside).

## Inputs
- `G`: float `(m, d)` with `m >= 1` and no all-zero rows. `h`: float `(m,)`.
- `X`: float `(n, d)` with `n >= 1`, one point per row.

Vectorise: no Python loops over points or faces.
"""

STARTER = '''import numpy as np

def polytope_membership(G, h, X, tol=1e-9):
    """Return (inside bool (n,), margin float (n,)) for P = {x : G x <= h}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def polytope_membership(G, h, X, tol=1e-9):
    G, h, X = (np.asarray(v, dtype=float) for v in (G, h, X))
    norms = np.linalg.norm(G, axis=1)                # (m,) length of each face normal
    dist = (X @ G.T - h) / norms                     # (n, m) signed distance to every face
    margin = dist.max(axis=1)                        # worst face decides
    return margin <= tol, margin
'''

HINTS = [
    'A point is in the intersection of half-spaces iff it satisfies every inequality, so look at the **worst** one.',
    '`X @ G.T` has shape `(n, m)`: entry `(i, j)` is $g_j^\\top x_i$. Subtract `h` (broadcasts over rows) and divide by the row norms of `G`.',
    'Take `.max(axis=1)` to get each point\'s worst face, then compare with `tol`. Both outputs come from that one `(n, m)` matrix.',
]

_r = np.random.default_rng(5)
_square = (np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), np.array([1.0, 1.0, 1.0, 1.0]))
_tri = (np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 1.0]))
_ang = np.linspace(0, 2 * np.pi, 7)[:-1]
_hex = (np.stack([np.cos(_ang), np.sin(_ang)], axis=1) * 3.0, np.full(6, 3.0))
TESTS = [
    (*_square, np.array([[0.0, 0.0], [2.0, 0.5], [1.0, 1.0]])),
    (*_tri, np.array([[0.25, 0.25], [1.0, 1.0], [-0.5, 0.2]])),
    (*_hex, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])),
    (*_square, np.array([[1.0 + 1e-12, -1.0]])),
    (np.array([[2.0, 0.0, 0.0], [0.0, 0.0, -5.0]]), np.array([4.0, 0.0]), np.array([[2.0, 9.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, -1e-12]])),
    (*_tri, np.array([[0.5, 0.5 + 1e-12], [0.5, 0.5 + 1e-6]])),
    (_r.normal(size=(8, 3)), _r.uniform(0.5, 2.0, size=8), _r.normal(size=(20, 3))),
]

EXPLANATION = r"""
## Why the max decides membership
$x \in P$ iff $g_j^\top x \le h_j$ for every $j$, iff the largest violation $\max_j (g_j^\top x - h_j)$ is $\le 0$. Dividing each row by $\lVert g_j\rVert$ doesn't change the sign, but it turns the raw residual into a distance in the units of $x$, so `tol` means the same thing for every face no matter how $G$ was scaled.

## What the margin measures
For a point **inside** $P$, $-\text{margin}$ is exactly its distance to the boundary: the nearest boundary point lies on some face, and the segment to any face hyperplane must cross the boundary first. For a point **outside**, the margin is only a lower bound on the distance to $P$; near a corner the true distance (a small quadratic program) is larger.

## Complexity
One $(n, d) \times (d, m)$ matrix product: $O(nmd)$ time, $O(nm)$ memory. For huge $n$, process points in chunks to cap memory.

## Common bugs
- Using `all(G @ x <= h)` with exact comparison, so points sitting on a face flicker in and out with round-off.
- Forgetting to normalise rows, so a face written as $10x \le 10$ gets 10x the weight of $x \le 1$.
- `X @ G` instead of `X @ G.T`: shapes only line up by accident when $m = d$.

## Interview follow-ups
- Projected gradient descent needs the **nearest point** in $P$, not just membership. Why is that easy for a box but a QP for a general polytope? (A box projects by clipping each coordinate independently.)
- Why is the intersection of convex sets convex, and why does that make $P$ convex? (If $x, y$ satisfy every inequality, so does $\theta x + (1-\theta)y$ by linearity.)
"""
