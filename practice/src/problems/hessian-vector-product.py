import numpy as np

META = dict(title='Hessian-vector products, forward over reverse', tags=['JAX', 'Optimization', 'Calculus'], diff='Medium',
            runtime='jax', lessons=['mlmaths:09.28'], related=['value-and-grad-of-loss', 'convexity-certificate-via-hessian'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
Second-order methods and curvature diagnostics need products $Hv$ with the Hessian $H = \nabla^2 f(w)$. Forming $H$ costs $O(D^2)$ memory, but $Hv$ is the directional derivative of the gradient, which JAX computes for about the cost of two gradients.

Use the regularised logistic loss with labels $y_i \in \{-1, +1\}$:

$$f(w) = \frac{1}{N} \sum_{i=1}^{N} \mathrm{softplus}(-y_i\, x_i^\top w) + \frac{\lambda}{2} \lVert w \rVert^2, \qquad \mathrm{softplus}(t) = \log(1 + e^t)$$

Write `grad_and_hvps(w, X, y, lam, V)`:

- `w`: `(D,)`; `X`: `(N, D)`; `y`: `(N,)` of $\pm 1$; `lam`: float.
- `V`: `(k, D)`: $k$ directions.

Return `(g, HV)`: `g` is $\nabla f(w)$ with shape `(D,)` and `HV` is `(k, D)` with `HV[j]` $= H\, V_j$. Both float32. Do not build the `(D, D)` Hessian: compute one HVP with `jax.jvp` of `jax.grad(f)` and batch it over the rows of `V` with `jax.vmap`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def grad_and_hvps(w, X, y, lam, V):
    """Return (gradient (D,), Hessian-vector products (k, D))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def grad_and_hvps(w, X, y, lam, V):
    w, X, y, V = (jnp.asarray(a, jnp.float32) for a in (w, X, y, V))

    def f(w):
        return jnp.mean(jax.nn.softplus(-y * (X @ w))) + 0.5 * lam * jnp.dot(w, w)

    g_fn = jax.grad(f)

    def hvp(v):
        # Forward-mode derivative of the gradient along v: d/dt grad f(w + t v) at t = 0 equals H v.
        return jax.jvp(g_fn, (w,), (v,))[1]

    return g_fn(w), jax.vmap(hvp)(V)
'''

HINTS = [
    '$Hv = \\frac{d}{dt} \\nabla f(w + t v)\\big|_{t=0}$: the directional derivative of the gradient function.',
    '`jax.jvp(fn, (w,), (v,))` returns `(fn(w), J_fn(w) v)`. Apply it to `fn = jax.grad(f)`.',
    'Wrap the single-vector HVP in a function of `v` and map it over the rows of `V` with `jax.vmap`.',
]

_r = np.random.default_rng(8)
_X = _r.normal(size=(30, 6))
_y = np.where(_r.random(30) < 0.5, -1.0, 1.0)
TESTS = [
    (np.zeros(2), np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, -1.0]), 0.0, np.eye(2)),
    (np.array([0.5, -0.5]), np.array([[1.0, 1.0]]), np.array([1.0]), 1.0, np.array([[1.0, 0.0]])),
    (_r.normal(size=6), _X, _y, 0.1, _r.normal(size=(3, 6))),
    (_r.normal(size=6), _X, _y, 0.0, np.zeros((2, 6))),
    (10 * _r.normal(size=6), _X, _y, 0.01, np.eye(6)),
    (np.zeros(4), _r.normal(size=(12, 4)), np.ones(12), 2.0, _r.normal(size=(5, 4))),
    (_r.normal(size=3), _r.normal(size=(50, 3)), np.where(_r.random(50) < 0.3, -1.0, 1.0), 0.05, _r.normal(size=(1, 3))),
]

EXPLANATION = r"""
## Why forward-over-reverse
`jax.grad(f)` is a function $\mathbb{R}^D \to \mathbb{R}^D$ whose Jacobian is $H$. A JVP of that function along $v$ is exactly $Hv$. Forward mode adds one tangent per primal value, so the cost is a small constant times one gradient evaluation, with $O(D)$ memory. Reverse-over-reverse (`grad` of `v · grad f`) also works but stores more intermediates; forward-over-reverse is the usual default.

## Sanity checks you can do
For this loss, $H = \frac{1}{N} X^\top \mathrm{diag}(s_i(1-s_i)) X + \lambda I$ with $s_i = \sigma(-y_i x_i^\top w)$. It is positive semidefinite, so $v^\top H v \ge 0$ for every row. With $\lambda > 0$ it is strictly positive. Large weights (test 5) saturate the sigmoid and shrink the data term, leaving mostly $\lambda I$.

## Complexity
$k$ HVPs cost about $2k$ gradient evaluations, i.e. $O(k N D)$ here, versus $O(N D^2)$ time and $O(D^2)$ memory to form $H$. `vmap` runs the $k$ directions as one batched computation instead of a Python loop.

## Common bugs
- `jax.jvp(f, ...)` on the loss itself: that gives the scalar directional derivative $\nabla f \cdot v$, not $Hv$.
- Passing tangents with a different dtype from the primal (float64 NumPy vs float32 JAX): `jvp` raises a dtype mismatch, so cast first.
- Calling `jax.hessian` and multiplying: correct values, but defeats the point at scale.

## Interview follow-ups
- How would you estimate the largest Hessian eigenvalue of a neural network loss? (Power iteration or Lanczos with HVPs.)
- How do conjugate-gradient Newton steps (Hessian-free optimisation) use only HVPs?
"""
