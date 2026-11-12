import numpy as np

META = dict(title='Gradient checking with central differences', tags=['Calculus', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.04'],
            related=['chain-rule-through-3-layer-network', 'forward-mode-autodiff-with-dual-numbers'])

STATEMENT = r"""
Before trusting a hand-written backward pass, compare it with a numerical gradient. For logistic regression with labels $y_n \in \{0, 1\}$ the mean loss is

$$L(w) = \frac1n \sum_{n} \Bigl(\log\bigl(1 + e^{z_n}\bigr) - y_n z_n\Bigr), \qquad z = Xw$$

Write `gradient_check(X, y, w, grad, h=1e-5)` that returns `(num_grad, rel_error)`:

- `num_grad`: the central-difference gradient, shape `(d,)`:
$$\hat g_j = \frac{L(w + h e_j) - L(w - h e_j)}{2h}$$
- `rel_error`: how far the supplied `grad` is from it, as a Python `float`:
$$\text{rel\_error} = \frac{\lVert \hat g - \text{grad} \rVert_2}{\max\bigl(\lVert \hat g\rVert_2 + \lVert \text{grad}\rVert_2,\ 10^{-12}\bigr)}$$

## Requirements
- `X`: `(n, d)` floats, `y`: `(n,)` of 0/1 floats, `w` and `grad`: `(d,)`.
- Compute $\log(1 + e^z)$ stably (`np.logaddexp(0, z)`), since tests use $|z|$ in the hundreds.
- Don't modify `w` in place.
"""

STARTER = '''import numpy as np

def gradient_check(X, y, w, grad, h=1e-5):
    """Return (central-difference gradient of the logistic loss at w, relative error of grad)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gradient_check(X, y, w, grad, h=1e-5):
    X, y, w, grad = (np.asarray(a, dtype=float) for a in (X, y, w, grad))
    d = len(w)
    # Perturb every coordinate at once: column j of W is w + h e_j (or w - h e_j).
    Z_plus = X @ (w[:, None] + h * np.eye(d))
    Z_minus = X @ (w[:, None] - h * np.eye(d))
    loss = lambda Z: np.mean(np.logaddexp(0, Z) - y[:, None] * Z, axis=0)   # one loss per column
    num = (loss(Z_plus) - loss(Z_minus)) / (2 * h)
    denom = max(np.linalg.norm(num) + np.linalg.norm(grad), 1e-12)
    return num, float(np.linalg.norm(num - grad) / denom)
'''

HINTS = [
    'Write a helper `loss(w)` first. The numerical gradient needs $2d$ calls to it, one pair per coordinate.',
    'Build the perturbed vector with a copy: `wp = w.copy(); wp[j] += h`. Or perturb all coordinates at once with `w[:, None] + h * np.eye(d)`, whose columns are the $d$ shifted vectors.',
    'The denominator uses the **sum** of the two norms, and the `max(..., 1e-12)` only matters when both gradients are exactly zero.',
]


def _true_grad(X, y, w):
    z = X @ w
    return X.T @ (1 / (1 + np.exp(-z)) - y) / len(y)


_r = np.random.default_rng(31)
_X = _r.normal(size=(10, 3))
_y = (_r.random(10) < 0.5).astype(float)
_w = _r.normal(size=3)
_X2 = _r.normal(size=(25, 6))
_y2 = (_r.random(25) < 0.4).astype(float)
_w2 = _r.normal(size=6)
TESTS = [
    (np.array([[1.0, 2.0]]), np.array([1.0]), np.array([0.0, 0.0]), np.array([-0.5, -1.0])),
    (np.array([[1.0, 2.0]]), np.array([1.0]), np.array([0.0, 0.0]), np.array([0.5, 1.0])),
    (_X, _y, _w, _true_grad(_X, _y, _w)),
    (_X, _y, _w, _true_grad(_X, _y, _w) * len(_y)),        # forgot the 1/n
    (_X, _y, _w, np.zeros(3)),
    (np.array([[100.0, -200.0], [-300.0, 50.0]]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0])),
    (_X2, _y2, _w2, _true_grad(_X2, _y2, _w2), 1e-4),
    (_X2, _y2, _w2, _true_grad(_X2, _y2, _w2) + np.array([0, 0, 1e-3, 0, 0, 0])),
]

EXPLANATION = r"""
## Why central differences
Taylor-expand: $L(w \pm h e_j) = L \pm h\,\partial_j L + \tfrac{h^2}{2}\partial_j^2 L \pm \tfrac{h^3}{6}\partial_j^3 L + \dots$ Subtracting cancels the even terms, so the central difference has error $O(h^2)$. The one-sided $(L(w+he_j) - L(w))/h$ only gets $O(h)$.

## Choosing h
Truncation error shrinks like $h^2$, but rounding error grows like $\varepsilon_{\text{mach}}/h$ because you subtract two nearly equal losses. The sweet spot in float64 is around $h \approx 10^{-5}$, giving relative errors near $10^{-10}$ for a correct gradient. A common rule: below $10^{-7}$ is fine, above $10^{-4}$ means a bug. Test 4 (forgot to divide by $n$) and test 8 (one wrong entry) show what bugs look like.

## Why a relative error
An absolute difference of $10^{-6}$ is a bug if the gradient is $10^{-6}$ and noise if it's $10^{3}$. Normalising by $\lVert\hat g\rVert + \lVert g\rVert$ makes the threshold scale-free and keeps it bounded by 1.

## Complexity
$2d$ loss evaluations, each $O(nd)$: $O(nd^2)$ total. That's why you gradient-check on tiny models, never on the real one.

## Common bugs
- `wp = w; wp[j] += h` mutates `w` itself, so every later coordinate is evaluated at the wrong point.
- Checking at $w = 0$ only, where many bugs (like a missing activation derivative) happen to vanish.
- Checking a ReLU network exactly at a kink, where the numerical and analytic gradients legitimately disagree.

## Interview follow-ups
- How would you gradient-check a model with dropout or batch norm? (Fix the random mask; use eval-mode statistics or a fixed batch.)
- What is complex-step differentiation, and why does it allow $h = 10^{-20}$?
"""
