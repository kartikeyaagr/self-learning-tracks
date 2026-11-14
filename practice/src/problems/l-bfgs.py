import numpy as np

META = dict(title='L-BFGS on the Rosenbrock function', tags=['Optimization'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:07.01', 'mlmaths:09.52'],
            related=['newton-s-method', 'rmsprop', 'optimizer-showdown'])

STATEMENT = r"""
L-BFGS approximates Newton's method without ever forming a Hessian. It stores the last $m$ pairs $s_i = x_{i+1} - x_i$, $y_i = g_{i+1} - g_i$ and uses them to apply an approximate inverse Hessian to the gradient with the **two-loop recursion**. Minimise the $d$-dimensional Rosenbrock function ($d \ge 2$), whose minimum is at $(1, \dots, 1)$:

$$f(x) = \sum_{i=0}^{d-2}\left[100\,(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right].$$

Write `lbfgs_rosenbrock(x0, m=5, max_iter=200, tol=1e-6)` that returns the tuple `(x, n_iter)`: the final point (float `(d,)`) and the number of updates taken (int).

## Algorithm (follow it exactly)
Start with $x = x_0$, $g = \nabla f(x)$, empty history. For each iteration:
1. If $\lVert g\rVert_2 \le$ `tol`, stop. Also stop once `max_iter` updates have been taken.
2. **Direction** $p = -r$, where $r$ comes from the two-loop recursion on $q = g$ with the stored pairs (newest first in the first loop):
   - for $i$ newest to oldest: $\rho_i = 1/(y_i^\top s_i)$, $\alpha_i = \rho_i s_i^\top q$, $q \leftarrow q - \alpha_i y_i$;
   - $r = \gamma q$ with $\gamma = \dfrac{s^\top y}{y^\top y}$ of the **newest** pair (or $\gamma = 1$ with no pairs);
   - for $i$ oldest to newest: $\beta = \rho_i y_i^\top r$, $r \leftarrow r + s_i(\alpha_i - \beta)$.
   If $g^\top p \ge 0$, fall back to $p = -g$.
3. **Weak-Wolfe line search** (bisection): set $t = 1$, $\text{lo} = 0$, $\text{hi} = \infty$, and repeat at most 50 times:
   - if $f(x + tp) > f(x) + 10^{-4}\, t\, g^\top p$ (not enough decrease): $\text{hi} = t$;
   - else if $\nabla f(x + tp)^\top p < 0.9\, g^\top p$ (still too steep, step too short): $\text{lo} = t$;
   - else accept $t$ and stop;
   - next trial: $t = (\text{lo} + \text{hi})/2$ if $\text{hi} < \infty$, otherwise $t = 2\,\text{lo}$.

   If 50 trials pass without acceptance, use the last trial $t$.
4. Update $s = tp$, $x_{\text{new}} = x + s$, $g_{\text{new}} = \nabla f(x_{\text{new}})$, $y = g_{\text{new}} - g$. Store $(s, y)$ only if $s^\top y > 10^{-10}$; keep at most $m$ pairs, dropping the oldest.
"""

STARTER = '''import numpy as np

def lbfgs_rosenbrock(x0, m=5, max_iter=200, tol=1e-6):
    """Return (x_final (d,), n_iter int) following the exact L-BFGS recipe."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _f(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

def _grad(x):
    r = x[1:] - x[:-1] ** 2                         # (d-1,) residuals of the curved valleys
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * r - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * r                              # each x_i also appears as x_{i+1} of the previous term
    return g

def _two_loop(g, S, Y):
    q, alphas = g.copy(), []
    for s, y in zip(reversed(S), reversed(Y)):      # newest to oldest
        a = (s @ q) / (y @ s)
        alphas.append(a)
        q = q - a * y
    r = q * ((S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0)   # scaled identity as H0
    for (s, y), a in zip(zip(S, Y), reversed(alphas)):          # oldest to newest
        b = (y @ r) / (y @ s)
        r = r + s * (a - b)
    return r

def lbfgs_rosenbrock(x0, m=5, max_iter=200, tol=1e-6):
    x = np.array(x0, dtype=float)
    g = _grad(x)
    S, Y, it = [], [], 0
    while it < max_iter and np.linalg.norm(g) > tol:
        p = -_two_loop(g, S, Y)
        if g @ p >= 0:
            p = -g                                  # safeguard: always move downhill
        fx, slope = _f(x), g @ p
        t, lo, hi = 1.0, 0.0, np.inf
        for _ in range(50):                         # weak Wolfe by bisection / doubling
            if _f(x + t * p) > fx + 1e-4 * t * slope:
                hi = t                              # overshot: too little decrease
            elif _grad(x + t * p) @ p < 0.9 * slope:
                lo = t                              # still descending steeply: go further
            else:
                break
            t = (lo + hi) / 2 if hi < np.inf else 2 * lo
        s = t * p
        x_new = x + s
        g_new = _grad(x_new)
        y = g_new - g
        if s @ y > 1e-10:                           # keep only pairs with positive curvature
            S.append(s); Y.append(y)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        x, g, it = x_new, g_new, it + 1
    return x, it
'''

HINTS = [
    'Get $f$ and $\\nabla f$ right first and check the gradient against finite differences. Every interior $x_i$ appears in two terms of the sum.',
    'Store the pairs in two Python lists, newest last. The first loop walks them in reverse and saves each $\\alpha_i$; the second loop walks forward and uses the $\\alpha$s in the opposite order.',
    'Line search: keep a bracket `[lo, hi]` with `hi = inf` at first. Too little decrease shrinks `hi`; a still-steep slope raises `lo`; double while `hi` is infinite, bisect once it is finite. The $\\gamma = s^\\top y / y^\\top y$ scaling makes $t = 1$ pass most of the time.',
]

TESTS = [
    (np.array([-1.2, 1.0]),),
    (np.array([1.0, 1.0]),),
    (np.array([-1.2, 1.0]), 3, 5),
    (np.array([0.0, 0.0, 0.0, 0.0]),),
    (np.array([2.0, -1.0, 0.5]), 10),
    (np.array([-1.2, 1.0, -1.2, 1.0, -1.2, 1.0]), 5, 300, 1e-8),
    (np.array([0.5, 0.5]), 1, 50),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## Why the two-loop recursion works
BFGS maintains an inverse-Hessian estimate $H_k$ via the update $H_{k+1} = (I - \rho s y^\top) H_k (I - \rho y s^\top) + \rho s s^\top$, which enforces the **secant condition** $H_{k+1} y = s$: the model reproduces the observed change in gradient. Unrolling the last $m$ updates applied to $H_0 = \gamma I$ and multiplying by $g$ gives exactly the two loops, in $O(md)$ time and memory instead of $O(d^2)$.

## Why a Wolfe line search
The update keeps $H$ positive definite only if $s^\top y > 0$. Plain Armijo backtracking doesn't ensure it: on Rosenbrock it happily accepts tiny steps whose curvature pairs are negative, those pairs get discarded, the memory never learns the valley's shape, and the method crawls for hundreds of iterations. The curvature condition $\nabla f(x + tp)^\top p \ge 0.9\, g^\top p$ forces $y^\top s = t(\nabla f_{\text{new}} - g)^\top p \ge t(0.9 - 1)\, g^\top p > 0$ on any smooth function. The $10^{-10}$ check and the $-g$ fallback remain as cheap safety nets.

## Behaviour
From the classic start $(-1.2, 1)$, gradient descent needs thousands of steps to follow the curved valley; L-BFGS needs a few dozen. The 2-D tests show `n_iter`, and test 2 (starting at the optimum) returns 0. Test 7 uses $m = 1$: still far better than GD, but noticeably slower than $m = 5$.

## Complexity
$O(md)$ per iteration for the recursion plus a handful of function evaluations in the line search; memory $O(md)$.

## Common bugs
- Using the $\alpha$s in the wrong order in the second loop.
- Skipping the $\gamma$ scaling, so the first trial step has the wrong length and the line search does all the work.
- Forgetting the $200(x_{i+1} - x_i^2)$ contribution of the previous term to the gradient of $x_{i+1}$.
- Using backtracking only (no curvature condition): it runs, but it can stall exactly as described above.
- Updating the history before checking $s^\top y$.

## Interview follow-ups
- Why is L-BFGS popular for full-batch problems (logistic regression, style transfer, physics fits) but rarely used to train deep nets? (Mini-batch noise breaks the secant pairs and the line search; it needs consistent gradients.)
- What does the **strong** Wolfe condition add, and why do production implementations prefer it? (It bounds $|\nabla f_{\text{new}}^\top p|$ from both sides, so steps can't land far past the 1-D minimum.)
"""
