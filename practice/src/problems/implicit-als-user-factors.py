import numpy as np

META = dict(title='Implicit-feedback ALS: user update with confidence weights', tags=['Recommender Systems', 'Linear Algebra'],
            diff='Hard', runtime='numpy', lessons=['sdsd:12.08', 'sdsd:16.05'],
            related=['als-explicit-sweep', 'ridge-regression'])

STATEMENT = r"""
With implicit feedback (plays, clicks, purchases) there are no ratings, only counts $r_{ui} \ge 0$. Weighted matrix factorisation for implicit data turns each count into a **preference** and a **confidence**:

$$p_{ui} = \mathbb{1}[r_{ui} > 0], \qquad c_{ui} = \begin{cases} 1 + \alpha\, r_{ui} & \text{scheme} = \texttt{"linear"} \\ 1 + \alpha \log(1 + r_{ui}/\epsilon) & \text{scheme} = \texttt{"log"} \end{cases}$$

and fits **every** user-item pair, observed or not, with those weights:

$$\min_X \sum_{u, i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda \sum_u \lVert x_u \rVert^2$$

With item factors $Y$ fixed, user $u$'s solution is

$$x_u = (Y^\top C_u Y + \lambda I)^{-1}\, Y^\top C_u\, p_u, \qquad C_u = \mathrm{diag}(c_{u1}, \dots, c_{uI})$$

Write `implicit_als_users(R, Y, lam, alpha, scheme='linear', eps=1.0)` that returns the new user factors $X$, shape `(n_users, f)`.

## Inputs
- `R`: `(n_users, n_items)` non-negative counts (int or float).
- `Y`: `(n_items, f)` float item factors.
- `lam` $> 0$, `alpha` $\ge 0$, `eps` $> 0$. `scheme` is `"linear"` or `"log"` (natural log).

## Efficiency requirement
Real catalogues have millions of items, so don't build an $I \times I$ matrix. Use the identity
$$Y^\top C_u Y = Y^\top Y + Y^\top (C_u - I) Y,$$
compute $Y^\top Y$ **once**, and touch only the items user $u$ interacted with ($C_u - I$ is zero elsewhere, and so is $p_u$). Solve the $f \times f$ system with `np.linalg.solve`.

A user with no interactions has $p_u = 0$, so $x_u = 0$.
"""

STARTER = '''import numpy as np

def implicit_als_users(R, Y, lam, alpha, scheme='linear', eps=1.0):
    """Return (n_users, f) user factors."""
    R, Y = np.asarray(R, dtype=float), np.asarray(Y, dtype=float)
    P = (R > 0).astype(float)
    f = Y.shape[1]
    return np.linalg.solve(Y.T @ Y + lam * np.eye(f), Y.T @ P.T).T     # ignores confidence
'''

SOLUTION = '''import numpy as np

def implicit_als_users(R, Y, lam, alpha, scheme='linear', eps=1.0):
    R, Y = np.asarray(R, dtype=float), np.asarray(Y, dtype=float)
    if scheme == 'linear':
        C = 1.0 + alpha * R
    else:
        C = 1.0 + alpha * np.log1p(R / eps)
    f = Y.shape[1]
    base = Y.T @ Y + lam * np.eye(f)                # shared by every user: computed once
    X = np.zeros((R.shape[0], f))
    for u in range(R.shape[0]):
        nz = np.flatnonzero(R[u] > 0)               # only these items differ from c = 1, p = 0
        if nz.size == 0:
            continue                                # p_u = 0 -> x_u = 0
        Yn, cn = Y[nz], C[u, nz]
        A = base + (Yn.T * (cn - 1.0)) @ Yn         # Y^T (C_u - I) Y restricted to nz
        b = Yn.T @ cn                               # Y^T C_u p_u, since p = 1 exactly on nz
        X[u] = np.linalg.solve(A, b)
    return X
'''

HINTS = [
    'Write $Y^\\top C_u Y$ as $Y^\\top Y + Y^\\top (C_u - I) Y$. The first term is the same for every user; the second only involves items with $r_{ui} > 0$.',
    'The right-hand side simplifies too: $p_{ui} = 1$ exactly where $r_{ui} > 0$ and 0 elsewhere, so $Y^\\top C_u p_u = \\sum_{i: r_{ui} > 0} c_{ui}\\, y_i$.',
    'For the log scheme use `np.log1p(R / eps)`. Multiply columns of `Y[nz].T` by `c - 1` with broadcasting rather than building a diagonal matrix.',
]

_r = np.random.default_rng(9)
_R = _r.poisson(0.6, size=(9, 12)) * (_r.random((9, 12)) < 0.5)
_R[4] = 0
_Y = _r.normal(0, 0.5, size=(12, 3))

TESTS = [
    (np.array([[2, 0], [0, 0]]), np.array([[1.0], [1.0]]), 1.0, 1.0),
    (np.array([[1, 0, 3]]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 0.5, 2.0, 'log', 1.0),
    (np.array([[1, 0, 3]]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 0.5, 2.0, 'linear'),
    (np.array([[5, 5], [5, 5]]), np.array([[0.5, 0.5], [-0.5, 0.5]]), 0.1, 0.0),
    (np.zeros((3, 4)), np.ones((4, 2)), 1.0, 40.0),
    (np.array([[100.0, 0.0, 1.0]]), np.array([[1.0], [2.0], [0.5]]), 0.01, 40.0, 'log', 0.01),
    (_R, _Y, 0.1, 40.0),
    (_R, _Y, 0.1, 5.0, 'log', 0.5),
]

EXPLANATION = r"""
## Why it works
Unobserved pairs aren't "unknown" here: they're weak evidence of dislike, with confidence 1, while observed pairs are strong evidence of liking, with confidence growing in the count. Because every pair is in the loss, the user's normal equations involve all $I$ items, but $C_u = I + (C_u - I)$ and $C_u - I$ is non-zero only on the user's $n_u$ interacted items. That turns a per-user cost of $O(I f^2)$ into $O(n_u f^2 + f^3)$, with one shared $O(I f^2)$ computation of $Y^\top Y$. Test 4 sets $\alpha = 0$: every confidence is 1 and the method reduces to ridge regression on the 0/1 preference matrix. The log scheme compresses heavy counts (test 6: 100 plays vs 1 play is a confidence ratio of about 98× under linear but only about 2× under log with these settings), which helps when a few users replay one song all day.

## Complexity
$O(I f^2)$ once, then $O(n_u f^2 + f^3)$ per user: $O(|\Omega| f^2 + U f^3)$ in total, linear in the number of interactions.

## Common bugs
- Building `np.diag(C[u])`, an $I \times I$ matrix per user. Correct, but impossible at catalogue scale.
- Using the counts $r_{ui}$ as the regression target instead of the binary preference $p_{ui}$.
- Forgetting the "$-1$": adding $Y^\top C_u Y$ restricted to observed items on top of $Y^\top Y$ double-counts them.
- Using `np.log(R / eps)` instead of `log1p`, which is $-\infty$ at zero counts and inaccurate for tiny ones.

## Interview follow-ups
- How do you pick $\alpha$, and why do typical values (around 10-40) look so large? (It's validated on held-out ranking metrics; it has to make a handful of positives outweigh thousands of confidence-1 negatives.)
- The item update is symmetric. How would you serve recommendations for a brand-new user who just played 3 songs, without retraining? (Fold-in: run exactly this user step with the trained $Y$, then score $Y x_u$ and take the top-k.)
"""
