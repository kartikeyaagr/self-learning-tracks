import numpy as np

META = dict(title='One ALS sweep for explicit-rating matrix factorisation', tags=['Recommender Systems', 'Linear Algebra', 'Optimization'],
            diff='Hard', runtime='numpy', lessons=['sdsd:12.08'],
            related=['ridge-regression', 'low-rank-approximation', 'implicit-als-user-factors'])

STATEMENT = r"""
Matrix factorisation models a rating as $\hat r_{ui} = u_u^\top v_i$ and fits only the **observed** entries $\Omega$:

$$\min_{U, V} \sum_{(u,i) \in \Omega} (r_{ui} - u_u^\top v_i)^2 + \lambda \Big(\sum_u \lVert u_u \rVert^2 + \sum_i \lVert v_i \rVert^2\Big)$$

The objective is not jointly convex, but with $V$ fixed each user row is an independent ridge regression, and vice versa. **Alternating least squares** solves them in closed form, one side at a time.

Write `als_sweep(R, M, U, V, lam)` that performs one sweep and returns `(U_new, V_new, rmse)`:

1. For every user $u$, with $\Omega_u$ the items $u$ rated: $\;u_u \leftarrow (V_{\Omega_u}^\top V_{\Omega_u} + \lambda I)^{-1} V_{\Omega_u}^\top r_{u,\Omega_u}$, using the **input** $V$.
2. For every item $i$, with $\Omega_i$ the users who rated $i$: $\;v_i \leftarrow (U_{\Omega_i}^\top U_{\Omega_i} + \lambda I)^{-1} U_{\Omega_i}^\top r_{\Omega_i, i}$, using the **new** $U$ from step 1.
3. `rmse` = root mean squared error of `U_new @ V_new.T` against $R$ over the observed entries only, as a Python float.

## Inputs
- `R`: `(n_users, n_items)` float; entries where `M == 0` are meaningless.
- `M`: same shape, 0/1, with at least one 1.
- `U`: `(n_users, f)`, `V`: `(n_items, f)` float initial factors.
- `lam`: $\lambda > 0$. It multiplies the identity as written; it is **not** scaled by $|\Omega_u|$.

## Rules
- A user (or item) with no observed ratings gets a zero row: the formula gives exactly that.
- Solve the $f \times f$ systems with `np.linalg.solve`; don't form explicit inverses. Don't modify the inputs.
"""

STARTER = '''import numpy as np

def als_sweep(R, M, U, V, lam):
    """Return (U_new, V_new, rmse) after one user pass then one item pass."""
    R, U, V = (np.asarray(x, dtype=float) for x in (R, U, V))
    f = U.shape[1]
    U_new = np.linalg.solve(V.T @ V + lam * np.eye(f), V.T @ R.T).T      # treats every entry as observed
    V_new = np.linalg.solve(U_new.T @ U_new + lam * np.eye(f), U_new.T @ R).T
    rmse = float(np.sqrt(np.mean((U_new @ V_new.T - R) ** 2)))
    return U_new, V_new, rmse
'''

SOLUTION = '''import numpy as np

def _half_step(R, M, F, lam):
    """Ridge-solve one row per row of R, using only the columns M marks as observed."""
    f = F.shape[1]
    out = np.zeros((R.shape[0], f))
    for r in range(R.shape[0]):
        obs = M[r] > 0
        Fo = F[obs]                                   # (|obs|, f) factors of the observed partners
        A = Fo.T @ Fo + lam * np.eye(f)
        out[r] = np.linalg.solve(A, Fo.T @ R[r, obs])
    return out

def als_sweep(R, M, U, V, lam):
    R, M = np.asarray(R, dtype=float), np.asarray(M)
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    U_new = _half_step(R, M, V, lam)                  # users, with the old item factors
    V_new = _half_step(R.T, M.T, U_new, lam)          # items, with the new user factors
    obs = M > 0
    rmse = float(np.sqrt(np.mean((U_new @ V_new.T - R)[obs] ** 2)))
    return U_new, V_new, rmse
'''

HINTS = [
    'Fix $V$. User $u$\'s problem only involves the items they rated, so it is ridge regression with design matrix `V[M[u] > 0]` and targets `R[u, M[u] > 0]`.',
    'The item pass is the same computation on the transposed problem: pass `R.T`, `M.T` and the **new** `U`. Write one helper and call it twice.',
    'Compute the RMSE with a boolean mask on the observed entries; averaging over all of `R` counts meaningless zeros.',
]

_r = np.random.default_rng(21)
_Ut, _Vt = _r.normal(size=(8, 2)), _r.normal(size=(7, 2))
_R = _Ut @ _Vt.T + 0.1 * _r.normal(size=(8, 7))
_M = (_r.random((8, 7)) < 0.55).astype(int)
_M[0] = 0                                                # a user with no ratings
_U0, _V0 = 0.1 * _r.normal(size=(8, 2)), 0.1 * _r.normal(size=(7, 2))

TESTS = [
    (np.array([[2.0, 0.0], [4.0, 1.0]]), np.array([[1, 0], [1, 1]]), np.array([[1.0], [1.0]]), np.array([[1.0], [1.0]]), 1.0),
    (np.array([[5.0, 3.0, 0.0], [4.0, 0.0, 1.0]]), np.array([[1, 1, 0], [1, 0, 1]]),
     np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.3]]), 0.5),
    (np.array([[3.0, 9.0]]), np.array([[1, 0]]), np.array([[2.0]]), np.array([[1.0], [1.0]]), 0.1),
    (np.ones((3, 3)), np.eye(3, dtype=int), np.ones((3, 2)), np.ones((3, 2)), 2.0),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones((2, 2), dtype=int), np.array([[1.0, 0.0], [0.0, 1.0]]),
     np.array([[1.0, 1.0], [1.0, -1.0]]), 1e-3),
    (_R, _M, _U0, _V0, 0.1),
    (_R, _M, _U0, _V0, 2.0),
]

EXPLANATION = r"""
## Why it works
With $V$ fixed, the objective splits into one term per user, each a ridge regression in $f$ unknowns whose design matrix is the factors of the items that user rated. Setting the gradient to zero gives the normal equations $(V_{\Omega_u}^\top V_{\Omega_u} + \lambda I)\,u_u = V_{\Omega_u}^\top r_{u,\Omega_u}$. Because each half-step exactly minimises the objective over one block, the objective never increases across sweeps, so ALS converges (to a local optimum) without a learning rate. Test 3 shows the masking: the unobserved 9.0 must have no effect.

## Complexity
Per sweep, $O(|\Omega| f^2 + (U + I) f^3)$: building each Gram matrix costs $|\Omega_u| f^2$ and each solve $f^3$. Every row is independent, which is why ALS parallelises so well (Spark MLlib's recommender is ALS).

## Common bugs
- Treating unobserved entries as zeros (the starter's bug). That fits a different model: "unrated means hated".
- Using the old $U$ in the item pass. It's still a valid block method, but not ALS, and the test expects the Gauss-Seidel order.
- Inverting with `np.linalg.inv`: slower and less stable than `solve`.
- RMSE over all entries instead of the observed ones.

## Interview follow-ups
- Many implementations scale $\lambda$ by $|\Omega_u|$ ("weighted-$\lambda$ regularisation"). What does that change? (Heavy raters get proportionally more shrinkage, so regularisation strength per rating is uniform; it often generalises better.)
- When would you pick SGD over ALS for matrix factorisation? (Many non-quadratic terms, e.g. biases with side features or ranking losses; ALS needs a quadratic loss for closed-form block updates, except in its implicit variant.)
"""
