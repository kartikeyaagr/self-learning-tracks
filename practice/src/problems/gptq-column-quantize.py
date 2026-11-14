import numpy as np

META = dict(title='GPTQ: quantise column by column with error feedback', tags=['Quantization', 'Linear Algebra'], diff='Hard', runtime='numpy',
            related=['int8-symmetric-quantize'])

STATEMENT = r"""
GPTQ quantises a weight matrix `W (rows, cols)` for inputs with Hessian $H = 2X^\top X$ (`cols x cols`) one column at a time, pushing each column's rounding error onto the columns not yet quantised.

Write `gptq(W, X, bits, damp)`:

1. Per-row symmetric grid from the **original** W: $\delta_r = \max_j |W_{rj}| / q_{max}$ with $q_{max} = 2^{b-1} - 1$ (1 if zero). $Q(w) = \mathrm{clip}(\mathrm{round}(w/\delta_r), -q_{max}, q_{max})\,\delta_r$.
2. $H = 2X^\top X + \lambda I$ with $\lambda = $ `damp` $\cdot \mathrm{mean}(\mathrm{diag}(2X^\top X))$.
3. $U$ = the upper Cholesky factor of $H^{-1}$ (so $H^{-1} = U^\top U$; use `np.linalg.cholesky(np.linalg.inv(H)).T`).
4. Work on a copy of W. For $i = 0..cols-1$: $q = Q(W_{:,i})$, $e = (W_{:,i} - q)/U_{ii}$, set column $i$ to $q$, then $W_{:,i+1:} \mathrel{-}= e\,U_{i,i+1:}$.

Return `{"Q": quantised W, "loss_gptq": ‖XW^T − XQ^T‖_F², "loss_rtn": the same for round-to-nearest Q(W)}` using the original W in both losses.
"""

STARTER = '''import numpy as np

def gptq(W, X, bits, damp):
    """GPTQ quantisation of W given calibration inputs X."""
    W = np.asarray(W, float)
    qm = 2 ** (bits - 1) - 1
    d = np.abs(W).max(1, keepdims=True) / qm
    Q = np.clip(np.round(W / d), -qm, qm) * d
    return {"Q": Q, "loss_gptq": 0.0, "loss_rtn": 0.0}
'''

SOLUTION = '''import numpy as np

def gptq(W, X, bits, damp):
    W0, X = np.asarray(W, float), np.asarray(X, float)
    qm = 2 ** (bits - 1) - 1
    d = np.abs(W0).max(1) / qm
    d[d == 0] = 1.0
    quant = lambda w: np.clip(np.round(w / d), -qm, qm) * d
    H = 2 * X.T @ X
    H = H + damp * np.mean(np.diag(H)) * np.eye(len(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    Wk = W0.copy()
    for i in range(W0.shape[1]):
        q = quant(Wk[:, i])
        e = (Wk[:, i] - q) / U[i, i]
        Wk[:, i] = q
        Wk[:, i + 1:] -= np.outer(e, U[i, i + 1:])
    loss = lambda Q: float(np.linalg.norm(X @ W0.T - X @ Q.T) ** 2)
    rtn = np.stack([quant(W0[:, i]) for i in range(W0.shape[1])], 1)
    return {"Q": Wk, "loss_gptq": loss(Wk), "loss_rtn": loss(rtn)}
'''

HINTS = [
    'The grid (the per-row scale) is fixed up front from the original weights; only the values being rounded change.',
    'Dividing the error by U[i, i] and spreading it along row i of U is the optimal update of the remaining weights for a quadratic loss.',
    'Dampening keeps H invertible when some input channels are nearly constant or correlated.',
]

def _case(seed, rows, cols, n, corr):
    r = np.random.default_rng(seed)
    Z = r.normal(size=(n, cols))
    X = Z + corr * Z[:, :1]
    return r.normal(size=(rows, cols)), X

TESTS = [
    _case(1, 4, 6, 32, 0.0) + (4, 0.01),
    _case(2, 8, 12, 64, 1.5) + (4, 0.01),
    _case(3, 8, 12, 64, 1.5) + (3, 0.01),
    _case(4, 6, 8, 40, 0.5) + (8, 0.01),
    _case(5, 5, 10, 50, 2.0) + (2, 0.1),
]

EXPLANATION = r"""
## The objective
Quantisation should preserve the layer's outputs on real inputs, $\|XW^\top - XQ^\top\|^2$, not the weights themselves. That loss is quadratic with Hessian $2X^\top X$, so when one weight is rounded, the best correction to the others is known in closed form (Optimal Brain Surgeon).

## Why column order and Cholesky
GPTQ quantises all rows in the same column order, so the inverse Hessian is shared across rows and the per-column updates come from one Cholesky factor. That makes the method fast enough for 100B-parameter models in a few GPU hours.

## What the tests show
With correlated inputs, round-to-nearest errors in correlated channels add up, and GPTQ's loss is much lower. At 8 bits both are small; the advantage grows at 3 and 4 bits.

## Practical details
Real implementations process columns in blocks for speed, use group-wise grids, and sometimes reorder columns by Hessian diagonal ("act order").

## Interview follow-ups
- Why does GPTQ need calibration data, and how many samples are usually enough?
"""
