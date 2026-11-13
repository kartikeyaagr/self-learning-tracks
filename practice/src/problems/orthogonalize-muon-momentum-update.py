import numpy as np

META = dict(title='Muon: orthogonalize a momentum update with Newton-Schulz', tags=['Training', 'Linear Algebra'],
            diff='Easy', runtime='numpy', lessons=['mlmaths:07.02', 'air-fanout-v1:05.06'],
            related=['adam-optimizer', 'gram-schmidt-orthogonalization', 'diagnose-mup-width-invariants'])

STATEMENT = r"""
Muon is an optimizer for 2-D weight matrices. It keeps an SGD momentum buffer, but instead of stepping along the momentum $M$ it steps along the nearest **orthogonal** matrix, $UV^\top$ from the SVD $M = U\Sigma V^\top$. That equalises the update's singular values, so rare directions get as much push as dominant ones. An SVD per step is too slow on a GPU, so Muon approximates $UV^\top$ with a few matrix-multiply-only **Newton-Schulz** iterations.

Write `muon_update(grad, momentum, beta=0.95, steps=5, nesterov=True)` that returns `(update, new_momentum)`.

## Steps (all float64)
1. Momentum: $M' = \beta M + G$, with $G$ = `grad` and $M$ = `momentum`, both `(m, n)`.
2. Direction: $Z = G + \beta M'$ if `nesterov`, else $Z = M'$.
3. Normalise: $X_0 = Z / (\lVert Z \rVert_F + 10^{-7})$. If $m > n$, work with $X_0^\top$ instead (so $X X^\top$ is the smaller Gram matrix) and transpose back at the end.
4. Iterate `steps` times, with $(a, b, c) = (3.4445, -4.7750, 2.0315)$:
   $$A = X X^\top, \qquad X \leftarrow a X + (bA + cA^2) X$$
5. Scale: `update` $= X \cdot \sqrt{\max(1, m/n)}$.

Return `update` `(m, n)` and $M'$ `(m, n)`, both float64.
"""

STARTER = '''import numpy as np

def muon_update(grad, momentum, beta=0.95, steps=5, nesterov=True):
    """Return (update (m, n), new_momentum (m, n))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

A_, B_, C_ = 3.4445, -4.7750, 2.0315        # quintic coefficients tuned to push singular values toward 1 fast

def _newton_schulz(Z, steps):
    X = Z / (np.linalg.norm(Z) + 1e-7)      # Frobenius norm >= spectral norm, so singular values start in [0, 1]
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                             # keep the Gram matrix X X^T as small as possible
    for _ in range(steps):
        A = X @ X.T
        X = A_ * X + (B_ * A + C_ * A @ A) @ X
    return X.T if tall else X

def muon_update(grad, momentum, beta=0.95, steps=5, nesterov=True):
    G, M = np.asarray(grad, dtype=float), np.asarray(momentum, dtype=float)
    M_new = beta * M + G
    Z = G + beta * M_new if nesterov else M_new
    m, n = G.shape
    return _newton_schulz(Z, steps) * np.sqrt(max(1.0, m / n)), M_new
'''

HINTS = [
    'Update the momentum first; the Nesterov direction uses the **new** buffer: `G + beta * M_new`.',
    'Divide by the Frobenius norm before iterating: it bounds the spectral norm, so every singular value starts in $[0, 1]$, where the iteration converges.',
    'Each iteration is three matmuls: `A = X @ X.T`, then `X = a*X + (b*A + c*A@A) @ X`. Transpose tall matrices before the loop and back after, then multiply by `sqrt(max(1, m/n))`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([[2.0, 0.0], [0.0, 1.0]]), np.zeros((2, 2)), 0.9, 5, False),
    (np.array([[1.0, 2.0, 0.0]]), np.zeros((1, 3))),
    (_r.normal(size=(4, 6)), _r.normal(size=(4, 6)) * 0.1),
    (_r.normal(size=(8, 3)), _r.normal(size=(8, 3)), 0.9, 5, True),
    (_r.normal(size=(5, 5)), _r.normal(size=(5, 5)), 0.95, 1, False),
    (_r.normal(size=(6, 4)) @ np.diag([10.0, 1.0, 0.1, 0.01]), np.zeros((6, 4)), 0.0, 8, False),
    (np.zeros((3, 2)), np.ones((3, 2)), 0.5, 3, True),
]

EXPLANATION = r"""
## What the iteration does
Write $X = U S V^\top$. Then $X X^\top X = U S^3 V^\top$ and $(X X^\top)^2 X = U S^5 V^\top$, so each step applies the scalar polynomial $\varphi(s) = a s + b s^3 + c s^5$ to every singular value while leaving $U$ and $V$ alone. The coefficients are chosen so that $\varphi$ maps the whole interval $(0, 1]$ into a band around 1 within about five steps. It doesn't converge to exactly 1 (the band is roughly $[0.7, 1.2]$): Muon trades exactness for speed, and that turns out not to hurt training. Test 6 has singular values spread over three orders of magnitude; after 8 steps (and before the $\sqrt{m/n} \approx 1.22$ shape factor) they all lie between about 0.98 and 1.13.

## Why orthogonalise
Gradients of weight matrices tend to be close to low-rank: a few directions dominate. SGD and Adam keep the update mostly in those directions. The orthogonal factor $UV^\top$ is the steepest-descent direction under the spectral norm, and it gives every singular direction a unit-sized step.

## Cost
For an $m \times n$ matrix with $m \le n$, each step costs $O(m^2 n)$ in three matmuls, which run well in bf16 on tensor cores. Transposing tall matrices keeps the Gram matrix at the small dimension.

## Common bugs
- Nesterov with the old momentum buffer instead of the updated one.
- Normalising by the spectral norm estimate from a single power-iteration step without a safety margin (singular values above 1 make the quintic diverge); the Frobenius norm is always safe.
- Forgetting the transpose for tall matrices: still correct, but the Gram matrix becomes $m \times m$.
- Applying Muon to embeddings, biases and gains: in practice those stay on AdamW.

## Interview follow-ups
- Why is the shape-dependent factor $\sqrt{\max(1, m/n)}$ there, and how does it relate to keeping the update's RMS consistent across layers (as in $\mu$P)?
- Muon needs no second-moment buffer. How much optimizer memory does that save relative to AdamW?
"""
