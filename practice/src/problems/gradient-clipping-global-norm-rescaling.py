import numpy as np

META = dict(title='Clip gradients by their global norm', tags=['Training', 'Optimization'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:02.04'],
            related=['adam-step', 'learning-rate-stability-and-convergence-bounds'])

STATEMENT = r"""
Gradient clipping caps the size of an update without changing its direction. Treat all parameter gradients as one long vector and measure its **global** L2 norm:

$$\lVert g\rVert = \sqrt{\sum_{\text{tensors } k}\ \sum_{\text{entries } i} g_{k,i}^2}$$

If $\lVert g\rVert >$ `max_norm`, multiply **every** tensor by `max_norm / ||g||`; otherwise leave them unchanged.

Write `clip_global_norm(grads, max_norm)` that returns a tuple `(clipped, total_norm)`:
- `clipped`: a list of new float arrays, same length, order and shapes as `grads`.
- `total_norm`: the global norm before clipping, as a Python `float`.

## Rules
- `grads`: a list of float arrays of any shapes (possibly empty arrays, possibly an empty list). `max_norm > 0`.
- The norm must not overflow: entries can be as large as $10^{200}$, whose squares are `inf` in float64. It must also not underflow for entries as small as $10^{-200}$.
- If any entry is `nan` or `±inf`, the step is skipped: return all-zero arrays (same shapes) and `total_norm = inf`.
- An empty list returns `([], 0.0)`. Don't modify the input arrays.
"""

STARTER = '''import numpy as np

def clip_global_norm(grads, max_norm):
    """Rescale a list of gradient arrays so their joint L2 norm is at most max_norm. Return (clipped, norm)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def clip_global_norm(grads, max_norm):
    grads = [np.asarray(g, dtype=float) for g in grads]
    flat = np.concatenate([g.ravel() for g in grads]) if grads else np.zeros(0)
    if not np.all(np.isfinite(flat)):
        return [np.zeros_like(g) for g in grads], float('inf')
    scale = np.max(np.abs(flat), initial=0.0)
    if scale == 0:
        return [g.copy() for g in grads], 0.0
    total = float(scale * np.sqrt(np.sum((flat / scale) ** 2)))   # divide by the max first: no overflow or underflow
    factor = max_norm / total if total > max_norm else 1.0
    return [g * factor for g in grads], total
'''

HINTS = [
    'Flatten and concatenate everything into one vector; the global norm is just its L2 norm. Handle the empty list before calling `np.concatenate`.',
    'Squaring $10^{200}$ overflows. Factor out the largest magnitude $s$: $\\lVert g\\rVert = s\\,\\sqrt{\\sum (g_i/s)^2}$, where every ratio is at most 1.',
    'Order of checks: non-finite entries first (zeros and `inf`), then an all-zero vector (norm 0, no division), then compute the norm and the factor `min(1, max_norm / norm)`. Return new arrays, e.g. `g * factor`.',
]

_r = np.random.default_rng(71)
TESTS = [
    ([np.array([3.0, 4.0])], 1.0),
    ([np.array([1.0, 2.0]), np.array([[2.0], [0.0]])], 10.0),
    ([np.array([3.0]), np.array([[4.0, 0.0], [0.0, 0.0]]), np.zeros(0)], 2.5),
    ([np.array([1e200, -1e200]), np.array([1e199])], 1.0),
    ([np.array([1e-200, 2e-200]), np.array([2e-200])], 1.0),
    ([np.array([1.0, np.nan]), np.array([2.0])], 1.0),
    ([np.array([np.inf]), np.ones((2, 2))], 1.0),
    ([], 1.0),
    ([np.zeros(3), np.zeros((2, 2))], 1.0),
    ([_r.normal(size=(4, 3)) * 5, _r.normal(size=7), _r.normal(size=(2, 2, 2))], 3.0),
]

EXPLANATION = r"""
## Why global, not per-tensor
Clipping each tensor separately changes the update's **direction**: a layer with a small gradient keeps it while a large one is shrunk. Global clipping scales everything by the same factor, so the update points the same way and only its length is capped. That's the behaviour of `clip_grad_norm_` in PyTorch and `clip_by_global_norm` in TensorFlow.

## Overflow-safe norms
$\sqrt{\sum g_i^2}$ overflows when any $|g_i| > 10^{154}$, and underflows to 0 when all $|g_i| < 10^{-162}$, even though the norm itself is representable. Dividing by $s = \max|g_i|$ first keeps every term in $[0, 1]$ and at least one term equal to 1. LAPACK's `nrm2` does a streaming version of the same trick. (`np.linalg.norm` on a flat vector does not.)

## Skipping non-finite steps
One `nan` makes the norm `nan`, and `g * (max_norm / nan)` turns every parameter into `nan` on the next update. Mixed-precision training scalers do what the statement asks: detect it, skip the step, and (there) lower the loss scale.

## Complexity
$O(P)$ in the total number of parameters, plus one concatenated copy.

## Common bugs
- Summing per-tensor norms instead of combining squared norms: $\lVert a\rVert + \lVert b\rVert \ne \sqrt{\lVert a\rVert^2 + \lVert b\rVert^2}$.
- Clipping in place and corrupting the caller's arrays.
- Scaling up gradients that are already below `max_norm`.

## Interview follow-ups
- Why does clipping help with exploding gradients in RNNs, and why doesn't it fix vanishing ones?
- Adaptive clipping (AGC) clips each unit's gradient relative to its weight norm. When is that better than a global threshold?
"""
