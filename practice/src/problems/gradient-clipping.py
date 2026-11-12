import numpy as np

META = dict(title='Gradient clipping: value, global norm, adaptive', tags=['Training', 'Optimization'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:02.04'],
            related=['gradient-accumulation', 'adam-optimizer'])

STATEMENT = r"""
Exploding gradients wreck training in a single step. Implement three ways to clip a model's gradients, given as a list of arrays (one per parameter tensor).

Write `clip_gradients(grads, mode, threshold, params=None, eps=1e-3)` that returns the tuple `(clipped, total_norm)`:

- `total_norm`: the **global** L2 norm of the original gradients, $\sqrt{\sum_k \lVert G_k\rVert_F^2}$, as a Python `float`, in every mode.
- `clipped`: a new list of float arrays, same shapes and order as `grads`. Don't modify the inputs.

## Modes (with $c$ = `threshold > 0`)
1. `'value'`: clip every element independently to $[-c, c]$.
2. `'norm'`: rescale all gradients by one shared factor $\min\!\left(1, \dfrac{c}{\text{total\_norm} + 10^{-6}}\right)$.
3. `'agc'` (adaptive gradient clipping): clip each **unit** relative to the size of its weights. `params` is a list of arrays matching `grads`.
   - For an array with `ndim >= 2`, each slice along axis 0 (each row / output unit) is a unit, with norms taken over all its remaining elements. For `ndim <= 1`, the whole array is a single unit.
   - For a unit with gradient $G_u$ and weights $W_u$: let $w = \max(\lVert W_u\rVert, \epsilon)$ and $g = \lVert G_u\rVert$. If $g > c\,w$, replace $G_u$ by $G_u \cdot \dfrac{c\,w}{g}$. Otherwise leave it unchanged.

`grads` has at least one array and every array has at least one element.
"""

STARTER = '''import numpy as np

def clip_gradients(grads, mode, threshold, params=None, eps=1e-3):
    """mode: 'value' | 'norm' | 'agc'. Return (list of clipped arrays, total_norm float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _unit_norms(a):
    """L2 norm of each unit, shaped to broadcast against `a`."""
    if a.ndim <= 1:
        return np.sqrt(np.sum(a * a)).reshape((1,) * a.ndim)
    axes = tuple(range(1, a.ndim))
    return np.sqrt(np.sum(a * a, axis=axes, keepdims=True))

def clip_gradients(grads, mode, threshold, params=None, eps=1e-3):
    grads = [np.asarray(g, dtype=float) for g in grads]
    total = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
    c = threshold
    if mode == 'value':
        return [np.clip(g, -c, c) for g in grads], total
    if mode == 'norm':
        scale = min(1.0, c / (total + 1e-6))          # one factor for every tensor keeps the direction
        return [g * scale for g in grads], total
    out = []
    for g, w in zip(grads, params):                    # 'agc': per-unit, relative to weight size
        wn = np.maximum(_unit_norms(np.asarray(w, dtype=float)), eps)
        gn = _unit_norms(g)
        limit = c * wn
        factor = np.where(gn > limit, limit / np.where(gn > 0, gn, 1.0), 1.0)
        out.append(g * factor)
    return out, total
'''

HINTS = [
    'Compute the global norm first; every mode returns it. `np.sqrt(sum(np.sum(g**2) for g in grads))`.',
    'Value clipping is `np.clip`. Norm clipping is one scalar factor applied to every array, which preserves the overall gradient direction.',
    'For AGC, write a helper that returns per-unit norms with `keepdims=True` (sum over axes 1..ndim-1, or over everything for 0-d/1-d arrays) so the clip factor broadcasts back over each unit.',
]

_r = np.random.default_rng(61)
_W = [_r.normal(size=(4, 3)), _r.normal(size=4) * 0.01, _r.normal(size=(2, 3, 3))]
_G = [_r.normal(size=(4, 3)) * 0.5, _r.normal(size=4), _r.normal(size=(2, 3, 3)) * 3]
TESTS = [
    ([np.array([3.0, -4.0]), np.array([[0.5, 2.0]])], 'value', 1.0),
    ([np.array([3.0, 4.0]), np.array([[0.0, 12.0]])], 'norm', 6.5),
    ([np.array([3.0, 4.0]), np.array([[0.0, 12.0]])], 'norm', 100.0),
    ([np.array([[3.0, 4.0], [0.3, 0.4]]), np.array([1.0, 1.0])], 'agc', 0.1,
     [np.array([[1.0, 0.0], [10.0, 0.0]]), np.array([0.0, 0.0])]),
    (_G, 'agc', 0.05, _W),
    (_G, 'agc', 0.05, _W, 0.5),
    (_G, 'norm', 1.0),
    (_G, 'value', 0.25),
    ([np.zeros((2, 2))], 'agc', 0.01, [np.ones((2, 2))]),
]

EXPLANATION = r"""
## Three different contracts
- **Value** clipping bounds each coordinate but changes the gradient's **direction** when only some coordinates are large.
- **Global norm** clipping (PyTorch's `clip_grad_norm_`) keeps the direction and caps the step length. It's the standard choice for RNNs and transformers: a rare spike can't take a huge step, but normal steps pass through untouched. The total norm it returns is a useful health metric to log.
- **AGC** (from the NFNets paper) bounds the **relative** update: a unit's gradient may be at most $c$ times its weight norm. Large weights can take large gradients; tiny or freshly initialised units (and biases, via $\epsilon$) are protected. It was introduced to train without batch normalisation at large batch sizes.

## Test walk-through
Test 2: the norm is $\sqrt{9 + 16 + 144} = 13$, so with $c = 6.5$ everything halves (minus a hair for the $10^{-6}$). Test 4: row 0 has $\lVert G\rVert = 5 > 0.1 \cdot 1$ and gets scaled to norm 0.1; row 1 has $0.5 \le 0.1 \cdot 10$ and survives; the bias has zero weight norm, so $\epsilon = 10^{-3}$ sets its limit to $10^{-4}$.

## Complexity
$O(P)$ for $P$ total gradient elements in every mode.

## Common bugs
- Norm-clipping each tensor separately with the same threshold. That changes the direction and isn't what `clip_grad_norm_` does.
- Dividing by $g$ when it's 0 in AGC (`nan` from `0/0`).
- Clipping **after** the optimizer's moment updates. Clip the raw gradients before `optimizer.step()`, and after unscaling when you use mixed-precision loss scaling.

## Interview follow-ups
- Does global norm clipping interact with Adam? (Less than with SGD: Adam's normalisation already caps the per-coordinate step at about $\eta$, but clipping still protects the moment estimates from a single spike.)
- How do you compute the global norm across data-parallel shards? (Sum squared norms locally, all-reduce the scalar, then take the root.)
"""
