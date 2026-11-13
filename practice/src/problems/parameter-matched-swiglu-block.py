import numpy as np

META = dict(title='SwiGLU feed-forward with a parameter-matched width', tags=['Transformers', 'Neural Networks'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:12.02', 'ieng:02.03'],
            related=['norm-gated-linear-transform', 'rmsnorm-forward-pass', 'transformer-training-flop-estimator'])

STATEMENT = r"""
A classic transformer MLP has two matrices, $d \to 4d \to d$: $8d^2$ parameters. A SwiGLU MLP has **three** (gate, up, down), so to keep the parameter count the same its hidden width shrinks to about $\tfrac{2}{3} \cdot 4d = \tfrac{8}{3}d$, then is rounded **up** to a multiple of a hardware-friendly number.

Write `swiglu_block(x, W_gate, W_up, W_down, multiple_of)`.

## Forward pass
- `x`: `float64` `(..., d)`. `W_gate`, `W_up`: `(d, h)`. `W_down`: `(h, d)`.
- $\mathrm{silu}(t) = t\,\sigma(t) = \dfrac{t}{1 + e^{-t}}$; it must not overflow or warn for $|t|$ up to 1000.
- $y = \bigl(\mathrm{silu}(x W_\text{gate}) \odot (x W_\text{up})\bigr) W_\text{down}$, shape `(..., d)`.

## Sizing report
For this $d$ (from `x.shape[-1]`) and the given positive int `multiple_of`:
- `hidden`: the recommended SwiGLU width $h^\* = \texttt{multiple\_of} \cdot \left\lceil \lfloor 8d/3 \rfloor / \texttt{multiple\_of} \right\rceil$ (integer arithmetic).
- `params`: parameters of **these** weights, $3 d h$ where $h$ = `W_gate.shape[1]` (no biases).
- `dense_params`: $8d^2$, the classic $4d$ MLP.
- `ratio`: float, `params / dense_params`.

The test weights do not always use $h^\*$, so read $h$ from the weights for `params`.

## Output
A dict `{'y': ndarray, 'hidden': int, 'params': int, 'dense_params': int, 'ratio': float}`.
"""

STARTER = '''import numpy as np

def swiglu_block(x, W_gate, W_up, W_down, multiple_of):
    """Return {'y', 'hidden', 'params', 'dense_params', 'ratio'}."""
    d = x.shape[-1]
    h = 4 * d                                   # TODO: a SwiGLU block is not 4d wide
    y = np.maximum(x @ W_gate, 0) @ W_down      # TODO: no gate, wrong activation
    return {'y': y, 'hidden': h, 'params': 2 * d * h, 'dense_params': 8 * d * d, 'ratio': 2 * d * h / (8 * d * d)}
'''

SOLUTION = '''import numpy as np

def _silu(t):
    # t * sigmoid(t) with exp of a non-positive number only, so nothing overflows
    e = np.exp(-np.abs(t))
    sig = np.where(t >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return t * sig

def swiglu_block(x, W_gate, W_up, W_down, multiple_of):
    d, h = x.shape[-1], W_gate.shape[1]
    y = (_silu(x @ W_gate) * (x @ W_up)) @ W_down
    base = 8 * d // 3                                        # floor(2/3 * 4d)
    hidden = multiple_of * (-(-base // multiple_of))         # integer ceil-division, then scale back up
    params, dense = 3 * d * h, 8 * d * d
    return {'y': y, 'hidden': hidden, 'params': params, 'dense_params': dense, 'ratio': params / dense}
'''

HINTS = [
    'The gate and up projections are independent matmuls of the same input; only the gate goes through SiLU, then they are multiplied elementwise.',
    'A stable sigmoid: with $e = e^{-|t|}$, $\\sigma(t) = 1/(1+e)$ for $t \\ge 0$ and $e/(1+e)$ for $t < 0$. Then $\\mathrm{silu}(t) = t\\,\\sigma(t)$.',
    'Round up with integers only: `multiple_of * (-(-base // multiple_of))`, with `base = 8 * d // 3`. Floats risk off-by-one at exact multiples.',
]

_r = np.random.default_rng(11)


def _w(d, h, s=0.3):
    return _r.normal(size=(d, h)) * s, _r.normal(size=(d, h)) * s, _r.normal(size=(h, d)) * s


TESTS = [
    (np.array([[1.0, -1.0, 0.5]]), np.ones((3, 2)), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.ones((2, 3)), 4),
    (np.array([[0.0, 2.0, 0.0, 0.0]]), *_w(4, 3), 1),
    (_r.normal(size=(2, 5, 6)), *_w(6, 16), 16),
    (_r.normal(size=(7, 12)), *_w(12, 32), 8),
    (np.array([[1000.0, -1000.0]]), np.eye(2), np.eye(2), np.eye(2), 2),
    (_r.normal(size=(3, 9)), *_w(9, 24), 24),
    (_r.normal(size=(1, 48)), *_w(48, 128, 0.1), 64),
    (_r.normal(size=(4, 3, 2, 10)), *_w(10, 7), 3),
]

EXPLANATION = r"""
## Why it works
A dense MLP with width $4d$ has $2 \cdot d \cdot 4d = 8d^2$ weights. A SwiGLU MLP with width $h$ has $3dh$. Setting $3dh = 8d^2$ gives $h = \tfrac{8}{3}d$. Rounding up to a multiple of 64 or 256 makes the matmul tiles divide evenly on tensor cores, at the cost of slightly more parameters than the dense block: the `ratio` is a little above 1.

For example, $d = 4096$ gives $\lfloor 8 \cdot 4096 / 3 \rfloor = 10922$, rounded up to $11008$ with `multiple_of = 256`, the width used by a well-known 7B model.

## The gate
$\mathrm{silu}(xW_\text{gate})$ acts as a soft, input-dependent mask on $xW_\text{up}$. The product of two linear functions makes the block quadratic in $x$ before the activation saturates, which empirically trains better than GELU at matched parameters and FLOPs.

## Complexity
Three matmuls: $O(n\,d\,h)$ for $n$ tokens, i.e. $6dh$ FLOPs per token, the same as the dense block when parameters match.

## Common bugs
- Applying SiLU to the up projection instead of the gate, or to both.
- `1 / (1 + np.exp(-t))` for $t = -1000$ (test 5): the forward value happens to come out right, but NumPy warns about overflow, and the same expression's gradient is `inf / inf = NaN`. The two-branch form never exponentiates a positive number.
- Rounding with floats: `math.ceil(8 * d / 3 / m) * m` is fine here but differs from the integer formula when $8d/3$ is just below a multiple.

## Interview follow-ups
- Why can the gate and up matrices be fused into one `(d, 2h)` matmul, and how does that interact with tensor parallelism? (Each rank must get matching column slices of both halves.)
- GeGLU vs SwiGLU vs ReGLU: what changes, and does the parameter matching change? (Only the activation; the sizing is the same.)
"""
