import numpy as np

META = dict(title='RMSNorm and gated MLP block', tags=['Transformers', 'Neural Networks'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:05.04', 'ieng:02.03'], related=['layer-norm', 'softmax-rows', 'sigmoid-stable'])

STATEMENT = r"""
Many modern transformer layers apply the same three-stage feed-forward block to every token: normalise, apply a **gated** linear expansion, project back and add the input (a residual connection). Implement it in NumPy for a whole batch at once.

Write `gated_block(x, g, W1, W3, W2, eps=1e-6)`.

## Shapes
- `x`: `(..., d)`, any number of leading axes (for example `(T, d)` or `(B, T, d)`).
- `g`: `(d,)`, the RMSNorm gain. `W1`, `W3`: `(h, d)`. `W2`: `(d, h)`.

## Steps (for every vector $x$ along the last axis)
1. **RMSNorm**: $\displaystyle \mathrm{rms}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_k x_k^2 + \varepsilon}$, then $\hat x = \dfrac{x}{\mathrm{rms}(x)} \odot g$. Note that $\varepsilon$ goes **inside** the square root.
2. **Two projections**: $a = W_1 \hat x$ and $b = W_3 \hat x$, each of length $h$.
3. **Gate**: $u = \mathrm{silu}(a) \odot b$, where $\mathrm{silu}(t) = t\,\sigma(t) = \dfrac{t}{1 + e^{-t}}$.
4. **Project back and add**: $\text{out} = x + W_2 u$.

## Requirements
- Return a `float64` array with the same shape as `x`.
- No loops over tokens or batch entries: use `@` with the right transposes.
- It must not overflow for large pre-activations: `a` values of $\pm 1000$ are valid. ($\sigma(t) = \tfrac12(1 + \tanh(t/2))$ is a stable form.)
"""

STARTER = '''import numpy as np

def gated_block(x, g, W1, W3, W2, eps=1e-6):
    """x: (..., d). RMSNorm(x)*g -> silu(W1 xn) * (W3 xn) -> W2 u, plus residual x. Same shape as x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gated_block(x, g, W1, W3, W2, eps=1e-6):
    x = np.asarray(x, dtype=float)
    # 1. RMSNorm over the feature axis; keepdims so the (..., 1) scale broadcasts back over d.
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    xn = x / rms * g
    # 2. Row-vector convention: (..., d) @ (d, h) applies W to every token at once.
    a, b = xn @ W1.T, xn @ W3.T
    # 3. silu(a) = a * sigmoid(a); the tanh form of sigmoid never overflows.
    u = a * 0.5 * (1.0 + np.tanh(0.5 * a)) * b
    # 4. Back to d features, plus the residual.
    return x + u @ W2.T
'''

HINTS = [
    'Treat it as four small vectorised steps. Each one maps an `(..., k)` array to an `(..., k\')` array, so `@` with a 2-D weight on the right handles any number of leading axes.',
    'The weights are stored as `(out, in)`, as in most frameworks, while the tokens are row vectors. So $W\\hat x$ for every token is `xn @ W.T`. For RMSNorm, reduce with `np.mean(x * x, axis=-1, keepdims=True)`.',
    '`1 / (1 + np.exp(-a))` overflows (with a warning) when `a` is very negative. Use `0.5 * (1 + np.tanh(a / 2))` for the sigmoid, multiply by `a` for silu and by `b` for the gate, then return `x + u @ W2.T`.',
]

_r = np.random.default_rng(71)
_d, _h = 6, 10
_W1, _W3, _W2 = (_r.normal(size=(_h, _d)) / np.sqrt(_d), _r.normal(size=(_h, _d)) / np.sqrt(_d),
                 _r.normal(size=(_d, _h)) / np.sqrt(_h))
TESTS = [
    (np.array([[3.0, 4.0]]), np.ones(2), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([[1.0], [-1.0]])),
    (np.array([[1.0, -1.0], [0.0, 2.0]]), np.array([1.0, 0.5]), np.eye(2), np.ones((2, 2)), np.eye(2)),
    (np.zeros((1, 3)), np.ones(3), np.ones((2, 3)), np.ones((2, 3)), np.ones((3, 2))),
    (np.array([[1.0, 1.0]]), np.array([2000.0, -2000.0]), np.eye(2), np.eye(2), np.eye(2)),
    (np.array([[1e-4, -2e-4, 3e-4]]), np.ones(3), np.eye(3), np.eye(3), np.eye(3), 1e-5),
    (_r.normal(size=(4, _d)), 1.0 + 0.1 * _r.normal(size=_d), _W1, _W3, _W2),
    (_r.normal(size=(2, 3, _d)) * 5.0, np.ones(_d), _W1, _W3, _W2),
]

EXPLANATION = r"""
## Why it works
Each stage is a shape-preserving map on the last axis, so the whole block vectorises over any leading axes.

- **RMSNorm** rescales each token vector to unit root-mean-square and then applies a learned per-feature gain. Unlike LayerNorm it does not subtract the mean, which saves a reduction and works as well in practice. `keepdims=True` keeps the per-token scale as `(..., 1)` so it divides every feature of its own token.
- **Gating**: `silu(a)` decides how much of each hidden unit passes, and `b` carries the content. The product of two linear maps with a smooth gate is more expressive per parameter than a single activation, which is why gated MLPs (SwiGLU) replaced plain ReLU MLPs in most current LLMs.
- **Residual**: adding `x` back means the block only has to learn a correction, and gradients have a direct path through deep stacks.

For the sigmoid, $\sigma(t) = \tfrac12(1+\tanh(t/2))$ is an algebraic identity. `tanh` saturates at $\pm1$ instead of overflowing, so $t = -2000$ gives exactly 0 with no warning.

## Complexity
$O(N \cdot d \cdot h)$ for $N$ tokens, dominated by the three matrix products. Memory is $O(N h)$ for the hidden activations.

## Common bugs
- `xn @ W1` instead of `xn @ W1.T`. With `h == d` it runs and is silently wrong.
- Putting $\varepsilon$ outside the root (`/ (sqrt(ms) + eps)`) or using the L2 norm instead of the RMS (off by $\sqrt d$).
- Applying silu to `b` as well, or to the product.
- Forgetting the residual.

## Interview follow-ups
- Why is the hidden size of a gated MLP usually about $\tfrac{8}{3}d$ rather than $4d$? (There are three weight matrices instead of two; shrinking $h$ keeps the parameter count equal to a $4d$ two-matrix MLP.)
- In inference, `W1` and `W3` are often fused into one `(2h, d)` matrix. Why does that help? (One larger GEMM uses the hardware better than two small ones and reads `xn` once.)
"""
