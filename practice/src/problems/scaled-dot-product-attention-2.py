import numpy as np

META = dict(title='Multi-head causal self-attention', tags=['Attention', 'Transformers', 'Linear Algebra'], diff='Hard', runtime='numpy',
            lessons=['aieng:05.02', 'aieng:05.01'], related=['scaled-dot-product-attention', 'softmax-rows', 'layer-norm'])

STATEMENT = r"""
A transformer's self-attention layer is a handful of matrix products around one softmax. Implement the full multi-head version, batched, with a causal mask, using only NumPy.

Write `mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True)`.

## Shapes
- `X`: `(B, T, D)`: a batch of $B$ sequences of $T$ tokens with model width $D$.
- `Wq`, `Wk`, `Wv`, `Wo`: `(D, D)` each. `n_heads` $= H$ divides $D$; head width $d_h = D / H$.

## Computation
1. Projections: $Q = XW_q$, $K = XW_k$, $V = XW_v$, each `(B, T, D)`.
2. Split heads: head $h$ uses feature columns $h\,d_h$ to $(h+1)\,d_h - 1$ of $Q$, $K$, $V$.
3. Per head: $S = Q_h K_h^\top / \sqrt{d_h}$, shape `(B, T, T)`. If `causal`, position $i$ may only attend to positions $j \le i$: set $S_{ij} = -\infty$ for $j > i$.
4. $A = \operatorname{softmax}(S)$ over the last axis (subtract the row max for stability), then $O_h = A V_h$.
5. Concatenate heads back along the feature axis in head order, giving `(B, T, D)`, and apply the output projection: $Y = O\,W_o$.

## Output
A tuple `(Y, A)`: `Y` of shape `(B, T, D)` and the attention weights `A` of shape `(B, H, T, T)`.

## Requirements
No Python loops over the batch, heads or positions: reshape and transpose instead.
"""

STARTER = '''import numpy as np

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """X: (B, T, D); W*: (D, D). Return (Y (B, T, D), attention weights (B, H, T, T))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    X = np.asarray(X, dtype=float)
    B, T, D = X.shape
    dh = D // n_heads

    def split(M):
        # (B, T, D) -> (B, T, H, dh) -> (B, H, T, dh): head h owns columns h*dh .. (h+1)*dh - 1
        return M.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)

    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)            # (B, H, T, T)
    if causal:
        future = np.triu(np.ones((T, T), dtype=bool), k=1)    # True where j > i
        S = np.where(future, -np.inf, S)                      # diagonal is always allowed, so no row is all -inf
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(axis=-1, keepdims=True)
    O = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D)        # merge heads back in order
    return O @ Wo, A
'''

HINTS = [
    'Do the three projections on the full width first. Splitting into heads is then just a reshape: `(B, T, D)` to `(B, T, H, d_h)`, followed by moving the head axis next to the batch axis.',
    'With `Q`, `K`, `V` shaped `(B, H, T, d_h)`, `Q @ K.transpose(0, 1, 3, 2)` gives all scores at once. The causal mask is `np.triu(ones((T, T), bool), k=1)`, broadcast over batch and heads.',
    'After the softmax, `(A @ V)` is `(B, H, T, d_h)`. Undo the split with `.transpose(0, 2, 1, 3).reshape(B, T, D)` (transpose **before** reshape), then multiply by `Wo`.',
]

_r = np.random.default_rng(103)
_D = 8
_W = [_r.normal(size=(_D, _D)) / np.sqrt(_D) for _ in range(4)]
_X1 = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
_I2 = np.eye(2)
TESTS = [
    (_X1, _I2, _I2, _I2, _I2, 1),
    (_X1, _I2, _I2, _I2, _I2, 2),
    (_X1, _I2, _I2, _I2, _I2, 1, False),
    (np.array([[[2.0, -1.0, 0.5, 3.0]]]), *[np.eye(4)] * 4, 2),
    (_r.normal(size=(2, 5, _D)), *_W, 2),
    (_r.normal(size=(3, 4, _D)), *_W, 4, False),
    (_r.normal(size=(1, 6, _D)) * 30, *_W, 1),
    (_r.normal(size=(2, 3, _D)), *_W, 8),
]

EXPLANATION = r"""
## Why the reshape splits heads correctly
`reshape(B, T, H, d_h)` on a C-ordered `(B, T, D)` array cuts the last axis into $H$ consecutive blocks of $d_h$ features, exactly "head $h$ owns columns $h d_h$ to $(h+1)d_h - 1$". Transposing to `(B, H, T, d_h)` lets one batched matmul handle every (batch, head) pair. Merging must undo the steps in reverse order: transpose back to `(B, T, H, d_h)`, then reshape. Reshaping a `(B, H, T, d_h)` array straight to `(B, T, D)` interleaves tokens and heads; it has the right shape and the wrong numbers.

## Causal mask
`np.triu(..., k=1)` marks $j > i$. Since $j = i$ is always allowed, every row keeps at least one finite score, so after subtracting the row max the softmax never sees an all-$-\infty$ row. Test 4 has $T = 1$: each token attends only to itself, so attention is the identity and $Y = XW_vW_o$.

## Complexity
Projections cost $O(BTD^2)$; scores and the weighted sum cost $O(BT^2D)$, and the `(B, H, T, T)` weights need $O(BHT^2)$ memory. The $T^2$ term is what FlashAttention avoids storing.

## Common bugs
- Scaling by $\sqrt D$ instead of $\sqrt{d_h}$.
- Masking with $j \ge i$ (blocking the diagonal), or using `np.tril` and masking the past.
- Merging heads with a reshape before the transpose.
- Skipping the max subtraction. Test 7 multiplies the inputs by 30, so scores reach the thousands and `np.exp` overflows to `inf`, giving `nan` weights.

## Interview follow-ups
- What changes at inference time with a KV cache? (Each new token computes only its own query row against cached $K, V$: $O(TD)$ per step instead of recomputing $O(T^2D)$.)
- How does grouped-query attention change the shapes? ($K$ and $V$ get fewer heads than $Q$; each group of query heads shares one key/value head, shrinking the cache.)
"""
