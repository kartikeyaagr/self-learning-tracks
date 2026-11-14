import numpy as np

META = dict(title='SmoothQuant: migrate activation outliers into the weights', tags=['Quantization'], diff='Medium', runtime='numpy',
            related=['int8-matmul-with-per-channel-scales'])

STATEMENT = r"""
For $Y = XW$ with `X (n, c_in)` and `W (c_in, c_out)`, SmoothQuant rescales each input channel $j$:

$$s_j = \frac{\max_i |X_{ij}|^{\alpha}}{\max_k |W_{jk}|^{1-\alpha}}, \qquad \hat X = X\,\mathrm{diag}(s)^{-1}, \quad \hat W = \mathrm{diag}(s)\,W$$

so $\hat X \hat W = XW$ exactly. Clamp each max to at least `1e-5` before the powers.

Write `smoothquant(X, W, alpha)`:

- Compute `s` and the smoothed tensors.
- Quantise a tensor per-tensor symmetric int8: $q = \mathrm{clip}(\mathrm{round}(t / \delta), -127, 127)$, $\delta = \max|t| / 127$ (1 if zero), dequantise $q\delta$.
- `err_plain` = $\|XW - Q(X)Q(W)\|_F$; `err_smooth` = $\|XW - Q(\hat X)Q(\hat W)\|_F$.

Return `{"s": s, "err_plain": float, "err_smooth": float, "act_range": [max|X|, max|X̂|]}`.
"""

STARTER = '''import numpy as np

def smoothquant(X, W, alpha):
    """Return smoothing factors and the int8 errors with and without smoothing."""
    X, W = np.asarray(X, float), np.asarray(W, float)
    s = np.ones(X.shape[1])
    return {"s": s, "err_plain": 0.0, "err_smooth": 0.0, "act_range": [float(np.abs(X).max())] * 2}
'''

SOLUTION = '''import numpy as np

def _q8(t):
    d = np.abs(t).max() / 127
    d = d if d > 0 else 1.0
    return np.clip(np.round(t / d), -127, 127) * d

def smoothquant(X, W, alpha):
    X, W = np.asarray(X, float), np.asarray(W, float)
    ax = np.maximum(np.abs(X).max(0), 1e-5)
    aw = np.maximum(np.abs(W).max(1), 1e-5)
    s = ax ** alpha / aw ** (1 - alpha)
    Xs, Ws = X / s, W * s[:, None]
    ref = X @ W
    return {"s": s, "err_plain": float(np.linalg.norm(ref - _q8(X) @ _q8(W))),
            "err_smooth": float(np.linalg.norm(ref - _q8(Xs) @ _q8(Ws))),
            "act_range": [float(np.abs(X).max()), float(np.abs(Xs).max())]}
'''

HINTS = [
    'Activation maxima are per column of X (input channel); weight maxima are per row of W (the same channel).',
    'Dividing X by s and multiplying W by s leaves the product unchanged, so only the quantisation error changes.',
    'alpha = 0.5 balances the difficulty between the two; larger alpha moves more of it into the weights.',
]

def _case(seed, n, cin, cout, outl):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, cin)); X[:, outl] *= 50
    return X, r.normal(size=(cin, cout)) * 0.2

TESTS = [
    _case(1, 32, 8, 6, [1]) + (0.5,),
    _case(2, 64, 16, 8, [2, 9]) + (0.5,),
    _case(2, 64, 16, 8, [2, 9]) + (0.8,),
    _case(2, 64, 16, 8, [2, 9]) + (0.0,),
    _case(3, 16, 4, 4, []) + (0.5,),
]

EXPLANATION = r"""
## The problem it solves
Weights are easy to quantise; activations in large models are not, because a few channels have values 50 to 100 times larger than the rest. With one int8 scale for the whole activation tensor, those outliers force a huge step size and the ordinary channels collapse to a few codes.

## The trick
The product $XW$ is unchanged if you divide a channel of $X$ by $s_j$ and multiply the matching row of $W$ by $s_j$. Choosing $s_j$ from both maxima spreads the outlier between activation and weight, so both become quantisable with simple per-tensor int8 and fast integer kernels. The scale can be folded into the previous LayerNorm, so it costs nothing at inference.

## Choosing alpha
$\alpha = 0$ leaves activations unchanged; $\alpha = 1$ moves all the range into the weights. Around 0.5 works for most models; models with stronger outliers need more.

## Interview follow-ups
- Why can the smoothing factors be folded into the preceding LayerNorm but not after a ReLU in general?
"""
