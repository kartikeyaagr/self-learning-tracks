import numpy as np

META = dict(title='Group-wise int4 weights with protected salient columns', tags=['Quantization'], diff='Medium', runtime='numpy',
            related=['group-wise-int4-quantization', 'affine-uint8-quantize'])

STATEMENT = r"""
Write `int4_protect(W, X, group, n_protect)` for a linear layer $y = XW^\top$ with `W (out, in)` and calibration inputs `X (n, in)`.

1. **Salience.** Input channel $j$'s salience is the mean absolute activation $\mathrm{mean}_i |X_{ij}|$. Protect the `n_protect` most salient channels (ties to the lower index); they stay in full precision.
2. **Quantise.** Split each row of `W` into consecutive groups of `group` input channels (the last group may be shorter). In each group, quantise with 4-bit affine quantisation: $lo = \min(\min w, 0)$, $hi = \max(\max w, 0)$, $scale = (hi - lo)/15$ (1 if equal), $zp = \mathrm{clip}(\mathrm{round}(-lo/scale), 0, 15)$, $q = \mathrm{clip}(\mathrm{round}(w/scale) + zp, 0, 15)$, $\hat w = (q - zp)\,scale$. Groups are computed over all their entries, including protected ones.
3. Put the original values back in the protected columns.

Return `{"protected": sorted list, "W_hat": array, "err": ‖XW^T − XŴ^T‖_F, "err_plain": same error with n_protect = 0}`.
"""

STARTER = '''import numpy as np

def int4_protect(W, X, group, n_protect):
    """Group-wise int4 with salient input channels kept in full precision."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _q4(w):
    lo, hi = min(w.min(), 0.0), max(w.max(), 0.0)
    s = (hi - lo) / 15 if hi > lo else 1.0
    zp = np.clip(np.round(-lo / s), 0, 15)
    return (np.clip(np.round(w / s) + zp, 0, 15) - zp) * s

def _quant(W, group):
    Wh = np.empty_like(W)
    for r in range(W.shape[0]):
        for g in range(0, W.shape[1], group):
            Wh[r, g:g + group] = _q4(W[r, g:g + group])
    return Wh

def int4_protect(W, X, group, n_protect):
    W, X = np.asarray(W, float), np.asarray(X, float)
    sal = np.abs(X).mean(0)
    prot = sorted(np.lexsort((np.arange(len(sal)), -sal))[:n_protect].tolist())
    plain = _quant(W, group)
    Wh = plain.copy()
    Wh[:, prot] = W[:, prot]
    ref = X @ W.T
    return {"protected": prot, "W_hat": Wh, "err": float(np.linalg.norm(ref - X @ Wh.T)),
            "err_plain": float(np.linalg.norm(ref - X @ plain.T))}
'''

HINTS = [
    'Salience comes from activations, not weights: a small weight on a huge activation matters more than a large weight on a tiny one.',
    'Quantise every group first, then overwrite the protected columns with their original values.',
    'Use np.lexsort((index, -salience)) to rank with a deterministic tie-break.',
]

def _case(seed, out, inp, n, outliers):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, inp))
    X[:, outliers] *= 20
    return r.normal(size=(out, inp)) * 0.1, X

TESTS = [
    _case(1, 4, 8, 16, [2]) + (4, 1),
    _case(2, 8, 32, 64, [3, 17]) + (8, 2),
    _case(2, 8, 32, 64, [3, 17]) + (8, 0),
    _case(3, 6, 20, 30, [0, 5, 19]) + (6, 3),
    _case(4, 3, 10, 12, []) + (4, 2),
]

EXPLANATION = r"""
## Why a few channels matter
In large language models a small number of input channels carry activations tens of times larger than the rest. Quantisation error in the weights of those channels is multiplied by those activations, so they dominate the output error. Keeping about 1% of channels in higher precision removes most of the damage, which is the observation behind LLM.int8() and AWQ.

## Groups
A per-row scale has to cover the largest weight in the row. Groups of 64 or 128 input channels give each slice its own scale and zero point, at a cost of about 0.25 to 0.5 extra bits per weight.

## Practical versions
Mixed-precision columns complicate kernels, so AWQ instead scales salient channels up before quantisation (and the activations down by the same factor), keeping a uniform int4 format with most of the benefit.

## Interview follow-ups
- Why do you need calibration data to choose the protected channels?
"""
