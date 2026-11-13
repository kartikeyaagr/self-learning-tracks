import numpy as np

META = dict(title='Coordinate check: diagnose width scaling for muP', tags=['Scaling Laws', 'Training'], diff='Medium',
            runtime='numpy', lessons=[],
            related=['power-law-log-fit', 'orthogonalize-muon-momentum-update', 'fit-learning-rate-and-batch-size-scaling'])

STATEMENT = r"""
Under the **maximal update parametrization** ($\mu$P), a model's hyperparameters tuned at a small width transfer to large widths. The standard sanity check is a *coordinate check*: train the model for a few steps at several widths and record the root-mean-square size of each layer's activations (or their change since initialisation). In a correct $\mu$P setup every curve is flat in width; in standard parametrization some of them grow or shrink as a power of the width, which is exactly what breaks hyperparameter transfer.

Write `coord_check(widths, rms, tol)` that fits those power laws and flags the offending layers.

## Inputs
- `widths`: float64 `(W,)`, distinct positive widths in increasing order, $W \ge 2$. The first is the **base width** $w_0$.
- `rms`: float64 `(L, W)`, strictly positive. Row $l$ holds layer $l$'s measured RMS at each width.
- `tol`: float, the largest slope magnitude still considered flat.

## Per layer
1. Fit $\log_2 \text{rms}_{l} = c_l + s_l \log_2 w$ by ordinary least squares with an intercept. $s_l$ is the **width exponent**: the RMS scales like $w^{s_l}$.
2. Classify: `"stable"` if $|s_l| \le$ `tol`, `"grows"` if $s_l >$ `tol`, `"shrinks"` if $s_l < -$`tol`.
3. The multiplier that would cancel the trend at width $w$ is $(w / w_0)^{-s_l}$ (1 at the base width).

## Output
A dict:

| key | value |
|---|---|
| `"slopes"` | float64 `(L,)` |
| `"verdict"` | list of `L` strings |
| `"multipliers"` | float64 `(L, W)` |
| `"transferable"` | bool, `True` iff every layer is `"stable"` |
| `"worst"` | int, the layer with the largest $|s_l|$ (lowest index on an exact tie) |
"""

STARTER = '''import numpy as np

def coord_check(widths, rms, tol):
    """Return a dict: slopes, verdict, multipliers, transferable, worst."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def coord_check(widths, rms, tol):
    x = np.log2(np.asarray(widths, dtype=float))
    y = np.log2(np.asarray(rms, dtype=float))                 # (L, W)
    xc = x - x.mean()
    slopes = (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)   # one OLS slope per row
    verdict = ["stable" if abs(s) <= tol else ("grows" if s > tol else "shrinks") for s in slopes]
    ratio = np.asarray(widths, dtype=float) / widths[0]
    multipliers = ratio[None, :] ** -slopes[:, None]
    return {"slopes": slopes, "verdict": verdict, "multipliers": multipliers,
            "transferable": all(v == "stable" for v in verdict), "worst": int(np.argmax(np.abs(slopes)))}
'''

HINTS = [
    'Take $\\log_2$ of both widths and RMS values; a power law becomes a straight line and its exponent is the slope.',
    'All $L$ slopes come from one matrix-vector product: centre $x = \\log_2 w$, centre each row of $y$, then `slopes = y_centred @ xc / (xc @ xc)`.',
    'Broadcast `(widths / widths[0])[None, :] ** -slopes[:, None]` for the multipliers, and use `np.argmax(np.abs(slopes))` (first maximum) for the worst layer.',
]

_w = np.array([128.0, 256.0, 512.0, 1024.0, 2048.0])
_r = np.random.default_rng(9)
_noise = np.exp(_r.normal(0.0, 0.03, size=(4, 5)))
TESTS = [
    (np.array([1.0, 2.0, 4.0]), np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]), 0.1),
    (np.array([64.0, 256.0]), np.array([[2.0, 1.0], [3.0, 3.1]]), 0.05),
    (_w, np.vstack([np.full(5, 0.9), 0.8 * (_w / 128) ** 0.5, 1.1 * (_w / 128) ** -0.5, 1.0 * (_w / 128) ** 1.0]) * _noise, 0.1),
    (_w, np.vstack([np.full(5, 0.9), np.full(5, 1.2), np.full(5, 0.7), np.full(5, 1.0)]) * _noise, 0.1),
    (_w, np.vstack([np.full(5, 0.9), np.full(5, 1.2), np.full(5, 0.7), np.full(5, 1.0)]) * _noise, 0.01),
    (np.array([32.0, 64.0, 128.0, 256.0]), np.array([[1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]]), 0.2),
    (np.array([100.0, 300.0, 900.0]), np.array([[0.5, 0.52, 0.49], [1e-3, 3e-3, 9.2e-3]]), 0.15),
]

EXPLANATION = r"""
## What the slopes mean
In standard parametrization with a width-independent learning rate, each hidden-layer update changes activations by an amount that grows with fan-in (a sum of $w$ correlated terms), so after a few steps the curves fan out like $w^{1/2}$ or $w^{1}$, and the output logits blow up. $\mu$P rescales initialisation, learning rates and the output multiplier per layer type so that every activation's update is $\Theta(1)$ in width. A coordinate check with all slopes near 0 is strong evidence the implementation is right; one slope near $+1$ points straight at the layer whose learning rate or multiplier is missing a $1/w$.

## Why fit in log space
A power law $w^{s}$ is a straight line in $\log$-$\log$ coordinates, and OLS on the logs treats a doubling and a halving symmetrically. Base 2 is convenient because widths are usually powers of 2, but the slope is base-independent. Test 3 has noise on top of exponents $0, \tfrac12, -\tfrac12, 1$; tests 4 and 5 show the same flat measurements passing with a loose tolerance and some failing with a strict one, which is why a coordinate check needs a sensible threshold.

## Complexity
$O(LW)$; the slopes are one matrix-vector product.

## Common bugs
- Fitting `rms` against `widths` linearly: the fit then depends on the absolute scale of the widths.
- Using the last width as the base for the multipliers.
- Testing the signed slope (`s <= tol` means stable): every shrinking layer then passes, like the second layer of test 6.

## Interview follow-ups
- Under $\mu$P with Adam, how should the learning rate of a hidden matrix scale with width, and why is it different from SGD?
- Coordinate checks look at a few steps only. What could still go wrong with hyperparameter transfer at long training horizons?
"""
