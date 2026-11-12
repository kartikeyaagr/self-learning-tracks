import numpy as np

META = dict(title='Weighted, masked log-sum-exp', tags=['Activation Functions', 'Loss Functions'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:02.02', 'ieng:00.03'], related=['softmax-rows', 'cross-entropy-logits'])

STATEMENT = r"""
Log-sum-exp is the workhorse behind log-softmax, cross-entropy, mixture likelihoods and attention masking. The weighted version is

$$\mathrm{LSE}_b(a) = \log \sum_i b_i\, e^{a_i}$$

Write `logsumexp(a, axis=-1, b=None)` that reduces along `axis`.

## Inputs
- `a`: float array, at least 1-D. Entries are finite or `-inf`. A `-inf` entry is **masked out**: $e^{-\infty} = 0$. There are no `nan` or `+inf` entries.
- `axis`: the axis to reduce (negative values allowed).
- `b`: `None` (all weights 1) or a non-negative float array with the same shape as `a`. A zero weight drops that term.

## Output
- An `np.ndarray` with `axis` removed (0-d for a 1-D input).
- A slice where every term is dropped (all `-inf`, or all zero weight) gives `-inf`, never `nan`, and no runtime warnings.
- Must be accurate for magnitudes like $\pm 10^4$, where `np.exp` alone overflows or underflows to 0.
- Don't call `scipy`.
"""

STARTER = '''import numpy as np

def logsumexp(a, axis=-1, b=None):
    """log(sum(b * exp(a))) along axis. Masked (-inf) or zero-weight terms are dropped."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def logsumexp(a, axis=-1, b=None):
    a = np.asarray(a, dtype=float)
    if b is not None:
        with np.errstate(divide='ignore'):
            a = a + np.log(np.asarray(b, dtype=float))   # b * e^a = e^(a + log b); b = 0 becomes -inf
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)    # an all -inf slice: shift by 0 so exp gives 0, not nan
    s = np.exp(a - m).sum(axis=axis, keepdims=True)
    with np.errstate(divide='ignore'):
        out = np.log(s) + m                  # log(0) = -inf for empty slices
    return np.asarray(np.squeeze(out, axis=axis))
'''

HINTS = [
    'The usual shift still works: $\\log\\sum_i e^{a_i} = m + \\log\\sum_i e^{a_i - m}$ for any finite $m$. Weights can be folded into the exponent: $b_i e^{a_i} = e^{a_i + \\log b_i}$.',
    'If every entry of a slice is `-inf`, its max is `-inf` and `a - m` becomes `-inf - (-inf) = nan`. Replace a non-finite max by 0 before subtracting.',
    'Wrap `np.log(b)` and the final `np.log(s)` in `np.errstate(divide="ignore")`: both legitimately see zeros. Reduce with `keepdims=True`, then `np.squeeze(out, axis=axis)`.',
]

_r = np.random.default_rng(13)
_masked = _r.normal(size=(4, 6)) * 50
_masked[_r.random((4, 6)) < 0.4] = -np.inf
_masked[2] = -np.inf
TESTS = [
    (np.array([1.0, 2.0, 3.0]), -1, None),
    (np.array([[0.0, 0.0], [1000.0, 1000.0]]), 1, None),
    (np.array([[-np.inf, -np.inf], [-np.inf, 0.0]]), 1, None),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), 0, None),
    (np.array([1.0, 2.0, 3.0]), 0, np.array([0.5, 0.0, 2.0])),
    (np.array([5.0, 7.0]), 0, np.array([0.0, 0.0])),
    (np.array([-1e4, -1e4 + 1.0, -1e4 - 3.0]), -1, None),
    (_masked, 1, None),
    (_r.normal(size=(3, 4, 5)) * 30, 1, _r.uniform(0, 2, size=(3, 4, 5))),
]

EXPLANATION = r"""
## Why the shift works
$\log\sum_i e^{a_i} = \log\left(e^m \sum_i e^{a_i - m}\right) = m + \log\sum_i e^{a_i - m}$. With $m = \max_i a_i$, every exponent is $\le 0$ (no overflow) and the largest term is exactly 1, so the sum is at least 1 and its log can't underflow to $-\infty$.

## Masks and weights
Folding $\log b_i$ into the exponent turns weighting into an additive shift and a zero weight into a mask. That's exactly how attention masks work: add `-inf` to forbidden logits before softmax. The only special case left is a slice with nothing in it, where $\log 0 = -\infty$ is the right answer and the max-shift would produce `nan` without the guard.

## Complexity
Two passes over the data (max, then exp-sum): $O(n)$ time, $O(n)$ temporary memory.

## Common bugs
- `np.log(np.sum(np.exp(a)))`: overflows at $a \approx 710$ and underflows to `-inf` at $a \approx -745$.
- Shifting by the global max instead of the per-slice max. A slice far below the global max underflows to 0.
- Shifting by the max of `a` but forgetting the weights: a huge `a` with weight 0 then sets $m$, and every other term underflows.

## Interview follow-ups
- Derive $\partial\,\mathrm{LSE}(a)/\partial a = \mathrm{softmax}(a)$. Why does that make LSE a "soft max"?
- How do you compute LSE in one streaming pass when the data arrives in chunks (the online-softmax trick in FlashAttention)?
"""
