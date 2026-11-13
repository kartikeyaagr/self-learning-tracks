import numpy as np

META = dict(title='Temperature, top-k and top-p sampling', tags=['Language Models', 'Sampling', 'Inference'], diff='Medium', runtime='numpy',
            lessons=['ieng:02.04', 'aieng:03.03'], related=['beam-search-decoding', 'softmax-rows', 'perplexity-computation'])

STATEMENT = r"""
Language models generate text by sampling the next token from a probability distribution. Three knobs reshape that distribution first. Write `sample_next(logits, u, temperature=1.0, top_k=0, top_p=1.0)` that returns `(token, probs)`.

## Steps
1. **Temperature.** If `temperature == 0`, the result is greedy: `token` is the index of the largest logit (smallest index on ties) and `probs` is one-hot at that index. Skip the remaining steps. Otherwise compute $p = \mathrm{softmax}(\text{logits} / \text{temperature})$.
2. **Top-k.** If `top_k > 0`, keep only the `top_k` tokens with the largest $p$ (ties go to the smaller index), set the rest to 0 and renormalize.
3. **Top-p (nucleus).** If `top_p < 1`, order the tokens that are still non-zero by probability, highest first (ties to the smaller index). Keep the shortest prefix of that order whose cumulative probability is $\ge$ `top_p` (at least one token), set the rest to 0 and renormalize.
4. **Sample.** With the final distribution `probs`, return the smallest token index $i$ with $\sum_{j \le i} \text{probs}_j > u$, scanning in **vocabulary order**. If rounding leaves no such index, return the largest index with non-zero probability.

## Inputs and output
- `logits`: float array `(V,)`. `u`: a float in $[0, 1)$ that plays the role of the random draw.
- `token`: Python int. `probs`: float array `(V,)` that sums to 1 and is 0 at filtered tokens.
"""

STARTER = '''import numpy as np

def sample_next(logits, u, temperature=1.0, top_k=0, top_p=1.0):
    """Return (sampled token id, final probability vector)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sample_next(logits, u, temperature=1.0, top_k=0, top_p=1.0):
    z = np.asarray(logits, dtype=float)
    if temperature == 0:
        t = int(np.argmax(z))
        probs = np.zeros_like(z)
        probs[t] = 1.0
        return t, probs
    z = z / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")          # descending, ties keep the smaller index first
    if top_k > 0:
        p[order[top_k:]] = 0.0
        p /= p.sum()
    if top_p < 1.0:
        keep = order[p[order] > 0]                 # still-alive tokens, same order
        n = int(np.searchsorted(np.cumsum(p[keep]), top_p)) + 1   # first prefix with cumsum >= top_p
        p[keep[n:]] = 0.0
        p /= p.sum()
    cdf = np.cumsum(p)
    idx = np.flatnonzero(cdf > u)
    token = int(idx[0]) if len(idx) else int(np.flatnonzero(p)[-1])
    return token, p
'''

HINTS = [
    'Sort the token indices once by descending probability with a stable sort: `np.argsort(-p, kind="stable")`. Both top-k and top-p cut a prefix of that order (top-k reorders nothing, so the order stays valid).',
    'For top-p, take the cumulative sum along the sorted order and find the first position where it reaches `top_p`; `np.searchsorted(cumsum, top_p)` gives exactly that index (the `side="left"` default matches "greater or equal").',
    'Sampling with a given `u` is inverse-CDF sampling in vocabulary order: `np.cumsum(probs)` and the first index where it exceeds `u`. Guard the rare case where rounding makes the total slightly below `u`.',
]

_r = np.random.default_rng(53)
_z = _r.normal(size=20) * 2
TESTS = [
    (np.array([2.0, 1.0, 0.0, -1.0]), 0.5),
    (np.array([2.0, 1.0, 0.0, -1.0]), 0.5, 1.0, 2),
    (np.array([2.0, 1.0, 0.0, -1.0]), 0.99, 0.0),
    (np.array([1.0, 3.0, 3.0, 0.0]), 0.6, 1.0, 1),
    (np.array([1.0, 3.0, 3.0, 0.0]), 0.3, 1.0, 0, 0.5),
    (np.array([0.0, 0.0, 0.0, 0.0]), 0.74, 1.0, 3, 0.6),
    (_z, 0.37, 0.7, 0, 0.9),
    (_z, 0.81, 1.5, 5, 0.75),
    (_z, 0.05, 1e-3),
]

EXPLANATION = r"""
## What each knob does
- **Temperature** divides the logits: $T < 1$ sharpens the distribution towards the top token, $T > 1$ flattens it, $T \to 0$ is greedy decoding. Test 9 uses $T = 10^{-3}$, which is numerically almost greedy but still goes through softmax (subtracting the max keeps it from overflowing).
- **Top-k** removes the long tail of unlikely tokens, which is where incoherent continuations come from. But a fixed $k$ is too many when the model is confident and too few when it is not.
- **Top-p** adapts: it keeps the smallest set of tokens covering probability mass `top_p`, so the candidate set shrinks when the model is sure and grows when it is uncertain.

## Why pass `u` in
Sampling is `cdf > u` for a uniform $u$, so passing $u$ makes the function deterministic and testable; a real loop draws `u = rng.random()` per step. Scanning in vocabulary order is a convention; any fixed order gives the same distribution over tokens.

## Ties
Test 4 has two equal logits and `top_k = 1`: the smaller index (1) survives. Test 6 is all ties, so top-k keeps tokens 0-2 and top-p keeps the first two of them (cumulative $1/3 + 1/3 \ge 0.6$).

## Complexity
$O(V \log V)$ for the sort. With only top-k, `np.argpartition` gives $O(V)$.

## Common bugs
- Applying top-p to the probabilities before renormalizing after top-k.
- Keeping tokens while the cumulative sum is $<$ `top_p` and forgetting the token that crosses the threshold, which can leave nothing.
- Dividing by `temperature = 0`.

## Interview follow-ups
- What is min-p sampling and why is it more robust than top-p at high temperature? (Keep tokens with $p \ge p_{\min} \cdot p_{\max}$.)
- How do repetition and frequency penalties modify the logits before these steps?
"""
