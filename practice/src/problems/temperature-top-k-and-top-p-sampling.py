import numpy as np

META = dict(title='Batched sampling with per-request temperature, top-k and top-p', tags=['Inference', 'Language Models', 'Sampling'],
            diff='Medium', runtime='torch', lessons=['ieng:02.04', 'aieng:03.03'],
            related=['text-generation-sampling', 'speculative-decoding-and-token-verification', 'softmax-rows'])

STATEMENT = r"""
A serving engine samples the next token for a whole batch at once, and every request brings its own sampling settings. A Python loop over requests would stall the GPU, so the filtering has to be vectorised across rows that each use a different temperature, `top_k` and `top_p`.

Write `sample_batch(logits, temperature, top_k, top_p, u)` in PyTorch and return `(tokens, probs)`.

## Inputs
- `logits`: float64 `(B, V)`.
- `temperature`: float64 `(B,)`, values $\ge 0$. `top_k`: int64 `(B,)`, `0` means off. `top_p`: float64 `(B,)` in $(0, 1]$, `1.0` means off.
- `u`: float64 `(B,)` in $[0, 1)$, one uniform draw per row (the randomness is passed in so results are reproducible).

## Per row, in this order
1. **Greedy.** If `temperature == 0`: the token is the index of the largest logit (smallest index on ties) and the row of `probs` is one-hot there. Skip the remaining steps for that row.
2. **Temperature.** $p = \mathrm{softmax}(\text{logits} / T)$.
3. **Top-k** (if `top_k > 0`). Rank tokens by $p$, highest first, ties to the smaller index. Keep the first `top_k`, zero the rest, renormalise.
4. **Top-p** (if `top_p < 1`). In the same ranking, keep the shortest prefix whose cumulative probability is $\ge$ `top_p` (at least one token), zero the rest, renormalise.
5. **Sample.** Return the smallest index $i$, in **vocabulary order**, with $\sum_{j \le i} p_j > u$. If rounding leaves no such index, return the largest index with $p > 0$.

## Output
- `tokens`: int64 `(B,)`.
- `probs`: float64 `(B, V)`, the final distribution of each row (sums to 1, exactly 0 at filtered tokens).

No Python loops over the batch.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def sample_batch(logits, temperature, top_k, top_p, u):
    """Return (tokens (B,) int64, probs (B,V) float64)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def sample_batch(logits, temperature, top_k, top_p, u):
    B, V = logits.shape
    greedy = temperature == 0
    t = torch.where(greedy, torch.ones_like(temperature), temperature)   # avoid dividing by 0 on greedy rows
    p = torch.softmax(logits / t[:, None], dim=-1)

    sp, order = torch.sort(p, dim=-1, descending=True, stable=True)      # stable: ties keep the smaller index first
    rank = torch.arange(V)[None, :]
    keep = (top_k[:, None] == 0) | (rank < top_k[:, None])               # top-k in rank space
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True)
    before = sp.cumsum(dim=-1) - sp                                      # mass strictly ahead in the ranking
    keep = keep & ((top_p[:, None] >= 1) | (before < top_p[:, None]))    # shortest prefix reaching top_p
    mask = torch.zeros_like(keep).scatter(1, order, keep)                # back to vocabulary order
    p = p * mask
    p = p / p.sum(dim=-1, keepdim=True)

    onehot = torch.zeros_like(p).scatter(1, logits.argmax(dim=-1, keepdim=True), 1.0)
    p = torch.where(greedy[:, None], onehot, p)

    tok = (p.cumsum(dim=-1) <= u[:, None]).sum(dim=-1)                   # first index whose CDF exceeds u
    last_nz = V - 1 - (p > 0).flip(-1).int().argmax(dim=-1)              # fallback when rounding leaves none
    return torch.where(tok < V, tok, last_nz), p
'''

HINTS = [
    'Sort each row once with `torch.sort(p, descending=True, stable=True)`; both filters are easy in that rank space. Compare `arange(V)` with `top_k[:, None]` for top-k.',
    'For top-p, a ranked token survives when the probability mass strictly ahead of it is still below `top_p`: `cumsum - p < top_p`. Treat `top_p >= 1` as "keep all" explicitly, or floating-point rounding may drop the tail.',
    'Map the rank-space keep mask back with `scatter(1, order, keep)`. For sampling, the number of CDF entries `<= u` is exactly the first index whose CDF exceeds `u`. Handle greedy rows last with `torch.where`.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([[2.0, 1.0, 0.0, -1.0]]), np.array([1.0]), np.array([2]), np.array([1.0]), np.array([0.5])),
    (np.array([[0.0, 3.0, 3.0], [1.0, 2.0, 3.0]]), np.array([0.0, 1.0]), np.array([0, 0]), np.array([1.0, 0.5]),
     np.array([0.9, 0.1])),
    (_r.normal(size=(4, 10)) * 2, np.array([0.7, 1.0, 1.3, 0.0]), np.array([3, 0, 5, 2]), np.array([1.0, 0.8, 0.6, 0.5]),
     _r.uniform(size=4)),
    (np.zeros((2, 5)), np.array([1.0, 1.0]), np.array([2, 0]), np.array([1.0, 0.5]), np.array([0.99, 0.75])),
    (_r.normal(size=(3, 8)), np.array([2.0, 0.5, 1.0]), np.array([1, 0, 8]), np.array([1.0, 0.01, 0.999]),
     np.array([0.3, 0.0, 0.6])),
    (_r.normal(size=(6, 50)) * 3, _r.uniform(0.3, 1.5, size=6), np.array([0, 10, 40, 1, 0, 7]),
     np.array([0.9, 1.0, 0.75, 1.0, 0.3, 0.95]), _r.uniform(size=6)),
    (np.array([[5.0, -5.0, 0.0]]), np.array([1e-3]), np.array([0]), np.array([1.0]), np.array([0.999999])),
]

EXPLANATION = r"""
## Why it works
Both filters are statements about a token's **rank**, so one stable sort per row turns them into elementwise comparisons. Top-k keeps ranks below `top_k`. For top-p, the shortest prefix with cumulative mass $\ge p$ is exactly the set of ranked tokens whose mass *before* them is $< p$: the first token always survives (nothing is ahead of it), and the token that crosses the threshold survives because the mass before it was still short. Scattering the mask back through `order` returns it to vocabulary order. Inverse-CDF sampling with a given `u` makes the draw deterministic and testable; `torch.multinomial` with a seeded generator does the same job in production.

## Complexity
The sort costs $O(BV\log V)$, which for $V \approx 128\text{k}$ is the most expensive part of sampling. Engines avoid a full sort with a top-k partial selection first (`torch.topk` or a radix select) and apply top-p within those candidates.

## Common bugs
- Applying top-p to probabilities that were not renormalised after top-k: thresholds then mean something different.
- Using inclusive cumsum `< top_p`: it drops the token that crosses the threshold and can return an empty set.
- Dividing by a temperature of 0 on greedy rows: the `nan`s then leak through `torch.where` gradients or masks in other implementations.
- Breaking ties by sort order of an unstable sort: results change between runs or devices.

## Interview follow-ups
- How would you add a repetition penalty or `min_p` filter to this vectorised pipeline, and where in the order does each go?
- Why does sampling on the GPU need care for reproducibility across batch compositions (batch-invariant kernels)?
"""
