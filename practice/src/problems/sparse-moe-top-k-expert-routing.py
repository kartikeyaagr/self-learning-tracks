import numpy as np

META = dict(title='Top-k MoE router with load statistics and balance loss', tags=['Inference', 'Transformers'], diff='Medium',
            runtime='torch', lessons=['ieng:19.02', 'aieng:06.02'],
            related=['moe-token-dispatch-and-expert-aggregation', 'expert-parallel-routing-across-gpus', 'softmax-rows'])

STATEMENT = r"""
A sparse mixture-of-experts layer sends each token to only $k$ of its $E$ expert MLPs. A small linear **router** scores the experts, and its choices decide both the output and how evenly work spreads across experts (and the GPUs that hold them).

Write `route(x, W_r, k, renormalize=True)` in PyTorch.

## Inputs
- `x`: float64 `(N, D)` token states (batch and sequence already flattened). `W_r`: float64 `(D, E)`. `k`: int, $1 \le k \le E$.

## Computation
1. Router probabilities $P = \mathrm{softmax}(x W_r)$ over experts, shape `(N, E)`.
2. For each token, pick the $k$ experts with the largest probability, **highest first**, ties to the smaller expert index.
3. Gates: the chosen probabilities. If `renormalize`, divide them by their sum so each token's gates sum to 1 (Mixtral-style); otherwise keep them as they are (they then sum to less than 1).
4. Load: `counts[e]` = number of (token, slot) assignments that went to expert $e$.
5. Balance loss (Switch Transformer form): $\;\mathcal{L} = E \sum_e f_e\, \bar P_e$, where $f_e = \texttt{counts}[e] / (N k)$ is the fraction of assignments to $e$ and $\bar P_e$ is the mean of $P_{:,e}$ over tokens.

## Output
A tuple `(idx, gates, counts, loss)`:
- `idx`: int64 `(N, k)`, `gates`: float64 `(N, k)`, `counts`: int64 `(E,)`, `loss`: Python float.

No Python loops over tokens.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def route(x, W_r, k, renormalize=True):
    """Return (expert idx (N,k) int64, gates (N,k), counts (E,) int64, balance loss float)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def route(x, W_r, k, renormalize=True):
    N, E = x.shape[0], W_r.shape[1]
    P = torch.softmax(x @ W_r, dim=-1)                               # (N, E)
    _, order = torch.sort(P, dim=-1, descending=True, stable=True)   # stable sort gives the tie rule
    idx = order[:, :k]
    gates = P.gather(1, idx)
    if renormalize:
        gates = gates / gates.sum(dim=-1, keepdim=True)
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    f = counts.to(P.dtype) / (N * k)                                 # fraction of assignments per expert
    loss = E * (f * P.mean(dim=0)).sum()                             # 1.0 when routing is perfectly uniform
    return idx, gates, counts, loss.item()
'''

HINTS = [
    'Softmax over the expert axis, then a stable descending sort: the first `k` columns of the sort indices are the chosen experts, highest first.',
    '`P.gather(1, idx)` picks the chosen probabilities. Renormalise by their row sum only when asked.',
    '`torch.bincount(idx.flatten(), minlength=E)` counts assignments, including experts nobody chose. The loss uses the fraction of assignments `counts / (N*k)` times the mean router probability, summed and multiplied by `E`.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]), 1),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]), 2),
    (_r.normal(size=(16, 8)), _r.normal(size=(8, 4)), 2),
    (_r.normal(size=(10, 6)), _r.normal(size=(6, 8)), 2, False),
    (np.zeros((4, 3)), np.zeros((3, 4)), 2),
    (_r.normal(size=(32, 16)), _r.normal(size=(16, 8)) * 3, 1),
    (_r.normal(size=(5, 4)), _r.normal(size=(4, 6)), 6),
    (_r.normal(size=(64, 8)), _r.normal(size=(8, 16)), 4, False),
]

EXPLANATION = r"""
## Why it works
The router is a softmax classifier over experts; top-$k$ selection makes the layer sparse, so compute per token is $k$ expert MLPs no matter how many experts exist. Renormalising keeps the output scale independent of how confident the router is, which is why Mixtral does it; DeepSeek-style routers often skip it and scale gates separately.

## The balance loss
$f_e$ (hard counts) has no gradient, but $\bar P_e$ does. The product is minimised, at value 1, when both are uniform at $1/E$; if an expert is over-used, its $f_e$ is large and the gradient pushes its router probability down. Without it, routers collapse onto a few experts, wasting capacity and, at inference time, overloading the GPUs that host the popular experts.

## Inference consequences
`counts` is what an expert-parallel deployment cares about: the most loaded expert sets the step time, because every other GPU waits for it. A perfectly balanced batch gives each expert $Nk/E$ assignments.

## Common bugs
- `torch.topk` for selection: it does not promise which index wins a tie, so outputs can differ across devices.
- Computing the softmax after top-k (over the chosen logits only) when the spec says before: that is a different, also common, design (it equals renormalised gates here, but the balance loss changes).
- Using `counts / N` instead of `counts / (N k)`: the loss is then off by a factor of $k$.

## Interview follow-ups
- What is "expert capacity", and what happens to tokens routed to a full expert? How do drop-less MoE implementations avoid dropping?
- DeepSeek-V3 balances load with a per-expert bias added to the routing scores instead of an auxiliary loss. How does that work, and why avoid the loss?
"""
