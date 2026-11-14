import numpy as np

META = dict(title='Auxiliary-loss-free MoE routing with bias balancing', tags=['Transformers', 'Inference'], diff='Medium',
            runtime='numpy', lessons=['aieng:06.02', 'ieng:19.04'],
            related=['top-k-moe-router-with-load-statistics', 'sparse-moe-top-k-expert-routing', 'moe-token-dispatch-and-expert-aggregation'])

STATEMENT = r"""
Most mixture-of-experts models keep experts evenly loaded with an auxiliary loss, which also pulls the router away from the language-modelling objective. DeepSeek-V3 balances load differently: each expert has a **bias** that is added to its score **only when choosing** the top-$k$ experts, and after each batch the bias is nudged up for under-loaded experts and down for over-loaded ones. The gate weights that scale expert outputs ignore the bias.

Write `route_aux_free(logits, bias, k, gamma)` returning a tuple `(idx, gates, new_bias)`.

## Rules
- Affinities $s_{t,e} = \sigma(\text{logits}_{t,e})$ (sigmoid, not softmax). `logits`: `(N, E)`.
- **Selection**: for each token, the $k$ experts with the largest $s_{t,e} + b_e$. `idx` is `(N, k)` int, ordered from the largest biased score down.
- **Gates**: $g_{t,j} = s_{t,\text{idx}_{t,j}} / \sum_{j'} s_{t,\text{idx}_{t,j'}}$, the **unbiased** affinities of the selected experts, normalised to sum to 1. `gates`: `(N, k)`, same order as `idx`.
- **Bias update**: let $c_e$ be the number of tokens that selected expert $e$ and $\bar c = Nk/E$. Then $b_e \leftarrow b_e + \gamma\,\mathrm{sign}(\bar c - c_e)$, with $\mathrm{sign}(0) = 0$.

Inputs never contain exact ties between biased scores within a token.
"""

STARTER = '''import numpy as np

def route_aux_free(logits, bias, k, gamma):
    """Return (idx (N, k) int, gates (N, k), new_bias (E,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def route_aux_free(logits, bias, k, gamma):
    logits, bias = np.asarray(logits, dtype=float), np.asarray(bias, dtype=float)
    N, E = logits.shape
    s = 1.0 / (1.0 + np.exp(-logits))
    idx = np.argsort(-(s + bias), axis=1, kind='stable')[:, :k]       # bias only affects the choice
    chosen = np.take_along_axis(s, idx, axis=1)
    gates = chosen / chosen.sum(axis=1, keepdims=True)                  # weights use the raw affinities
    load = np.bincount(idx.ravel(), minlength=E)
    new_bias = bias + gamma * np.sign(N * k / E - load)
    return idx, gates, new_bias
'''

HINTS = [
    'Compute $s = \\sigma(\\text{logits})$ once. Sort by `s + bias` to choose experts, then gather from `s` (no bias) to build the gates.',
    '`np.bincount(idx.ravel(), minlength=E)` gives the per-expert token counts. The bias moves by exactly $\\pm\\gamma$ or 0, not in proportion to the imbalance.',
]

_r = np.random.default_rng(2412)
_L1 = _r.normal(size=(6, 4))
_L2 = _r.normal(size=(16, 8)) + np.linspace(1.5, -1.5, 8)       # experts 0..2 are popular
TESTS = [
    (_L1, np.zeros(4), 1, 0.01),
    (_L1, np.zeros(4), 2, 0.01),
    (_L1, np.array([0.0, 0.0, 5.0, 0.0]), 2, 0.1),
    (_L2, np.zeros(8), 2, 0.001),
    (_L2, -np.linspace(0.3, -0.3, 8), 3, 0.001),
    (_r.normal(size=(9, 6)), _r.normal(size=6) * 0.05, 3, 0.02),
]

EXPLANATION = r"""
## Why separate choosing from weighting
If the bias also scaled the outputs, balancing would change what the model computes for a token, which is exactly the interference an auxiliary loss causes. Using the bias only in the top-$k$ choice means it shifts *which* experts a borderline token visits, while the contribution of each chosen expert still reflects the learned affinity.

## Why a sign update
The update is a fixed step per batch, like a simple integral controller: persistent overload keeps pushing an expert's bias down until its traffic falls back to the mean. It needs no gradient and no extra loss term. DeepSeek-V3 still kept a very small sequence-level balance loss as a safeguard against extreme imbalance within single sequences.

## Sigmoid affinities
With sigmoid scores each expert's affinity is independent of the others, so normalising over only the selected $k$ is well defined. With a softmax over all $E$ experts, the selected gates would not sum to 1 without renormalising either.

## Complexity
$O(NE\log E)$ for the sort (an `argpartition` brings it to $O(NE)$), $O(Nk)$ for gates and counts.

## Common bugs
- Using `s + bias` for the gates.
- Updating the bias in proportion to $\bar c - c_e$ (works, but it is a different controller with a different step-size scale).
- Counting load per token rather than per (token, expert) assignment.

## Interview follow-ups
- What happens to the bias of an expert that no token wants, even with a large bias? Can it overshoot?
- How would you add DeepSeek-V3's node-limited routing (each token uses experts on at most $M$ devices)?
"""
