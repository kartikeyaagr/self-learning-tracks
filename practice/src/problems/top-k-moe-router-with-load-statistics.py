import numpy as np

META = dict(title='Capacity-limited top-k MoE routing with balance and z-losses', tags=['Transformers', 'Language Models'],
            diff='Hard', runtime='numpy', lessons=['aieng:06.02', 'ieng:19.04'],
            related=['sparse-moe-top-k-expert-routing', 'moe-token-dispatch-and-expert-aggregation', 'softmax-rows'])

STATEMENT = r"""
In training, every expert of a mixture-of-experts layer gets a fixed-size buffer so that tensor shapes stay static. If too many tokens pick the same expert, the overflow is **dropped** (those tokens skip that expert and rely on the residual path). Two auxiliary losses keep the router healthy: a balance loss that pushes load toward uniform, and a z-loss that keeps router logits small.

Write `route_with_capacity(logits, k, capacity_factor)`.

## Inputs
- `logits`: `float64` `(N, E)` router logits for $N$ tokens and $E$ experts. `k`: int, $1 \le k \le E$.
- `capacity_factor`: positive float. Expert capacity $C = \left\lceil \texttt{capacity\_factor} \cdot N k / E \right\rceil$.

## Routing
1. $P = \operatorname{softmax}(\text{logits})$ per token (stable).
2. Each token chooses its $k$ highest-probability experts, **highest first**; ties go to the smaller expert index. Choice $j$ of a token is its "slot $j$".
3. Gates: the chosen probabilities divided by their sum over the token's $k$ choices.
4. Capacity, in priority order: **all slot-0 choices first** (token order $0, 1, \dots$), then all slot-1 choices, and so on. A choice is kept if its expert has fewer than $C$ kept choices so far; otherwise it is dropped and its gate becomes 0. Kept gates are **not** renormalised.

## Statistics
- `load`: `int64` `(E,)`, kept choices per expert.
- `dropped_frac`: float, dropped choices / $(Nk)$.
- `aux_loss`: float, $E \sum_e f_e \bar P_e$ where $f_e$ = (choices for expert $e$ **before** dropping) / $(Nk)$ and $\bar P_e$ = mean of $P_{:,e}$ over tokens.
- `z_loss`: float, mean over tokens of $\bigl(\log \sum_e e^{\text{logits}_{n,e}}\bigr)^2$.

## Output
A dict with keys `'experts'` (`int64` `(N, k)`), `'gates'` (`float64` `(N, k)`), `'kept'` (`bool` `(N, k)`), `'load'`, `'dropped_frac'`, `'aux_loss'`, `'z_loss'`.
"""

STARTER = '''import numpy as np

def route_with_capacity(logits, k, capacity_factor):
    """Return dict: experts, gates, kept, load, dropped_frac, aux_loss, z_loss."""
    N, E = logits.shape
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    experts = np.argsort(-P, axis=1)[:, :k]               # TODO: unstable softmax, no capacity, no losses
    gates = np.take_along_axis(P, experts, axis=1)
    load = np.bincount(experts.ravel(), minlength=E)
    return {'experts': experts, 'gates': gates, 'kept': np.ones((N, k), dtype=bool), 'load': load,
            'dropped_frac': 0.0, 'aux_loss': 0.0, 'z_loss': 0.0}
'''

SOLUTION = '''import math
import numpy as np

def route_with_capacity(logits, k, capacity_factor):
    N, E = logits.shape
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))         # (N,) log-sum-exp
    P = np.exp(logits - lse[:, None])

    # stable sort on -P: equal probabilities keep index order, so ties go to the smaller expert
    experts = np.argsort(-P, axis=1, kind='stable')[:, :k]
    chosen = np.take_along_axis(P, experts, axis=1)
    gates = chosen / chosen.sum(axis=1, keepdims=True)

    C = math.ceil(capacity_factor * N * k / E)
    kept = np.zeros((N, k), dtype=bool)
    fill = np.zeros(E, dtype=np.int64)
    for j in range(k):                       # slot-major priority: every token's first choice beats any second choice
        for n in range(N):
            e = experts[n, j]
            if fill[e] < C:
                fill[e] += 1
                kept[n, j] = True

    f = np.bincount(experts.ravel(), minlength=E) / (N * k)        # demand before dropping
    return {
        'experts': experts.astype(np.int64),
        'gates': np.where(kept, gates, 0.0),
        'kept': kept,
        'load': fill,
        'dropped_frac': float((~kept).sum() / (N * k)),
        'aux_loss': float(E * np.sum(f * P.mean(axis=0))),
        'z_loss': float(np.mean(lse ** 2)),
    }
'''

HINTS = [
    'Compute the log-sum-exp once with the max trick; it gives both the stable softmax ($P = e^{\\text{logits} - \\text{lse}}$) and the z-loss.',
    '`np.argsort(-P, axis=1, kind="stable")[:, :k]` sorts by probability, highest first, and keeps equal probabilities in index order. Normalise the chosen probabilities to get the gates **before** applying capacity.',
    'Loop over slots on the outside and tokens on the inside, with a per-expert fill counter. The balance loss uses the demand counts (all choices), not the kept load.',
]

_r = np.random.default_rng(42)
TESTS = [
    (np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]), 1, 1.0),
    (np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]), 2, 1.0),
    (np.zeros((3, 4)), 2, 2.0),
    (_r.normal(size=(16, 4)) * 2, 2, 1.25),
    (_r.normal(size=(10, 8)), 2, 1.0),
    (np.array([[1000.0, 999.0, -1000.0], [5.0, 5.0, 5.0]]), 2, 10.0),
    (np.tile([[3.0, 0.0, 0.0, 0.0]], (6, 1)), 1, 1.0),
    (_r.normal(size=(32, 6)) * 0.5, 3, 0.5),
]

EXPLANATION = r"""
## Why it works
Capacity makes every expert's input buffer exactly $C$ tokens long, so dispatch is a fixed-shape gather that compiles well. The slot-major priority means a token's best expert is honoured before anybody's second choice, which drops the least important assignments first.

The balance loss $E\sum_e f_e \bar P_e$ equals 1 when both demand and probability mass are uniform and grows when the router concentrates on a few experts. $f_e$ is not differentiable (it comes from an argmax), but $\bar P_e$ is, so the gradient lowers the probability of experts that are already overloaded. The z-loss penalises large logits, which otherwise drift upward and make the softmax brittle in bf16.

## Complexity
Softmax and sort are $O(NE \log E)$; the capacity pass is $O(Nk)$ with a sequential dependency (each decision depends on earlier fills). On accelerators it is done with a cumulative sum over a one-hot `(k, N, E)` tensor in the same priority order: position-in-expert = cumsum $-$ 1, kept if $< C$.

## Edge cases
- Equal logits (test 3): ties broken by index, so every token picks experts 0 and 1; with $C = 3$ nothing is dropped, even though experts 2 and 3 get nothing.
- Huge logits (test 6): the max trick keeps everything finite, and the z-loss is about $10^6$, a strong signal to shrink them.
- A collapsed router (test 7): every token wants expert 0, $C = 2$, so 4 of 6 tokens are dropped.

## Interview follow-ups
- Why do inference engines usually drop the capacity limit (dropless MoE), and what replaces static shapes? (Grouped GEMM over variable-length expert batches.)
- How does expert-choice routing (experts pick their top-$C$ tokens) guarantee perfect balance, and why is it awkward for causal decoding?
"""
