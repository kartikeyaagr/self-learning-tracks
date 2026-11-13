import numpy as np

META = dict(title='MoE dispatch with expert capacity and weighted combine', tags=['Inference', 'Transformers'], diff='Hard',
            runtime='torch', lessons=['ieng:19.05', 'ieng:19.04'],
            related=['sparse-moe-top-k-expert-routing', 'expert-parallel-routing-across-gpus'])

STATEMENT = r"""
After routing, a mixture-of-experts layer must **group** tokens by expert, run each expert once on its group (one dense matmul per expert, not one per token), and **scatter** the results back, weighted by the gates. Fixed-size kernels also impose an expert **capacity**: an expert accepts at most $C$ assignments per batch and the rest are dropped.

Write `moe_forward(x, idx, gates, W1, W2, capacity=None)` in PyTorch.

## Inputs (float64 unless stated)
- `x`: `(N, D)` tokens. `idx`: int64 `(N, k)`, the experts chosen for each token (distinct within a row). `gates`: `(N, k)`.
- `W1`: `(E, D, F)`, `W2`: `(E, F, D)`. Expert $e$ computes $f_e(u) = \mathrm{relu}(u\,W_1[e])\,W_2[e]$.
- `capacity`: int $C \ge 0$, or `None` for no limit.

## Capacity rule
Process assignments in **priority order**: slot 0 of tokens $0, 1, \dots, N-1$, then slot 1 of tokens $0, \dots, N-1$, and so on (every token's first choice beats anyone's second choice). Each expert keeps the first $C$ assignments it receives in that order and drops the rest.

## Output
A tuple `(y, kept)`:
- `y`: `(N, D)`, with $y_n = \sum_{s\,:\,\text{kept}[n,s]} \texttt{gates}[n,s]\; f_{\texttt{idx}[n,s]}(x_n)$. A token whose assignments were all dropped gets a zero row (the residual connection outside the layer carries it).
- `kept`: bool `(N, k)`.

Loop over experts if you like, but not over tokens.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def moe_forward(x, idx, gates, W1, W2, capacity=None):
    """Return (y (N,D), kept (N,k) bool)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def moe_forward(x, idx, gates, W1, W2, capacity=None):
    N, k = idx.shape
    E = W1.shape[0]
    # Flatten slot-major: priority order is slot 0 of every token, then slot 1, ...
    expert = idx.T.reshape(-1)                         # (k*N,)
    token = torch.arange(N).repeat(k)                  # token id of each flat assignment
    gate = gates.T.reshape(-1)

    onehot = torch.nn.functional.one_hot(expert, E)    # (k*N, E)
    rank = (onehot.cumsum(dim=0) - 1).gather(1, expert[:, None]).squeeze(1)   # position in its expert's queue
    keep = torch.ones_like(expert, dtype=torch.bool) if capacity is None else rank < capacity

    y = torch.zeros_like(x)
    for e in range(E):                                 # one dense GEMM pair per expert
        sel = keep & (expert == e)
        if not sel.any():
            continue
        t = token[sel]
        h = torch.relu(x[t] @ W1[e]) @ W2[e]
        y.index_add_(0, t, gate[sel, None] * h)        # scatter back, weighted by the gate
    return y, keep.view(k, N).T
'''

HINTS = [
    'Flatten the assignments slot-major (`idx.T.reshape(-1)`) so their order is the priority order; `torch.arange(N).repeat(k)` gives the matching token ids.',
    "An assignment's position in its expert's queue is a running count: one-hot the expert ids, take a cumulative sum down the rows, and read the column of its own expert (minus 1). Keep it when that position is below the capacity.",
    'For each expert, gather its kept tokens, run `relu(x @ W1[e]) @ W2[e]` once, multiply by the gates and `index_add_` into `y`. Reshape the flat keep mask back with `.view(k, N).T`.',
]

_r = np.random.default_rng(17)


def _case(N, D, F, E, k, cap, s=0.5):
    idx = np.stack([_r.permutation(E)[:k] for _ in range(N)]).astype(np.int64)
    g = _r.uniform(0.1, 1.0, size=(N, k))
    g /= g.sum(1, keepdims=True)
    return (_r.normal(size=(N, D)), idx, g, _r.normal(size=(E, D, F)) * s, _r.normal(size=(E, F, D)) * s, cap)


_W1 = np.array([[[1.0]], [[2.0]]])
_W2 = np.array([[[1.0]], [[1.0]]])
TESTS = [
    (np.array([[1.0], [2.0], [-1.0]]), np.array([[0], [1], [0]]), np.ones((3, 1)), _W1, _W2),
    (np.array([[1.0], [2.0], [3.0]]), np.array([[0, 1], [0, 1], [1, 0]]), np.full((3, 2), 0.5), _W1, _W2, 2),
    _case(8, 4, 6, 4, 2, None),
    _case(12, 4, 8, 3, 2, 5),
    _case(10, 6, 5, 4, 1, 2),
    _case(6, 3, 4, 2, 2, 0),
    _case(20, 8, 16, 8, 2, 6),
    _case(9, 5, 7, 6, 3, 3),
]

EXPLANATION = r"""
## Why it works
Slot-major flattening puts the assignments in priority order, and a cumulative sum of one-hot expert ids counts, for each assignment, how many earlier assignments went to the same expert. That count is its slot in the expert's buffer, so the capacity check is one comparison. After that, each expert sees a dense block of tokens, which is the whole point: $E$ well-shaped GEMMs instead of $Nk$ tiny ones. `index_add_` sums a token's contributions from its different experts.

## Complexity
Expert compute is $O(Nk\,DF)$, the same as $k$ dense MLPs per token. The bookkeeping (one-hot cumsum) is $O(NkE)$, cheap next to the GEMMs. Production kernels replace the per-expert loop with one **grouped GEMM** over all experts and replace the one-hot with a sort by expert id.

## Capacity in practice
With capacity factor $c$, $C = c \cdot Nk/E$. At $c = 1$ any imbalance drops tokens; at $c = 1.25$ to $2$ fewer drop but padded buffers waste compute. Inference engines prefer drop-less routing (variable-size groups), because dropping changes model output depending on what else is in the batch.

## Common bugs
- Token-major priority (`idx.reshape(-1)`): later tokens' first choices lose to earlier tokens' second choices.
- `y[t] += ...` with repeated indices: plain indexed assignment does not accumulate duplicates; `index_add_` does.
- Forgetting the gate, or applying it before the nonlinearity.

## Interview follow-ups
- How does batch composition affect a given request's output when capacity is finite, and why is that a problem for serving?
- Replace the per-expert loop with a sort plus a grouped GEMM. What metadata (offsets per expert) does the grouped kernel need?
"""
