import numpy as np

META = dict(title='Expert-parallel all-to-all: send counts and GPU load', tags=['Distributed Computing', 'Inference'], diff='Hard',
            runtime='numpy', lessons=['ieng:19.03', 'ieng:19.05'],
            related=['sparse-moe-top-k-expert-routing', 'moe-token-dispatch-and-expert-aggregation', 'tensor-parallel-communication-cost'])

STATEMENT = r"""
With **expert parallelism**, each GPU holds a subset of a mixture-of-experts layer's experts. After routing, every GPU must send each of its tokens to the GPUs that own the chosen experts (an all-to-all), and the busiest GPU sets the pace for everyone. Compute the traffic matrix and the load before any data moves.

Write `ep_dispatch(idx, placement, n_gpus, dedup=True)` in NumPy.

## Inputs
- `idx`: int array `(N, k)`, the experts chosen for each token (distinct within a row).
- `placement`: int array `(E,)`, `placement[e]` = the GPU that owns expert $e$ (any mapping, e.g. to move a hot expert).
- `n_gpus` $G$ divides $N$. Tokens are spread contiguously: token $n$ starts on GPU $\lfloor n / (N/G) \rfloor$.
- `dedup`: if true, a token routed to **several** experts on the same destination GPU is sent there **once** (the destination fans it out locally). If false, one copy per (token, expert) assignment.

## Output
A tuple `(send, load, imbalance, remote_fraction)`:
- `send`: int64 `(G, G)`, `send[s, d]` = token copies GPU $s$ sends to GPU $d$ (the diagonal counts tokens that stay local).
- `load`: int64 `(G,)`, the number of (token, expert) assignments each GPU computes (independent of `dedup`).
- `imbalance`: float, `load.max() / load.mean()` (1.0 is perfect balance).
- `remote_fraction`: float, off-diagonal copies divided by all copies in `send`.

No Python loops over tokens.
"""

STARTER = '''import numpy as np

def ep_dispatch(idx, placement, n_gpus, dedup=True):
    """Return (send (G,G) int64, load (G,) int64, imbalance float, remote_fraction float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ep_dispatch(idx, placement, n_gpus, dedup=True):
    N, k = idx.shape
    G = n_gpus
    src = np.arange(N) // (N // G)                      # (N,) GPU each token starts on
    dst = placement[idx]                                # (N, k) GPU of each chosen expert

    if dedup:
        # one-hot of destinations per token: duplicates collapse because assignment is idempotent
        hit = np.zeros((N, G), dtype=bool)
        hit[np.arange(N)[:, None], dst] = True
        s, d = np.nonzero(hit)
        s = src[s]
    else:
        s, d = np.repeat(src, k), dst.ravel()
    send = np.zeros((G, G), dtype=np.int64)
    np.add.at(send, (s, d), 1)                          # unbuffered: repeated (s, d) pairs all count

    load = np.bincount(dst.ravel(), minlength=G).astype(np.int64)
    total = send.sum()
    return send, load, float(load.max() / load.mean()), float((total - np.trace(send)) / total)
'''

HINTS = [
    '`placement[idx]` turns expert ids into destination GPUs in one step; `np.arange(N) // (N // G)` gives each token\'s source GPU.',
    'For dedup, mark a boolean `(N, G)` matrix with fancy indexing (`hit[rows, dst] = True`): writing True twice is the same as once. `np.nonzero` then lists unique (token, GPU) pairs.',
    'Accumulate the `(G, G)` matrix with `np.add.at`, because plain `send[s, d] += 1` drops repeated index pairs. The load comes from `np.bincount` over all destinations.',
]

_r = np.random.default_rng(81)


def _route(N, E, k, hot=None):
    p = np.ones(E) if hot is None else np.where(np.arange(E) == hot, 6.0, 1.0)
    return np.array([_r.choice(E, size=k, replace=False, p=p / p.sum()) for _ in range(N)])


_idx = _route(32, 8, 2, hot=3)
TESTS = [
    (np.array([[0, 1], [2, 3], [0, 3], [1, 2]]), np.array([0, 0, 1, 1]), 2),
    (np.array([[0, 1], [2, 3], [0, 3], [1, 2]]), np.array([0, 0, 1, 1]), 2, False),
    (_route(16, 8, 2), np.arange(8) // 2, 4),
    (_idx, np.arange(8) // 2, 4),
    (_idx, np.array([0, 0, 1, 2, 1, 2, 3, 3]), 4),
    (_idx, np.arange(8) // 2, 4, False),
    (_route(12, 6, 3), np.array([2, 0, 1, 2, 0, 1]), 3),
    (np.array([[0], [0], [0], [0]]), np.array([0, 1]), 2),
]

EXPLANATION = r"""
## Why it works
Each (token, chosen expert) pair needs the token on the expert's GPU. Mapping experts to GPUs with `placement` turns routing into a list of (source, destination) pairs, and the traffic matrix is their histogram. Dedup is a property of the token, not the assignment: if two of a token's experts live on the same GPU, sending the activation once and letting that GPU feed both experts halves that traffic (DeepSeek-style routing even limits how many GPUs or nodes each token may touch to bound this).

## Load and imbalance
The all-to-all and the expert matmuls run in lockstep, so a layer takes as long as its busiest GPU. Test 4 has a hot expert (chosen six times as often): with the default placement its GPU is overloaded; test 5 moves experts so the hot one shares a GPU with cold ones, and the imbalance drops. Real systems also replicate hot experts (redundant experts) or add an auxiliary balance loss in training.

## Common bugs
- `send[s, d] += 1` with repeated index pairs: NumPy applies it once per unique pair. Use `np.add.at` or `np.bincount` on `s * G + d`.
- Deduplicating per (token, expert) instead of per (token, GPU).
- Forgetting that local traffic (the diagonal) is still traffic for the kernel, just not over the network; `remote_fraction` separates the two.

## Interview follow-ups
- With $E$ experts spread over $G$ GPUs and uniform routing, what fraction of copies is remote? (About $1 - 1/G$, which is why expert parallelism needs fast interconnect.)
- Why does expert parallelism usually pair with data parallelism for attention? (Attention has no experts; each GPU runs attention on its own tokens, then the all-to-all regroups them by expert.)
"""
