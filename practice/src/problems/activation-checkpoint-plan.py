import math

META = dict(title='Plan activation checkpoints for the lowest peak memory', tags=['Training'], diff='Medium', runtime='python',
            related=['transformer-activation-memory'])

STATEMENT = r"""
With checkpointing, a network of $n$ identical layers is cut into segments of $k$ layers. Only the input of each segment (a boundary tensor of `b` bytes) is kept during the forward pass. In the backward pass one segment at a time is recomputed, which needs that segment's full activations, $k \cdot a$ bytes, where `a` is one layer's activations.

Peak activation memory for segment size $k$: $\lceil n/k \rceil \cdot b + k \cdot a$.

Write `plan_checkpoints(n, a, b, budget, fwd_cost)`:

- `best_k`: the $k$ in $1..n$ with the lowest peak (ties to the smaller $k$).
- `peak`: that peak; `no_ckpt`: $n \cdot a$ (store everything).
- `fits_k`: the **largest** $k$ whose peak is at most `budget` (larger segments mean fewer stored boundaries to manage), or `None`.
- `extra_compute`: recomputation as a fraction of one training step, assuming backward costs twice the forward: $1/3$ if the best peak is below `no_ckpt` (every segment is recomputed once), else 0.0 because checkpointing would not help. (`fwd_cost` only scales both sides and cancels.)
- `sqrt_k`: `round(math.sqrt(n * b / a))` clipped to $[1, n]$, the continuous optimum of the peak formula.
"""

STARTER = '''import math

def plan_checkpoints(n, a, b, budget, fwd_cost):
    """Choose a checkpoint segment size."""
    k = max(1, int(math.sqrt(n)))
    return {"best_k": k, "peak": math.ceil(n / k) * b + k * a}
'''

SOLUTION = '''import math

def plan_checkpoints(n, a, b, budget, fwd_cost):
    peak = lambda k: math.ceil(n / k) * b + k * a
    best = min(range(1, n + 1), key=lambda k: (peak(k), k))
    fits = [k for k in range(1, n + 1) if peak(k) <= budget]
    sk = min(max(round(math.sqrt(n * b / a)), 1), n)
    return {"best_k": best, "peak": peak(best), "no_ckpt": n * a, "fits_k": max(fits) if fits else None,
            "extra_compute": (1 / 3) if peak(best) < n * a else 0.0, "sqrt_k": sk}
'''

HINTS = [
    'Evaluate the peak formula for every k from 1 to n; n is small enough that brute force is exact.',
    'Minimising (n/k) b + k a over continuous k gives k = sqrt(n b / a), the classic square-root rule.',
    'Recomputing every segment once adds one forward pass, about a third of a forward-plus-backward step.',
]

TESTS = [
    (16, 10.0, 1.0, 20.0, 1.0),
    (100, 4.0, 1.0, 50.0, 1.0),
    (24, 1.0, 1.0, 5.0, 1.0),
    (8, 1.0, 10.0, 100.0, 1.0),
    (48, 34.0, 2.0, 200.0, 1.0),
]

EXPLANATION = r"""
## The trade
Storing every layer's activations costs $n a$. Storing nothing but segment boundaries and recomputing costs one extra forward pass. With segments of $\sqrt{n b / a}$ layers the peak falls from linear in $n$ to about $2\sqrt{n a b}$, the square-root rule from Chen et al. (2016).

## In real frameworks
PyTorch's `torch.utils.checkpoint` and JAX's `jax.checkpoint` wrap a function so its internals are recomputed in the backward pass. Transformer training usually checkpoints at the layer boundary ($k = 1$, boundary = layer input), because the boundary is small relative to a layer's activations and the bookkeeping is simple.

## When not to checkpoint
If activations already fit, recomputation is pure overhead. Selective recomputation (only attention scores) is often the better middle ground.

## Interview follow-ups
- How does checkpointing interact with pipeline parallelism, where several micro-batches are in flight?
"""
