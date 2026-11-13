import numpy as np

META = dict(title='Data-parallel SGD step with uneven shards', tags=['Distributed Computing', 'Training'], diff='Medium',
            runtime='torch', lessons=['ieng:18.07', 'air-fanout-v1:03.09'],
            related=['simulate-distributed-collectives', 'gradient-accumulation-equivalence', 'data-parallel-step-with-psum'])

STATEMENT = r"""
In data parallelism every rank holds a full replica of the model, computes gradients on its own slice of the batch, and an all-reduce combines them so that every replica takes the **same** optimizer step. The step must equal the one a single process would take on the whole batch, even when the last batch of an epoch leaves ranks with different numbers of samples, or none.

Simulate it in one process with PyTorch: one parameter replica per rank, autograd per rank, a summed "all-reduce", and an SGD update on every replica.

Write `dp_sgd_step(X, y, counts, W, b, lr)`.

## Inputs
- `X`: `float64` `(N, D)`; `y`: `int64` `(N,)` class labels in $[0, C)$.
- `counts`: list of $R \ge 1$ non-negative ints summing to $N \ge 1$: rank $r$ owns the next `counts[r]` rows of the batch, in order.
- `W`: `float64` `(D, C)`, `b`: `float64` `(C,)`: the shared initial parameters. `lr`: float.

## Model and loss
Logits $= XW + b$; the batch loss is the **mean** cross-entropy over all $N$ samples.

## Per step
1. Each rank $r$ with $n_r > 0$ makes its own leaf copies of `W` and `b`, computes the mean cross-entropy over its $n_r$ samples, and backpropagates to get local gradients $g_r$. A rank with $n_r = 0$ contributes zero gradients (and must not produce NaN).
2. All-reduce: the global gradient is $g = \dfrac{1}{N}\sum_r n_r\, g_r$.
3. Every replica applies $\theta \leftarrow \theta - \text{lr}\cdot g$.

## Output
A tuple `(W_all, b_all, flat_grad)`:
- `W_all`: `(R, D, C)` and `b_all`: `(R, C)`, the updated parameters of every replica (they should all be identical);
- `flat_grad`: `(D C + C,)`, the global gradient as one flat bucket: `W`'s gradient flattened in row-major order, followed by `b`'s.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
import torch.nn.functional as F

def dp_sgd_step(X, y, counts, W, b, lr):
    """Return (W_all (R, D, C), b_all (R, C), flat_grad (D*C + C,))."""
    grads_W, grads_b, start = [], [], 0
    for n in counts:
        Wr = W.clone().requires_grad_(True)
        br = b.clone().requires_grad_(True)
        F.cross_entropy(X[start:start + n] @ Wr + br, y[start:start + n]).backward()
        grads_W.append(Wr.grad); grads_b.append(br.grad)
        start += n
    gW = sum(grads_W) / len(counts)              # TODO: equal weights per rank; NaN for empty ranks
    gb = sum(grads_b) / len(counts)
    R = len(counts)
    return (W - lr * gW).expand(R, *W.shape), (b - lr * gb).expand(R, *b.shape), torch.cat([gW.flatten(), gb])
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def dp_sgd_step(X, y, counts, W, b, lr):
    N = sum(counts)
    flat_parts, start = [], 0
    for n in counts:                                       # each iteration plays one rank
        Wr = W.detach().clone().requires_grad_(True)
        br = b.detach().clone().requires_grad_(True)
        if n > 0:
            F.cross_entropy(X[start:start + n] @ Wr + br, y[start:start + n]).backward()
            g = torch.cat([Wr.grad.flatten(), br.grad])   # pack the rank's gradients into one bucket
        else:
            g = torch.zeros(W.numel() + b.numel(), dtype=W.dtype)
        flat_parts.append(n * g)                           # re-weight the local mean by the local count
        start += n

    flat = torch.stack(flat_parts).sum(dim=0) / N          # the all-reduce (sum), then the global mean
    gW, gb = flat[:W.numel()].view_as(W), flat[W.numel():]
    W_all = torch.stack([W - lr * gW for _ in counts])     # every replica applies the same update
    b_all = torch.stack([b - lr * gb for _ in counts])
    return W_all, b_all, flat
'''

HINTS = [
    'A local mean over $n_r$ samples is $\\frac{1}{n_r}\\sum$. Multiply it back by $n_r$ before summing across ranks and divide by the global $N$, and you get the full-batch mean gradient.',
    'Skip the forward pass on ranks with no data and use zeros: `cross_entropy` on an empty batch returns NaN, and NaN would poison the all-reduce for **every** rank.',
    'Flatten each rank\'s gradients into one vector (`torch.cat([gW.flatten(), gb])`), sum those vectors, then split the result back with `view_as`. That is what DDP buckets do.',
]

_r = np.random.default_rng(14)


def _case(N, D, C):
    return (_r.normal(size=(N, D)), _r.integers(0, C, size=N).astype(np.int64))


_X3, _y3 = _case(10, 3, 4)
_W3, _b3 = _r.normal(size=(3, 4)) * 0.5, _r.normal(size=4) * 0.1
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), [1, 1], np.zeros((2, 2)), np.zeros(2), 1.0),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([0, 1, 1]), [2, 1], np.zeros((2, 2)), np.zeros(2), 0.5),
    (_X3, _y3, [5, 5], _W3, _b3, 0.1),
    (_X3, _y3, [7, 2, 1], _W3, _b3, 0.1),          # same result as test 3: only the split differs
    (_X3, _y3, [0, 10, 0], _W3, _b3, 0.1),
    (_X3, _y3, [10], _W3, _b3, 0.1),
    (*_case(23, 5, 3), [6, 6, 6, 5], _r.normal(size=(5, 3)), _r.normal(size=3), 0.05),
    (*_case(8, 4, 6), [1, 0, 3, 0, 4], _r.normal(size=(4, 6)), np.zeros(6), 1.0),
]

EXPLANATION = r"""
## Why it works
The full-batch loss is $\frac1N \sum_i \ell_i = \sum_r \frac{n_r}{N}\cdot\frac{1}{n_r}\sum_{i \in r}\ell_i$. Gradients are linear, so $g = \sum_r \frac{n_r}{N} g_r$. With equal shards this is the plain average that DDP computes (sum, then divide by world size). With uneven shards, the plain average gives each rank's samples a different weight. Tests 3, 4, 5 and 6 split the same 10 samples four ways and all produce the same update.

Since every replica starts from the same parameters and applies the same averaged gradient, the replicas stay bit-identical without ever exchanging parameters. That invariant is what makes data parallelism cheap: only gradients cross the network.

## Cost
Each rank does $1/R$ of the compute. Communication is one all-reduce of the full gradient per step, $2(R-1)/R$ times the model size per rank, independent of the batch size. DDP overlaps it with the backward pass by all-reducing fixed-size buckets as soon as their gradients are ready.

## Common bugs
- Averaging local means equally across ranks when shards are uneven (what `drop_last=False` with DDP's default does).
- Running the forward pass on an empty shard: `mean` over 0 elements is NaN.
- Letting replicas diverge (different random seeds for dropout are fine; different initial weights or an un-synchronised gradient are not).

## Interview follow-ups
- DDP divides by the world size, not by the sample count. How do you make the update exact for uneven batches without changing DDP? (Use a summed loss scaled by world size / global count, which needs an all-reduce of the count.)
- How does gradient bucketing let communication overlap with backward, and why are buckets filled in reverse layer order?
"""
