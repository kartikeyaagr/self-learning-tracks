import numpy as np

META = dict(title='Capstone: train on many workers and check every split', tags=['Distributed Computing', 'Training'], diff='Hard', runtime='numpy',
            related=['ring-vs-parameter-server-bytes', 'simulate-distributed-collectives', 'account-for-zero-memory-by-stage', 'tensor-parallel-mlp', 'diloco-outer-step'])

STATEMENT = r"""
Model: $h = \mathrm{relu}(XW_1)$, $\hat y = hw_2$ with `W1 (d, f)` and `w2 (f,)`; loss $L = \frac{1}{2n}\sum (\hat y - y)^2$. Write `check_parallel(X, y, W1, w2, K, tp, lr, H, outer_lr, mu)` that implements each parallel scheme and checks it against the single-device computation:

1. **Data parallel.** Split the rows into `K` equal consecutive shards. Each worker computes the gradient of its shard's mean loss. Average with a simulated **ring all-reduce**: flatten each worker's gradient `[dW1.ravel(), dw2]`, split it with `np.array_split` into K chunks; reduce-scatter for K−1 steps (at step $s$ worker $i$ sends chunk $(i - s) \bmod K$ to worker $(i+1) \bmod K$, which adds it); then all-gather for K−1 steps (at step $s$ worker $i$ sends chunk $(i + 1 - s) \bmod K$, which the receiver overwrites); divide by K. `dp_diff` = max abs difference between worker 0's result and the full-batch gradient.
2. **Tensor parallel.** Split the columns of `W1` and the entries of `w2` into `tp` parts with `np.array_split`; each part computes a partial output; sum them. `tp_diff` = max abs difference from the full output.
3. **ZeRO-1 step.** Split the flattened parameters into K shards (`np.array_split`); worker $k$ applies SGD (`lr`) to shard $k$ only, using the averaged gradient; concatenate. `zero_diff` = max abs difference from a full SGD step.
4. **DiLoCo.** Each worker starts from the same parameters and takes `H` local SGD steps (`lr`) on its own shard; then one outer step: $\Delta = \theta - \mathrm{mean}_k\theta_k$, $v = \mu\cdot 0 + \Delta$, $\theta \leftarrow \theta - $ `outer_lr` $(\Delta + \mu v)$.

Return `{"dp_diff", "tp_diff", "zero_diff", "loss_before", "loss_after"}` where the losses are full-batch losses before and after the DiLoCo round.
"""

STARTER = '''import numpy as np

def check_parallel(X, y, W1, w2, K, tp, lr, H, outer_lr, mu):
    """Implement four parallel schemes and compare with one device."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _loss_grad(X, y, W1, w2):
    a = X @ W1
    h = np.maximum(a, 0)
    r = h @ w2 - y
    n = len(y)
    g2 = h.T @ r / n
    g1 = X.T @ (np.outer(r, w2) * (a > 0)) / n
    return float((r ** 2).sum() / (2 * n)), g1, g2

def _ring_allreduce(vecs):
    K = len(vecs)
    ch = [[c.copy() for c in np.array_split(v, K)] for v in vecs]
    for s in range(K - 1):
        sent = [ch[i][(i - s) % K].copy() for i in range(K)]
        for i in range(K):
            ch[(i + 1) % K][(i - s) % K] += sent[i]
    for s in range(K - 1):
        sent = [ch[i][(i + 1 - s) % K].copy() for i in range(K)]
        for i in range(K):
            ch[(i + 1) % K][(i + 1 - s) % K] = sent[i]
    return [np.concatenate(c) / K for c in ch]

def check_parallel(X, y, W1, w2, K, tp, lr, H, outer_lr, mu):
    X, y, W1, w2 = (np.asarray(a, float) for a in (X, y, W1, w2))
    flat = lambda g1, g2: np.concatenate([g1.ravel(), g2])
    unflat = lambda v: (v[:W1.size].reshape(W1.shape), v[W1.size:])
    L0, G1, G2 = _loss_grad(X, y, W1, w2)
    full = flat(G1, G2)
    Xs, ys = np.split(X, K), np.split(y, K)
    local = [flat(*_loss_grad(Xs[k], ys[k], W1, w2)[1:]) for k in range(K)]
    avg = _ring_allreduce(local)[0]
    dp = float(np.abs(avg - full).max())
    ref = np.maximum(X @ W1, 0) @ w2
    cols = np.array_split(np.arange(W1.shape[1]), tp)
    part = sum(np.maximum(X @ W1[:, c], 0) @ w2[c] for c in cols)
    tpd = float(np.abs(part - ref).max())
    theta = flat(W1, w2)
    shards = np.array_split(np.arange(theta.size), K)
    z = np.concatenate([theta[s] - lr * avg[s] for s in shards])
    zd = float(np.abs(z - (theta - lr * full)).max())
    ws = []
    for k in range(K):
        t = theta.copy()
        for _ in range(H):
            a, b = unflat(t)
            t = t - lr * flat(*_loss_grad(Xs[k], ys[k], a, b)[1:])
        ws.append(t)
    delta = theta - np.mean(ws, 0)
    v = delta
    new = theta - outer_lr * (delta + mu * v)
    L1 = _loss_grad(X, y, *unflat(new))[0]
    return {"dp_diff": dp, "tp_diff": tpd, "zero_diff": zd, "loss_before": L0, "loss_after": float(L1)}
'''

HINTS = [
    'Write the single-device loss and gradient first; every check compares against it.',
    'Equal shards matter: the average of shard means equals the full mean only when the shards have the same size.',
    'In tensor parallelism the ReLU is elementwise on columns of XW1, so splitting columns needs no communication until the final sum.',
    'The difference checks should come out at the level of floating-point round-off, around 1e-15.',
]

def _case(seed, n, d, f):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)); W1 = r.normal(size=(d, f)) / np.sqrt(d); w2 = r.normal(size=f) / np.sqrt(f)
    y = np.maximum(X @ r.normal(size=(d, f)), 0) @ r.normal(size=f) / f
    return X, y, W1, w2

TESTS = [
    _case(1, 16, 4, 6) + (4, 2, 0.1, 5, 0.7, 0.9),
    _case(2, 24, 5, 8) + (3, 4, 0.05, 10, 1.0, 0.0),
    _case(3, 32, 6, 10) + (8, 3, 0.1, 20, 0.7, 0.9),
    _case(4, 12, 3, 5) + (2, 5, 0.2, 1, 1.0, 0.0),
    _case(5, 40, 8, 12) + (5, 2, 0.05, 50, 0.7, 0.9),
]

EXPLANATION = r"""
## Every scheme is the same computation, rearranged
Data parallelism splits the batch and sums gradients; tensor parallelism splits weight matrices and sums partial outputs; ZeRO splits the optimiser update. Done correctly, each reproduces single-device training to round-off. That is the first test any distributed training setup should pass, and these checks are exactly it.

## Where they differ
DiLoCo is the exception: local steps make workers diverge, so the result is a different (communication-cheap) optimiser rather than the same computation. Its loss after one round is the thing to compare against synchronous training for the same number of gradient evaluations.

## Common bugs these checks catch
Unequal shards weighted as if equal, forgetting to divide by the number of workers, a chunk index off by one in the ring, and applying the ReLU after summing tensor-parallel partials instead of before.

## Interview follow-ups
- Why does gradient accumulation over micro-batches also need equal-sized micro-batches to match a big batch exactly?
"""
