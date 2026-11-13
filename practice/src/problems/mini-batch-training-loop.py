import numpy as np

META = dict(title='A mini-batch training loop with validation checkpoints', tags=['Training', 'Classification'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:05.07', 'air-fanout-v1:12.01'],
            related=['batch-vs-stochastic-vs-mini-batch-gd', 'early-stopping', 'logistic-regression'])

STATEMENT = r"""
Write the skeleton every training script shares: shuffle, cut into mini-batches, step, evaluate on a validation set after each epoch, and keep the best weights seen so far. The model is binary logistic regression, $p = \sigma(Xw + b)$, trained on the mean binary cross-entropy

$$L(w, b) = -\frac1m \sum_{i=1}^m \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big].$$

Write `train(X, y, Xv, yv, lr, epochs, batch_size, seed, drop_last=False)` that returns a dict.

## Exact procedure
1. `rng = np.random.default_rng(seed)`, created once. Start from `w = np.zeros(d)`, `b = 0.0`.
2. Each epoch: `perm = rng.permutation(n)`. Cut `perm` into consecutive batches of `batch_size`. The final batch may be smaller; if `drop_last` is `True`, skip it when it is smaller than `batch_size` (so an epoch can contain zero steps).
3. For each batch $B$, with $p_B = \sigma(X_B w + b)$, take a plain gradient step on the batch-mean loss:
$$w \leftarrow w - \eta\,\tfrac{1}{|B|} X_B^\top (p_B - y_B), \qquad b \leftarrow b - \eta\,\tfrac{1}{|B|}\textstyle\sum (p_B - y_B).$$
4. At the end of every epoch, record the full training loss $L$ on `(X, y)` and the validation loss on `(Xv, yv)`.
5. The **best epoch** is the 1-based epoch with the lowest validation loss; on an exact tie keep the **earliest**. Snapshot `w` and `b` at that moment.

## Output
A dict with keys:
- `'train_loss'`, `'val_loss'`: lists of Python floats, one per epoch.
- `'best_epoch'`: int. `'w'`: float array `(d,)` and `'b'`: float, the snapshot from the best epoch.

## Conventions
`X`: float `(n, d)`; `y`: 0/1 floats `(n,)`; same for `Xv`, `yv`. Compute $\log p$ and $\log(1-p)$ stably from the logits $s$: $\log\sigma(s) = -\operatorname{softplus}(-s)$ and $\log(1-\sigma(s)) = -\operatorname{softplus}(s)$.
"""

STARTER = '''import numpy as np

def train(X, y, Xv, yv, lr, epochs, batch_size, seed, drop_last=False):
    """Return {'train_loss': [...], 'val_loss': [...], 'best_epoch': int, 'w': (d,), 'b': float}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _bce(X, y, w, b):
    s = X @ w + b
    # -[y log sig(s) + (1-y) log(1-sig(s))] = y softplus(-s) + (1-y) softplus(s)
    return float(np.mean(y * np.logaddexp(0, -s) + (1 - y) * np.logaddexp(0, s)))

def _sigmoid(s):
    e = np.exp(-np.abs(s))
    return np.where(s >= 0, 1 / (1 + e), e / (1 + e))

def train(X, y, Xv, yv, lr, epochs, batch_size, seed, drop_last=False):
    X, y, Xv, yv = (np.asarray(a, dtype=float) for a in (X, y, Xv, yv))
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w, b = np.zeros(d), 0.0
    out = {'train_loss': [], 'val_loss': [], 'best_epoch': 0, 'w': w.copy(), 'b': b}
    best = np.inf
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                continue
            err = _sigmoid(X[idx] @ w + b) - y[idx]          # dL/ds for each sample in the batch
            w = w - lr * X[idx].T @ err / len(idx)
            b = b - lr * float(err.mean())
        out['train_loss'].append(_bce(X, y, w, b))
        v = _bce(Xv, yv, w, b)
        out['val_loss'].append(v)
        if v < best:                                         # strict: ties keep the earliest epoch
            best = v
            out.update(best_epoch=epoch, w=w.copy(), b=float(b))
    return out
'''

HINTS = [
    'Draw one permutation per epoch from a single generator, then slice it: `perm[start:start + batch_size]` for `start` in `range(0, n, batch_size)`.',
    'The gradient of BCE with respect to the logit is just `p - y`, so a batch step is `w -= lr * X_B.T @ (p_B - y_B) / len(B)`. Evaluate both losses only after the epoch\'s last step.',
    'Track the best validation loss with a strict `<` so an equal later value does not replace the earlier epoch, and store **copies** of `w` and `b` at that point.',
]

_r = np.random.default_rng(13)
_wt = np.array([1.5, -2.0, 0.5])
_X = _r.normal(size=(40, 3)); _y = (_X @ _wt + _r.normal(scale=1.0, size=40) > 0).astype(float)
_Xv = _r.normal(size=(20, 3)); _yv = (_Xv @ _wt + _r.normal(scale=1.0, size=20) > 0).astype(float)
TESTS = [
    (np.array([[1.0], [-1.0], [2.0]]), np.array([1.0, 0.0, 1.0]), np.array([[1.0], [-1.0]]), np.array([1.0, 0.0]), 0.5, 2, 2, 0),
    (np.array([[1.0], [-1.0], [2.0]]), np.array([1.0, 0.0, 1.0]), np.array([[1.0], [-1.0]]), np.array([1.0, 0.0]), 0.5, 2, 2, 0, True),
    (_X, _y, _Xv, _yv, 0.3, 8, 8, 1),
    (_X, _y, _Xv, _yv, 0.3, 8, 7, 1, True),
    (_X, _y, _Xv, _yv, 2.0, 25, 4, 2),
    (_X, _y, _Xv, _yv, 0.1, 3, 40, 3),
    (_X[:5], _y[:5], _Xv, _yv, 0.1, 3, 8, 4, True),
    (_X, _y, _Xv, _yv, 0.5, 6, 1, 5),
]

EXPLANATION = r"""
## The loop every framework hides
`DataLoader(shuffle=True, drop_last=...)` produces exactly the batches of step 2, and the checkpoint logic is what early-stopping callbacks implement. Writing it once by hand makes the details visible: one generator for the whole run (re-seeding each epoch would repeat the same order), evaluation after the epoch rather than per step, and copying the weights you keep.

## Why the gradient is $p - y$
$\partial\,\mathrm{BCE}/\partial s = \sigma(s) - y$ for logit $s$. The mean over the batch then gives $\frac{1}{|B|}X_B^\top(p_B - y_B)$.

## drop_last
Without it, the smaller final batch has a noisier gradient but the same learning rate. With BatchNorm a batch of size 1 even breaks training. Test 7 has fewer samples than `batch_size` with `drop_last=True`: no steps happen, every validation loss is exactly $\log 2$, and the tie rule picks epoch 1 with zero weights.

## Complexity
$O(\text{epochs} \cdot n d)$ for the steps plus $O(\text{epochs}\,(n + n_v) d)$ for the evaluations.

## Common bugs
- Storing `w` itself as the best weights and then mutating it in place, so the "snapshot" follows training.
- Evaluating the training loss as the average of batch losses during the epoch; that measures moving weights.
- Using `<=` for the best epoch (picks the latest tie).

## Interview follow-ups
- Why shuffle every epoch instead of once? What goes wrong if the data is sorted by label?
- How would you add gradient accumulation so the effective batch is $4\times$ `batch_size`?
"""
