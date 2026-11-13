import numpy as np

META = dict(title='Gradient accumulation with padding masks', tags=['Training', 'Optimization'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:12.01'],
            related=['batch-vs-stochastic-vs-mini-batch-gd', 'gradient-clipping'])

STATEMENT = r"""
When a batch doesn't fit in memory, you split it into micro-batches, run backward on each, **add** the gradients, and step the optimizer once. Done right, the result is identical to one big batch. Done wrong (averaging per-micro-batch means), it silently reweights samples whenever micro-batches contain different amounts of real data.

Train a linear model $\hat y = x^\top w$ with SGD. The data arrives as a list of micro-batches `(X_k, y_k, m_k)`: `X_k` float `(b_k, d)`, `y_k` float `(b_k,)`, and `m_k` float `(b_k,)` of non-negative weights (0 marks a padding row, 1 a real one; other weights are allowed). Every micro-batch has at least one row.

Optimizer step $s$ ($s = 0, 1, \dots$) consumes micro-batches $sA, \dots, sA + A - 1$ where $A$ = `accum`. If the number of micro-batches isn't a multiple of $A$, the last step uses the leftovers. For the micro-batches $\mathcal B_s$ of step $s$, with $w$ the weights at the start of the step:

$$L_s = \frac{\sum_{k \in \mathcal B_s}\sum_i m_{ki}\,(x_{ki}^\top w - y_{ki})^2}{\sum_{k\in\mathcal B_s}\sum_i m_{ki}}, \qquad w \leftarrow w - \eta\, \nabla_w L_s.$$

That is, the loss is the weighted mean over **all** real rows in the step, not a mean of micro-batch means.

If a step's total weight is 0 (all padding), skip the update and record its loss as `0.0`.

Write `accumulated_sgd(w0, micro_batches, accum, lr)` returning `(w, losses)`: the final weights `(d,)` and a float array with one loss $L_s$ per optimizer step (computed before that step's update).
"""

STARTER = '''import numpy as np

def accumulated_sgd(w0, micro_batches, accum, lr):
    """micro_batches: list of (X_k, y_k, m_k). Return (w (d,), losses (num_steps,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def accumulated_sgd(w0, micro_batches, accum, lr):
    w = np.array(w0, dtype=float)
    losses = []
    for s in range(0, len(micro_batches), accum):
        # Accumulate un-normalised sums; divide once by the step's total weight.
        sq_sum, grad_sum, weight = 0.0, np.zeros_like(w), 0.0
        for X, y, m in micro_batches[s:s + accum]:
            X, y, m = (np.asarray(a, dtype=float) for a in (X, y, m))
            r = X @ w - y                          # residuals at the step's starting weights
            sq_sum += np.sum(m * r * r)
            grad_sum += 2.0 * X.T @ (m * r)
            weight += m.sum()
        if weight == 0:
            losses.append(0.0)                     # nothing real in this step: no update
            continue
        losses.append(sq_sum / weight)
        w = w - lr * grad_sum / weight
    return w, np.array(losses)
'''

HINTS = [
    'Inside one optimizer step, the weights don\'t change. Every micro-batch in the step uses the same `w`.',
    'Accumulate three running sums across the step\'s micro-batches: weighted squared error, weighted gradient numerator $2X^\\top(m \\odot r)$, and total weight. Divide by the total weight only at the end.',
    'Loop `for s in range(0, len(micro_batches), accum)` and slice `micro_batches[s:s + accum]`; slicing hands you the short last group for free. Check for zero total weight before dividing.',
]

_r = np.random.default_rng(47)
def _mb(b, pad, d=3):
    X = _r.normal(size=(b, d))
    y = X @ np.array([1.0, -1.0, 0.5]) + 0.05 * _r.normal(size=b)
    m = np.ones(b)
    m[b - pad:] = 0.0                               # padding rows at the end
    return (X, y, m)
_uneven = [_mb(4, 0), _mb(4, 3), _mb(2, 0), _mb(4, 1), _mb(3, 0), _mb(4, 2), _mb(1, 0)]
TESTS = [
    (np.array([0.0]), [(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), np.array([1.0, 1.0])),
                       (np.array([[3.0]]), np.array([6.0]), np.array([1.0]))], 2, 0.05),
    (np.array([0.0]), [(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), np.array([1.0, 0.0])),
                       (np.array([[3.0]]), np.array([6.0]), np.array([1.0]))], 1, 0.05),
    (np.zeros(3), _uneven, 2, 0.1),
    (np.zeros(3), _uneven, 3, 0.1),
    (np.zeros(3), _uneven, 7, 0.1),
    (np.zeros(3), _uneven, 1, 0.05),
    (np.array([1.0, 1.0]), [(np.array([[1.0, 0.0]]), np.array([5.0]), np.array([0.0])),
                            (np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0]), np.array([0.0, 0.0])),
                            (np.array([[2.0, 0.0]]), np.array([1.0]), np.array([0.5]))], 2, 0.1),
]

EXPLANATION = r"""
## Why sums, not means
The big-batch gradient is $\frac{1}{M}\sum_{\text{all rows}} m_i \nabla \ell_i$ with $M = \sum m_i$. Splitting the sum across micro-batches is exact. Averaging micro-batch means instead computes $\frac1A\sum_k \frac{1}{M_k}\sum_{i\in k} m_i\nabla\ell_i$, which gives each row weight $1/(A M_k)$: rows in a mostly-padding micro-batch count several times more than rows in a full one. In language-model training with variable-length sequences this is a real bug that has shipped in popular libraries; the fix is to normalise by the step's **total** token count.

Test 2 vs a naive version: with `accum=1` each step is a single micro-batch, so both agree. Test 3's micro-batches have 4, 1, 2, 3, 3, 2, 1 real rows, so a mean-of-means version drifts.

## Memory and compute
Accumulation trades time for memory: activations for only one micro-batch are alive at once, while the gradient buffer is the same size as the weights. Compute per step is unchanged; the number of optimizer steps falls by $A$.

## Complexity
$O(Nd)$ per epoch for $N$ total rows, $O(d)$ extra memory for the gradient accumulator.

## Common bugs
- Updating `w` after every micro-batch (that's just a smaller batch, not accumulation).
- Dividing each micro-batch's loss by `accum` and by its own row count: correct only when every micro-batch has the same real-row count.
- Dropping the leftover micro-batches when the count isn't a multiple of `accum`.

## Interview follow-ups
- Does gradient accumulation exactly reproduce large-batch training when the model has BatchNorm? (No: BatchNorm statistics are computed per micro-batch. LayerNorm/RMSNorm models are unaffected.)
- In data-parallel training, how does this relate to all-reduce? (Each worker is a "micro-batch"; the all-reduce must sum gradients and divide by the **global** token count, not average per-worker means.)
"""
