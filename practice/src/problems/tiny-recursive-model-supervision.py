import numpy as np

META = dict(title='Tiny Recursive Model: recursion, deep supervision and halting', tags=['Neural Networks', 'Training'],
            diff='Hard', runtime='numpy',
            related=['multi-layer-perceptron-forward-pass', 'cross-entropy-logits', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
The Tiny Recursive Model (TRM) solves puzzles with one very small network applied many times. It keeps two states besides the embedded question $x$: a current answer $y$ and a latent scratchpad $z$. A **latent recursion** refines $z$ several times and then updates $y$; a **deep recursion** repeats that; and **deep supervision** runs the whole thing up to $N_{sup}$ times, computing a loss after each and letting a halting head stop early.

Write `trm_run(x, y0, z0, y_true, p, n, T_rec, n_sup)` returning `(losses, steps, preds)`.

## Pieces (all states are `(L, D)`; $L$ = number of puzzle cells)
- Network (one set of weights, used for both updates): $\mathrm{net}(u) = u + \tanh(uW_1)\,W_2$.
- **Latent recursion**: repeat $n$ times $z \leftarrow \mathrm{net}(x + y + z)$; then $y \leftarrow \mathrm{net}(y + z)$.
- **Deep recursion**: apply the latent recursion $T_{rec}$ times. (In training, the first $T_{rec}-1$ run without gradients; the forward values are the same.)
- Heads after each deep recursion: logits $= yW_{out}$ `(L, V)`; halting logit $q = \mathrm{mean}_{cells}(y)\cdot w_q + b_q$ (a scalar).
- **Step loss** $= \mathrm{mean}_{cells}\,\mathrm{CE}(\text{logits}, y_{true}) + \mathrm{BCE}(\sigma(q),\ c)$, where $c = 1$ if every cell's argmax equals `y_true`, else 0. Compute BCE from the logit stably: $\max(q, 0) - qc + \log(1 + e^{-|q|})$.

## Deep supervision loop
For step $s = 1..n_{sup}$: run a deep recursion from the current $(y, z)$ (carried over from the previous step), compute the step loss and append it to `losses`. Stop after this step if $q > 0$.
- `steps`: number of steps run. `preds`: `(L,)` argmax of the last step's logits.
- `p` keys: `W1` `(D, F)`, `W2` `(F, D)`, `W_out` `(D, V)`, `w_q` `(D,)`, `b_q` (float).
"""

STARTER = '''import numpy as np

def trm_run(x, y0, z0, y_true, p, n, T_rec, n_sup):
    """Return (losses: list of floats, steps: int, preds: (L,) int)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _net(u, p):
    return u + np.tanh(u @ p['W1']) @ p['W2']

def _latent(x, y, z, p, n):
    for _ in range(n):
        z = _net(x + y + z, p)
    return _net(y + z, p), z

def trm_run(x, y0, z0, y_true, p, n, T_rec, n_sup):
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y0, z0))
    y_true = np.asarray(y_true)
    losses, preds = [], None
    for step in range(1, n_sup + 1):
        for _ in range(T_rec):
            y, z = _latent(x, y, z, p, n)
        logits = y @ p['W_out']
        zl = logits - logits.max(1, keepdims=True)
        logp = zl - np.log(np.exp(zl).sum(1, keepdims=True))
        ce = -logp[np.arange(len(y_true)), y_true].mean()
        preds = logits.argmax(1)
        c = float(np.all(preds == y_true))
        q = float(y.mean(0) @ p['w_q'] + p['b_q'])
        bce = max(q, 0.0) - q * c + np.log1p(np.exp(-abs(q)))
        losses.append(float(ce + bce))
        if q > 0:
            break
    return losses, step, preds
'''

HINTS = [
    'Write `net`, then `latent_recursion(x, y, z)` returning the new `(y, z)`. A deep recursion is a loop of `T_rec` latent recursions, and the supervision loop wraps that.',
    'The states are carried across supervision steps: the next step starts from the $(y, z)$ the previous one ended with, not from $(y_0, z_0)$.',
    'The halting target $c$ is whether the whole puzzle is right, computed from this step\'s argmax predictions. Break out of the loop after appending the loss when $q > 0$.',
]

_r = np.random.default_rng(2510)
def _p(D, F, V, bq):
    return dict(W1=_r.normal(size=(D, F)) / np.sqrt(D), W2=_r.normal(size=(F, D)) / np.sqrt(F) * 0.5,
                W_out=_r.normal(size=(D, V)) / np.sqrt(D), w_q=_r.normal(size=D), b_q=bq)
_pA, _pB = _p(4, 8, 3, -3.0), _p(6, 12, 5, 0.0)
_xA, _xB = _r.normal(size=(5, 4)), _r.normal(size=(7, 6))
TESTS = [
    (_xA, np.zeros((5, 4)), np.zeros((5, 4)), np.array([0, 1, 2, 1, 0]), _pA, 2, 1, 1),
    (_xA, np.zeros((5, 4)), np.zeros((5, 4)), np.array([0, 1, 2, 1, 0]), _pA, 3, 2, 4),
    (_xB, np.zeros((7, 6)), np.zeros((7, 6)), _r.integers(0, 5, size=7), _pB, 2, 2, 6),
    (_xB, _r.normal(size=(7, 6)) * 0.1, _r.normal(size=(7, 6)) * 0.1, _r.integers(0, 5, size=7), _pB, 6, 3, 3),
    (_xA, np.zeros((5, 4)), np.zeros((5, 4)), np.array([2, 2, 2, 2, 2]), _p(4, 8, 3, 3.0), 1, 1, 5),
    (_xB, np.zeros((7, 6)), np.zeros((7, 6)), _r.integers(0, 5, size=7), _p(6, 12, 5, -1.0), 1, 3, 8),
]

EXPLANATION = r"""
## The idea
Instead of a deep network, TRM uses one two-layer network and buys depth with recursion: $n$ latent updates plus one answer update per latent recursion, $T_{rec}$ of those per supervision step, and up to $N_{sup}$ steps. The effective depth is about $N_{sup} \cdot T_{rec}(n + 1)$ network applications, from a model with a few million parameters.

## Why carry states across steps
Each supervision step starts from the previous answer and scratchpad, so later steps only need to *improve* a solution. Gradients are cut between steps (and within a deep recursion only the last latent recursion is backpropagated), which keeps memory constant no matter how many recursions run.

## Halting
The halting head learns to predict whether the current answer is already fully correct. At training time it lets easy examples stop early so the batch spends more steps on hard ones; at test time the model can simply run all steps. TRM simplified the earlier HRM's Q-learning halting to this single binary cross-entropy.

## Common bugs
- Resetting $y$ and $z$ at each supervision step.
- Updating $y$ inside the $n$-loop instead of once after it.
- Computing the BCE as `log(sigmoid(q))`, which underflows for large $|q|$.

## Interview follow-ups
- Why might recursion with a 2-layer network generalise better on small puzzle datasets than a 16-layer network trained once?
- What would change if $y$ and $z$ had their own separate networks?
"""
