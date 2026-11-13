import numpy as np

META = dict(title='Combined policy-value loss for self-play training', tags=['Loss Functions', 'Reinforcement Learning'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:01.11', 'air-fanout-v1:02.04'],
            related=['turn-root-visits-into-search-policy', 'apply-one-policy-value-head-training-step',
                     'cross-entropy-logits'])

STATEMENT = r"""
Each training example is a position with two targets produced by self-play: the search policy $\pi$ and the final game result $z \in \{-1, 0, 1\}$ from the point of view of the player to move. The network is trained to predict both, with L2 regularisation:

$$\ell = \frac{1}{B}\sum_{i=1}^{B} \Big[(z_i - v_i)^2 \;-\; \sum_a \pi_{i,a} \log p_{i,a}\Big] \;+\; c \sum_{\theta \in \text{weights}} \lVert\theta\rVert_2^2$$

where $p_i = \mathrm{softmax}(\text{logits}_i)$.

Write `policy_value_loss(logits, values, target_pi, z, weights, c)` returning the tuple `(total, value_loss, policy_loss, l2)` of Python floats, where `value_loss` and `policy_loss` are the batch means of the two bracketed terms, `l2` is the regularisation term (already multiplied by `c`), and `total` is their sum.

## Inputs
- `logits`: float `(B, A)`. Illegal moves may be masked with `-inf`; every row has at least one finite logit.
- `values`: float `(B,)`, the value head outputs. `z`: float `(B,)`.
- `target_pi`: float `(B, A)`, each row non-negative and summing to 1, and `0` wherever the logit is `-inf`.
- `weights`: list of float arrays of any shapes (biases are usually excluded by the caller). `c`: float.

## Numerics
Compute $\log p$ with a stable log-softmax. Terms with $\pi_{i,a} = 0$ contribute exactly `0` (do not let $0 \cdot (-\infty)$ become `nan`). Logits can be large (hundreds).
"""

STARTER = '''import numpy as np

def policy_value_loss(logits, values, target_pi, z, weights, c):
    """Return (total, value_loss, policy_loss, l2) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def policy_value_loss(logits, values, target_pi, z, weights, c):
    logits = np.asarray(logits, dtype=float)
    pi = np.asarray(target_pi, dtype=float)
    m = logits.max(axis=1, keepdims=True)                         # finite: each row has a legal move
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    ce = -(pi * np.where(pi > 0, logp, 0.0)).sum(axis=1)          # never form 0 * (-inf)
    value_loss = float(np.mean((np.asarray(z, dtype=float) - np.asarray(values, dtype=float)) ** 2))
    policy_loss = float(ce.mean())
    l2 = float(c * sum(np.sum(np.asarray(w, dtype=float) ** 2) for w in weights))
    return value_loss + policy_loss + l2, value_loss, policy_loss, l2
'''

HINTS = [
    'Split it into three numbers: mean squared value error, mean cross-entropy against `target_pi`, and `c` times the sum of squared weights.',
    'Stable log-softmax per row: `logits - m - log(sum(exp(logits - m)))` with `m` the row max. Masked `-inf` logits give `exp(-inf) = 0` and `logp = -inf`.',
    'Never let `0 * -inf` happen: replace `logp` by 0 wherever `pi == 0` before multiplying, e.g. `-(pi * np.where(pi > 0, logp, 0.0)).sum(axis=1)`, then average over the batch.',
]

_rng = np.random.default_rng(47)
_B, _A = 4, 10
_lg = _rng.normal(0, 2, (_B, _A))
_pi = _rng.random((_B, _A)); _pi /= _pi.sum(axis=1, keepdims=True)
_v = np.tanh(_rng.normal(size=_B)); _z = np.array([1.0, -1.0, -1.0, 1.0])
_ws = [_rng.normal(0, 0.1, (3, 4)), _rng.normal(0, 0.1, 5)]
_mask = _rng.random((_B, _A)) < 0.3; _mask[:, 0] = False
_lgm = np.where(_mask, -np.inf, _lg)
_pim = np.where(_mask, 0.0, _pi); _pim /= _pim.sum(axis=1, keepdims=True)

TESTS = [
    (np.array([[0.0, 0.0]]), np.array([0.5]), np.array([[0.5, 0.5]]), np.array([1.0]), [np.array([1.0, 2.0])], 0.1),
    (np.array([[2.0, 0.0, -np.inf]]), np.array([0.0]), np.array([[1.0, 0.0, 0.0]]), np.array([-1.0]), [], 0.0),
    (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), np.array([0.2, -0.4]), np.array([[0.0, 0.0, 1.0], [0.2, 0.3, 0.5]]),
     np.array([1.0, 0.0]), [np.ones((2, 2))], 1e-4),
    (np.array([[500.0, 0.0, -500.0]]), np.array([0.99]), np.array([[0.0, 1.0, 0.0]]), np.array([1.0]), [], 0.0),
    (_lg, _v, _pi, _z, _ws, 1e-4),
    (_lgm, _v, _pim, _z, _ws, 1e-2),
]

EXPLANATION = r"""
## What each term does
- $(z - v)^2$ teaches the value head to predict the game outcome. It is regression on a $\pm 1$ target; with a `tanh` output the prediction lives in the same range.
- $-\pi^\top \log p$ is the cross-entropy to the search policy. Its gradient with respect to the logits is $p - \pi$, so the network is pulled toward what search found, which is the policy-improvement step of the loop.
- $c \lVert\theta\rVert^2$ is weight decay. AlphaGo Zero used $c = 10^{-4}$.

The two prediction terms are weighted equally. Later work reduced the value weight, since $z$ is one noisy bit per game and the value head can overfit to it when many positions share the same outcome.

## Numerics
The log-sum-exp shift by the row max keeps `exp` from overflowing (test 4 has logits of 500). Masked moves have $\log p = -\infty$ but $\pi = 0$, and the `where` makes their contribution exactly zero rather than `nan`.

## Complexity
$O(BA)$ for the losses plus $O(|\theta|)$ for the regulariser.

## Common bugs
- `np.log(softmax(logits))`, which returns `-inf` (and then `nan`) when probabilities underflow.
- Summing the policy term over the batch but averaging the value term, which silently re-weights them.
- Regularising biases and batch-norm parameters (the caller should not pass them).

## Interview follow-ups
- The policy loss can be written as $H(\pi) + \mathrm{KL}(\pi \,\Vert\, p)$. Why is minimising it the same as minimising the KL divergence?
- If self-play games are long and all positions share one $z$, how does that correlation affect value-head overfitting, and what do engines do about it?
"""
