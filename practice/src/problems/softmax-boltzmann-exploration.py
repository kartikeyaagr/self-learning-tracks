import numpy as np

META = dict(title='Boltzmann exploration with a temperature', tags=['Reinforcement Learning', 'Probability'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.02'],
            related=['epsilon-greedy-action-selection', 'softmax-rows', 'expected-sarsa'])

STATEMENT = r"""
Boltzmann (softmax) exploration picks actions with probability that grows with their value:

$$\pi(a \mid s) = \frac{\exp(Q(s,a) / \tau)}{\sum_b \exp(Q(s,b) / \tau)}$$

The temperature $\tau > 0$ sets how greedy it is: small $\tau$ is nearly greedy, large $\tau$ nearly uniform.

Write `boltzmann(Q, tau, u)` and return `(probs, actions)`.

## Inputs
- `Q`: float array `(B, A)`.
- `tau`: float, `> 0`. It may be tiny (e.g. `1e-3`), so compute stably: no `inf` or `nan`.
- `u`: float array `(B,)` of uniform draws in `[0, 1)`.

## Sampling rule (inverse CDF)
Row $i$ takes the **first** action $a$ whose cumulative probability `cumsum(probs[i])[a]` is strictly greater than `u[i]`. If rounding leaves no such action, take the last action.

## Output
`probs`: float array `(B, A)`, each row summing to 1. `actions`: int array `(B,)`.
"""

STARTER = '''import numpy as np

def boltzmann(Q, tau, u):
    """Q: (B, A), u: (B,). Return (probs (B, A), sampled actions (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def boltzmann(Q, tau, u):
    z = np.asarray(Q, dtype=float) / tau
    z -= z.max(axis=1, keepdims=True)                  # stable: largest logit becomes 0
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    actions = (cdf <= np.asarray(u)[:, None]).sum(axis=1)   # count of cdf entries not above u = first index above u
    return p, np.minimum(actions, p.shape[1] - 1)
'''

HINTS = [
    'Divide by $\\tau$, then subtract the row max before `exp` so a tiny temperature cannot overflow.',
    'For sampling, take the cumulative sum along each row. The first index with `cdf > u` is the number of entries with `cdf <= u`.',
    'Clip the result to `A - 1` to cover the case where rounding makes the last cumulative value slightly below `u`.',
]

_r = np.random.default_rng(131)
QR = _r.normal(size=(6, 4))
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0, np.array([0.7, 0.7])),
    (np.array([[1.0, 2.0, 3.0]]), 0.5, np.array([0.1])),
    (np.array([[1.0, 2.0, 3.0]]), 1e-3, np.array([0.99])),
    (np.array([[1.0, 2.0, 3.0]]), 1e6, np.array([0.5])),
    (np.array([[1000.0, 1001.0]]), 1.0, np.array([0.2])),
    (np.array([[3.0]]), 0.1, np.array([0.999])),
    (QR, 0.7, _r.uniform(size=6)),
    (QR * 5, 2.0, _r.uniform(size=6)),
]

EXPLANATION = r"""
## Why softmax instead of epsilon-greedy
$\epsilon$-greedy explores uniformly, so a terrible action is tried as often as a nearly-best one. Boltzmann exploration ranks actions: the closer an action's value is to the best, the more often it is tried. The price is a temperature that depends on the scale of Q (multiplying Q by 5 is the same as dividing $\tau$ by 5, compare tests 7 and 8).

## Stability
At $\tau = 10^{-3}$ the logits are around 3000, and `exp(3000)` overflows. Subtracting the row maximum leaves the probabilities unchanged and makes the largest exponent $e^0 = 1$ (tests 3 and 5). As $\tau \to 0$ the policy becomes the argmax; as $\tau \to \infty$ it becomes uniform (test 4).

## Inverse-CDF sampling
Splitting $[0, 1)$ into consecutive intervals of length $\pi(a)$ and seeing which one contains $u$ gives action $a$ with probability $\pi(a)$. The clip handles a final cumulative value of $0.9999999999999999$ with $u$ above it.

## Complexity
$O(BA)$ time and memory.

## Common bugs
- Using `cdf >= u` (off by one at exact boundaries) or `argmax(cdf > u)` without handling the all-False row.
- Normalising across the whole batch instead of per row.

## Interview follow-ups
- How is this the same as sampling from an LLM with temperature?
- Soft Q-learning and SAC use an entropy-regularised objective whose optimal policy is exactly a Boltzmann policy. Why?
"""
