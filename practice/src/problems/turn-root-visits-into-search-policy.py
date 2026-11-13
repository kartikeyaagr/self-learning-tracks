import numpy as np

META = dict(title='Root visit counts to a move policy with temperature', tags=['Reinforcement Learning'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:02.02'],
            related=['mix-dirichlet-noise-into-root-priors', 'alphago-policy-value-loss',
                     'run-deterministic-neural-guided-mcts'])

STATEMENT = r"""
After the simulations at a root finish, the visit counts $N(a)$ are a sharper policy than the network's prior. They are both the training target and the distribution the self-play move is drawn from:

$$\pi(a) = \frac{N(a)^{1/\tau}}{\sum_b N(b)^{1/\tau}}$$

Early in a game AlphaGo Zero used $\tau = 1$ (play proportionally to visits, for diverse openings), later $\tau \to 0$ (play the most visited move).

Write `search_policy(visits, tau, u)` returning `(pi, action)`.

## Rules
- `visits`: int array `(A,)`, non-negative, at least one entry positive.
- `tau > 0`: use the formula above. It must not overflow for tiny `tau` (e.g. `tau = 0.01` with counts in the hundreds): $N^{100}$ is far beyond the float range, but the ratios are not.
- `tau == 0`: the limit of the formula. `pi` puts equal mass on every action whose count equals the maximum and `0` elsewhere.
- Actions with zero visits always get `pi = 0`.
- `action`: draw from `pi` by inversion with the given uniform number `u` in $[0, 1)$: the smallest index `a` with $\sum_{b \le a} \pi(b) > u$. If rounding leaves no such index, return the last action with `pi > 0`.

Return `pi` as a float array `(A,)` and `action` as a Python `int`.
"""

STARTER = '''import numpy as np

def search_policy(visits, tau, u):
    """Return (pi float (A,), action int)."""
    v = np.asarray(visits, dtype=float)
    pi = v ** (1.0 / tau)                           # overflows for small tau, and tau == 0 divides by zero
    pi = pi / pi.sum()
    return pi, int(np.argmax(pi))
'''

SOLUTION = '''import numpy as np

def search_policy(visits, tau, u):
    v = np.asarray(visits, dtype=float)
    if tau == 0:
        best = v == v.max()
        pi = best / best.sum()                      # split evenly among the most visited moves
    else:
        pi = (v / v.max()) ** (1.0 / tau)           # ratios <= 1 cannot overflow; 0 stays 0
        pi = pi / pi.sum()
    cum = np.cumsum(pi)
    action = int(np.searchsorted(cum, u, side='right'))   # first index with cum > u
    return pi, min(action, int(np.flatnonzero(pi)[-1]))
'''

HINTS = [
    'Raising counts to $1/\\tau$ overflows, but dividing every count by the largest one first changes nothing after normalisation and keeps every value in $[0, 1]$.',
    'Handle `tau == 0` separately: mark the entries equal to the maximum and divide that boolean mask by its sum.',
    'For sampling, `np.searchsorted(np.cumsum(pi), u, side="right")` returns the first index whose cumulative sum is greater than `u`; clamp it to the last index with positive probability.',
]

_rng = np.random.default_rng(43)
_v = _rng.integers(0, 400, 26); _v[[3, 9]] = 0

TESTS = [
    (np.array([10, 30, 60]), 1.0, 0.5),
    (np.array([5, 9, 9, 0]), 0, 0.75),
    (np.array([10, 30, 60]), 0.5, 0.05),
    (np.array([120, 480, 400, 0]), 0.01, 0.3),
    (np.array([0, 7, 0, 3]), 10.0, 0.6),
    (np.array([0, 0, 4]), 1.0, 0.999),
    (np.array([1, 1, 1, 1]), 0, 0.3),
    (_v, 1.0, 0.618),
    (_v, 0.25, 0.42),
]

EXPLANATION = r"""
## Why visits, not values
Visit counts aggregate the whole search: PUCT keeps sending simulations to a move only while its $Q + U$ stays competitive, so $N(a)$ reflects both the network's prior and the evidence gathered. They are also robust: a move whose $Q$ is high from one lucky simulation has few visits. Training the policy head toward $\pi$ is what turns search into policy improvement.

## Temperature without overflow
$\pi(a) \propto N(a)^{1/\tau} = N_\text{max}^{1/\tau} (N(a)/N_\text{max})^{1/\tau}$, and the constant factor cancels in the normalisation. After dividing by the max, every base is in $[0, 1]$, so powers can only underflow toward 0, which is the correct limit. As $\tau \to 0$ only entries equal to the maximum keep base 1, hence the even split among ties.

## Sampling by inversion
For $u \sim U[0, 1)$, the first index whose cumulative probability exceeds $u$ is distributed exactly as $\pi$. Zero-probability actions have zero-width intervals and are never chosen, which `side="right"` handles.

## Common bugs
- `visits ** (1 / tau)` overflowing to `inf` and producing `nan` after normalisation.
- `np.argmax` for $\tau = 0$, which is fine for play but wrong as a training target when two moves tie.
- `side="left"` in `searchsorted`, which can select a zero-probability action whose cumulative sum equals `u`.

## Interview follow-ups
- Why train on the full visit distribution rather than a one-hot of the move actually played?
- KataGo prunes "forced playout" visits from the target. Why would visits that exploration forced onto a move distort the policy target?
"""
