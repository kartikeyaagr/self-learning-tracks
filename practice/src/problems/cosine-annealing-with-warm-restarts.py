import numpy as np

META = dict(title='Cosine annealing with warm restarts', tags=['Training', 'Optimization'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.05'],
            related=['linear-warmup', 'lr-range-test-and-one-cycle-policy'])

STATEMENT = r"""
SGDR (stochastic gradient descent with warm restarts) anneals the learning rate along a half cosine, then jumps back up and starts over. Cycle $k = 0, 1, 2, \dots$ lasts $T_k = T_0 \cdot m^k$ steps, where $m$ = `t_mult` is a positive integer. Inside a cycle, at offset $t_{\text{cur}} \in \{0, \dots, T_k - 1\}$:

$$\eta = \eta_{\min} + \tfrac12\left(\eta^{(k)}_{\max} - \eta_{\min}\right)\left(1 + \cos\frac{\pi\, t_{\text{cur}}}{T_k}\right), \qquad \eta^{(k)}_{\max} = \eta_{\max}\cdot \gamma^k,$$

with $\gamma$ = `decay` shrinking the peak after every restart.

Write `sgdr_lr(steps, lr_max, lr_min, t0, t_mult=1, decay=1.0)` that returns the learning rate for every global step in `steps` (a float array of the same shape).

## Details
- `steps`: int array `(k,)` of global step indices `>= 0`, any order. Step 0 is the first step of cycle 0.
- Example with `t0=2, t_mult=2`: cycles cover steps `[0, 1]`, `[2..5]`, `[6..13]`, `[14..29]`, ...
- `t0 >= 1`; `t_mult >= 1` (int); `0 < decay <= 1`; `lr_min <= lr_max`.
- Cycle boundaries must be exact. Compute them with integer arithmetic, not with logarithms.
"""

STARTER = '''import numpy as np

def sgdr_lr(steps, lr_max, lr_min, t0, t_mult=1, decay=1.0):
    """Return the SGDR learning rate at each global step, float array (k,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sgdr_lr(steps, lr_max, lr_min, t0, t_mult=1, decay=1.0):
    t = np.asarray(steps, dtype=np.int64)
    cycle = np.zeros_like(t)
    start = np.zeros_like(t)                 # first global step of each element's cycle
    length = np.full_like(t, t0)
    if t.size:
        # Walk cycles until every step is covered; the number of cycles is logarithmic when t_mult > 1.
        # ponytail: for t_mult == 1 this loops max(t)/t0 times; a floor-divide shortcut would be O(1).
        begin, T, k = 0, t0, 0
        while begin <= t.max():
            inside = (t >= begin) & (t < begin + T)
            cycle[inside], start[inside], length[inside] = k, begin, T
            begin, T, k = begin + T, T * t_mult, k + 1
    cur = t - start
    peak = lr_max * float(decay) ** cycle
    return lr_min + 0.5 * (peak - lr_min) * (1 + np.cos(np.pi * cur / length))
'''

HINTS = [
    'For each step you need three things: which cycle it is in, where that cycle started, and how long it is.',
    'Generate cycles one at a time with integers (`begin += T; T *= t_mult`) until `begin` passes the largest step, and assign them to steps with a boolean mask.',
    'With `t_mult = 1` it simplifies to `k = t // t0`, `t_cur = t % t0`. A log-based formula for `k` can land one cycle off exactly at a restart because of round-off.',
]

TESTS = [
    (np.arange(8), 1.0, 0.0, 4),
    (np.arange(14), 1.0, 0.0, 2, 2),
    (np.array([0, 2, 6, 14, 30, 29]), 0.1, 0.001, 2, 2, 0.5),
    (np.array([0, 9, 10, 99, 100, 101]), 0.05, 0.0, 10, 1, 0.9),
    (np.array([0]), 0.3, 0.3, 1),
    (np.array([5, 0, 3]), 1.0, 0.1, 1, 3),
    (np.array([120, 121, 122, 1092, 1093]), 1e-3, 1e-5, 1, 3, 0.8),
]

EXPLANATION = r"""
## Why the cosine shape
The half cosine stays near $\eta_{\max}$ for a while (fast exploration), drops through the middle, and flattens near $\eta_{\min}$ (careful convergence). Compared with step decay it has no hyperparameters beyond the endpoints and length, and empirically matches or beats hand-tuned step schedules.

## Why restart
Each restart kicks the iterate out of the current basin with a large step. Ending every cycle at a low rate gives a good model at each cycle boundary, so the snapshots can be ensembled for free ("snapshot ensembles"). Growing cycles ($m = 2$) spend more time refining as training goes on, and $\gamma < 1$ makes later restarts gentler.

## Finding the cycle exactly
With $m > 1$ the cycle starts are $T_0 (m^k - 1)/(m - 1)$. Inverting that with $\log$ works mathematically, but at the boundary $t = 121$ for $T_0 = 1, m = 3$, `math.log(2*121 + 1) / math.log(3)` evaluates to `4.999999999999999`, so `floor` puts step 121 in cycle 4 instead of 5 (test 7 checks it). Integer cycle walking, or a log estimate followed by an integer correction, avoids it.

## Complexity
$O(k \cdot C)$ for $k$ steps and $C$ cycles, with $C = O(\log_m t_{\max})$ when $m > 1$.

## Common bugs
- Treating the last step of a cycle as $t_{\text{cur}} = T_k$ (rate exactly $\eta_{\min}$) instead of $T_k - 1$: the minimum is approached, never reached, before the restart.
- Applying `decay` to $\eta_{\min}$ as well as the peak.
- Mixing up per-epoch and per-step restarts when wiring it into a training loop.

## Interview follow-ups
- Why do people now prefer a single cosine decay with warmup over restarts for LLM pretraining? (Restarts waste compute re-climbing; one long anneal to near zero is what matters most for final loss.)
- How would you combine linear warmup with this schedule? (Prepend a warmup segment and shift the global step used for the cosine by the warmup length.)
"""
