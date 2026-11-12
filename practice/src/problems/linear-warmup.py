import numpy as np

META = dict(title='Linear warmup, then linear decay', tags=['Training', 'Optimization'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:05.05', 'air-fanout-v1:10.04'],
            related=['cosine-annealing-with-warm-restarts', 'lr-range-test-and-one-cycle-policy'])

STATEMENT = r"""
Transformers are usually trained with a learning rate that ramps up linearly for a few thousand steps, then decays. Implement the common "warmup + linear decay" schedule, evaluated at many steps at once.

For an integer step $t \ge 0$ (the update about to happen), with $W$ = `warmup` and $T$ = `total`:

$$\eta(t) = \begin{cases} \eta_{\text{base}}\,\dfrac{t + 1}{W} & 0 \le t < W \\[2mm] \eta_{\text{base}} + (\eta_{\text{final}} - \eta_{\text{base}})\,\dfrac{t - W}{T - W} & W \le t < T \\[2mm] \eta_{\text{final}} & t \ge T \end{cases}$$

Write `warmup_decay_lr(steps, base_lr, warmup, total, final_lr=0.0)` that returns a float array the same shape as `steps`.

## Inputs
- `steps`: int array `(k,)` of non-negative steps, in any order, repeats allowed.
- Integers `0 <= warmup <= total` and `total >= 1`. With `warmup == 0` there is no ramp.
- Vectorise with `np.where` or masks, not a Python loop over steps, and don't divide by zero when `warmup == 0` or `warmup == total`.
"""

STARTER = '''import numpy as np

def warmup_decay_lr(steps, base_lr, warmup, total, final_lr=0.0):
    """steps: int array (k,). Return the learning rate at each step, float array (k,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def warmup_decay_lr(steps, base_lr, warmup, total, final_lr=0.0):
    t = np.asarray(steps, dtype=float)
    ramp = base_lr * (t + 1) / max(warmup, 1)              # max(.,1): unused when warmup == 0
    frac = (t - warmup) / max(total - warmup, 1)           # unused when warmup == total
    decay = base_lr + (final_lr - base_lr) * frac
    return np.where(t < warmup, ramp, np.where(t < total, decay, float(final_lr)))
'''

HINTS = [
    'Compute all three pieces for every step, then pick per step with nested `np.where(t < warmup, ..., np.where(t < total, ..., final))`.',
    'The ramp uses `t + 1`, so step 0 already has a nonzero rate and step `warmup - 1` reaches exactly `base_lr`.',
    '`np.where` evaluates both branches, so guard the denominators: `max(warmup, 1)` and `max(total - warmup, 1)` are only used where their branch is selected anyway.',
]

TESTS = [
    (np.array([0, 1, 2, 3, 4, 5, 6, 7]), 1.0, 3, 7),
    (np.array([0, 2, 4, 9, 10, 15]), 0.1, 0, 10, 0.01),
    (np.array([0, 1, 4, 5, 6]), 2.0, 5, 5, 0.5),
    (np.array([100, 0, 50, 3]), 3e-4, 4, 80, 3e-5),
    (np.array([0]), 1.0, 1, 1),
    (np.arange(0, 4000, 250), 6e-4, 1000, 3000, 6e-5),
]

EXPLANATION = r"""
## Why warm up
At initialisation, Adam's second-moment estimate $v$ is built from very few gradients, and in transformers the early gradients are large and poorly aligned. A full-size step in that state can push weights into a region the model never recovers from (loss spikes, divergence, or dead attention heads). Ramping the rate linearly gives $v$ time to settle and lets the network reach a smoother region before taking big steps. Post-LayerNorm transformers in particular barely train without it.

## Why decay
Near a minimum, SGD noise keeps the iterate bouncing in a ball whose radius scales with $\eta$. Decaying $\eta$ shrinks the ball, which is why loss typically drops sharply as the rate falls toward `final_lr`.

## Complexity
$O(k)$ for $k$ queried steps, fully vectorised.

## Common bugs
- Off-by-one in the ramp: using $t/W$ makes step 0 have rate 0, wasting an update. (Both conventions exist; read the spec.)
- Dividing by `total - warmup` when it is 0. `np.where` evaluates every branch, so the warning fires even if the branch is never selected.
- Clamping only `t >= total` and forgetting that the decay segment should end exactly at `final_lr`.

## Interview follow-ups
- How long should warmup be? (Common heuristics: 1-5% of training, or a fixed few thousand steps; with Adam, about $2/(1-\beta_2)$ steps has been suggested.)
- What is the "inverse square root" schedule from the original transformer paper? ($\eta \propto \min(t^{-1/2},\, t \cdot W^{-3/2})$: linear warmup, then $1/\sqrt t$ decay.)
"""
