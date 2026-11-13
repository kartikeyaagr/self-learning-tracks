import numpy as np

META = dict(title='Warmup-stable-decay learning-rate schedule', tags=['Training', 'Scaling Laws'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:05.05'],
            related=['linear-warmup', 'cosine-annealing-with-warm-restarts', 'fit-learning-rate-and-batch-size-scaling'])

STATEMENT = r"""
A cosine schedule must know the total number of steps in advance, so every point on a scaling curve needs its own training run. The **warmup-stable-decay** (WSD) schedule avoids that: it holds the peak learning rate for most of training and only decays in a short final phase. One long stable run can then be branched at any checkpoint into a short decay, giving a finished model for each token budget.

Write `wsd_schedule(total_steps, peak_lr, warmup_steps, decay_steps, min_lr, shape)` that returns a float64 array of length `total_steps` holding the learning rate used at steps $t = 0, 1, \dots, T-1$.

## Phases
With $T$ = `total_steps`, $W$ = `warmup_steps`, $K$ = `decay_steps` and $W + K \le T$:

1. **Warmup**, $t < W$: $\eta_t = \eta_\text{peak} \cdot \frac{t + 1}{W}$, so the last warmup step reaches the peak.
2. **Stable**, $W \le t < T - K$: $\eta_t = \eta_\text{peak}$.
3. **Decay**, $t \ge T - K$: with progress $s = \frac{t - (T - K) + 1}{K} \in (0, 1]$,
   $$\eta_t = \eta_\text{min} + (\eta_\text{peak} - \eta_\text{min})\, f(s)$$
   so the final step lands exactly on $\eta_\text{min}$.

## Decay shapes
| `shape` | $f(s)$ |
|---|---|
| `"linear"` | $1 - s$ |
| `"cosine"` | $\tfrac12\big(1 + \cos(\pi s)\big)$ |
| `"sqrt"` | $1 - \sqrt{s}$ |

$W = 0$ means no warmup and $K = 0$ means no decay. `total_steps` $\ge 1$; the step counts are ints, the rates floats. No Python loop over steps is needed.
"""

STARTER = '''import numpy as np

def wsd_schedule(total_steps, peak_lr, warmup_steps, decay_steps, min_lr, shape):
    """Return a float64 array (total_steps,) of learning rates."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

_SHAPES = {
    "linear": lambda s: 1.0 - s,
    "cosine": lambda s: 0.5 * (1.0 + np.cos(np.pi * s)),
    "sqrt": lambda s: 1.0 - np.sqrt(s),
}

def wsd_schedule(total_steps, peak_lr, warmup_steps, decay_steps, min_lr, shape):
    t = np.arange(total_steps, dtype=float)
    lr = np.full(total_steps, float(peak_lr))            # stable phase by default

    if warmup_steps > 0:
        w = t < warmup_steps
        lr[w] = peak_lr * (t[w] + 1) / warmup_steps

    if decay_steps > 0:
        start = total_steps - decay_steps
        d = t >= start
        s = (t[d] - start + 1) / decay_steps             # in (0, 1]; s = 1 on the last step
        lr[d] = min_lr + (peak_lr - min_lr) * _SHAPES[shape](s)
    return lr
'''

HINTS = [
    'Start from an array full of the peak rate and overwrite the warmup and decay slices with boolean masks on `t = np.arange(T)`.',
    'Both phases use a "+1": warmup step $t$ uses $(t+1)/W$ and decay progress is $(t - (T-K) + 1)/K$. That way neither phase starts at a zero rate or ends before reaching its target.',
    'Keep the three decay shapes in a dict of functions of $s$; guard $W = 0$ and $K = 0$ so you never divide by zero.',
]

TESTS = [
    (10, 1.0, 2, 4, 0.0, 'linear'),
    (6, 0.5, 0, 3, 0.1, 'cosine'),
    (12, 3e-4, 3, 4, 3e-5, 'sqrt'),
    (5, 1.0, 5, 0, 0.0, 'linear'),
    (5, 2.0, 0, 5, 0.5, 'linear'),
    (1, 1e-3, 0, 1, 1e-5, 'cosine'),
    (1, 1e-3, 0, 0, 1e-5, 'sqrt'),
    (1000, 6e-4, 50, 200, 6e-5, 'cosine'),
    (300, 1e-2, 10, 290, 0.0, 'sqrt'),
]

EXPLANATION = r"""
## Why a long plateau works
At a high, constant learning rate the loss stalls on a noisy plateau: the iterate bounces around a valley floor it can't settle into. Decaying the rate averages out that noise, and the loss drops quickly. Most of the "decay benefit" comes in the last 10-20% of training, which is why WSD's short cooldown matches a full cosine schedule. And because the stable phase doesn't depend on the final step count, one run can be cooled down from several checkpoints to get a whole data-scaling curve for the price of one run plus a few short decays.

## Decay shapes
- **linear**: simple, a strong default.
- **cosine**: gentle at both ends.
- **sqrt** ($1-\sqrt{s}$): drops fastest at the start of the cooldown; it often edges out linear in practice.

Tests 3 and 9 use `sqrt`; test 2 uses `cosine` with a non-zero floor.

## Complexity
$O(T)$ time and memory, fully vectorised.

## Common bugs
- Off-by-one at the phase boundaries: warmup that starts at exactly 0 (the first step does nothing) or a decay whose last step is above $\eta_\text{min}$.
- Computing decay progress from the warmup end instead of from $T - K$.
- Dividing by $W$ or $K$ when they are 0 (tests 2, 4 and 7).

## Interview follow-ups
- You branched three cooldowns from one stable run, at 25%, 50% and 100% of the tokens. How do you turn their final losses into a data-scaling law, and what bias does the shared trajectory introduce?
- Why might a WSD run look worse than a cosine run for most of training, then end up better?
"""
