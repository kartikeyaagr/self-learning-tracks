import numpy as np

META = dict(title='Warmup plus cosine decay with LambdaLR', tags=['Training', 'Language Models'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:05.05', 'air-fanout-v1:10.04'],
            related=['linear-warmup', 'cosine-annealing-with-warm-restarts', 'optimizer-scheduler'])

STATEMENT = r"""
Transformer training almost always ramps the learning rate up linearly, then decays it along a cosine to a floor. Implement that schedule as a `torch.optim.lr_scheduler.LambdaLR` over an optimizer with several parameter groups.

## The multiplier
For update $s = 0, 1, 2, \ldots$ with warmup length $w \ge 0$, total length $S$ and floor ratio $\rho \in [0, 1]$:

$$\text{factor}(s) = \begin{cases} \dfrac{s+1}{w} & s < w \\[2mm] \rho + (1-\rho)\,\tfrac12\big(1 + \cos(\pi\, p)\big), \quad p = \min\!\Big(1,\ \dfrac{s - w}{\max(1,\ S - w)}\Big) & s \ge w \end{cases}$$

So the first update already uses $1/w$ of the peak (not 0), the peak is reached at update $w-1$ or $w$, and after update $S$ the multiplier stays at $\rho$. With $w = 0$ there is no warmup.

## What's graded
Write `warmup_cosine_lrs(base_lrs, warmup, total, min_ratio, steps)`:

- `base_lrs`: list of floats, one per parameter group (each group may hold a single dummy parameter).
- Build a `torch.optim.SGD` (or AdamW) with one group per base lr, wrap it in `LambdaLR` with your multiplier, and run `steps` updates: read every group's lr, then `optimizer.step()`, then `scheduler.step()`.
- Return a float64 tensor `(steps, G)`: row `s` holds the learning rate of each group at update `s`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import math
import torch

def warmup_cosine_lrs(base_lrs, warmup, total, min_ratio, steps):
    """Return (steps, G) float64: lr of each param group at each update."""
    params = [torch.zeros(1, requires_grad=True) for _ in base_lrs]
    opt = torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, base_lrs)])

    def factor(s):
        return 1.0          # TODO: warmup, then cosine down to min_ratio

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    rows = []
    for _ in range(steps):
        rows.append([g['lr'] for g in opt.param_groups])
        opt.step()
        sched.step()
    return torch.tensor(rows, dtype=torch.float64)
'''

SOLUTION = '''import math
import torch

def warmup_cosine_lrs(base_lrs, warmup, total, min_ratio, steps):
    params = [torch.zeros(1, requires_grad=True) for _ in base_lrs]
    opt = torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, base_lrs)])

    def factor(s):                                          # s = number of scheduler.step() calls so far
        if s < warmup:
            return (s + 1) / warmup                         # never 0: the first update still learns
        p = min(1.0, (s - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)  # lr = initial_lr * factor(s), per group
    rows = []
    for _ in range(steps):
        rows.append([g['lr'] for g in opt.param_groups])
        opt.step()                                          # no grads here, so this is a no-op update
        sched.step()
    return torch.tensor(rows, dtype=torch.float64)
'''

HINTS = [
    '`LambdaLR(opt, f)` sets each group\'s lr to `initial_lr * f(s)`, where `s` counts `scheduler.step()` calls (0 right after construction). One function serves every group.',
    'Warmup is `(s + 1) / warmup` for `s < warmup`. After that, measure progress from the end of warmup and clamp it at 1 so the lr stays at the floor.',
    'Guard the denominator with `max(1, total - warmup)` so `total == warmup` doesn\'t divide by zero. With `warmup = 0` the first branch never runs.',
]

TESTS = [
    ([1.0], 2, 6, 0.0, 8),
    ([0.1, 0.01], 0, 4, 0.1, 6),
    ([3e-4], 5, 20, 0.1, 25),
    ([1.0, 2.0, 0.5], 3, 3, 0.2, 6),
    ([1.0], 1, 2, 0.5, 4),
    ([6e-4, 6e-5], 10, 100, 0.0, 120),
    ([1.0], 4, 10, 1.0, 12),
]

EXPLANATION = r"""
## Why it works
`LambdaLR` stores each group's `initial_lr` when it's created and from then on sets `lr = initial_lr * f(s)`. Because the multiplier is relative, one function gives every group the same shape at its own scale: test 2 shows the 0.01 group tracking the 0.1 group at a tenth of the size, which is how per-group lrs (e.g. a smaller lr for embeddings or a pretrained backbone) survive the schedule.

## Why warm up
Early in training, Adam's second-moment estimates are based on a handful of gradients and the weights are far from any stable region. A large step then can blow up the loss or push attention logits into saturation. Ramping the lr over the first few hundred or thousand steps avoids that; cosine decay to a floor then gives a long, smooth anneal that tends to end lower than step decay.

## Edge cases
- $w = 0$: the cosine starts at the peak immediately.
- $S = w$: there is no decay phase: the multiplier is 1 at update $w$ and the floor from then on (test 4).
- $\rho = 1$: a constant lr after warmup (test 7).

## Complexity
$O(1)$ per update per group.

## Common bugs
- `s / warmup` instead of `(s + 1) / warmup`: the first update has lr 0 and is wasted.
- Measuring cosine progress from 0 instead of from the end of warmup, which starts the decay below the peak.
- Not clamping $p$: past $S$ the cosine comes back up.

## Interview follow-ups
- How do you resume this schedule from a checkpoint at step 37,000? (Save and load the scheduler's `state_dict`, or construct with `last_epoch`.)
- What is the "warmup-stable-decay" schedule and why is it convenient for scaling-law experiments?
"""
