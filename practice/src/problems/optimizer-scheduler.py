import numpy as np

META = dict(title='Step-decay learning rates with an optimizer and scheduler', tags=['Training', 'Optimization'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:05.05', 'air-fanout-v1:10.04'],
            related=['learning-rate-warmup-scheduler', 'linear-warmup', 'mini-training-loop'])

STATEMENT = r"""
Pair `torch.optim.SGD` with `torch.optim.lr_scheduler.MultiStepLR` and get the call order right, then report the learning rate each step actually used.

Write `multistep_sgd(w0, target, lr, milestones, gamma, steps)`:

- `w0`, `target`: float64 tensors `(D,)`. Minimise $f(w) = \tfrac12\lVert w - \text{target}\rVert^2$, starting from `w0`.
- `lr`: base learning rate; `gamma`: decay factor; `milestones`: a strictly increasing list of ints, each $\ge 1$; `steps`: number of updates.
- At update $s$ (counting from 0) the learning rate must be $\text{lr}\cdot\gamma^{k}$, where $k$ is the number of milestones $\le s$.

Use the real objects: one `SGD` optimizer (no momentum) and one `MultiStepLR` scheduler that you step **once per update**. Before each update, read the learning rate from the optimizer.

Return `(lrs, w)`: `lrs` a float64 tensor `(steps,)` of the learning rate used at each update, `w` the final weights `(D,)`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def multistep_sgd(w0, target, lr, milestones, gamma, steps):
    """Return (lrs (steps,), final w (D,))."""
    w = w0.clone().requires_grad_()
    opt = torch.optim.SGD([w], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    lrs = []
    for _ in range(steps):
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
        opt.zero_grad()
        (0.5 * ((w - target) ** 2).sum()).backward()
        opt.step()
    return torch.tensor(lrs, dtype=torch.float64), w.detach()
'''

SOLUTION = '''import torch

def multistep_sgd(w0, target, lr, milestones, gamma, steps):
    w = w0.detach().clone().requires_grad_()
    opt = torch.optim.SGD([w], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]['lr'])      # the lr this update will use
        opt.zero_grad()
        loss = 0.5 * ((w - target) ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()                               # after optimizer.step(): advance to the next update's lr
    return torch.tensor(lrs, dtype=torch.float64), w.detach()
'''

HINTS = [
    'The optimizer owns the current learning rate: `opt.param_groups[0]["lr"]`. The scheduler only rewrites that value when you call `sched.step()`.',
    'The order inside the loop is `zero_grad`, `backward`, `opt.step()`, then `sched.step()`. Stepping the scheduler first skips the base learning rate entirely (and PyTorch warns about it).',
    'Check yourself: with this objective one update maps $w - t$ to $(1 - \\text{lr}_s)(w - t)$.',
]

TESTS = [
    (np.array([1.0]), np.array([0.0]), 0.5, [2], 0.1, 4),
    (np.array([2.0, -2.0]), np.array([1.0, 1.0]), 0.1, [1, 3], 0.5, 5),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]), 0.3, [5, 10], 0.2, 12),
    (np.array([5.0]), np.array([5.0]), 1.0, [1], 0.5, 3),
    (np.array([1.0, 1.0]), np.array([-1.0, 3.0]), 0.2, [1, 2], 0.5, 4),
    (np.array([4.0, -1.0]), np.array([0.0, 0.0]), 0.25, [20], 0.1, 6),
    (np.array([3.0, 1.0, -2.0, 0.5]), np.array([0.0, 1.0, 2.0, 3.0]), 0.9, [2, 4, 6], 0.3, 8),
]

EXPLANATION = r"""
## Why it works
A scheduler is a small state machine attached to an optimizer. When it is created it records each group's `initial_lr` and sets step 0's learning rate; every `sched.step()` advances its counter and rewrites `param_groups[i]["lr"]`. `MultiStepLR` multiplies by $\gamma$ when the counter reaches a milestone, so update $s$ runs at $\text{lr}\cdot\gamma^{\#\{m \le s\}}$.

## Order of calls
Since PyTorch 1.1 the contract is `optimizer.step()` then `scheduler.step()`. Reversing them shifts the whole schedule one update early. For epoch-based schedules (`StepLR(step_size=30)` on ImageNet) the scheduler steps once per epoch, outside the batch loop; here we step per update.

## Complexity
The scheduler is $O(1)$ per step; the cost is all in the optimizer, $O(D)$ here.

## Common bugs
- Calling `sched.step()` before `opt.step()` (the starter): the base learning rate is never used.
- Reading `sched.get_last_lr()` after stepping and logging it as the lr just used.
- Creating a new scheduler after loading a checkpoint without restoring its `state_dict`: the decay restarts.

## Interview follow-ups
- Why did step decay give way to cosine schedules with warmup for training transformers?
- How do you save and resume a run so the scheduler continues exactly where it stopped?
"""
