import numpy as np

META = dict(title='One round of the cross-entropy method', tags=['Reinforcement Learning', 'Optimization'], diff='Medium', runtime='numpy',
            related=['random-shooting-planner'])

STATEMENT = r"""
The cross-entropy method (CEM) keeps a Gaussian over action sequences and moves it towards the best samples.

Same model and reward as in random shooting. Write `cem_round(A, B, s0, mean, std, noise, lam, n_elite, alpha)`:

1. Candidates: `mean + std * noise`, with `mean` and `std` of shape `(H, du)` and `noise` of shape `(N, H, du)`.
2. Score each candidate by its total reward.
3. Elites: the `n_elite` highest scores (ties broken by lower index).
4. New mean $= \alpha\,\text{elite mean} + (1-\alpha)\,\text{mean}$; new std $= \alpha\,\text{elite std} + (1-\alpha)\,\text{std}$, using the population std over elites, per entry.

Return a dict with `mean`, `std` (nested lists), `best_return` (highest score this round) and `elite_idx` (the elite indices, best first).
"""

STARTER = '''import numpy as np

def cem_round(A, B, s0, mean, std, noise, lam, n_elite, alpha):
    """One CEM iteration over action sequences under a linear model."""
    C = np.asarray(mean) + np.asarray(std) * np.asarray(noise)
    return {"mean": C.mean(0).tolist(), "std": C.std(0).tolist(), "best_return": 0.0, "elite_idx": []}
'''

SOLUTION = '''import numpy as np

def _ret(A, B, s0, seq, lam):
    s, R = np.asarray(s0, float), 0.0
    for a in seq:
        s = A @ s + B @ a
        R -= float(s @ s + lam * (a @ a))
    return R

def cem_round(A, B, s0, mean, std, noise, lam, n_elite, alpha):
    A, B = np.asarray(A, float), np.asarray(B, float)
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    C = mean + std * np.asarray(noise, float)
    R = np.array([_ret(A, B, s0, c, lam) for c in C])
    order = np.lexsort((np.arange(len(R)), -R))[:n_elite]
    E = C[order]
    new_mean = alpha * E.mean(0) + (1 - alpha) * mean
    new_std = alpha * E.std(0) + (1 - alpha) * std
    return {"mean": new_mean.tolist(), "std": new_std.tolist(), "best_return": float(R.max()), "elite_idx": order.tolist()}
'''

HINTS = [
    'np.lexsort sorts by its last key first; pass (index, -score) to get best-first with index tie-breaks.',
    'Smoothing with alpha < 1 stops the distribution from collapsing onto a few lucky samples in one round.',
    'If std shrinks to zero, CEM stops exploring; real implementations put a floor under it.',
]

def _case(seed, N, H, ne, alpha):
    g = np.random.default_rng(seed)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    return (A, B, [1.0, 0.0], np.zeros((H, 1)), np.ones((H, 1)), g.normal(size=(N, H, 1)), 0.01, ne, alpha)

TESTS = [
    ([[1.0]], [[1.0]], [1.0], [[0.0]], [[1.0]], [[[0.0]], [[-1.0]], [[1.0]], [[-0.5]]], 0.0, 2, 1.0),
    ([[1.0]], [[1.0]], [1.0], [[0.0]], [[1.0]], [[[0.0]], [[-1.0]], [[1.0]], [[-0.5]]], 0.0, 2, 0.5),
    _case(1, 32, 5, 4, 1.0),
    _case(2, 64, 10, 8, 0.9),
    _case(3, 16, 3, 1, 1.0),
]

EXPLANATION = r"""
## Why CEM beats random shooting
Each round concentrates the sampling distribution on the region that scored well, so later rounds spend their samples near the optimum instead of everywhere. A few rounds of a few hundred samples is enough for many continuous-control tasks, which is why model-based agents such as PETS and the planning in TD-MPC use CEM or a close variant (MPPI).

## The knobs
- **Elite fraction:** too small and the distribution follows noise; too large and it barely moves.
- **Smoothing (alpha):** mixes in the previous distribution to keep it from collapsing in one round.
- **A single elite** (the last test) makes the new std zero, so the next round would not explore at all.

## Interview follow-ups
- How would you warm-start CEM at the next time step from the plan found at this one?
"""
