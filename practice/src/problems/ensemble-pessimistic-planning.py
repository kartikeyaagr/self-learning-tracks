import numpy as np

META = dict(title='Plan cautiously with a model ensemble', tags=['Reinforcement Learning', 'Optimization'], diff='Medium', runtime='numpy',
            related=['random-shooting-planner', 'rollout-error-compounding'])

STATEMENT = r"""
Train several dynamics models and plan against all of them. Where they disagree, the data was thin, so a careful planner should not trust any one of them.

Write `pessimistic_plan(models, s0, candidates, lam, beta)`:

- `models` is a list of `(A, B)` pairs for linear models; reward as in random shooting.
- For each candidate sequence, compute its return under every model.
- Score $= \text{mean} - \beta\,\text{std}$ over models (population std).
- Return a dict with `mean`, `std`, `score` (lists over candidates) and `best` (index of the highest score, lowest on ties).
"""

STARTER = '''import numpy as np

def pessimistic_plan(models, s0, candidates, lam, beta):
    """Pick the action sequence with the best mean-minus-disagreement return."""
    A, B = (np.asarray(m, float) for m in models[0])
    R = []
    for seq in np.asarray(candidates, float):
        s, r = np.asarray(s0, float), 0.0
        for a in seq:
            s = A @ s + B @ a
            r -= float(s @ s + lam * (a @ a))
        R.append(r)
    return {"mean": R, "std": [0.0] * len(R), "score": R, "best": int(np.argmax(R))}
'''

SOLUTION = '''import numpy as np

def _ret(A, B, s0, seq, lam):
    s, R = np.asarray(s0, float), 0.0
    for a in seq:
        s = A @ s + B @ a
        R -= float(s @ s + lam * (a @ a))
    return R

def pessimistic_plan(models, s0, candidates, lam, beta):
    ms = [(np.asarray(A, float), np.asarray(B, float)) for A, B in models]
    C = np.asarray(candidates, float)
    R = np.array([[_ret(A, B, s0, c, lam) for A, B in ms] for c in C])
    mu, sd = R.mean(1), R.std(1)
    score = mu - beta * sd
    return {"mean": mu.tolist(), "std": sd.tolist(), "score": score.tolist(), "best": int(np.argmax(score))}
'''

HINTS = [
    'Build a (candidates, models) matrix of returns; mean and std are then one call each along axis 1.',
    'beta = 0 is ordinary planning on the ensemble mean; larger beta prefers plans the models agree on.',
]

def _case(seed, M, N, H, spread, beta):
    g = np.random.default_rng(seed)
    A0 = np.array([[1.0, 0.1], [0.0, 1.0]]); B0 = np.array([[0.0], [0.1]])
    models = [(A0 + spread * g.normal(size=(2, 2)), B0 + spread * g.normal(size=(2, 1))) for _ in range(M)]
    return (models, [1.0, 0.0], g.normal(size=(N, H, 1)), 0.01, beta)

TESTS = [
    ([([[1.0]], [[1.0]]), ([[1.0]], [[2.0]])], [1.0], [[[-0.55]], [[-0.6667]], [[0.0]]], 0.0, 0.0),
    ([([[1.0]], [[1.0]]), ([[1.0]], [[2.0]])], [1.0], [[[-0.55]], [[-0.6667]], [[0.0]]], 0.0, 2.0),
    _case(1, 5, 16, 5, 0.05, 1.0),
    _case(6, 3, 32, 8, 0.1, 0.0),
    _case(6, 3, 32, 8, 0.1, 3.0),
]

EXPLANATION = r"""
## Disagreement as a warning
Models trained on different bootstraps or seeds agree where they saw lots of data and diverge where they did not. Their spread is a cheap estimate of epistemic uncertainty: what the agent does not know because it has not looked.

## What the first two tests show
The two models disagree about how strongly the action acts. With $\beta = 0$ the planner picks the action that is best on average. With $\beta = 2$ it picks $a = -0.6667$, which lands on almost the same state under both models: a slightly lower expected return in exchange for a plan both models agree on.

## Where this is used
Offline model-based RL (MOPO, MOReL) penalises disagreement so the policy stays near the data it was trained on. Online methods sometimes do the opposite and add a bonus for disagreement, to explore where the model is unsure.

## Interview follow-ups
- When would you want an optimistic planner instead of a pessimistic one?
"""
