import numpy as np

META = dict(title='Plan by random shooting', tags=['Reinforcement Learning', 'Optimization'], diff='Easy', runtime='numpy',
            related=['rollout-error-compounding'])

STATEMENT = r"""
Random shooting samples many action sequences, imagines each one with the model, and keeps the best.

The model is linear, $s_{t+1} = A s_t + B a_t$, and the reward of a step is $-(\lVert s_{t+1}\rVert^2 + \lambda \lVert a_t\rVert^2)$: drive the state to zero without large actions.

Write `shoot(A, B, s0, candidates, lam)` where `candidates` has shape `(N, H, du)`:

- Roll out every candidate from `s0` and sum its rewards over the $H$ steps (no discount).
- Return a dict with `returns` (list of $N$), `best` (index of the highest return; lowest index on ties) and `first_action` (the best candidate's first action, a list).
"""

STARTER = '''import numpy as np

def shoot(A, B, s0, candidates, lam):
    """Evaluate sampled action sequences under a linear model and pick the best."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    C = np.asarray(candidates, float)
    rets = [-float(np.sum(c ** 2)) for c in C]
    b = int(np.argmax(rets))
    return {"returns": rets, "best": b, "first_action": C[b, 0].tolist()}
'''

SOLUTION = '''import numpy as np

def shoot(A, B, s0, candidates, lam):
    A, B = np.asarray(A, float), np.asarray(B, float)
    C = np.asarray(candidates, float)
    rets = []
    for seq in C:
        s, R = np.asarray(s0, float), 0.0
        for a in seq:
            s = A @ s + B @ a
            R -= float(s @ s + lam * (a @ a))
        rets.append(R)
    b = int(np.argmax(rets))
    return {"returns": rets, "best": b, "first_action": C[b, 0].tolist()}
'''

HINTS = [
    'The reward uses the state after the action, s_{t+1}.',
    'Only the first action is executed; then the planner runs again from the new state.',
]

def _case(seed, N, H):
    g = np.random.default_rng(seed)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    return (A, B, [1.0, 0.0], g.normal(size=(N, H, 1)), 0.01)

TESTS = [
    ([[1.0]], [[1.0]], [1.0], [[[0.0]], [[-1.0]], [[-2.0]]], 0.0),
    ([[1.0]], [[1.0]], [1.0], [[[-1.0]], [[-0.5]]], 1.0),
    _case(1, 8, 5),
    _case(2, 64, 10),
    _case(3, 16, 3),
]

EXPLANATION = r"""
## The simplest planner
Random shooting needs nothing but a model you can simulate: no gradients, no value function. It works surprisingly well on short horizons and small action spaces, and it parallelises perfectly on a GPU.

## Where it breaks
The chance that a random sequence is close to the optimum falls exponentially with the horizon and the action dimension. With 10 steps of a 6-dimensional action, almost all samples are poor. The cross-entropy method (the next step) fixes this by refining the sampling distribution.

## Interview follow-ups
- Why does a model-predictive controller execute only the first action of its plan, and what does that buy you when the model is wrong?
"""
