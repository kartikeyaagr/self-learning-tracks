import numpy as np

META = dict(title='The DiLoCo outer step with Nesterov momentum', tags=['Distributed Computing', 'Optimization'], diff='Medium', runtime='numpy')

STATEMENT = r"""
DiLoCo trains with infrequent communication: each of $K$ workers takes $H$ local optimiser steps from the shared parameters $\theta$, then the workers synchronise once. Write `diloco_outer(theta, workers, v, lr, mu)`:

1. Outer gradient: $\Delta = \theta - \frac{1}{K}\sum_k \theta_k$ (the average distance the workers moved, as a gradient-like quantity).
2. Nesterov momentum on $\Delta$: $v \leftarrow \mu v + \Delta$, then $\theta \leftarrow \theta - lr\,(\Delta + \mu v)$.

`theta` and `v` are 1-D arrays, `workers` is a list of arrays. Return `(theta_new, v_new, delta_norm)` with `delta_norm` = $\|\Delta\|_2$.

With `lr = 1` and `mu = 0` this reduces to plain parameter averaging.
"""

STARTER = '''import numpy as np

def diloco_outer(theta, workers, v, lr, mu):
    """One DiLoCo outer step."""
    avg = np.mean(workers, axis=0)
    return avg, np.asarray(v), 0.0
'''

SOLUTION = '''import numpy as np

def diloco_outer(theta, workers, v, lr, mu):
    theta = np.asarray(theta, float)
    delta = theta - np.mean(np.asarray(workers, float), axis=0)
    v = mu * np.asarray(v, float) + delta
    return theta - lr * (delta + mu * v), v, float(np.linalg.norm(delta))
'''

HINTS = [
    'The outer gradient points from the average worker back to the starting point, so subtracting it moves toward the workers.',
    'Nesterov momentum applies the momentum twice: once into v, and once as a look-ahead in the update.',
    'Check the special case lr = 1, mu = 0: the result should be exactly the average of the workers.',
]

_r = np.random.default_rng(10)
TESTS = [
    (np.zeros(3), [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])], np.zeros(3), 1.0, 0.0),
    (np.zeros(3), [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])], np.zeros(3), 0.7, 0.9),
    (_r.normal(size=5), [_r.normal(size=5) for _ in range(4)], _r.normal(size=5) * 0.1, 0.7, 0.9),
    (np.ones(2), [np.ones(2)] * 3, np.array([0.5, -0.5]), 0.7, 0.9),
    (_r.normal(size=4), [_r.normal(size=4) for _ in range(8)], np.zeros(4), 0.5, 0.5),
]

EXPLANATION = r"""
## Why communicate less
Data parallelism all-reduces gradients every step, which needs fast interconnects. DiLoCo lets each worker (or each cluster) run hundreds of local AdamW steps and synchronise only the parameter change, cutting communication by the same factor. That makes training across weakly connected sites practical.

## Treating the change as a gradient
The averaged change $\Delta$ is fed to an outer optimiser. Plain averaging ($lr = 1$, $\mu = 0$) works but is slow; the paper found Nesterov momentum with an outer learning rate around 0.7 and momentum 0.9 works well, close to fully synchronous training with 500 times less communication.

## Interview follow-ups
- What happens if the workers' data distributions differ (non-IID shards)?
"""
