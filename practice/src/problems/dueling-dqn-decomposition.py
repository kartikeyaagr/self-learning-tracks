import numpy as np

META = dict(title='Dueling network aggregation', tags=['Reinforcement Learning', 'Neural Networks'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.03'],
            related=['double-dqn-target', 'dqn-loss-with-target-network'])

STATEMENT = r"""
A dueling Q network splits its head into a scalar **state value** $V(s)$ and per-action **advantages** $A(s, a)$, then combines them into Q values. Plain $V + A$ is not identifiable: adding a constant to $V$ and subtracting it from every $A$ gives the same Q. So one of two normalisations is used:

$$\text{mean:}\quad Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal A|}\sum_{a'} A(s,a')$$
$$\text{max:}\quad Q(s,a) = V(s) + A(s,a) - \max_{a'} A(s,a')$$

Write `dueling_q(value, advantages, aggregation="mean")` and return the Q values.

## Inputs
- `value`: float `(B,)`, one state value per row.
- `advantages`: float `(B, A)`.
- `aggregation`: `"mean"` or `"max"`.

## Output
Float array `(B, A)`.

With `"max"`, the greedy action's Q value equals $V(s)$ exactly; with `"mean"`, the row average of Q equals $V(s)$.
"""

STARTER = '''import numpy as np

def dueling_q(value, advantages, aggregation="mean"):
    """value: (B,), advantages: (B, A). Return Q (B, A)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dueling_q(value, advantages, aggregation="mean"):
    adv = np.asarray(advantages, dtype=float)
    if aggregation == "mean":
        base = adv.mean(axis=1, keepdims=True)
    elif aggregation == "max":
        base = adv.max(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    return np.asarray(value, dtype=float)[:, None] + adv - base   # (B, 1) broadcasts across actions
'''

HINTS = [
    'Reduce the advantages per row (mean or max) with `keepdims=True` so the result stays `(B, 1)`.',
    'Turn `value` from `(B,)` into a column with `value[:, None]` before adding it to a `(B, A)` array.',
    'Q is `V[:, None] + A - reduce(A)`. Check: with "max" the largest entry of each Q row equals V.',
]

_r = np.random.default_rng(191)
VR, AR = _r.normal(size=5), _r.normal(size=(5, 4))
TESTS = [
    (np.array([1.0]), np.array([[1.0, 2.0, 3.0]])),
    (np.array([1.0]), np.array([[1.0, 2.0, 3.0]]), "max"),
    (np.array([0.0, 10.0]), np.array([[5.0, 5.0], [-1.0, 1.0]]), "mean"),
    (np.array([2.0]), np.array([[7.0]]), "max"),
    (np.array([-3.0, 0.5]), np.zeros((2, 3)), "mean"),
    (VR, AR, "mean"),
    (VR, AR, "max"),
]

EXPLANATION = r"""
## Why split V and A
In many states the action barely matters (no enemy on screen, nothing to dodge). A dueling head can learn $V(s)$ from every transition, whichever action was taken, while the advantage stream only has to learn the *differences* between actions. That shared value estimate generalises better and speeds learning, especially with many similar actions.

## Why normalise
$V$ and $A$ are only defined up to a constant, so without a constraint the network can drift both in opposite directions without changing Q. Fixing $\max_a A = 0$ makes $V$ the value of the greedy action, which matches the definition $V^*(s) = \max_a Q^*(s,a)$. Fixing $\text{mean}_a A = 0$ is what the paper recommends in practice: the subtracted term moves smoothly (no jump when the argmax changes), so optimisation is more stable, at the cost of $V$ being offset by a constant.

## Edge cases
With a single action (test 4), both normalisations give $Q = V$. Equal advantages (tests 3 and 5) give $Q = V$ in every action.

## Complexity
$O(BA)$.

## Common bugs
- Missing `keepdims` (or `[:, None]`), so `(B,)` broadcasts against the action axis of `(B, A)` instead of the batch axis, or fails when B differs from A.
- Subtracting the mean over the whole batch instead of per row.

## Interview follow-ups
- How would you backpropagate through the mean aggregation? What gradient does $V$ receive?
- Why doesn't the dueling architecture change the DQN loss at all?
"""
