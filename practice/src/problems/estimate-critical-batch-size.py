import numpy as np

META = dict(title='Critical batch size from the gradient noise scale', tags=['Scaling Laws', 'Training'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:12.01'],
            related=['gradient-accumulation', 'batch-vs-stochastic-vs-mini-batch-gd', 'fit-learning-rate-and-batch-size-scaling'])

STATEMENT = r"""
Doubling the batch size halves the number of optimizer steps only while the gradient estimate is dominated by noise. Past a **critical batch size** the extra examples are mostly wasted. A cheap estimate of that point is the *simple gradient noise scale*

$$B_\text{noise} = \frac{\operatorname{tr}(\Sigma)}{\lVert G \rVert^2}$$

where $G$ is the true (full-batch) gradient and $\Sigma$ the covariance of a single example's gradient. In the same model, training with batch size $B$ needs

$$\frac{S}{S_\text{min}} = 1 + \frac{B_\text{noise}}{B} \quad\text{times more steps}, \qquad \frac{E}{E_\text{min}} = 1 + \frac{B}{B_\text{noise}} \quad\text{times more examples}$$

than the two extremes (infinite batch for steps, batch size 1 for examples).

Write `noise_scale(grads, batch_sizes)`.

## Inputs
- `grads`: float64 array `(n, d)` of per-example gradients, $n \ge 2$. Estimate $G$ by the row mean $\bar g$ and $\operatorname{tr}(\Sigma)$ by the sum over the $d$ coordinates of the **unbiased** sample variance (`ddof=1`). The tests guarantee $\bar g \ne 0$.
- `batch_sizes`: float64 array `(k,)` of positive batch sizes to evaluate.

## Output
A dict:

| key | value |
|---|---|
| `"g_norm2"` | float, $\lVert \bar g \rVert^2$ |
| `"trace_cov"` | float, $\operatorname{tr}(\hat\Sigma)$ |
| `"b_noise"` | float, their ratio |
| `"step_factor"` | float64 `(k,)`, $1 + B_\text{noise}/B$ |
| `"example_factor"` | float64 `(k,)`, $1 + B/B_\text{noise}$ |
"""

STARTER = '''import numpy as np

def noise_scale(grads, batch_sizes):
    """Return a dict with g_norm2, trace_cov, b_noise, step_factor, example_factor."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def noise_scale(grads, batch_sizes):
    g = np.asarray(grads, dtype=float)
    bs = np.asarray(batch_sizes, dtype=float)
    mean = g.mean(axis=0)                            # estimate of the true gradient G
    g_norm2 = float(mean @ mean)
    trace_cov = float(g.var(axis=0, ddof=1).sum())   # tr(Sigma) = sum of per-coordinate variances
    b_noise = trace_cov / g_norm2
    return {
        "g_norm2": g_norm2,
        "trace_cov": trace_cov,
        "b_noise": b_noise,
        "step_factor": 1.0 + b_noise / bs,           # noise-dominated: more examples per step still helps
        "example_factor": 1.0 + bs / b_noise,        # curvature-dominated: extra examples are wasted
    }
'''

HINTS = [
    'You never need the $d \\times d$ covariance matrix: its trace is just the sum of the per-coordinate variances.',
    'Use `grads.var(axis=0, ddof=1).sum()` for the trace and `mean @ mean` for the squared norm of the mean gradient.',
    'Both factors are elementwise formulas in `batch_sizes`; at $B = B_\\text{noise}$ each equals 2.',
]

_r = np.random.default_rng(7)
_g = np.array([0.5, -1.0, 2.0]) + _r.normal(0.0, 3.0, size=(64, 3))
TESTS = [
    (np.array([[1.0, 0.0], [3.0, 0.0]]), np.array([1.0, 2.0, 4.0])),
    (np.array([[2.0, 1.0], [2.0, 1.5], [2.0, 0.5]]), np.array([8.0])),
    (np.array([[1.0, 2.0, -1.0], [0.0, 2.0, 1.0], [2.0, -1.0, 0.0], [1.0, 1.0, 0.0]]), np.array([1.0, 3.0, 10.0])),
    (np.array([[10.0], [-9.0]]), np.array([0.5, 1.0, 1000.0])),
    (_g, np.array([1.0, 16.0, 256.0, 4096.0])),
    (np.array([0.01, 0.02]) + _r.normal(0.0, 1.0, size=(200, 2)), np.array([32.0, 1024.0])),
    (np.tile(np.array([[1.0, -1.0, 0.5, 0.0]]), (5, 1)) + np.eye(5, 4) * 0.1, np.array([2.0])),
]

EXPLANATION = r"""
## Why the ratio predicts the critical batch
A step of SGD with batch $B$ sees the gradient $G$ plus noise of covariance $\Sigma / B$. Expanding the loss to second order around the current point and choosing the best learning rate gives an improvement per step proportional to $1 / (1 + \operatorname{tr}(H\Sigma) / (B\, G^\top H G))$. Taking the Hessian $H$ to be a multiple of the identity leaves $\operatorname{tr}(\Sigma)/\lVert G\rVert^2$: the batch at which the noise in a mean gradient is about as large as the signal. Below it, doubling $B$ nearly halves the steps; above it, the step count saturates and the example count grows linearly.

## Cost
One pass over the `(n, d)` matrix: $O(nd)$ time and $O(d)$ extra memory. In practice you never store per-example gradients. You compare gradient norms measured at two batch sizes $B_\text{small}$ and $B_\text{big}$ (for example per-device and after the all-reduce), using $\mathbb{E}\lVert g_B\rVert^2 = \lVert G\rVert^2 + \operatorname{tr}(\Sigma)/B$ to solve for both unknowns.

## Common bugs
- Using `ddof=0`: biases $\operatorname{tr}(\Sigma)$ low by a factor $(n-1)/n$, noticeable for small $n$ (test 1).
- Using $\lVert \bar g \rVert^2$ as $\lVert G \rVert^2$ is itself biased upward by $\operatorname{tr}(\Sigma)/n$. That's fine here because the formula is specified, but it explains why noisy estimates of $B_\text{noise}$ come out too small.
- Taking the full covariance matrix with `np.cov` and then its trace: correct, but $O(d^2)$ memory, which is impossible for real models.

## Interview follow-ups
- The noise scale grows as training progresses and the loss falls. What does that suggest about batch-size warmup schedules?
- How would you estimate $B_\text{noise}$ during data-parallel training without per-example gradients?
"""
