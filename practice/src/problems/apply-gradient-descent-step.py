import numpy as np

META = dict(title='Fit a line with hand-derived gradient steps', tags=['Optimization', 'Regression'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.04'],
            related=['vanilla-gradient-descent', 'measure-scalar-expression-partials', 'apply-one-full-sgd-training-step'])

STATEMENT = r"""
Once you know the gradient, learning is just: step every parameter a little *against* it. Do that for a one-feature linear model $\hat y = w x + b$ with the mean squared error

$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - y_i)^2$$

Write `fit_line(xs, ys, w, b, lr, steps)` that runs `steps` iterations of

$$w \leftarrow w - \eta\, \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta\, \frac{\partial L}{\partial b}$$

with both partials evaluated at the **same** current $(w, b)$ (a simultaneous update), and returns

```
{"w": final w, "b": final b, "losses": [L before step 1, L after step 1, ..., L after the last step]}
```

`losses` has `steps + 1` entries; with `steps == 0` it's just the starting loss.

## Inputs
- `xs`, `ys`: lists of $N \ge 1$ floats. `w`, `b`, `lr` ($\eta > 0$): floats. `steps`: int $\ge 0$.
- Derive the partials yourself. Plain Python, no libraries.
"""

STARTER = '''def fit_line(xs, ys, w, b, lr, steps):
    """Gradient descent on the MSE of y ~ w*x + b. Return {"w", "b", "losses"}."""
    raise NotImplementedError
'''

SOLUTION = '''def fit_line(xs, ys, w, b, lr, steps):
    n = len(xs)

    def loss_and_grads(w, b):
        loss = dw = db = 0.0
        for x, y in zip(xs, ys):
            r = w * x + b - y                 # residual
            loss += r * r
            dw += 2 * r * x                   # d(r^2)/dw = 2r * x
            db += 2 * r                       # d(r^2)/db = 2r
        return loss / n, dw / n, db / n

    loss, dw, db = loss_and_grads(w, b)
    losses = [loss]
    for _ in range(steps):
        w, b = w - lr * dw, b - lr * db       # simultaneous: both use the old (w, b)
        loss, dw, db = loss_and_grads(w, b)
        losses.append(loss)
    return {"w": w, "b": b, "losses": losses}
'''

HINTS = [
    'Write one helper that, for a given $(w, b)$, loops over the data once and returns the loss and both partials.',
    'With residual $r_i = w x_i + b - y_i$: $\\partial L / \\partial w = \\frac{2}{N}\\sum r_i x_i$ and $\\partial L / \\partial b = \\frac{2}{N}\\sum r_i$.',
    'Update with a tuple assignment, `w, b = w - lr * dw, b - lr * db`, so the new $w$ is not used to compute the step for $b$. Record the loss once before the loop and once after every step.',
]

_r = np.random.default_rng(42)
_x = [float(round(v, 2)) for v in _r.uniform(-2, 2, size=12)]
_y = [float(round(1.7 * x - 0.4 + n, 3)) for x, n in zip(_x, _r.normal(0, 0.1, size=12))]
TESTS = [
    ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 0.0, 0.0, 0.1, 3),
    ([0.0, 1.0], [1.0, 3.0], 0.0, 0.0, 0.25, 2),
    ([1.0, 2.0], [3.0, 5.0], 2.0, 1.0, 0.1, 4),
    ([1.0, -1.0, 2.0], [0.5, 0.0, 1.0], 0.3, -0.2, 0.05, 0),
    ([5.0], [1.0], 1.0, 0.0, 0.01, 5),
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0, 0.0, 0.3, 6),
    (_x, _y, 0.0, 0.0, 0.1, 25),
]

EXPLANATION = r"""
## Why the update decreases the loss
For a small step, $L(\theta - \eta g) \approx L(\theta) - \eta \lVert g \rVert^2$, so moving against the gradient lowers the loss whenever $\eta$ is small enough. The gradient points uphill. Its negative is the steepest way down.

## When it goes wrong
"Small enough" is relative to the curvature. For this loss the curvature along $w$ is about $2\,\overline{x^2}$; once $\eta$ exceeds $2 / \lambda_{\max}$ of the Hessian, each step overshoots further than the last. Test 6 uses a learning rate that's too large for its data: watch `losses` grow instead of shrink. Test 3 starts at the exact solution, so the gradient is zero and nothing moves.

## Complexity
$O(N)$ per step for the loss and both partials, $O(N \cdot \text{steps})$ overall, $O(\text{steps})$ memory for the history.

## Common bugs
- Updating `w` first and then computing the `b` gradient from the new `w` (a sequential, not simultaneous, update).
- Forgetting the factor 2 or the $1/N$. The direction is still right, so it "works", but the effective learning rate is wrong and the test values differ.
- Adding the gradient instead of subtracting it: gradient *ascent*.

## Interview follow-ups
- What is the largest stable learning rate for this problem, in terms of the data? (It depends on the largest eigenvalue of $\frac{2}{N}\begin{bmatrix}\sum x^2 & \sum x \\ \sum x & N\end{bmatrix}$.)
- How does centering the inputs ($\bar x = 0$) change the conditioning and the speed of convergence?
"""
