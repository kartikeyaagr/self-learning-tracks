import numpy as np

META = dict(title="Squeezing the sigmoid's vanishing gradient", tags=['Activation Functions', 'Calculus'], diff='Medium',
            runtime='python', lessons=['air-fanout-v1:01.02'],
            related=['sigmoid-stable', 'derivatives-of-common-activation-functions'])

STATEMENT = r"""
The sigmoid's derivative $\sigma'(x) = \sigma(x)\,(1-\sigma(x)) = \dfrac{e^{-|x|}}{(1+e^{-|x|})^2}$ has a denominator between 1 and 4, which squeezes it between two simple curves:

$$\tfrac14\, e^{-|x|} \;\le\; \sigma'(x) \;\le\; e^{-|x|}$$

Both bounds go to 0 as $|x| \to \infty$, so by the squeeze theorem $\sigma'(x) \to 0$ too: the vanishing gradient.

Given a gradient threshold $\tau$, the bounds bracket the region where the gradient is still at least $\tau$. Write `grad_radii(tau)` that returns a tuple of three floats `(inner, exact, outer)`:

- `inner`: the radius from the **lower** bound. $\tfrac14 e^{-|x|} \ge \tau$ for $|x| \le$ `inner`, so $\sigma'(x) \ge \tau$ is guaranteed there.
- `exact`: the $x \ge 0$ where $\sigma'(x) = \tau$ exactly.
- `outer`: the radius from the **upper** bound. Beyond it, $e^{-|x|} < \tau$, so $\sigma'(x) < \tau$ is guaranteed.

The squeeze gives `inner <= exact <= outer`.

## Rules
- $0 < \tau \le 0.25$ (0.25 is the peak of $\sigma'$, at $x = 0$).
- `exact` must stay accurate for tiny thresholds: `tau = 1e-20` must give a finite answer correct to about 1e-6 relative. Solving the quadratic for $\sigma(x)$ the textbook way loses all precision there.
"""

STARTER = '''import math

def grad_radii(tau):
    """Return (inner, exact, outer) radii for the threshold tau, 0 < tau <= 0.25."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def grad_radii(tau):
    inner = math.log(1 / (4 * tau))        # (1/4) e^{-r} = tau
    outer = math.log(1 / tau)              #       e^{-r} = tau
    # s(1 - s) = tau with s = sigma(x) >= 1/2. The roots are (1 +- sqrt(1 - 4 tau)) / 2.
    root = math.sqrt(1 - 4 * tau)
    s = (1 + root) / 2
    one_minus_s = 2 * tau / (1 + root)     # = (1 - root) / 2, rewritten so nothing cancels
    exact = math.log(s) - math.log(one_minus_s)   # logit(s)
    return inner, exact, outer
'''

HINTS = [
    'The two radii come from solving $\\tfrac14 e^{-r} = \\tau$ and $e^{-r} = \\tau$ for $r$. The exact one comes from $s(1-s) = \\tau$ with $s = \\sigma(x)$, then $x = \\log\\frac{s}{1-s}$.',
    'The quadratic gives $s = \\frac{1 + \\sqrt{1 - 4\\tau}}{2}$ and $1 - s = \\frac{1 - \\sqrt{1-4\\tau}}{2}$. For tiny $\\tau$ the second is the difference of two numbers that are both almost 1.',
    'Multiply $1 - \\sqrt{1-4\\tau}$ by its conjugate: $1 - s = \\frac{2\\tau}{1 + \\sqrt{1 - 4\\tau}}$. Then `exact = log(s) - log(1 - s)`.',
]

TESTS = [
    (0.1,),
    (0.25,),
    (0.01,),
    (0.2499,),
    (1e-6,),
    (1e-12,),
    (1e-20,),
    (1e-300,),
]

EXPLANATION = r"""
## Why the bounds hold
With $u = e^{-|x|} \in (0, 1]$, $\sigma'(x) = u/(1+u)^2$ and $(1+u)^2 \in (1, 4]$. Dividing $u$ by something between 1 and 4 lands between $u/4$ and $u$. Each bound is invertible in closed form, so they bracket the true radius without solving anything hard. That's the squeeze theorem used quantitatively, not just for the limit.

## The cancellation trap
For $\tau = 10^{-12}$, $\sqrt{1 - 4\tau} \approx 1 - 2\times10^{-12}$, and $1 - \sqrt{\cdot}$ keeps only about 4 significant digits. For $\tau = 10^{-20}$, $1 - 4\tau$ rounds to exactly 1 and the naive $1 - s$ is 0, so `log` returns $-\infty$. The conjugate form $2\tau / (1 + \sqrt{1-4\tau})$ has no subtraction, so it's accurate everywhere. Note $\text{exact} \approx \log(1/\tau)$ for small $\tau$: the upper bound is tight in the tails.

## Complexity
$O(1)$.

## Common bugs
- Solving for the root $s < \tfrac12$, which gives the negative $x$ (the same radius, wrong sign).
- Computing $\sigma^{-1}$ as `-log(1/s - 1)`. `1/s - 1` cancels just like $1 - s$ does.

## Interview follow-ups
- Repeat the squeeze for $\tanh'(x) = 1 - \tanh^2 x$. (It's $4\sigma'(2x)$, so the bounds are $e^{-2|x|} \le \tanh'(x) \le 4 e^{-2|x|}$.)
- In a 10-layer sigmoid network, the backward signal is multiplied by at most 0.25 per layer. What's the best case factor, and how do residual connections change it?
"""
