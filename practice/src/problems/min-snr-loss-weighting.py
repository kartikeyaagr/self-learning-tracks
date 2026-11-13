import numpy as np

META = dict(title='Min-SNR loss weighting', tags=['Diffusion Models', 'Loss Functions', 'Training'], diff='Medium',
            runtime='numpy', lessons=[], related=['noise-prediction-loss', 'prediction-parameterization'])

STATEMENT = r"""
Diffusion training mixes many noise levels in one batch, and they fight: low-noise steps have easy, high-magnitude gradients that can dominate. **Min-SNR-$\gamma$** caps each step's influence using its signal-to-noise ratio

$$\mathrm{SNR}_t = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

The weight is $\min(\mathrm{SNR}_t, \gamma)$ **expressed in $x_0$-space**. Since an $\varepsilon$-error equals an $x_0$-error times $\mathrm{SNR}_t$ (and a $v$-error times $\mathrm{SNR}_t/(\mathrm{SNR}_t + 1)$), the weight on the loss you actually compute depends on the prediction target:

| target | weight $w_t$ |
| `"eps"` | $\min(\mathrm{SNR}_t, \gamma)\,/\,\mathrm{SNR}_t$ |
| `"x0"` | $\min(\mathrm{SNR}_t, \gamma)$ |
| `"v"` | $\min(\mathrm{SNR}_t, \gamma)\,/\,(\mathrm{SNR}_t + 1)$ |

Write `min_snr_loss(pred, target, t, alpha_bars, gamma=5.0, kind="eps")` and return `(weights, loss)`.

## Inputs
- `pred`, `target`: `(B, D)`; `t`: `(B,)` integer steps; `alpha_bars`: `(T,)` with every value strictly inside $(0, 1)$.
- `gamma`: positive float; `kind`: one of `"eps"`, `"x0"`, `"v"`.

## Output
- `weights`: `(B,)` float array $w_{t_b}$.
- `loss`: Python `float`, $\frac1B\sum_b w_{t_b}\cdot\frac1D\lVert \text{pred}_b - \text{target}_b\rVert^2$ (per-example mean squared error, weighted, then averaged over the batch).
"""

STARTER = '''import numpy as np

def min_snr_loss(pred, target, t, alpha_bars, gamma=5.0, kind="eps"):
    """Return (weights (B,), weighted mean loss as float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def min_snr_loss(pred, target, t, alpha_bars, gamma=5.0, kind="eps"):
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    ab = np.asarray(alpha_bars, dtype=float)[t]
    snr = ab / (1.0 - ab)
    capped = np.minimum(snr, gamma)
    w = {'eps': capped / snr, 'x0': capped, 'v': capped / (snr + 1.0)}[kind]
    per_example = ((pred - target) ** 2).mean(axis=1)
    return w, float(np.mean(w * per_example))
'''

HINTS = [
    'Gather `alpha_bars[t]`, turn it into SNR, and cap it: `np.minimum(snr, gamma)`.',
    'For `"eps"` the weight is `1` whenever SNR is below $\\gamma$ (noisy steps) and shrinks as $\\gamma/\\mathrm{SNR}$ for clean steps.',
    'Take the mean squared error **per example** with `.mean(axis=1)`, multiply by the weights, then average over the batch.',
]

_r = np.random.default_rng(37)
_ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
_t = np.array([0, 10, 100, 300, 600, 999])
_p, _y = _r.normal(size=(6, 4)), _r.normal(size=(6, 4))
TESTS = [
    (np.array([[1.0], [1.0]]), np.array([[0.0], [0.0]]), np.array([0, 1]), np.array([0.9, 0.5]), 5.0, 'eps'),
    (np.array([[1.0], [1.0]]), np.array([[0.0], [0.0]]), np.array([0, 1]), np.array([0.9, 0.5]), 5.0, 'x0'),
    (_p, _y, _t, _ab, 5.0, 'eps'),
    (_p, _y, _t, _ab, 5.0, 'x0'),
    (_p, _y, _t, _ab, 5.0, 'v'),
    (_p, _y, _t, _ab, 1.0, 'v'),
    (_p[:3], _y[:3], np.array([0, 0, 0]), _ab, 1e6, 'eps'),
]

EXPLANATION = r"""
## Why it works
Think of every step's loss in $x_0$-space. The $\varepsilon$-MSE is $\mathrm{SNR}_t$ times the $x_0$-MSE, so plain $\varepsilon$-training already weights clean steps by their (huge) SNR. Capping that factor at $\gamma$ stops nearly-clean steps from dominating, and dividing by $\mathrm{SNR}_t$ converts the capped $x_0$-weight back into a weight on the $\varepsilon$ loss you actually compute. With $\gamma = 5$ this converges several times faster than uniform weighting in the original experiments.

Test 7 sets $\gamma$ huge: nothing is capped and $\varepsilon$ weights are exactly $1$, the plain DDPM loss.

## Complexity
$O(BD)$.

## Common bugs
- Applying $\min(\mathrm{SNR}, \gamma)$ directly to the $\varepsilon$ loss, which inverts the intent and up-weights clean steps.
- Using $\mathrm{SNR}$ for $v$ instead of $\mathrm{SNR} + 1$: the $v$-error equals the $x_0$-error times $(\mathrm{SNR} + 1)$.
- Averaging over all entries before weighting, so the weights mix across examples.

## Interview follow-ups
- Derive the factor relating the $v$ loss to the $x_0$ loss.
- With zero terminal SNR, $\mathrm{SNR} = 0$ at the last step. Which of the three formulas break, and how would you guard them?
"""
