import numpy as np

META = dict(title='Respace a diffusion schedule to K steps', tags=['Diffusion Models', 'Noise Schedules'], diff='Medium', runtime='numpy',
            related=['cosine-noise-schedule', 'ddpm-ancestral-step'])

STATEMENT = r"""
A model trained with $T$ noise levels can sample with far fewer steps by picking a subsequence of timesteps and recomputing the betas between them.

Write `respace(betas, K)` for a training schedule `betas` of length $T$:

1. $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$, for $t = 0, \dots, T-1$.
2. Timesteps: `np.round(np.linspace(0, T - 1, K)).astype(int)`, with duplicates removed (keep order).
3. New betas: $\beta'_i = 1 - \bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}$, with $\bar\alpha_{\tau_{-1}} = 1$.

Return a dict with `timesteps`, `alpha_bar` ($\bar\alpha$ at the chosen steps) and `betas` (the new betas), all lists.
"""

STARTER = '''import numpy as np

def respace(betas, K):
    """Subsample timesteps and recompute betas so the cumulative products match."""
    b = np.asarray(betas, float)
    t = np.round(np.linspace(0, len(b) - 1, K)).astype(int)
    return {"timesteps": t.tolist(), "alpha_bar": np.cumprod(1 - b)[t].tolist(), "betas": b[t].tolist()}
'''

SOLUTION = '''import numpy as np

def respace(betas, K):
    b = np.asarray(betas, float)
    ab = np.cumprod(1.0 - b)
    t = np.round(np.linspace(0, len(b) - 1, K)).astype(int)
    _, first = np.unique(t, return_index=True)
    t = t[np.sort(first)]
    a = ab[t]
    prev = np.r_[1.0, a[:-1]]
    return {"timesteps": t.tolist(), "alpha_bar": a.tolist(), "betas": (1.0 - a / prev).tolist()}
'''

HINTS = [
    'The new schedule must reproduce the same alpha_bar at the kept steps; that is what the trained model expects at those noise levels.',
    'Taking the original betas at the kept steps is the common mistake: it adds far too little noise per step.',
    'Asking for more steps than T makes linspace repeat integers; remove them.',
]

def _cos(T, s=0.008):
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    t = np.arange(T + 1)
    return np.clip(1 - f(t[1:]) / f(t[:-1]), 0, 0.999)

TESTS = [
    ([0.1, 0.1, 0.1, 0.1, 0.1], 5),
    ([0.1, 0.1, 0.1, 0.1, 0.1], 3),
    (np.linspace(1e-4, 0.02, 1000), 10),
    (_cos(100), 8),
    ([0.2, 0.3], 5),
]

EXPLANATION = r"""
## Why it works
A diffusion model is trained to denoise at each noise level $\bar\alpha_t$; it does not care which steps came before. If the shorter chain reaches exactly the same $\bar\alpha$ values at its chosen steps, the trained network can be used unchanged. The new betas are whatever makes the cumulative product match.

## The trade-off
Each step of a short chain has to remove much more noise, and the Gaussian assumption behind each DDPM step becomes less accurate. Quality falls as K shrinks; deterministic samplers such as DDIM and higher-order ODE solvers degrade more gracefully, which is why they are used for 10 to 50 step sampling.

## Interview follow-ups
- Is it better to space the K steps evenly in $t$, or evenly in $\log$ signal-to-noise ratio? Why?
"""
