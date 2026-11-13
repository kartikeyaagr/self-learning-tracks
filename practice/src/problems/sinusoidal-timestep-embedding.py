import numpy as np

META = dict(title='Sinusoidal timestep embedding for a denoiser', tags=['Conditioning', 'Diffusion Models'], diff='Easy',
            runtime='numpy', lessons=['aieng:04.01'], related=['sinusoidal-positional-encoding', 'time-conditioned-residual-block'])

STATEMENT = r"""
A diffusion denoiser must know **how noisy** its input is. The step $t$ (an integer, or a continuous noise level) is turned into a vector with sinusoids at geometrically spaced frequencies, then fed through a small MLP.

With $h = \lfloor \texttt{dim}/2 \rfloor$ frequencies

$$\omega_i = \exp\!\left(-\ln(P)\,\frac{i}{h}\right), \quad i = 0, \dots, h-1$$

each timestep $t$ maps to $\big[\cos(t\,\omega_0), \dots, \cos(t\,\omega_{h-1}),\ \sin(t\,\omega_0), \dots, \sin(t\,\omega_{h-1})\big]$.

Write `timestep_embedding(t, dim, max_period=10000.0)`.

## Inputs
- `t`: `(B,)` array of timesteps, integer **or float** (fractional steps are allowed).
- `dim`: embedding size, `dim >= 2`. `max_period` is $P$.

## Output
`(B, dim)` float array laid out as **all cosines, then all sines** (two contiguous halves, not interleaved). If `dim` is odd, append one column of zeros at the end.
"""

STARTER = '''import numpy as np

def timestep_embedding(t, dim, max_period=10000.0):
    """t: (B,) timesteps. Return (B, dim): [cos half | sin half], zero-padded if dim is odd."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def timestep_embedding(t, dim, max_period=10000.0):
    t = np.asarray(t, dtype=float)
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half) / half)    # omega_0 = 1 down to ~1/P
    args = t[:, None] * freqs[None, :]                              # (B, half)
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros((len(t), 1))], axis=1)
    return emb
'''

HINTS = [
    'Build the `half` frequencies once with `np.exp(-np.log(max_period) * np.arange(half) / half)`. The first one is exactly 1.',
    'An outer product `t[:, None] * freqs[None, :]` gives every (timestep, frequency) angle in one `(B, half)` array.',
    'Concatenate `cos` then `sin` along axis 1, and for odd `dim` add a zero column with `np.zeros((B, 1))`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([0, 1]), 4),
    (np.array([2.0]), 2, 1.0),
    (np.array([0, 10, 500, 999]), 8),
    (np.array([3, 7]), 5),
    (np.array([0.25, 0.5]), 6, 100.0),
    (_r.uniform(0, 1000, size=5), 16),
]

EXPLANATION = r"""
## Why it works
Low frequencies vary slowly and separate distant timesteps; high frequencies separate neighbours. Together they give every $t$ a distinct, smooth code with bounded entries, which a network handles far better than the raw integer $t$ (whose scale runs to 1000). The embedding of $t + \delta$ is a rotation of the embedding of $t$ within each (cos, sin) pair, so relative offsets are easy to read off linearly.

## Differences from the Transformer positional encoding
The values are the same family, but the layout is two halves rather than interleaved pairs, the frequency exponent is $i/h$ rather than $2i/d$, and inputs may be fractional (continuous-time models embed $\log\sigma$ or $t \in [0, 1]$, often scaled by 1000 first). Weights trained with one layout are useless with the other, so match the convention exactly.

## Complexity
$O(B \cdot \texttt{dim})$.

## Common bugs
- Integer division in `np.arange(half) / half` (not a problem in NumPy, but it is in some frameworks with integer tensors).
- Interleaving `sin`/`cos` when the checkpoint expects halves, or putting `sin` first.
- Shape `(dim,)` instead of `(B, dim)` when `B == 1`.

## Interview follow-ups
- In a U-Net, where does this embedding go after the MLP? (It is projected and added to every residual block's features.)
- Why do flow-matching models with $t \in [0, 1]$ usually multiply $t$ by 1000 before embedding it?
"""
