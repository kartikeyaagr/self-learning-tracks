import numpy as np

META = dict(title='GELU kernel: exact and tanh-approximate', tags=['GPU Programming', 'Activation Functions'],
            diff='Easy', runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.04', 'ieng:13.06'],
            related=['swish', 'tanh'])

STATEMENT = r"""
GELU, the activation in BERT and GPT-2's MLP blocks, weights each input by the probability that a standard normal falls below it:

$$\mathrm{GELU}(x) = x\,\Phi(x) = \tfrac12\, x\left(1 + \operatorname{erf}\!\left(\tfrac{x}{\sqrt 2}\right)\right)$$

GPT-2 was trained with a cheaper approximation, and checkpoints must be served with the same formula they were trained with:

$$\mathrm{GELU}_{\tanh}(x) = \tfrac12\, x\left(1 + \tanh\!\left(\sqrt{2/\pi}\,\bigl(x + 0.044715\,x^3\bigr)\right)\right)$$

Write `gelu(x, approximate)` for a 1-D `float32` array of length `n` (possibly 0):

- `approximate` is a Python `bool`. `False` → exact formula using `math.erf`; `True` → tanh formula using `math.tanh`.
- Return a `float32` array of shape `(n,)`.
- Pass `approximate` to the kernel as a scalar argument; every thread reads the same flag, so there is no divergence.
- 128 threads per block, $\lceil n/128 \rceil$ blocks; no launch for `n == 0`.
- Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda

TPB = 128

def gelu(x, approximate):
    """x: float32 (n,), approximate: bool. Returns GELU(x) as float32 (n,)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    gelu_kernel[(n + TPB - 1) // TPB, TPB](d_x, approximate, d_out)
    return d_out.copy_to_host()

@cuda.jit
def gelu_kernel(x, approximate, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        v = x[i]
        # TODO: exact (erf) or tanh-approximate GELU depending on the flag
        out[i] = v
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda

TPB = 128
INV_SQRT2 = 0.7071067811865476        # 1 / sqrt(2)
SQRT_2_OVER_PI = 0.7978845608028654   # sqrt(2 / pi)

def gelu(x, approximate):
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    gelu_kernel[(n + TPB - 1) // TPB, TPB](d_x, approximate, d_out)
    return d_out.copy_to_host()

@cuda.jit
def gelu_kernel(x, approximate, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        v = x[i]
        if approximate:               # uniform across the grid: no warp divergence
            inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)
            out[i] = 0.5 * v * (1.0 + math.tanh(inner))
        else:
            out[i] = 0.5 * v * (1.0 + math.erf(v * INV_SQRT2))
'''

HINTS = [
    'Define the constants $1/\\sqrt 2$ and $\\sqrt{2/\\pi}$ at module level; Numba bakes globals into the kernel as compile-time constants.',
    'Exact: `0.5 * v * (1 + math.erf(v / sqrt(2)))`. Approximate: `0.5 * v * (1 + math.tanh(c * (v + 0.044715 * v**3)))` with $c = \\sqrt{2/\\pi}$.',
    'Write `v * v * v` rather than `v ** 3`; it is cheap and exact for floats. Both `math.erf` and `math.tanh` saturate cleanly at $\\pm 1$, so large inputs need no special case.',
]

_r = np.random.default_rng(7)
_x = (_r.standard_normal(200) * 3).astype(np.float32)
TESTS = [
    (np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), False),
    (np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), True),
    (np.zeros(0, dtype=np.float32), True),
    (np.array([-30.0, -6.0, 6.0, 30.0], dtype=np.float32), False),
    (np.array([-30.0, -6.0, 6.0, 30.0], dtype=np.float32), True),
    (_x, False),
    (_x, True),
    (np.linspace(-4, 4, 129, dtype=np.float32), True),
]

EXPLANATION = r"""
## Why it works
$\Phi(x) = \tfrac12(1 + \operatorname{erf}(x/\sqrt2))$ is the standard normal CDF, so the exact branch is the definition. The tanh branch fits $\Phi$ with a scaled tanh of a cubic; the two agree to about $10^{-3}$ in absolute terms, which is visible in tests 1 and 2 (same input, different flag) and far larger than the test tolerance. That is why serving a model with the wrong variant is a real, if small, accuracy bug.

## Uniform branches are free
`approximate` has the same value in every thread, so each warp takes one path and nothing is masked. Contrast with the data-dependent branch in the sigmoid problem. A more common production choice is to compile two specialised kernels (in Triton or C++ templates, a constexpr parameter) so the branch disappears entirely.

## Performance
Still memory-bound, but now with a real transcendental per element. `tanh` and `erf` are a few dozen instructions each. GELU is almost always **fused** into the matmul epilogue of the first MLP projection, so it costs no extra trip through memory at all.

## Common bugs
- Using `math.sqrt(2/math.pi)` inside the kernel: correct, but recomputed per thread (the compiler usually folds it, but a module constant makes the intent explicit).
- Forgetting the leading $x$ (returning $\Phi(x)$).
- `0.044715 * v ** 2` instead of the cube.

## Interview follow-ups
- Derive the derivative of exact GELU: $\Phi(x) + x\,\phi(x)$. Which intermediate would you save for the backward pass?
- GELU vs SwiGLU in modern LLMs: what changed and why does SwiGLU need a third weight matrix?
"""
