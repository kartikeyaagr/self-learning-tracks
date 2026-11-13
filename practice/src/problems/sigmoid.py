import numpy as np

META = dict(title='Overflow-safe sigmoid kernel', tags=['GPU Programming', 'Activation Functions'], diff='Easy',
            runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.04'], related=['swish', 'tanh', 'sigmoid-stable'])

STATEMENT = r"""
The logistic sigmoid is

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

In `float32`, $e^{v}$ overflows once $v > 88.7$. The textbook formula evaluates $e^{-x}$, which overflows for every $x < -88.7$. On a GPU that intermediate quietly becomes `inf` (the final result happens to round to 0, but the rearranged form $e^{x}/(1+e^{x})$ would give `inf / inf = NaN`); here, on the simulator, it raises `OverflowError`. The fix is to branch on the sign so the exponent is always $\le 0$:

$$\sigma(x) = \begin{cases} \dfrac{1}{1 + e^{-x}} & x \ge 0 \\[2mm] \dfrac{e^{x}}{1 + e^{x}} & x < 0 \end{cases}$$

Write `sigmoid(x)` for a 1-D `float32` array of length `n` (possibly 0) and return a `float32` array of shape `(n,)`.

## Requirements
- One thread per element, 128 threads per block, $\lceil n/128 \rceil$ blocks; return an empty array for `n == 0` without launching.
- Use `math.exp` inside the kernel and never call it with an argument greater than 0.
- Inputs may be as large as $\pm 1000$. Results are compared with a relative tolerance of $10^{-5}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda

TPB = 128

def sigmoid(x):
    """x: float32 (n,). Returns 1 / (1 + exp(-x)) as float32 (n,)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    sigmoid_kernel[(n + TPB - 1) // TPB, TPB](d_x, d_out)
    return d_out.copy_to_host()

@cuda.jit
def sigmoid_kernel(x, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = 1.0 / (1.0 + math.exp(-x[i]))   # TODO: overflows for very negative x
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda

TPB = 128

def sigmoid(x):
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    sigmoid_kernel[(n + TPB - 1) // TPB, TPB](d_x, d_out)
    return d_out.copy_to_host()

@cuda.jit
def sigmoid_kernel(x, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        v = x[i]
        if v >= 0:
            out[i] = 1.0 / (1.0 + math.exp(-v))   # exponent <= 0: no overflow
        else:
            e = math.exp(v)                       # v < 0, so e is in (0, 1)
            out[i] = e / (1.0 + e)
'''

HINTS = [
    'Try the starter on `x = -1000`: $e^{1000}$ is far beyond the `float32` range.',
    'Both branches are the same function: multiply the top and bottom of $1/(1+e^{-x})$ by $e^{x}$.',
    'Branch on `v >= 0`. Each branch calls `math.exp` on a non-positive number, so the result is in $(0, 1]$.',
]

_r = np.random.default_rng(4)
TESTS = [
    (np.array([0.0, 1.0, -1.0, 4.0], dtype=np.float32),),
    (np.array([-1000.0, -90.0, 90.0, 1000.0], dtype=np.float32),),
    (np.zeros(0, dtype=np.float32),),
    (np.array([-88.0, -50.0, -20.0, 20.0, 50.0], dtype=np.float32),),
    ((_r.standard_normal(300) * 5).astype(np.float32),),
    ((_r.standard_normal(129) * 300).astype(np.float32),),
    (np.linspace(-10, 10, 64, dtype=np.float32),),
]

EXPLANATION = r"""
## Why it works
For $x < 0$, multiply numerator and denominator of $1/(1+e^{-x})$ by $e^{x}$ to get $e^{x}/(1+e^{x})$. Same value, but now the exponent is negative, so $e^{x} \in (0,1)$ and nothing overflows. For very negative $x$ the result underflows gracefully towards 0, which is the correct limit.

## Divergence
The `if` is data-dependent, so a warp whose 32 inputs have mixed signs runs **both** branches one after the other, with half the lanes masked each time. Here each branch is a handful of instructions and the kernel is memory-bound anyway, so the cost is invisible. In compute-heavy kernels, prefer branch-free forms when possible, e.g. $e = e^{-|x|}$ followed by a select:
`out = 1/(1+e) if x >= 0 else e/(1+e)`, which compiles to one `exp` plus a cheap select.

## Precision
`float32` has a 24-bit mantissa, so $\sigma(x)$ rounds to exactly 1.0 once $x \gtrsim 17$. That is fine for a forward activation. If you need $\log \sigma(x)$ (binary cross-entropy), compute it directly as $-\mathrm{softplus}(-x)$ instead of taking the log of a saturated sigmoid.

## Common bugs
- The naive formula: an `inf` intermediate on a GPU, `OverflowError` on the simulator.
- Using `abs(x)` in both branches but forgetting to flip the result for negative inputs ($\sigma(-x) = 1 - \sigma(x)$).
- Writing `1 / (1 + math.exp(-x))` with integer literals in an integer-typed kernel: always keep the arithmetic in floating point.

## Interview follow-ups
- Derive the sigmoid derivative and show it can be computed from the forward output alone: $\sigma' = \sigma(1-\sigma)$. Why does that matter for memory in training?
- How does the hardware evaluate `exp` quickly? (Special function units compute $2^x$; `__expf` / fast-math trades a few ulps of accuracy for speed.)
"""
