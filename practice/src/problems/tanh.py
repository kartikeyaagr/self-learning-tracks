import numpy as np

META = dict(title='Tanh from exponentials, without overflow', tags=['GPU Programming', 'Activation Functions'],
            diff='Easy', runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.04'],
            related=['sigmoid', 'gelu', 'differentiate-tanh-activation'])

STATEMENT = r"""
When you fuse an activation into a larger kernel you often write it from primitives rather than calling a library routine. Implement the hyperbolic tangent from `math.exp` only (do **not** call `math.tanh`):

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

Evaluated as written, both exponentials overflow for $|x| > 88$ and the result becomes `inf / inf = NaN`. Use the identity, valid for every real $x$,

$$\tanh(x) = \operatorname{sign}(x)\, \frac{1 - e^{-2|x|}}{1 + e^{-2|x|}}$$

which only ever exponentiates a non-positive number.

## Task
Write `tanh(x)` for a 1-D `float32` array of length `n` (possibly 0) and return a `float32` array of shape `(n,)`.

- Launch with a fixed grid of **2 blocks of 64 threads** and a grid-stride loop, so any `n` works (including 0, with no special case).
- Inputs range up to $\pm 10^4$. `tanh(0)` must be exactly `0`.
- Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda

BLOCKS, TPB = 2, 64

def tanh(x):
    """x: float32 (n,). Returns tanh(x) as float32 (n,), built from math.exp."""
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(x.shape[0], dtype=np.float32)
    tanh_kernel[BLOCKS, TPB](d_x, d_out)
    return d_out.copy_to_host()

@cuda.jit
def tanh_kernel(x, out):
    for i in range(cuda.grid(1), x.shape[0], cuda.gridsize(1)):
        v = x[i]
        # TODO: the textbook formula overflows for |v| > 88
        out[i] = (math.exp(v) - math.exp(-v)) / (math.exp(v) + math.exp(-v))
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda

BLOCKS, TPB = 2, 64

def tanh(x):
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(x.shape[0], dtype=np.float32)
    tanh_kernel[BLOCKS, TPB](d_x, d_out)       # fixed grid; the kernel loops over n
    return d_out.copy_to_host()

@cuda.jit
def tanh_kernel(x, out):
    for i in range(cuda.grid(1), x.shape[0], cuda.gridsize(1)):
        v = x[i]
        e = math.exp(-2.0 * abs(v))           # in (0, 1]: never overflows
        t = (1.0 - e) / (1.0 + e)             # tanh(|v|)
        out[i] = t if v >= 0 else -t          # tanh is odd
'''

HINTS = [
    'Divide the top and bottom of the textbook formula by $e^{|x|}$. What is left only involves $e^{-2|x|}$.',
    'Compute `t = tanh(|v|)` from `e = math.exp(-2.0 * abs(v))`, then restore the sign because tanh is odd.',
    'Check the extremes: at $v = 0$, $e = 1$ and $t = 0$ exactly; for huge $|v|$, $e$ underflows to 0 and $t = 1$.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32),),
    (np.array([-100.0, -1.0, 1.0, 100.0], dtype=np.float32),),
    (np.zeros(0, dtype=np.float32),),
    (np.array([10000.0, -10000.0, 89.0, -89.0, 0.0], dtype=np.float32),),
    (np.array([1e-4, -1e-4, 1e-2, -1e-2], dtype=np.float32),),
    ((_r.standard_normal(300) * 3).astype(np.float32),),
    ((_r.standard_normal(200) * 200).astype(np.float32),),
]

EXPLANATION = r"""
## Why it works
Dividing numerator and denominator by $e^{|x|}$ gives $\tanh(|x|) = (1 - e^{-2|x|}) / (1 + e^{-2|x|})$, and $\tanh(-x) = -\tanh(x)$ restores the sign. The single exponential is always in $(0, 1]$, so there is no overflow, and one `exp` per element instead of four is cheaper too.

## Accuracy near zero
For tiny $x$, $1 - e^{-2|x|}$ subtracts two numbers close to 1 and loses relative precision (about $10^{-7}/|x|$ in `float32`). Test 5 sits at $|x| = 10^{-4}$, where the error is still within tolerance. A production implementation switches to a polynomial ($x - x^3/3 + \dots$) or uses `expm1` for $|x| < 0.5$-ish: $\tanh(x) = -\mathrm{expm1}(-2|x|)/(2 + \mathrm{expm1}(-2|x|))$, with the sign restored.

## Performance
Memory-bound like every element-wise op, but transcendental functions are not free: `exp` runs on the special function units, which have a fraction of the throughput of regular FP32 ALUs. For a single activation this is hidden behind memory traffic; inside a fused kernel it can start to matter.

## Common bugs
- The textbook formula: `NaN` for $|x| > 88$ (`inf/inf`), `OverflowError` on the simulator.
- Forgetting the sign: every output is non-negative.
- Using `math.exp(-2 * v)` without the absolute value: overflows for very negative inputs.

## Interview follow-ups
- Express tanh through the sigmoid ($\tanh x = 2\sigma(2x) - 1$). Is that formula accurate near zero? (No: it also subtracts nearly equal numbers.)
- Why did LSTMs use tanh for the cell candidate but sigmoid for the gates?
"""
