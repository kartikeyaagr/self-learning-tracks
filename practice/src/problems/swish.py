import numpy as np

META = dict(title='Swish with a reusable device function', tags=['GPU Programming', 'Activation Functions'],
            diff='Easy', runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.04', 'ieng:13.06'],
            related=['sigmoid', 'gelu'])

STATEMENT = r"""
Swish (SiLU when $\beta = 1$) is the activation inside the SwiGLU blocks of Llama-style models:

$$\mathrm{swish}_\beta(x) = x\,\sigma(\beta x), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Kernels are easier to read and test when shared maths lives in **device functions**: functions decorated with `@cuda.jit(device=True)` that kernels call like normal Python functions. They are inlined at compile time, so they cost nothing.

## Task
Write `swish(x, beta)`:
- `x`: 2-D `float32` array of shape `(rows, cols)`, either dimension possibly 0. `beta`: Python float (any sign).
- Return a `float32` array of the same shape.
- Put an overflow-safe sigmoid (never call `math.exp` on a positive argument) in a device function `stable_sigmoid(z)` and call it from the kernel.
- Launch a 1-D kernel over all `rows * cols` elements of the flattened array, 128 threads per block, no launch when the array is empty.

Inputs may be as large as $\pm 500$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda

TPB = 128

def swish(x, beta):
    """x: float32 (rows, cols), beta: float. Returns x * sigmoid(beta * x), same shape, float32."""
    flat = np.ascontiguousarray(x).ravel()
    if flat.size == 0:
        return flat.reshape(x.shape)
    d_x = cuda.to_device(flat)
    d_out = cuda.device_array(flat.size, dtype=np.float32)
    swish_kernel[(flat.size + TPB - 1) // TPB, TPB](d_x, beta, d_out)
    return d_out.copy_to_host().reshape(x.shape)

@cuda.jit(device=True)
def stable_sigmoid(z):
    # TODO: overflow-safe sigmoid
    return 0.5

@cuda.jit
def swish_kernel(x, beta, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = x[i] * stable_sigmoid(beta * x[i])
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda

TPB = 128

def swish(x, beta):
    flat = np.ascontiguousarray(x).ravel()
    if flat.size == 0:
        return flat.reshape(x.shape)
    d_x = cuda.to_device(flat)
    d_out = cuda.device_array(flat.size, dtype=np.float32)
    swish_kernel[(flat.size + TPB - 1) // TPB, TPB](d_x, beta, d_out)
    return d_out.copy_to_host().reshape(x.shape)

@cuda.jit(device=True)
def stable_sigmoid(z):
    """Inlined into every kernel that calls it. The exponent is always <= 0."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

@cuda.jit
def swish_kernel(x, beta, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        v = x[i]
        out[i] = v * stable_sigmoid(beta * v)
'''

HINTS = [
    'A device function is declared with `@cuda.jit(device=True)` and can `return` a value, unlike a kernel.',
    'Inside `stable_sigmoid`, branch on the sign of `z` so the exponent passed to `math.exp` is never positive.',
    'The kernel body is one line: `out[i] = v * stable_sigmoid(beta * v)`. The device function must be defined before the kernel is first **launched**, not before it is written.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32), 1.0),
    (np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32), 2.0),
    (np.array([[0.0]], dtype=np.float32), 1.0),
    (np.array([[-500.0, 500.0], [-100.0, 100.0]], dtype=np.float32), 1.0),
    (np.array([[3.0, -3.0]], dtype=np.float32), -1.0),         # negative beta
    (np.array([[3.0, -3.0, 0.5]], dtype=np.float32), 0.0),     # beta = 0 gives x / 2
    ((_r.standard_normal((12, 20)) * 4).astype(np.float32), 1.702),
]

EXPLANATION = r"""
## Why it works
Swish is an element-wise product of $x$ and a sigmoid, so it shares the overflow problem of the sigmoid: for $\beta x < -88$ the naive $e^{-\beta x}$ overflows `float32`. Moving the safe sigmoid into a device function fixes it once for every kernel that needs it. Note that with a negative `beta`, a **positive** `x` is the dangerous case; the sign branch on `z = beta * x` handles both.

## Device functions
`@cuda.jit(device=True)` compiles a function that can only be called from GPU code. Numba inlines it, so there is no call overhead, no stack frame and no extra memory traffic. They are the natural place for numerics you want to unit-test once (stable sigmoid, softplus, a bf16 rounding helper) and reuse across kernels.

## Special values of $\beta$
- $\beta = 1$: SiLU, used in Llama, Mistral and most modern LLMs.
- $\beta = 0$: $\sigma(0) = 1/2$, so swish is $x/2$.
- $\beta \to \infty$: $\sigma(\beta x)$ becomes a step function and swish tends to ReLU. $\beta \approx 1.702$ makes swish a close approximation to GELU (test 7).

## Common bugs
- Calling a kernel from a kernel (dynamic parallelism is not what you want here) instead of a device function.
- Using the naive sigmoid: `OverflowError` on test 4 in the simulator. On a GPU the overflow to `inf` happens to round to the right limit here, but the same habit in the rearranged form $e^{z}/(1+e^{z})$ gives `inf / inf = NaN`.
- Forgetting to reshape back to `(rows, cols)`.

## Interview follow-ups
- In a SwiGLU MLP, $\mathrm{out} = (\mathrm{silu}(xW_1) \odot xW_3) W_2$. Which of these ops would you fuse into one kernel, and what does it save?
- What is the derivative of swish, and can it be computed from $\sigma(\beta x)$ alone?
"""
