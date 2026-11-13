import numpy as np

META = dict(title='Fused Leaky ReLU forward and gradient', tags=['GPU Programming', 'Activation Functions'],
            diff='Easy', runtime='cuda', lessons=['ieng:13.04', 'ieng:13.06'], related=['relu', 'gelu'])

STATEMENT = r"""
Leaky ReLU keeps a small slope $\alpha$ for negative inputs so their gradient never vanishes completely:

$$y = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases} \qquad \frac{dy}{dx} = \begin{cases} 1 & x > 0 \\ \alpha & x \le 0 \end{cases}$$

Training needs both, and reading `x` once to produce both is cheaper than two separate kernels. Write `leaky_relu(x, alpha)` that launches **one** kernel writing two outputs, and returns the tuple `(y, dydx)`.

## Inputs and outputs
- `x`: 1-D `float32` array of length `n` (possibly 0). `alpha`: Python float (may be 0 or negative; just apply the formula).
- `y`, `dydx`: `float32` arrays of shape `(n,)`.
- At exactly `x == 0` use the second branch: `y = 0`, `dydx = alpha`.

## Launch
128 threads per block, $\lceil n/128 \rceil$ blocks. For `n == 0` return two empty `float32` arrays without launching. `alpha` is passed to the kernel as an ordinary scalar argument.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

TPB = 128

def leaky_relu(x, alpha):
    """x: float32 (n,), alpha: float. Returns (y, dydx), both float32 (n,), from one kernel launch."""
    n = x.shape[0]
    y = np.zeros(n, dtype=np.float32)
    dydx = np.zeros(n, dtype=np.float32)
    # TODO: allocate both outputs on the device and launch leaky_kernel once
    return y, dydx

@cuda.jit
def leaky_kernel(x, alpha, y, dydx):
    # TODO
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda

TPB = 128

def leaky_relu(x, alpha):
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_y = cuda.device_array(n, dtype=np.float32)
    d_g = cuda.device_array(n, dtype=np.float32)
    leaky_kernel[(n + TPB - 1) // TPB, TPB](d_x, alpha, d_y, d_g)   # one pass over x, two outputs
    return d_y.copy_to_host(), d_g.copy_to_host()

@cuda.jit
def leaky_kernel(x, alpha, y, dydx):
    i = cuda.grid(1)
    if i < x.shape[0]:
        v = x[i]                       # read once, reuse for both outputs
        if v > 0:
            y[i] = v
            dydx[i] = 1.0
        else:                          # x == 0 falls here by convention
            y[i] = alpha * v
            dydx[i] = alpha
'''

HINTS = [
    'Allocate two device outputs and pass both, plus `alpha`, to a single launch: `leaky_kernel[blocks, TPB](d_x, alpha, d_y, d_g)`.',
    'Load `v = x[i]` into a local variable once; both outputs come from it.',
    'The comparison is strict: `if v > 0` takes the identity branch, so `x == 0` gets `y = 0` and `dydx = alpha`.',
]

_r = np.random.default_rng(6)
TESTS = [
    (np.array([-2.0, 0.0, 3.0], dtype=np.float32), 0.1),
    (np.array([1.0, -1.0, -0.5, 0.25], dtype=np.float32), 0.01),
    (np.zeros(0, dtype=np.float32), 0.2),
    (np.zeros(4, dtype=np.float32), 0.3),                   # every element hits the tie rule
    (np.array([-5.0, 5.0], dtype=np.float32), 0.0),         # alpha = 0 is plain ReLU
    (_r.standard_normal(300).astype(np.float32), 0.2),
    (_r.standard_normal(129).astype(np.float32), -0.5),
]

EXPLANATION = r"""
## Why it works
Each thread reads one input, decides the branch once, and writes the value and the local derivative. Because the derivative only depends on the sign of $x$, the backward pass later computes `grad_x = grad_y * dydx` with no transcendental work.

## Why fuse
Two separate kernels would read `x` from global memory twice and pay two launch overheads. Fused, the kernel moves 4 bytes in and 8 bytes out per element instead of 8 in and 8 out. For memory-bound ops, bytes moved is runtime, so this is a ~25% saving. In practice frameworks go further: they don't store `dydx` at all and recompute it from the sign of `x` (or `y`) during backward, trading 4 bytes of storage for one comparison.

## The tie at zero
Leaky ReLU is not differentiable at 0, so any value in $[\alpha, 1]$ is a valid subgradient. Frameworks pick one side by convention; what matters is that the forward and backward agree and that tests pin it down.

## Common bugs
- `if v >= 0` puts the tie on the wrong side.
- Writing `dydx[i] = alpha * v` (the output) instead of `alpha` (the slope).
- Two launches with `x` passed as a NumPy array each time: two host-to-device copies.

## Interview follow-ups
- Under what condition can ReLU-family backward be computed from the **output** instead of the input, and why does that let frameworks run the op in place? (When the output's sign determines the branch; for $\alpha > 0$, $y > 0 \iff x > 0$.)
- How would you vectorise the loads, e.g. 4 floats per thread? What alignment do you need?
"""
