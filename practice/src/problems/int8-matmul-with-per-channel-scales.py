import numpy as np

META = dict(title='W8A8 matmul with per-channel and per-token scales', tags=['Quantization', 'JAX', 'Inference'],
            diff='Medium', runtime='jax', lessons=['ieng:11.04', 'aieng:07.01'],
            related=['int8-symmetric-quantize', 'straight-through-estimator-with-custom-vjp', 'mixed-precision-with-loss-scaling'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
An int8 linear layer quantizes the weights once (offline) and the activations on the fly, multiplies in integers with int32 accumulation, and rescales the result back to float. One scale per tensor is too coarse, so use one scale **per output channel** for the weights and one **per token** (row) for the activations.

Write `int8_linear(x, w)` for `x` `(B, K)` and `w` `(K, N)`, both float32:

1. Weight scales, one per column $n$: $s^w_n = \max_k |w_{kn}| / 127$. Use $s^w_n = 1$ for an all-zero column.
2. Activation scales, one per row $b$: $s^x_b = \max_k |x_{bk}| / 127$, with the same zero rule.
3. Quantize: $q = \mathrm{clip}(\mathrm{round}(v / s), -127, 127)$ cast to int8, where `round` is round-half-to-even (`jnp.round`). Divide by the scale; do not multiply by its reciprocal.
4. Integer matmul: `acc = jax.lax.dot(qx, qw, preferred_element_type=jnp.int32)` gives exact int32 sums.
5. Dequantize: $y_{bn} = \text{acc}_{bn} \cdot s^x_b \cdot s^w_n$.

Return `(y, acc, qw, sw)`: `y` `(B, N)` float32, `acc` `(B, N)` int32, `qw` `(K, N)` int8 and `sw` `(N,)` float32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def int8_linear(x, w):
    """Return (y float32 (B, N), acc int32 (B, N), qw int8 (K, N), sw float32 (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _quantize(v, axis):
    """Symmetric int8 with one scale per slice, reducing over `axis`."""
    amax = jnp.max(jnp.abs(v), axis=axis, keepdims=True)
    s = jnp.where(amax > 0, amax / 127.0, 1.0)
    q = jnp.clip(jnp.round(v / s), -127, 127).astype(jnp.int8)    # clip before the cast: saturate, don't wrap
    return q, s

def int8_linear(x, w):
    x, w = jnp.asarray(x, jnp.float32), jnp.asarray(w, jnp.float32)
    qw, sw = _quantize(w, axis=0)                  # (K, N), (1, N): one scale per output channel
    qx, sx = _quantize(x, axis=1)                  # (B, K), (B, 1): one scale per token
    acc = jax.lax.dot(qx, qw, preferred_element_type=jnp.int32)    # int8 x int8 -> exact int32
    y = acc.astype(jnp.float32) * sx * sw          # outer product of scales broadcasts to (B, N)
    return y, acc, qw, sw[0]
'''

HINTS = [
    'Per-column weight scales reduce over axis 0 (`keepdims=True` gives `(1, N)`); per-row activation scales reduce over axis 1 (`(B, 1)`).',
    'Round, clip to `[-127, 127]`, then `.astype(jnp.int8)`. `jax.lax.dot(..., preferred_element_type=jnp.int32)` accumulates in int32 instead of overflowing int8.',
    'Because the scales factor out of each dot product, $y = \\text{acc} \\cdot s^x s^w$ with `(B, 1)` and `(1, N)` broadcasting.',
]

_r = np.random.default_rng(23)
_wo = _r.normal(size=(16, 4))
_wo[:, 1] *= 50.0                              # one outlier channel
TESTS = [
    (np.array([[1.0, -1.0]]), np.array([[1.0, 0.5], [2.0, -0.5]])),
    (np.array([[0.0, 0.0], [127.0, 63.5]]), np.array([[1.0, 0.0], [0.0, 0.0]])),
    (_r.normal(size=(3, 8)), _r.normal(size=(8, 5))),
    (_r.normal(size=(5, 16)), _wo),
    (np.vstack([_r.normal(size=(1, 32)), 100 * _r.normal(size=(1, 32))]), _r.normal(size=(32, 6))),
    (_r.normal(size=(1, 64)) * 1e-3, _r.normal(size=(64, 3)) * 1e-3),
    (_r.normal(size=(4, 256)), _r.normal(size=(256, 2))),
]

EXPLANATION = r"""
## Why the scales factor out
$y_{bn} = \sum_k x_{bk} w_{kn} \approx \sum_k (s^x_b q^x_{bk})(s^w_n q^w_{kn}) = s^x_b s^w_n \sum_k q^x_{bk} q^w_{kn}$. A per-row scale on $x$ and a per-column scale on $w$ are constant inside each dot product, so they can be applied once to the int32 result. A per-**input**-channel scale on the weights (one per $k$) would not factor out, which is why weight scales are per output channel.

## Accumulation width
Each product is at most $127^2 = 16129$; a 256-term sum (test 7) can reach about $4 \times 10^6$, far beyond int8 and beyond fp16's exact-integer range. int32 holds sums of up to about 133,000 such products exactly, which covers realistic $K$.

## Granularity and outliers
Test 4 has one weight column 50 times larger than the rest. Per-channel scales keep that column from wrecking the others' precision, which a single per-tensor scale would do. Test 5 does the same for one large token. Activation outliers in fixed **channels** (the LLM case) are not helped by per-token scales, which is what SmoothQuant addresses by moving the scale into the weights.

## Common bugs
- Casting to int8 before clipping: 128 wraps to −128.
- Accumulating in int8 or float16.
- Using `v * (127 / amax)`: occasionally rounds differently from `v / s`, so results are not bit-identical to the reference kernel.

## Interview follow-ups
- Why is W8A8 a win for prefill and large batches but weight-only int4 popular for single-stream decode?
- How would you fold the activation scale computation into the previous layer's kernel to avoid an extra pass over memory?
"""
