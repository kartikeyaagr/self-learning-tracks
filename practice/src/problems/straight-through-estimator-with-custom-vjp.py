import numpy as np

META = dict(title='Straight-through fake quantization with custom_vjp', tags=['JAX', 'Quantization', 'Training'],
            diff='Medium', runtime='jax', lessons=['ieng:11.02'],
            related=['int8-symmetric-quantize', 'int8-matmul-with-per-channel-scales'], tol=[1e-4, 1e-5])

STATEMENT = r"""
Quantization-aware training runs the forward pass with weights rounded to a low-bit grid. Rounding has zero gradient almost everywhere, so training would stall. The **straight-through estimator** (STE) pretends rounding is the identity in the backward pass, and zeroes the gradient where the weight was clipped.

With `qmax` $= 2^{\text{bits}-1} - 1$, fake quantization is

$$\mathrm{fq}(w) = s \cdot \mathrm{clip}\bigl(\mathrm{round}(w / s),\, -q_{\max},\, q_{\max}\bigr)$$

Implement it with `jax.custom_vjp` so that its backward pass is

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial\, \mathrm{fq}(w)} \cdot \mathbb{1}\bigl[\,|w / s| \le q_{\max}\bigr]$$

(the boundary $|w/s| = q_{\max}$ counts as inside). Rounding is half to even, as `jnp.round` does.

Write `qat_loss_and_grad(w, x, y, scale, bits)`:

- `w`: `(D, K)` float weights; `x`: `(B, D)`; `y`: `(B, K)`; `scale`: float $s > 0$; `bits`: int $\ge 2$.
- Loss: $\mathcal{L} = \mathrm{mean}\bigl((x\, \mathrm{fq}(w) - y)^2\bigr)$ over all $B \cdot K$ entries.

Return `(loss, grad_w, w_fq)`: float32 scalar, `(D, K)` STE gradient and `(D, K)` fake-quantized weights.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def qat_loss_and_grad(w, x, y, scale, bits):
    """Return (loss, STE gradient w.r.t. w (D, K), fake-quantized w (D, K))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def qat_loss_and_grad(w, x, y, scale, bits):
    w, x, y = (jnp.asarray(a, jnp.float32) for a in (w, x, y))
    qmax = 2 ** (bits - 1) - 1

    @jax.custom_vjp
    def fake_quant(w):
        return scale * jnp.clip(jnp.round(w / scale), -qmax, qmax)

    def fq_fwd(w):
        inside = jnp.abs(w / scale) <= qmax          # residual saved for the backward pass
        return fake_quant(w), inside

    def fq_bwd(inside, g):
        return (jnp.where(inside, g, 0.0),)          # identity inside the range, zero where clipped

    fake_quant.defvjp(fq_fwd, fq_bwd)

    def loss_fn(w):
        wq = fake_quant(w)
        return jnp.mean((x @ wq - y) ** 2), wq

    (loss, wq), g = jax.value_and_grad(loss_fn, has_aux=True)(w)
    return loss, g, wq
'''

HINTS = [
    'Decorate the forward function with `@jax.custom_vjp`, then call `fake_quant.defvjp(fwd, bwd)`. `fwd` returns `(output, residuals)`; `bwd(residuals, g)` returns a **tuple** with one cotangent per input.',
    'The only residual you need is the boolean mask `|w / s| <= qmax`, computed in `fwd`.',
    'Wrap the loss in `jax.value_and_grad(..., has_aux=True)` and return the fake-quantized weights as the aux output.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([[0.3], [-0.6]]), np.array([[1.0, 1.0]]), np.array([[0.0]]), 0.5, 2),
    (np.array([[0.5, 2.0], [-0.25, -3.0]]), np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.5, 2),
    (_r.normal(size=(4, 3)), _r.normal(size=(5, 4)), _r.normal(size=(5, 3)), 0.1, 4),
    (_r.normal(size=(4, 3)), _r.normal(size=(5, 4)), _r.normal(size=(5, 3)), 0.02, 4),
    (_r.normal(size=(6, 2)), _r.normal(size=(8, 6)), _r.normal(size=(8, 2)), 0.01, 8),
    (3 * _r.normal(size=(3, 3)), _r.normal(size=(4, 3)), np.zeros((4, 3)), 1.0, 2),
    (_r.normal(size=(8, 4)) * 0.05, _r.normal(size=(16, 8)), _r.normal(size=(16, 4)), 0.05 / 7, 4),
]

EXPLANATION = r"""
## Why the STE trains
The true derivative of `round` is zero almost everywhere, so plain autodiff returns zero gradients and the weights never move. The STE replaces it with the identity, treating `fq(w)` as `w` plus bounded rounding noise. The latent float weights accumulate small updates until they cross a rounding boundary. Outside the clip range, more movement can't change the output, so the gradient is zeroed; otherwise those weights would drift off without bound.

## `custom_vjp` mechanics
- `fwd(w)` runs the primal and returns residuals for later.
- `bwd(res, g)` maps the output cotangent `g` to a tuple of input cotangents.
- JAX uses your rule wherever `fake_quant` appears inside `grad`, `jit` or `vmap`. Closing over `scale` and `qmax` is fine because they are Python constants; to differentiate with respect to a learnable scale (LSQ) you would make it an input and return its cotangent too.

## Cost
Same as the unquantized layer plus elementwise ops. The saved residual is a boolean mask of the weight's shape.

## Common bugs
- `bwd` returning `g` instead of `(g,)`: JAX expects a tuple.
- Using `jax.lax.stop_gradient(round(w) - w) + w` but forgetting the clip mask: gradients flow into saturated weights.
- Strict `<` at the boundary: a weight exactly at $q_{\max} s$ (test 2) should keep its gradient.

## Interview follow-ups
- How does LSQ make the scale learnable, and what is the gradient with respect to $s$?
- Why does QAT usually beat post-training quantization at 4 bits but rarely matter at 8 bits?
"""
