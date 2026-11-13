import numpy as np

META = dict(title='Spectral norm by power iteration with fori_loop', tags=['JAX', 'Linear Algebra'], diff='Medium',
            runtime='jax', lessons=['air-fanout-v1:01.13'], related=['hessian-vector-product', 'exponential-moving-average-with-scan'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
Spectral normalisation divides a weight matrix by its largest singular value $\sigma_{\max}$ to bound the layer's Lipschitz constant. A full SVD is too slow to run every training step, so $\sigma_{\max}$ is estimated by a few rounds of power iteration.

Write `spectral_norm(W, u0, n_iters)`:

- `W`: `(M, N)` float matrix; `u0`: `(M,)` starting vector (not zero); `n_iters`: int $\ge 1$.
- Let $\mathrm{nrm}(z) = z / (\lVert z \rVert_2 + 10^{-12})$.
- Start with $u = \mathrm{nrm}(u_0)$, then repeat `n_iters` times:
  1. $v \leftarrow \mathrm{nrm}(W^\top u)$
  2. $u \leftarrow \mathrm{nrm}(W v)$
- Finally $\sigma = u^\top W v$ using the last $u$ and $v$.

Return `(sigma, u, v)`: a float32 scalar and float32 vectors of shapes `(M,)` and `(N,)`. Run the loop with `jax.lax.fori_loop`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def spectral_norm(W, u0, n_iters):
    """Return (sigma, u (M,), v (N,)) after n_iters power-iteration rounds."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _nrm(z):
    return z / (jnp.linalg.norm(z) + 1e-12)

def spectral_norm(W, u0, n_iters):
    W = jnp.asarray(W, jnp.float32)
    u = _nrm(jnp.asarray(u0, jnp.float32))
    v = jnp.zeros(W.shape[1], jnp.float32)           # placeholder with the right shape: the loop carry is (u, v)

    def body(_, uv):
        u, _ = uv
        v = _nrm(W.T @ u)
        return _nrm(W @ v), v

    u, v = jax.lax.fori_loop(0, n_iters, body, (u, v))
    return u @ W @ v, u, v
'''

HINTS = [
    '`jax.lax.fori_loop(lower, upper, body, init)` calls `body(i, carry)` and returns the final carry. Carry the pair `(u, v)`.',
    'The carry needs a fixed shape from the start, so initialise `v` as zeros of shape `(N,)`; the first iteration overwrites it.',
    'After the loop, $\\sigma = u^\\top W v$ is `u @ W @ v`. Normalise with the `1e-12` epsilon exactly as stated.',
]

_r = np.random.default_rng(19)
_A = _r.normal(size=(6, 4))
TESTS = [
    (np.diag([3.0, 1.0]), np.array([1.0, 1.0]), 1),
    (np.diag([3.0, 1.0]), np.array([1.0, 1.0]), 20),
    (np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]), np.array([1.0, 0.0]), 3),
    (_A, _r.normal(size=6), 1),
    (_A, _r.normal(size=6), 50),
    (_r.normal(size=(3, 8)), np.ones(3), 10),
    (np.outer([1.0, 2.0, 2.0], [0.0, 3.0, 4.0]), np.array([0.0, 0.0, 1.0]), 2),
    (-2.0 * np.eye(4), np.array([1.0, -1.0, 0.5, 0.0]), 5),
]

EXPLANATION = r"""
## Why power iteration converges
$W^\top W$ has eigenvalues $\sigma_i^2$. Each round applies $W^\top W$ to $v$ (and $W W^\top$ to $u$), so components along smaller singular directions shrink by $(\sigma_i / \sigma_1)^2$ per round. After normalising, $v$ and $u$ approach the top right and left singular vectors and $u^\top W v \to \sigma_1$. A rank-1 matrix (test 7) converges in one round; equal singular values never "converge" on a direction but $\sigma$ is still exact (test 8).

## In practice: one iteration per step
SN-GAN keeps $u$ in the training state and runs **one** iteration per optimiser step. Because $W$ changes slowly, the warm-started $u$ stays close to the top singular vector, so a single round is enough. The estimate is also treated as a constant in the backward pass (`stop_gradient` on $u$ and $v$).

## `fori_loop` vs `scan`
`fori_loop` with Python-int bounds lowers to a `scan`; with traced bounds it lowers to a `while_loop`, which cannot be reverse-differentiated. Use `scan` when you need gradients through the loop and only the final carry from `fori_loop` otherwise.

## Cost
Two matrix-vector products per round: $O(MN)$ each, versus $O(MN\min(M,N))$ for an SVD.

## Common bugs
- Normalising `u` but not `v` (or the reverse): the final $u^\top W v$ is then not $\sigma$.
- Initialising the carry `v` with a different shape or dtype than the body returns.
- Forgetting that the sign of $u, v$ is arbitrary: $\sigma$ is always $\ge 0$ when both come from the same iteration.

## Interview follow-ups
- Why does spectral normalisation stabilise GAN discriminators? (It bounds the Lipschitz constant, preventing sharp gradients that let the discriminator overpower the generator.)
- How would you estimate $\sigma_{\max}$ of a convolution without building its matrix? (Power iteration with the conv and its transpose, i.e. a VJP.)
"""
