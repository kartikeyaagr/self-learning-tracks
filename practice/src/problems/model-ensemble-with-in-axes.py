import numpy as np

META = dict(title='Run a model ensemble with vmap in_axes', tags=['JAX', 'Neural Networks'], diff='Easy', runtime='jax',
            lessons=['air-fanout-v1:04.07'], related=['prng-key-splitting', 'batched-attention-with-vmap'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
An ensemble of $E$ networks with the same architecture is one stacked set of parameters with a leading axis of size $E$. `jax.vmap` over that axis runs all members at once, and `in_axes` decides whether the input is shared or different per member.

Each member is a two-layer MLP classifier:

$$h = \mathrm{relu}(x W_1 + b_1), \qquad p = \mathrm{softmax}(h W_2 + b_2)$$

Write `ensemble_predict(params, x, per_member)`:

- `params`: dict `{'w1': (E, D, H), 'b1': (E, H), 'w2': (E, H, K), 'b2': (E, K)}`.
- `per_member`: Python bool. If `False`, `x` is `(N, D)` and every member sees the same inputs. If `True`, `x` is `(E, N, D)` and member `e` sees `x[e]` (as in bagging, where each member gets its own batch).
- Return `(probs, mean_probs)`: `probs` is `(E, N, K)`, member `e`'s class probabilities; `mean_probs` is `(N, K)`, their average over members. float32.

Write the forward pass for **one** member with unbatched parameters, then `vmap` it. No Python loop over members.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def ensemble_predict(params, x, per_member):
    """Return (probs (E, N, K), mean over members (N, K))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def forward(p, x):
    """One member: p holds unbatched weights, x is (N, D)."""
    h = jax.nn.relu(x @ p['w1'] + p['b1'])
    return jax.nn.softmax(h @ p['w2'] + p['b2'], axis=-1)

def ensemble_predict(params, x, per_member):
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    x = jnp.asarray(x, jnp.float32)
    # in_axes for a dict argument: one int maps every leaf along axis 0. per_member is a Python bool,
    # so the choice is made once, at trace time.
    probs = jax.vmap(forward, in_axes=(0, 0 if per_member else None))(params, x)
    return probs, probs.mean(axis=0)
'''

HINTS = [
    'Write `forward(p, x)` for a single member as if `w1` were `(D, H)`. `vmap` will slice every leaf of `params` along axis 0.',
    'A single `0` in `in_axes` applies to every leaf of the `params` dict. For `x`, use `None` when it is shared and `0` when each member has its own.',
    '`mean_probs` is just `probs.mean(axis=0)`.',
]

_r = np.random.default_rng(14)

def _p(E, D, H, K, s=1.0):
    return {'w1': s * _r.normal(size=(E, D, H)), 'b1': _r.normal(size=(E, H)),
            'w2': s * _r.normal(size=(E, H, K)), 'b2': _r.normal(size=(E, K))}

_eye = {'w1': np.stack([np.eye(2), 2 * np.eye(2)]), 'b1': np.zeros((2, 2)),
        'w2': np.stack([np.eye(2), np.eye(2)]), 'b2': np.zeros((2, 2))}
TESTS = [
    (_eye, np.array([[1.0, 0.0], [0.0, 0.0]]), False),
    (_eye, np.array([[[1.0, 0.0]], [[0.0, 1.0]]]), True),
    (_p(3, 4, 5, 3), _r.normal(size=(6, 4)), False),
    (_p(3, 4, 5, 3), _r.normal(size=(3, 6, 4)), True),
    (_p(1, 2, 3, 2), _r.normal(size=(4, 2)), False),
    (_p(5, 3, 8, 4, 3.0), _r.normal(size=(2, 3)), False),
    (_p(4, 6, 2, 5), _r.normal(size=(4, 1, 6)), True),
]

EXPLANATION = r"""
## How `in_axes` works on a pytree
`in_axes` mirrors the arguments. A bare `0` for `params` means "every leaf is batched along axis 0", so the dict of `(E, ...)` arrays is sliced into $E$ dicts of unbatched weights. `None` for `x` broadcasts the same input to every member; `0` slices it like the parameters. The output gets the batch axis in front (`out_axes=0`), giving `(E, N, K)`.

## Why this beats a loop
A Python loop over members traces $E$ copies of the network and launches $E$ small matmuls. `vmap` turns `x @ w1` into one batched matmul over $E$, which keeps the hardware busy. The stacked-parameter layout is also how ensembles are trained: initialise with `vmap` over split PRNG keys, then `vmap` the training step.

## Averaging probabilities, not logits
The ensemble prediction averages probabilities. Averaging logits is a different model (a product of experts after renormalising) and is usually more overconfident.

## Common bugs
- `in_axes=(0, 0)` with a shared `(N, D)` input: JAX maps over the $N$ axis and fails with a size mismatch (or silently pairs member $e$ with row $e$ when $N = E$).
- Writing `forward` for batched weights with `einsum('nd,edh->enh')` and then also wrapping it in `vmap`: the batch axis gets added twice.

## Interview follow-ups
- How would you initialise the $E$ members differently? (Split one PRNG key into $E$ keys and `vmap` the init function.)
- What changes to train the ensemble with one jitted step? (`vmap` the loss-and-grad function over params and per-member batches; the optimizer state gets the same leading axis.)
"""
