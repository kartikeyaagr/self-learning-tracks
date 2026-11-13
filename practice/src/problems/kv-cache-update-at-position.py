import numpy as np

META = dict(title='Write new keys into a batched KV cache', tags=['Inference', 'JAX'], diff='Easy', runtime='jax',
            lessons=['ieng:03.04', 'aieng:04.04'], related=['grouped-query-decode-step', 'paged-kv-cache-gather',
                                                            'functional-array-update'], tol=[1e-6, 1e-7])

STATEMENT = r"""
A JAX KV cache is a fixed-size array (shapes must be static for `jit`), and writing into it is a functional update that returns a new array. In a batch, every sequence is at a **different** position, so each row gets its own write offset.

Write `update_cache(k_cache, v_cache, k_new, v_new, pos)`:

- `k_cache`, `v_cache`: `(B, S_max, H, d)` float32.
- `k_new`, `v_new`: `(B, T_new, H, d)` float32, the keys and values of the new tokens.
- `pos`: `(B,)` int, the current length of each sequence. Row `b`'s new entries go to positions `pos[b] .. pos[b] + T_new - 1`. It is guaranteed that `pos[b] + T_new <= S_max`.

Return `(k_cache, v_cache, lengths, valid)`:

- the updated caches (everything outside the written slots unchanged);
- `lengths = pos + T_new`, `(B,)` int;
- `valid`: `(B, S_max)` bool, `True` at positions `< lengths[b]`.

Use `jax.lax.dynamic_update_slice` with `jax.vmap` over the batch. No Python loop over `B`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def update_cache(k_cache, v_cache, k_new, v_new, pos):
    """Return (k_cache, v_cache, lengths (B,), valid (B, S_max) bool)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _write(cache, new, p):
    # One sequence: cache (S_max, H, d), new (T_new, H, d). Start indices are (p, 0, 0); they may be traced values.
    return jax.lax.dynamic_update_slice(cache, new, (p, 0, 0))

def update_cache(k_cache, v_cache, k_new, v_new, pos):
    k_cache, v_cache, k_new, v_new = (jnp.asarray(a, jnp.float32) for a in (k_cache, v_cache, k_new, v_new))
    pos = jnp.asarray(pos, jnp.int32)
    write = jax.vmap(_write)                       # each row has its own offset
    lengths = pos + k_new.shape[1]
    valid = jnp.arange(k_cache.shape[1])[None, :] < lengths[:, None]
    return write(k_cache, k_new, pos), write(v_cache, v_new, pos), lengths, valid
'''

HINTS = [
    '`jax.lax.dynamic_update_slice(operand, update, start_indices)` writes `update` into `operand` at a start index that can be a traced value. You need one start index per axis.',
    'Write the update for one sequence (`(S_max, H, d)` cache, `(T_new, H, d)` update, scalar start), then `jax.vmap` it over the batch.',
    '`valid` is a broadcast comparison: `arange(S_max)[None, :] < lengths[:, None]`.',
]

_r = np.random.default_rng(26)
TESTS = [
    (np.zeros((1, 4, 1, 1)), np.zeros((1, 4, 1, 1)), np.array([[[[1.0]]]]), np.array([[[[2.0]]]]), np.array([2])),
    (np.zeros((2, 3, 1, 2)), np.zeros((2, 3, 1, 2)), np.ones((2, 2, 1, 2)), 2 * np.ones((2, 2, 1, 2)), np.array([0, 1])),
    (_r.normal(size=(3, 8, 2, 4)), _r.normal(size=(3, 8, 2, 4)), _r.normal(size=(3, 1, 2, 4)), _r.normal(size=(3, 1, 2, 4)),
     np.array([0, 5, 7])),
    (_r.normal(size=(2, 6, 1, 3)), _r.normal(size=(2, 6, 1, 3)), _r.normal(size=(2, 6, 1, 3)), _r.normal(size=(2, 6, 1, 3)),
     np.array([0, 0])),
    (np.zeros((2, 10, 2, 2)), np.zeros((2, 10, 2, 2)), _r.normal(size=(2, 4, 2, 2)), _r.normal(size=(2, 4, 2, 2)),
     np.array([6, 3])),
    (_r.normal(size=(4, 5, 3, 2)), _r.normal(size=(4, 5, 3, 2)), _r.normal(size=(4, 2, 3, 2)), _r.normal(size=(4, 2, 3, 2)),
     np.array([3, 0, 1, 2])),
]

EXPLANATION = r"""
## Functional writes, in-place execution
`dynamic_update_slice` returns a new array, so the Python code is pure. Under `jit`, when the old cache is not used afterwards (and especially with `donate_argnums` on the cache), XLA performs the update in place, so a decode step costs a write of $T_\text{new}$ rows, not a copy of the whole cache.

## Why `vmap` and not a slice
`cache[b, p:p+T]` with a traced `p` is not allowed (slice sizes must be static and starts must be concrete for basic indexing). `dynamic_update_slice` takes traced starts but one offset for the whole array; `vmap` turns it into a batched scatter with one offset per row.

## The clamping trap
If `pos + T_new > S_max`, `dynamic_update_slice` does **not** raise: it clamps the start index so the update fits, silently overwriting earlier positions. Production code must enforce the bound (stop generation, evict, or allocate more pages) before calling it.

## Complexity
$O(B\, T_\text{new}\, H d)$ written per call; the `valid` mask is $O(B\, S_\text{max})$ and feeds the attention mask.

## Interview follow-ups
- How large is the KV cache for a 32-layer, 8-KV-head, $d = 128$ model at 8k context in bf16, per sequence? ($2 \cdot 32 \cdot 8 \cdot 128 \cdot 8192 \cdot 2$ bytes $= 1$ GiB.)
- Why do serving engines move from a contiguous cache like this one to paged blocks?
"""
