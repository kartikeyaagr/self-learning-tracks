import numpy as np

META = dict(title='Temperature, top-k and top-p sampling with PRNG keys', tags=['Language Models', 'Inference', 'JAX'],
            diff='Medium', runtime='jax', lessons=['ieng:02.04', 'ieng:00.04'],
            related=['text-generation-sampling', 'prng-key-splitting', 'greedy-decode-loop-with-scan'], tol=[1e-4, 1e-5])

STATEMENT = r"""
Write the token sampler used in a JAX decode loop. Everything must be array ops (no Python loops over the batch or vocabulary), so it can run under `jit`.

Write `sample_tokens(logits, seed, temperature, top_k, top_p)`:

- `logits`: `(B, V)` float32. `temperature > 0`. `top_k`: int, `0` means no top-k filter. `top_p`: float in `(0, 1]`, `1.0` means no top-p filter.

Steps, per row:

1. Divide the logits by `temperature`.
2. **Top-k** (if `top_k > 0`): keep every token whose scaled logit is `>=` the `top_k`-th largest. Ties at the boundary are all kept.
3. **Top-p** (if `top_p < 1`): take the softmax over the tokens still kept. Sort them by probability, descending; equal probabilities are ordered by **lower token id first**. Keep a token if the total probability of the tokens strictly before it in that order is `< top_p`. (So the top token is always kept.)
4. The filtered log-probabilities are the `log_softmax` over the kept tokens, with `-inf` for removed tokens.
5. Sample with exactly `jax.random.categorical(jax.random.PRNGKey(seed), filtered, axis=-1)`.

Return `(tokens, filtered)`: `(B,)` int32 and `(B, V)` float32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def sample_tokens(logits, seed, temperature, top_k, top_p):
    """Return (tokens (B,), filtered log-probs (B, V) with -inf for removed tokens)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def sample_tokens(logits, seed, temperature, top_k, top_p):
    z = jnp.asarray(logits, jnp.float32) / temperature
    if top_k > 0:                                                    # Python ints: chosen at trace time
        kth = jax.lax.top_k(z, top_k)[0][:, -1:]                     # (B, 1) k-th largest value
        z = jnp.where(z >= kth, z, -jnp.inf)
    if top_p < 1.0:
        probs = jax.nn.softmax(z, axis=-1)
        order = jnp.argsort(-probs, axis=-1, stable=True)            # descending, ties by lower id
        sorted_p = jnp.take_along_axis(probs, order, axis=-1)
        before = jnp.cumsum(sorted_p, axis=-1) - sorted_p            # mass strictly before each token
        keep_sorted = before < top_p
        keep = jnp.zeros_like(keep_sorted).at[jnp.arange(z.shape[0])[:, None], order].set(keep_sorted)
        z = jnp.where(keep, z, -jnp.inf)
    filtered = jax.nn.log_softmax(z, axis=-1)                        # -inf stays -inf
    tokens = jax.random.categorical(jax.random.PRNGKey(seed), filtered, axis=-1)
    return tokens.astype(jnp.int32), filtered
'''

HINTS = [
    'Top-k by threshold: `jax.lax.top_k(z, k)[0][:, -1:]` is the k-th largest value per row; mask everything below it with `-inf`.',
    'For top-p, sort with `jnp.argsort(-probs, stable=True)`, compute `cumsum - p` (the mass before each token), compare with `top_p`, and scatter the keep mask back to vocabulary order with `.at[rows, order].set(...)`.',
    '`jax.nn.log_softmax` of a row with `-inf` entries renormalises over the rest and leaves them at `-inf`. Pass that to `jax.random.categorical` with `PRNGKey(seed)`.',
]

_r = np.random.default_rng(27)
TESTS = [
    (np.array([[2.0, 1.0, 0.0, -1.0]]), 0, 1.0, 2, 1.0),
    (np.array([[np.log(0.5), np.log(0.3), np.log(0.2)]]), 1, 1.0, 0, 0.6),
    (_r.normal(size=(4, 10)), 2, 0.7, 5, 0.9),
    (np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 3.0, 3.0, 3.0]]), 3, 1.0, 2, 1.0),
    (np.array([[0.0, 0.0, 0.0, 0.0]]), 4, 1.0, 0, 0.5),
    (3 * _r.normal(size=(3, 50)), 5, 1.5, 0, 0.8),
    (_r.normal(size=(2, 20)), 6, 0.1, 0, 1.0),
    (_r.normal(size=(5, 8)), 7, 1.0, 1, 0.3),
]

EXPLANATION = r"""
## Order of operations
Temperature first (it changes the distribution the filters see), then top-k, then top-p on the **renormalised** survivors, then sample. Different libraries order top-k and top-p differently, which is why "the same settings" give different text across engines.

## The top-p boundary
Keeping a token when the mass **before** it is below $p$ includes the token that crosses the threshold, so the kept set always has mass $\ge p$ and is never empty. In test 2 the sorted probabilities are 0.5, 0.3, 0.2; the mass before them is 0, 0.5, 0.8, so the first two are kept. Test 5 has four equal probabilities, where the tie rule decides that ids 0 and 1 survive.

## Why threshold top-k
Comparing against the k-th value keeps all tied tokens (test 4 keeps three tokens with `top_k=2`). The alternative, taking exactly the indices from `top_k`, has to break ties somehow, and that choice is rarely documented.

## Randomness
`jax.random.categorical` adds Gumbel noise from the key and takes the argmax, so `-inf` tokens can never win. The key is explicit: the same seed reproduces the same tokens, and a decode loop must split a fresh key every step.

## Complexity
$O(BV \log V)$ for the sort, which dominates for large vocabularies. Engines often apply top-k first (a partial selection) so top-p sorts only $k$ entries.

## Interview follow-ups
- What is min-p sampling and why is it more robust to temperature than top-p?
- Why does greedy decoding equal sampling at temperature $\to 0$, and what happens numerically at `temperature = 0.1` (test 7)?
"""
