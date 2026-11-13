import numpy as np

META = dict(title='One vectorised beam-search step with finished beams', tags=['Language Models', 'JAX'], diff='Hard',
            runtime='jax', lessons=['ieng:02.04'],
            related=['beam-search-decoding', 'greedy-decode-loop-with-scan', 'top-k-and-top-p-sampling'], tol=[1e-5, 1e-6])

STATEMENT = r"""
Beam search keeps the $K$ highest-scoring partial sequences per example. Write one step of it as pure array code (it will sit inside a `scan`), including beams that have already finished.

Write `beam_step(scores, tokens, finished, log_probs, eos_id)`:

- `scores`: `(B, K)` float32, cumulative log-probabilities. May contain `-inf` (for example, at the first step every beam but one is set to `-inf` so the identical start beams are not selected $K$ times).
- `tokens`: `(B, K, t)` int, the tokens so far (`t >= 0`).
- `finished`: `(B, K)` bool.
- `log_probs`: `(B, K, V)` float32, next-token log-probabilities for each beam.

Rules:

1. Candidate $(k, v)$ has score `scores[b, k] + log_probs[b, k, v]` if beam $k$ is not finished. A finished beam has exactly one candidate, $v$ = `eos_id`, with its score **unchanged**; its other candidates score `-inf`.
2. Flatten the candidates to index $k \cdot V + v$ and keep the $K$ best by score, descending. Equal scores are ordered by **lower flat index first**.
3. For each kept candidate: `parent = index // V`, `token = index % V`; the new history is `tokens[b, parent]` with `token` appended; it is finished if its parent was finished or `token == eos_id`.

Return `(new_scores, new_tokens, new_finished, parent)` with shapes `(B, K)`, `(B, K, t + 1)`, `(B, K)`, `(B, K)`. Scores float32, tokens and parents int32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def beam_step(scores, tokens, finished, log_probs, eos_id):
    """Return (new_scores (B, K), new_tokens (B, K, t+1), new_finished (B, K), parent (B, K))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def beam_step(scores, tokens, finished, log_probs, eos_id):
    scores = jnp.asarray(scores, jnp.float32)
    lp = jnp.asarray(log_probs, jnp.float32)
    tokens, finished = jnp.asarray(tokens, jnp.int32), jnp.asarray(finished, bool)
    B, K, V = lp.shape

    # Finished beams may only "extend" with EOS at zero cost, so they keep their score and compete as-is.
    only_eos = jnp.where(jnp.arange(V) == eos_id, 0.0, -jnp.inf)                 # (V,)
    cand = scores[:, :, None] + jnp.where(finished[:, :, None], only_eos, lp)     # (B, K, V)
    flat = cand.reshape(B, K * V)

    top = jnp.argsort(-flat, axis=-1, stable=True)[:, :K]                        # descending, ties by lower index
    new_scores = jnp.take_along_axis(flat, top, axis=-1)
    parent, tok = top // V, (top % V).astype(jnp.int32)

    hist = jnp.take_along_axis(tokens, parent[:, :, None], axis=1)               # (B, K, t) reordered
    new_tokens = jnp.concatenate([hist, tok[:, :, None]], axis=-1)
    new_finished = jnp.take_along_axis(finished, parent, axis=1) | (tok == eos_id)
    return new_scores, new_tokens, new_finished, parent.astype(jnp.int32)
'''

HINTS = [
    'Build a `(B, K, V)` candidate array: unfinished beams add `log_probs`; finished beams add a row that is 0 at `eos_id` and `-inf` elsewhere. Then reshape to `(B, K * V)`.',
    '`jnp.argsort(-flat, stable=True)[:, :K]` gives the top $K$ with ties broken by lower index (negation keeps the stable order among equal values).',
    'Reorder the histories and finished flags with `jnp.take_along_axis(..., parent[..., None], axis=1)`, then append the new tokens.',
]

_r = np.random.default_rng(32)

def _lp(B, K, V, s=1.0):
    z = s * _r.normal(size=(B, K, V))
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

_ninf = -np.inf
TESTS = [
    (np.array([[0.0, _ninf]]), np.zeros((1, 2, 0), int), np.array([[False, False]]),
     np.log(np.array([[[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]])), 2),
    (np.array([[-1.0, -2.0]]), np.array([[[0], [1]]]), np.array([[True, False]]),
     np.log(np.array([[[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]])), 2),
    (np.array([[0.0, _ninf, _ninf]]), np.zeros((1, 3, 0), int), np.zeros((1, 3), bool),
     np.log(np.full((1, 3, 4), 0.25)), 0),
    (np.log(_r.random((2, 3))), _r.integers(0, 5, size=(2, 3, 2)), np.zeros((2, 3), bool), _lp(2, 3, 5), 4),
    (np.log(_r.random((3, 4))), _r.integers(0, 6, size=(3, 4, 3)), _r.random((3, 4)) < 0.4, _lp(3, 4, 6, 2.0), 1),
    (np.array([[-0.5, -0.7]]), np.array([[[3, 1], [2, 1]]]), np.array([[True, True]]), _lp(1, 2, 4), 1),
    (np.log(_r.random((2, 2))), _r.integers(0, 3, size=(2, 2, 4)), np.array([[False, True], [True, False]]),
     _lp(2, 2, 3), 0),
]

EXPLANATION = r"""
## One sort over $K \cdot V$ candidates
Every beam proposes every token; the best $K$ of the $K V$ pairs survive, possibly several from the same parent and none from others. The flat index encodes both: `parent = i // V`, `token = i % V`. This is why beam search is a reorder (`take_along_axis` on the parent) followed by an append.

## Finished beams
A finished beam must stay in the race with its final score, otherwise the search would discard complete hypotheses in favour of worse unfinished ones. Giving it a single candidate (EOS, cost 0) does exactly that without changing shapes. In test 2 the finished beam (score −1) beats every extension of the other beam. Test 6 has only finished beams, which just carry over.

## Ties and the start state
Test 3 starts with one live beam and four equally likely tokens: the scores tie, and the rule keeps tokens 0, 1 and 2 of beam 0 in that order. Setting all but one start beam to `-inf` is the standard trick; if all $K$ start beams were live and identical, the search would pick the same continuation $K$ times.

## Complexity
$O(B K V \log(K V))$ for a full sort; `lax.top_k` brings that to roughly $O(BKV)$, at the price of an unspecified tie order. Reordering the history costs $O(BKt)$ per step, which is why real implementations store only parent pointers and backtrack at the end, and reorder the KV cache with the same `parent` indices.

## Interview follow-ups
- Where does length normalisation enter, and why does unnormalised beam search prefer short outputs?
- Why does beam search help translation but often hurt open-ended generation?
"""
