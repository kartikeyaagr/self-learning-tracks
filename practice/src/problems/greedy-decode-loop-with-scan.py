import numpy as np

META = dict(title='Batched greedy decoding with EOS in lax.scan', tags=['Language Models', 'JAX', 'Inference'],
            diff='Medium', runtime='jax', lessons=['ieng:02.05'],
            related=['exponential-moving-average-with-scan', 'top-k-and-top-p-sampling', 'kv-cache-update-at-position'],
            tol=[1e-6, 1e-7])

STATEMENT = r"""
A compiled decode loop has a fixed number of steps and fixed shapes, so sequences that finish early keep running but emit padding. Write that loop with `jax.lax.scan` for a toy recurrent language model.

Model, with `params = {'emb': (V, D), 'w': (D, D), 'out': (D, V)}` and a hidden state $h$ of shape `(D,)` per sequence, starting at zeros:

- Consume a token $t$: $h \leftarrow \tanh(\text{emb}[t] + h\, W)$.
- Predict: logits $= h\, W_\text{out}$; the greedy token is the `argmax` (ties go to the lower id).

Write `greedy_decode(params, prompt, n_steps, eos_id, pad_id)`:

1. `prompt`: `(B, P)` int, `P >= 1`. Consume the prompt tokens in order, for every sequence (one scan).
2. Then run `n_steps` (a Python int) generation steps (a second scan). At each step, predict a token from the current state. If the sequence has **already** emitted `eos_id`, the step's token is `pad_id` instead. Record the token, mark the sequence done if the token is `eos_id`, and consume the token to update $h$.
3. Return `(tokens, lengths)`: `tokens` `(B, n_steps)` int32; `lengths` `(B,)` int32, the number of non-padding tokens generated per sequence, **including** the EOS token.

`0 <= eos_id, pad_id < V`. Process the whole batch at once (batched ops or `vmap`), no Python loops.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def greedy_decode(params, prompt, n_steps, eos_id, pad_id):
    """prompt (B, P). Return (tokens (B, n_steps) int32, lengths (B,) int32)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def greedy_decode(params, prompt, n_steps, eos_id, pad_id):
    p = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    prompt = jnp.asarray(prompt, jnp.int32)
    B = prompt.shape[0]

    def consume(h, tok):                                   # h (B, D), tok (B,)
        return jnp.tanh(p['emb'][tok] + h @ p['w'])

    # 1. Prefill: scan over prompt positions (time-major, so transpose to (P, B)).
    h, _ = jax.lax.scan(lambda h, t: (consume(h, t), None), jnp.zeros((B, p['w'].shape[0])), prompt.T)

    # 2. Generate: the carry holds the state and the done flags; each step emits one token per row.
    def step(carry, _):
        h, done = carry
        tok = jnp.argmax(h @ p['out'], axis=-1).astype(jnp.int32)    # first max wins ties
        tok = jnp.where(done, pad_id, tok)
        emitted = ~done                                               # counts toward the length, EOS included
        done = done | (tok == eos_id)
        return (consume(h, tok), done), (tok, emitted)

    _, (toks, emitted) = jax.lax.scan(step, (h, jnp.zeros(B, bool)), None, length=n_steps)
    return toks.T, emitted.sum(axis=0).astype(jnp.int32)
'''

HINTS = [
    'Scan over `prompt.T` (shape `(P, B)`) so each step consumes one position for the whole batch. The carry is the `(B, D)` hidden state.',
    'For generation, call `jax.lax.scan(step, init, None, length=n_steps)`: no per-step inputs, the carry is `(h, done)` and each step outputs `(token, emitted)`.',
    'Decide the token, then replace it with `pad_id` where `done` was already true, then update `done`. A step counts toward the length when the sequence was not done **before** it.',
]

_r = np.random.default_rng(29)

def _p(V, D, s=1.5):
    return {'emb': s * _r.normal(size=(V, D)), 'w': _r.normal(size=(D, D)) / np.sqrt(D), 'out': s * _r.normal(size=(D, V))}

# Hand-made model: state after token t is tanh(emb[t] + ...) and out picks token (t + 1) mod 3.
_cyc = {'emb': 3 * np.eye(3), 'w': np.zeros((3, 3)), 'out': np.roll(np.eye(3), 1, axis=1)}
_P1 = _p(6, 4)
TESTS = [
    (_cyc, np.array([[0]]), 4, 2, 0),
    (_cyc, np.array([[1], [0]]), 3, 0, 1),
    (_P1, _r.integers(0, 6, size=(3, 4)), 6, 5, 0),
    (_P1, _r.integers(0, 6, size=(4, 1)), 8, 0, 1),
    (_p(10, 8), _r.integers(0, 10, size=(5, 3)), 10, 3, 9),
    (_p(4, 3), _r.integers(0, 4, size=(2, 2)), 1, 1, 0),
    (_p(12, 6), _r.integers(0, 12, size=(6, 5)), 12, 11, 0),
]

EXPLANATION = r"""
## Fixed shapes, data-dependent stopping
`jit` needs static shapes, so the loop always runs `n_steps` iterations and every row produces `n_steps` tokens. "Stopping" is a mask: once a row is done, its tokens become `pad_id`. In test 1 the hand-made model cycles $0 \to 1 \to 2$; with `eos_id = 2`, the outputs are `[1, 2, 0, 0]` and the length is 2. Engines that serve many requests avoid this wasted work with continuous batching, which swaps finished rows for new requests between steps.

## Two scans
Prefill and generation are both scans, but with different inputs: prefill scans over known tokens, generation scans over nothing (`xs=None`, `length=n_steps`) and feeds each output back through the carry. A real transformer would prefill all prompt positions in parallel instead of recurrently; the structure of the generation loop is the same.

## Complexity
$O(n \cdot B (D^2 + DV))$ for $n$ = prompt length + `n_steps`. The loop compiles once regardless of `n_steps`.

## Common bugs
- Counting the length after updating `done`, which drops the EOS token from the count.
- Using a Python `while` loop: it cannot run under `jit` with a traced condition, and `lax.while_loop` cannot return per-step outputs (it has no stacked outputs and no reverse-mode gradient).
- Forgetting the transpose: scanning over `prompt` directly iterates over batch rows.

## Interview follow-ups
- How would you stop early when **every** row is done? (`lax.while_loop` with a preallocated output buffer written by `dynamic_update_slice`.)
- How does this loop change for a transformer with a KV cache, and which arrays move into the carry?
"""
