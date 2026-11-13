import numpy as np

META = dict(title='Greedy generation with a KV cache: prefill once, then decode', tags=['Inference', 'Language Models', 'Attention'],
            diff='Medium', runtime='torch', lessons=['ieng:03.04', 'ieng:02.05'],
            related=['scaled-dot-product-attention-3', 'multi-query-attention-mqa', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Without a cache, generating token $t$ recomputes keys and values for all $t - 1$ earlier tokens. With one, the prompt is processed once (**prefill**), and every later step pushes only the newest token through the model (**decode**), appending its key and value to the cache.

Write `generate(prompt, n_new, E, Pos, Wq, Wk, Wv, Wo, W1, W2)` in PyTorch for this tiny one-layer, one-head model.

## Model (float64; projections are $hW$, no biases)
For a token $t$ at position $i$:
1. $x = E[t] + \text{Pos}[i]$, with `E`: `(V, D)` token embeddings and `Pos`: `(max_len, D)` position embeddings.
2. $q = xW_q$, $k = xW_k$, $v = xW_v$ (all `(D, D)`). Causal attention over positions $\le i$: $a = \mathrm{softmax}(q K^\top / \sqrt{D})\,V$, then $h = x + a W_o$.
3. $h \leftarrow h + \mathrm{relu}(h W_1) W_2$, with `W1`: `(D, F)`, `W2`: `(F, D)`.
4. $\text{logits} = h E^\top$ (tied output embedding), shape `(V,)`.

## Generation
- `prompt`: int64 `(T_0,)` with $T_0 \ge 1$. `n_new` $\ge 1$. $T_0 + n_{\text{new}} \le$ `max_len`.
- **Prefill:** run the whole prompt in one batched pass (causal mask), store its keys and values, and take the logits at the last prompt position.
- **Decode:** the next token is the argmax of those logits (smallest index on ties). Feed **only** that token through steps 1-4, using and extending the cache, to get the next logits. Repeat until `n_new` tokens have been produced. The last produced token is never fed back.

## Output
A tuple `(tokens, logits, K)`:
- `tokens`: int64 `(n_new,)`, the generated tokens.
- `logits`: `(n_new, V)`, row $s$ holds the logits that produced `tokens[s]`.
- `K`: `(T_0 + n_{\text{new}} - 1, D)`, the final key cache.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def generate(prompt, n_new, E, Pos, Wq, Wk, Wv, Wo, W1, W2):
    """Return (tokens (n_new,) int64, logits (n_new, V), key cache (T0 + n_new - 1, D))."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def generate(prompt, n_new, E, Pos, Wq, Wk, Wv, Wo, W1, W2):
    D = E.shape[1]

    def forward(tokens, start, K, V):
        """Run tokens at positions start.. through the layer; return last logits and the extended cache."""
        T = len(tokens)
        x = E[tokens] + Pos[start:start + T]                      # (T, D)
        K = torch.cat([K, x @ Wk]); V = torch.cat([V, x @ Wv])    # append before attending (a token sees itself)
        s = (x @ Wq) @ K.T / math.sqrt(D)                         # (T, start + T)
        future = torch.arange(K.shape[0])[None, :] > torch.arange(start, start + T)[:, None]
        h = x + torch.softmax(s.masked_fill(future, float('-inf')), dim=-1) @ V @ Wo
        h = h + torch.relu(h @ W1) @ W2
        return h[-1] @ E.T, K, V

    empty = torch.zeros(0, D, dtype=E.dtype)
    logits, K, V = forward(prompt, 0, empty, empty)              # prefill: the whole prompt at once
    toks, rows = [], []
    for step in range(n_new):
        tok = int(logits.argmax())                               # first max wins ties
        toks.append(tok); rows.append(logits)
        if step + 1 < n_new:                                     # decode: one token in, one row of K/V out
            logits, K, V = forward(torch.tensor([tok]), len(prompt) + step, K, V)
    return torch.tensor(toks, dtype=torch.int64), torch.stack(rows), K
'''

HINTS = [
    'Write one `forward(tokens, start, K, V)` that handles any number of new tokens at positions `start, start+1, ...`: prefill calls it with the prompt and `start=0`, decode with one token.',
    'Append the new keys and values to the cache **before** attending, so each token can see itself. The causal mask compares absolute key positions `arange(len(K))` with query positions `arange(start, start+T)`.',
    'Only the logits of the last row matter for generation. Do not feed the final generated token back: the cache ends with `T0 + n_new - 1` rows.',
]

_r = np.random.default_rng(23)


def _model(V, D, F, L, s=0.5):
    return (_r.normal(size=(V, D)), _r.normal(size=(L, D)) * 0.3, _r.normal(size=(D, D)) * s, _r.normal(size=(D, D)) * s,
            _r.normal(size=(D, D)) * s, _r.normal(size=(D, D)) * s, _r.normal(size=(D, F)) * s, _r.normal(size=(F, D)) * s)


_tiny = (np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), np.zeros((4, 2)), np.eye(2), np.eye(2), np.eye(2),
         np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((1, 2)))
TESTS = [
    (np.array([0]), 2, *_tiny),
    (np.array([1, 2]), 2, *_tiny),
    (np.array([3, 1, 4]), 5, *_model(8, 6, 12, 10)),
    (np.array([0]), 1, *_model(5, 4, 8, 4)),
    (np.array([2, 7, 1, 8, 2, 8]), 4, *_model(10, 8, 16, 12)),
    (np.array([5]), 8, *_model(6, 4, 4, 9, 1.0)),
    (np.array([9, 9, 9]), 3, *_model(12, 6, 6, 8)),
]

EXPLANATION = r"""
## Why it works
A token's key and value depend only on that token and its position, never on later tokens, so they can be computed once and reused. A decode step therefore needs only the newest token's query against the cached keys. The result is identical to rerunning the full prompt-plus-generated sequence every step; you can check that by writing the uncached version and comparing logits.

## Cost
Without a cache, step $t$ costs $O(t D^2 + t^2 D)$ and generating $n$ tokens costs $O(n^2 D^2 + n^3 D)$. With it, a decode step costs $O(D^2 + tD)$: one token through the projections and MLP, plus one row of attention. The price is memory: $2 D$ numbers per token per layer, which at real model sizes is what limits batch size and context length.

## Prefill vs decode
Prefill processes $T_0$ tokens in one matrix-matrix pass and is compute-bound. Decode processes one token per sequence, a matrix-vector pass that must read every weight for one row of output, so it is memory-bandwidth-bound. Batching many sequences' decode steps together is how servers recover efficiency.

## Common bugs
- Attending before appending, so the new token cannot see itself.
- Restarting positions at 0 for each decode step instead of continuing at $T_0 + s$.
- Feeding the last generated token back (one extra forward pass, one extra cache row).

## Interview follow-ups
- How would you batch decode steps for several sequences with different lengths in the same forward pass?
- Where does the cache live in memory, and why does a contiguous `torch.cat` per step become a problem at scale? (Reallocation and fragmentation, which paged KV caches solve.)
"""
