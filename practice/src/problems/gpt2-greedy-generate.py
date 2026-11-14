import numpy as np

META = dict(title='GPT-2: forward pass and greedy generation', tags=['Language Models', 'Transformers'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:06.03', 'aieng:05.01'],
            related=['scaled-dot-product-attention-2', 'layer-norm', 'text-generation-sampling', 'apply-ranked-bpe-merges'])

STATEMENT = r"""
GPT-2 is a stack of pre-LN decoder blocks over learned token and position embeddings, with a final LayerNorm and an output layer tied to the token embeddings. Implement the forward pass and use it to generate text greedily.

Write `gpt2_generate(prompt, p, n_heads, n_new)` returning a tuple `(tokens, last_logits)`:
- `tokens`: int array with the prompt followed by `n_new` generated ids;
- `last_logits`: the `(V,)` logits used to choose the final generated token.

## Parameters (names follow the released checkpoint)
`wte` `(V, D)`, `wpe` `(n_ctx, D)`, `ln_f` (gamma, beta), and `blocks`, each with `ln_1`, `ln_2` (gamma, beta), `c_attn_w` `(D, 3D)`, `c_attn_b` `(3D,)`, `c_proj_w` `(D, D)`, `c_proj_b`, `c_fc_w` `(D, 4D)`, `c_fc_b`, `mlp_proj_w` `(4D, D)`, `mlp_proj_b`.

## Model
- $x = \text{wte}[ids] + \text{wpe}[0..T)$.
- Block: $x = x + \mathrm{Attn}(\mathrm{LN}_1(x))$, $x = x + \mathrm{MLP}(\mathrm{LN}_2(x))$.
- Attn: `qkv = z @ c_attn_w + c_attn_b`, split into $Q, K, V$ (in that order, each $D$ wide), then into heads (contiguous blocks of $D/H$); causal mask; $1/\sqrt{d_h}$; concatenate; `@ c_proj_w + c_proj_b`.
- MLP: $\mathrm{GELU}(z\,W_{fc} + b_{fc})\,W_{proj} + b_{proj}$, tanh-approximate GELU.
- Logits $= \mathrm{LN}_f(x)\,\text{wte}^\top$. LayerNorm: population variance, $\epsilon = 10^{-5}$.

## Generation
Repeat `n_new` times: run the model on the current tokens, but if there are more than `n_ctx` of them, only on the **last `n_ctx`** (positions restart at 0 for that window); append the argmax of the last row of logits (lowest id on an exact tie).
"""

STARTER = '''import numpy as np

def gpt2_generate(prompt, p, n_heads, n_new):
    """Return (tokens, last_logits)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(z, gb, eps=1e-5):
    g, b = gb
    mu = z.mean(-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(-1, keepdims=True) + eps) + b

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _attn(z, B, H):
    T, D = z.shape
    dh = D // H
    q, k, v = np.split(z @ B['c_attn_w'] + B['c_attn_b'], 3, axis=-1)
    heads = lambda a: a.reshape(T, H, dh).transpose(1, 0, 2)
    q, k, v = heads(q), heads(k), heads(v)
    S = np.where(np.tril(np.ones((T, T), dtype=bool)), q @ k.transpose(0, 2, 1) / np.sqrt(dh), -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ v).transpose(1, 0, 2).reshape(T, D) @ B['c_proj_w'] + B['c_proj_b']

def _forward(ids, p, H):
    x = p['wte'][ids] + p['wpe'][:len(ids)]
    for B in p['blocks']:
        x = x + _attn(_ln(x, B['ln_1']), B, H)
        x = x + _gelu(_ln(x, B['ln_2']) @ B['c_fc_w'] + B['c_fc_b']) @ B['mlp_proj_w'] + B['mlp_proj_b']
    return _ln(x, p['ln_f']) @ p['wte'].T

def gpt2_generate(prompt, p, n_heads, n_new):
    toks = [int(t) for t in prompt]
    n_ctx = p['wpe'].shape[0]
    last = None
    for _ in range(n_new):
        last = _forward(np.array(toks[-n_ctx:]), p, n_heads)[-1]
        toks.append(int(np.argmax(last)))
    return np.array(toks), last
'''

HINTS = [
    'Write `forward(ids)` returning `(T, V)` logits first and check it on its own; generation is a short loop around it.',
    'The fused `c_attn` projection gives `(T, 3D)`: `np.split(..., 3, axis=-1)` returns $Q$, $K$, $V$ in that order.',
    'When the sequence outgrows the position table, crop to the last `n_ctx` tokens before every forward pass. `np.argmax` already returns the lowest index on a tie.',
]

_r = np.random.default_rng(2019)
def _lnp(D):
    return (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))
def _model(V, D, n_ctx, nb):
    s = 1 / np.sqrt(D)
    blocks = [dict(ln_1=_lnp(D), ln_2=_lnp(D), c_attn_w=_r.normal(size=(D, 3 * D)) * s, c_attn_b=0.1 * _r.normal(size=3 * D),
                   c_proj_w=_r.normal(size=(D, D)) * s / np.sqrt(2 * nb), c_proj_b=0.1 * _r.normal(size=D),
                   c_fc_w=_r.normal(size=(D, 4 * D)) * s, c_fc_b=0.1 * _r.normal(size=4 * D),
                   mlp_proj_w=_r.normal(size=(4 * D, D)) / np.sqrt(4 * D) / np.sqrt(2 * nb), mlp_proj_b=0.1 * _r.normal(size=D))
              for _ in range(nb)]
    return {'wte': _r.normal(size=(V, D)), 'wpe': _r.normal(size=(n_ctx, D)) * 0.3, 'ln_f': _lnp(D), 'blocks': blocks}

_m1 = _model(9, 4, 8, 1)
_m2 = _model(16, 8, 6, 2)
TESTS = [
    (np.array([1, 2]), _m1, 1, 1),
    (np.array([1, 2]), _m1, 2, 4),
    (np.array([3, 0, 7]), _m2, 2, 2),
    (np.array([3, 0, 7]), _m2, 4, 6),
    (np.array([5, 9, 12, 1, 4]), _m2, 2, 5),
    (np.array([8]), _m1, 2, 9),
]

EXPLANATION = r"""
## What GPT-2 contributed
Architecturally GPT-2 is GPT-1 with the LayerNorm moved to the input of each sublayer (pre-LN), an extra LayerNorm after the last block, a 1024-token context, and residual projections initialised with an extra $1/\sqrt{2N}$ factor for $N$ blocks (the test weights mimic that). The paper's real point was scale: a 1.5B model trained on a large web corpus does many tasks zero-shot when prompted.

## Generation details
- **Cropping**: positions are a learned table of size `n_ctx`, so a longer sequence has no embedding for its later positions. Sliding the window keeps generation going; the model simply forgets the earliest tokens.
- **Recomputation**: this loop reruns the whole prefix each step, $O(T^2)$ forward passes' worth of attention. A KV cache stores each layer's keys and values so each step only processes one new token.
- **Greedy decoding** is deterministic but tends to loop; GPT-2's samples in the paper used top-$k$ sampling with $k = 40$.

## Common bugs
- Using `wte` without a transpose for the output layer, or adding a separate output matrix.
- Splitting `c_attn` into heads before splitting into $Q, K, V$ (the layout is $[Q | K | V]$, each with heads inside).
- Keeping absolute positions after cropping (position $T$ doesn't exist in `wpe`).

## Interview follow-ups
- Add a KV cache to this loop. Which arrays grow, and what changes when the window starts sliding?
- Why does the tied output layer make sense, and when might you untie it?
"""
