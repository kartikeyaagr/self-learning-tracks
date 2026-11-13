META = dict(title='Count transformer training FLOPs, matmul by matmul', tags=['Scaling Laws', 'Transformers'], diff='Easy',
            runtime='python', lessons=['ieng:12.07', 'ieng:00.01'],
            related=['training-compute-budget', 'mixed-precision-training-memory-accountant', 'isoflop-sweep-exponent'])

STATEMENT = r"""
The rule "training costs $6ND$ FLOPs" hides where the compute actually goes. Count it matmul by matmul for a decoder-only transformer and compare with the rule.

Write `training_flops(n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab, seq_len, n_tokens, gated=False, causal=False)`.

## Convention
A matmul of an $(m \times k)$ matrix by a $(k \times n)$ matrix costs $2mkn$ FLOPs (one multiply and one add per term). Count only matmuls: ignore norms, softmax, activations, biases and embedding lookups. The head width is $d_h = d_\text{model} / n_\text{heads}$, and the key/value width is $d_{kv} = n_\text{kv\_heads} \cdot d_h$ (grouped-query attention when $n_\text{kv\_heads} < n_\text{heads}$).

## Forward FLOPs per token, per layer
| key | matmul | FLOPs per token |
|---|---|---|
| `'qkv'` | project to $Q$, $K$, $V$ | $2\,d_\text{model}(d_\text{model} + 2 d_{kv})$ |
| `'scores'` | $QK^\top$ | $2\,c\,d_\text{model}$ |
| `'attn_v'` | probabilities times $V$ | $2\,c\,d_\text{model}$ |
| `'out_proj'` | attention output projection | $2\,d_\text{model}^2$ |
| `'mlp'` | feed-forward | $2 \cdot m \cdot d_\text{model}\, d_\text{ff}$, with $m = 3$ if `gated` (SwiGLU: gate, up, down) else $m = 2$ |

$c$ is the number of keys a query attends to, averaged over positions: $c = $ `seq_len` if not `causal`; if `causal`, $c = (\text{seq\_len} + 1)/2$ (only the computed lower triangle counts).

Plus, once per token (not per layer), `'logits'` $= 2\, d_\text{model}\, \text{vocab}$ for the output head.

## Output
A dict with:
- the six component keys above: per-token forward FLOPs **summed over all layers** (`'logits'` once);
- `'forward_per_token'`: their sum; `'train_per_token'`: three times that (backward costs about twice the forward);
- `'total'`: `train_per_token * n_tokens`;
- `'n_params'`: non-embedding parameters, $n_\text{layers}\,\bigl(d_\text{model}(d_\text{model} + 2 d_{kv}) + d_\text{model}^2 + m\, d_\text{model}\, d_\text{ff}\bigr)$;
- `'six_nd'`: `6 * n_params * n_tokens`, and `'ratio'`: `total / six_nd`.

Return every value as a float. All inputs are positive ints; `n_heads` divides `d_model` and `n_kv_heads` divides `n_heads`.
"""

STARTER = '''def training_flops(n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab, seq_len, n_tokens, gated=False, causal=False):
    """Return a dict of per-token forward FLOPs by component, plus totals and the 6ND comparison."""
    n_params = 12 * n_layers * d_model ** 2      # TODO: the textbook shortcut, wrong for GQA or SwiGLU
    total = 6.0 * n_params * n_tokens
    return {'total': total, 'n_params': float(n_params), 'six_nd': total, 'ratio': 1.0}
'''

SOLUTION = '''def training_flops(n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab, seq_len, n_tokens, gated=False, causal=False):
    d_kv = n_kv_heads * (d_model // n_heads)        # width of K (and of V)
    m = 3 if gated else 2                            # weight matrices in the MLP
    ctx = (seq_len + 1) / 2 if causal else seq_len   # keys per query, on average
    L = n_layers
    out = {
        'qkv': 2.0 * L * d_model * (d_model + 2 * d_kv),
        'scores': 2.0 * L * ctx * d_model,           # each head: 2 * ctx * d_h, times n_heads
        'attn_v': 2.0 * L * ctx * d_model,
        'out_proj': 2.0 * L * d_model ** 2,
        'mlp': 2.0 * L * m * d_model * d_ff,
        'logits': 2.0 * d_model * vocab,             # once, after the last layer
    }
    fwd = sum(out.values())
    n_params = float(L * (d_model * (d_model + 2 * d_kv) + d_model ** 2 + m * d_model * d_ff))
    out['forward_per_token'] = fwd
    out['train_per_token'] = 3.0 * fwd               # backward = grads w.r.t. inputs + grads w.r.t. weights
    out['total'] = out['train_per_token'] * n_tokens
    out['n_params'] = n_params
    out['six_nd'] = 6.0 * n_params * n_tokens
    out['ratio'] = out['total'] / out['six_nd']
    return out
'''

HINTS = [
    'Per token, a weight matrix of shape $(k, n)$ costs $2kn$ forward FLOPs: the token is a $1 \\times k$ row. Sum that over every weight matrix and you have $2N$ for the parameter matmuls.',
    'Attention scores have no weights: each query dots against $c$ keys of width $d_h$ in each of $n_\\text{heads}$ heads, so $2 c\\, d_h n_\\text{heads} = 2 c\\, d_\\text{model}$. The same again for multiplying by $V$.',
    'GQA shrinks only the K and V projections: $d_{kv} = n_\\text{kv\\_heads}\\, d_h$. The `ratio` is above 1 because 6ND leaves out the attention-score matmuls and the logits.',
]

TESTS = [
    (2, 64, 4, 4, 256, 1000, 128, 1_000_000),
    (2, 64, 4, 2, 172, 1000, 128, 1_000_000, True, True),
    (32, 4096, 32, 8, 14336, 128256, 8192, 15_000_000_000_000, True, True),     # Llama-3-8B-like
    (12, 768, 12, 12, 3072, 50257, 1024, 300_000_000_000),                      # GPT-2-small-like
    (12, 768, 12, 12, 3072, 50257, 1024, 300_000_000_000, False, True),
    (1, 8, 1, 1, 8, 2, 1, 1),
    (80, 8192, 64, 8, 28672, 128256, 131072, 1_000_000_000, True, True),        # long-context 70B-like
]

EXPLANATION = r"""
## Why $6N$
Per token, every weight matrix with $k \cdot n$ parameters costs $2kn$ FLOPs forward, so the parameter matmuls cost $2N$. The backward pass computes two matmuls per forward matmul (gradient with respect to the input and with respect to the weight), so it costs $4N$. Total: $6N$ per token.

## What $6N$ leaves out
- **Attention scores and the weighted sum of values** have no parameters. They cost $4\,c\,d_\text{model}$ per layer per token, which grows with context. In test 7 (128k context) this term dominates and the ratio is far above 1.
- **The output head** costs $2\,d_\text{model}\,\text{vocab}$ per token. With a 128k vocabulary on a small model it is a sizeable fraction.
- **Causal masking** halves the useful score FLOPs, but only if the kernel actually skips the masked half (FlashAttention does, a dense `QK^T` does not).

## Complexity
$O(1)$ arithmetic. The point is the bookkeeping, and having a number to plug into MFU: $\text{MFU} = \text{achieved FLOP/s} / \text{peak}$, where achieved FLOP/s uses exactly this count divided by the step time.

## Common bugs
- Counting $d_\text{model}^2$ for the K and V projections under GQA.
- Using 2 matrices for a SwiGLU MLP (it has 3) and therefore the wrong `n_params`.
- Multiplying the logits by `n_layers`.

## Interview follow-ups
- With activation checkpointing, the forward is recomputed during backward. What becomes the multiplier (about $8N$ instead of $6N$), and does MFU count the recompute?
- At what context length do the attention-score FLOPs equal the parameter FLOPs for a 4096-wide, 32-layer model? ($4c\,d = 2 \cdot 12 d^2$ roughly, so $c \approx 6d \approx 25\text{k}$.)
"""
