META = dict(title='Count the parameters of a GPT', tags=['Language Models', 'Transformers'], diff='Easy', runtime='python')

STATEMENT = r"""
Write `gpt_params(vocab, n_ctx, d, n_layer, d_ff, tied, bias)` for a GPT-2 style decoder (learned positions, pre-LayerNorm, GELU MLP).

Count, with biases only where `bias` is true (LayerNorms always have a scale and a shift):

- `tok_emb` = vocab·d, `pos_emb` = n_ctx·d
- per layer: `ln` = 2·(2d) for the two LayerNorms; `attn` = 3d·d + d·d weights, plus 3d + d biases; `mlp` = d·d_ff + d_ff·d weights, plus d_ff + d biases
- `ln_f` = 2d
- `lm_head` = 0 if `tied`, else vocab·d (no bias)

Return a dict with keys `tok_emb`, `pos_emb`, `per_layer`, `ln_f`, `lm_head`, `total`, `non_embedding` (total minus both embedding tables and the lm_head) and `approx` = 12·n_layer·d² (with integer arithmetic).
"""

STARTER = '''def gpt_params(vocab, n_ctx, d, n_layer, d_ff, tied, bias):
    """Parameter counts for a GPT-2 style model."""
    per_layer = 12 * d * d
    return {"total": vocab * d + n_layer * per_layer}
'''

SOLUTION = '''def gpt_params(vocab, n_ctx, d, n_layer, d_ff, tied, bias):
    b = 1 if bias else 0
    ln = 4 * d
    attn = 4 * d * d + b * 4 * d
    mlp = 2 * d * d_ff + b * (d_ff + d)
    per_layer = ln + attn + mlp
    tok, pos, ln_f = vocab * d, n_ctx * d, 2 * d
    head = 0 if tied else vocab * d
    total = tok + pos + n_layer * per_layer + ln_f + head
    return {"tok_emb": tok, "pos_emb": pos, "per_layer": per_layer, "ln_f": ln_f, "lm_head": head,
            "total": total, "non_embedding": total - tok - pos - head, "approx": 12 * n_layer * d * d}
'''

HINTS = [
    'Attention has four d x d matrices (query, key, value, output); the MLP has two d x d_ff matrices.',
    'With d_ff = 4d the weight matrices give 4d^2 + 8d^2 = 12d^2 per layer, which is where the approximation comes from.',
    'Tied embeddings reuse the token table as the output projection, so the head adds nothing.',
]

TESTS = [
    (50257, 1024, 768, 12, 3072, True, True),
    (50257, 1024, 1600, 48, 6400, True, True),
    (50257, 1024, 768, 12, 3072, False, True),
    (65, 256, 64, 4, 256, True, False),
    (32000, 4096, 4096, 32, 11008, False, False),
]

EXPLANATION = r"""
## Where the parameters are
Almost all of a large transformer is in its layer matrices: $12 L d^2$ when $d_{ff} = 4d$. For GPT-2 small (124M total) the token embeddings are still 39M, a large share, but by GPT-2 XL (1.5B) the layers dominate.

## Why "non-embedding" parameters
Scaling-law work counts non-embedding parameters because embedding tables add memory without adding much compute per token, and the fits are cleaner without them. When comparing model sizes, check which count is being quoted.

## Useful checks
- GPT-2 small: 124,439,808 parameters with tied embeddings and biases.
- Training FLOPs are roughly $6 \times$ parameters $\times$ tokens, so getting this count right is the first step of any compute budget.

## Interview follow-ups
- How does the count change for a SwiGLU MLP with three matrices and $d_{ff} \approx \frac{8}{3}d$?
"""
