META = dict(title='Total versus active parameters of a mixture-of-experts model', tags=['Language Models', 'Transformers'], diff='Easy', runtime='python',
            related=['gpt-parameter-count'])

STATEMENT = r"""
Write `moe_params(vocab, d, n_layer, n_dense, d_ff, d_expert, n_experts, n_shared, top_k)` for a decoder whose first `n_dense` layers use a dense FFN and the rest use mixture-of-experts FFNs. Ignore norms and biases.

- Attention per layer: $4d^2$.
- A SwiGLU FFN with hidden size $h$ has $3dh$ parameters. Dense layers use $h$ = `d_ff`; each expert (routed or shared) uses $h$ = `d_expert`.
- Each MoE layer has a router of $d \cdot$ `n_experts` parameters.
- Embeddings: an input table and an untied output head, $2 \cdot$ vocab $\cdot d$.

**Total** counts every expert. **Active** (per token) counts attention, routers, dense FFNs, the shared experts and only `top_k` routed experts per MoE layer.

Return a dict with `total`, `active`, `total_non_emb`, `active_non_emb` and `active_ratio` = active_non_emb / total_non_emb.
"""

STARTER = '''def moe_params(vocab, d, n_layer, n_dense, d_ff, d_expert, n_experts, n_shared, top_k):
    """Return total and active parameter counts."""
    total = n_layer * (4 * d * d + n_experts * 3 * d * d_expert)
    return {"total": total, "active": total}
'''

SOLUTION = '''def moe_params(vocab, d, n_layer, n_dense, d_ff, d_expert, n_experts, n_shared, top_k):
    n_moe = n_layer - n_dense
    attn = n_layer * 4 * d * d
    dense = n_dense * 3 * d * d_ff
    expert = 3 * d * d_expert
    router = n_moe * d * n_experts
    total_ne = attn + dense + router + n_moe * (n_experts + n_shared) * expert
    active_ne = attn + dense + router + n_moe * (top_k + n_shared) * expert
    emb = 2 * vocab * d
    return {"total": total_ne + emb, "active": active_ne + emb, "total_non_emb": total_ne,
            "active_non_emb": active_ne, "active_ratio": active_ne / total_ne}
'''

HINTS = [
    'Separate the layer count into dense and MoE layers first.',
    'The router runs for every token, so it is always active; only routed experts are sparse.',
    'Shared experts process every token, so they count in both totals.',
]

TESTS = [
    (32000, 4096, 32, 0, 14336, 14336, 8, 0, 2),
    (129280, 7168, 61, 3, 18432, 2048, 256, 1, 8),
    (50000, 1024, 12, 12, 4096, 1024, 8, 0, 2),
    (32000, 2048, 24, 1, 8192, 1408, 64, 2, 6),
    (1000, 64, 4, 2, 256, 64, 16, 0, 1),
]

EXPLANATION = r"""
## Two numbers, two costs
Total parameters set the memory needed to hold the model. Active parameters set the compute per token, and so the training FLOPs and the decode cost. A mixture of experts decouples them: a model with 8 experts and top-2 routing stores about four times the FFN weights it uses per token.

## Fine-grained and shared experts
Recent designs use many small experts with a larger top-k, plus one or more shared experts that every token uses. With the same active budget, more, smaller experts give the router more combinations to choose from, and shared experts capture common knowledge so routed experts can specialise.

## What the ratio hides
Serving an MoE still needs every expert in memory, and with small batches each expert sees few tokens, so the hardware efficiency of an MoE at decode time can be poor even though its FLOPs are low.

## Interview follow-ups
- Why do many MoE models keep the first few layers dense?
"""
