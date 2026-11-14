META = dict(title='Activation memory of a transformer under recomputation and parallelism', tags=['Training', 'Distributed Computing', 'Transformers'], diff='Medium', runtime='python')

STATEMENT = r"""
Korthikanti et al. (2022) count the activation memory of one transformer layer in 16-bit training with sequence length $s$, micro-batch $b$, hidden size $h$, $a$ heads and tensor-parallel size $t$. Write `activation_bytes(s, b, h, a, L, t, sp, recompute)` returning bytes for `L` layers:

| recompute | no sequence parallelism | sequence parallelism (`sp`) |
|---|---|---|
| `"none"` | $sbh\,(10 + 24/t + 5as/(ht))$ | $sbh\,(34/t + 5as/(ht))$ |
| `"selective"` | $sbh\,(10 + 24/t)$ | $34\,sbh/t$ |
| `"full"` | $2sbh$ | $2sbh/t$ |

`"selective"` recomputes only the attention scores (the $5as/h$ term); `"full"` keeps just each layer's input.

Return a dict with `per_layer` and `total` (floats, bytes) and `gib` (total / 2**30).
"""

STARTER = '''def activation_bytes(s, b, h, a, L, t, sp, recompute):
    """Activation memory in bytes for L transformer layers."""
    per = s * b * h * 34
    return {"per_layer": per, "total": per * L, "gib": per * L / 2 ** 30}
'''

SOLUTION = '''def activation_bytes(s, b, h, a, L, t, sp, recompute):
    sbh = s * b * h
    attn = 5 * a * s / (h * t)
    if recompute == "full":
        per = 2 * sbh / (t if sp else 1)
    elif recompute == "selective":
        per = sbh * (34 / t if sp else 10 + 24 / t)
    else:
        per = sbh * ((34 / t if sp else 10 + 24 / t) + attn)
    per = float(per)
    return {"per_layer": per, "total": per * L, "gib": per * L / 2 ** 30}
'''

HINTS = [
    "The 34 sbh part is the layer's linear-size activations; 5as^2 b is the attention score and dropout masks, quadratic in s.",
    'Without sequence parallelism, 10 sbh (LayerNorm inputs, dropout masks) is replicated on every tensor-parallel rank.',
    'Full recomputation stores only the 2-byte input of each layer and recomputes everything else in the backward pass.',
]

TESTS = [
    (2048, 1, 12288, 96, 96, 8, False, "none"),
    (2048, 1, 12288, 96, 96, 8, True, "none"),
    (2048, 1, 12288, 96, 96, 8, True, "selective"),
    (2048, 1, 12288, 96, 96, 8, False, "full"),
    (8192, 2, 4096, 32, 32, 1, False, "none"),
    (8192, 2, 4096, 32, 32, 1, False, "selective"),
]

EXPLANATION = r"""
## Where activation memory comes from
The backward pass needs the inputs of every matmul, every LayerNorm and every nonlinearity, plus dropout masks. For one layer that is about $34\,sbh$ bytes of linear-size tensors and $5\,a s^2 b$ bytes for the attention scores, softmax output and its dropout mask. At long sequence lengths the quadratic term dominates.

## Three levers
- Tensor parallelism splits most activations across $t$ ranks, but not the LayerNorm and dropout regions; sequence parallelism splits those along the sequence.
- Selective recomputation drops only the attention-score tensors, which are large but cheap to recompute, removing the quadratic term for a few percent of extra compute.
- Full recomputation keeps only layer inputs and repeats the forward pass during backward: about a third more compute for the smallest memory.

FlashAttention never materialises the score matrix at all, which removes the $5as/h$ term in practice.

## Interview follow-ups
- Why is selective recomputation almost free in FLOPs but large in memory saved?
"""
