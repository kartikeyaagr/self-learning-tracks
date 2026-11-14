import numpy as np

META = dict(title='GPT-2 initialisation plan with scaled residual projections', tags=['Language Models', 'Training', 'Neural Networks'], diff='Easy', runtime='python')

STATEMENT = r"""
GPT-2 initialises weights from $\mathcal N(0, 0.02^2)$, except the projections that write into the residual stream, which are scaled by $1/\sqrt{2L}$ for $L$ layers.

Write `init_plan(names, n_layer, base=0.02)` that maps parameter names to an initialisation, checking rules in this order:

1. A name whose last two dot-separated parts are a LayerNorm (`ln_1`, `ln_2`, `ln_f`) followed by `weight` → `["const", 1.0]`; followed by `bias` → `["const", 0.0]`.
2. Any other name ending in `.bias` → `["const", 0.0]`.
3. A name ending in `attn.c_proj.weight` or `mlp.c_proj.weight` → `["normal", base / sqrt(2 * n_layer)]`.
4. Any other name ending in `weight` → `["normal", base]`.
5. Anything else raises `ValueError`.

Return a dict from name to that two-element list.
"""

STARTER = '''def init_plan(names, n_layer, base=0.02):
    """Map parameter names to ["normal", std] or ["const", value]."""
    return {n: ["normal", base] for n in names}
'''

SOLUTION = '''import math

def init_plan(names, n_layer, base=0.02):
    plan = {}
    for n in names:
        parts = n.split(".")
        if len(parts) >= 2 and parts[-2] in ("ln_1", "ln_2", "ln_f") and parts[-1] in ("weight", "bias"):
            plan[n] = ["const", 1.0 if parts[-1] == "weight" else 0.0]
        elif n.endswith(".bias"):
            plan[n] = ["const", 0.0]
        elif n.endswith("attn.c_proj.weight") or n.endswith("mlp.c_proj.weight"):
            plan[n] = ["normal", base / math.sqrt(2 * n_layer)]
        elif n.endswith("weight"):
            plan[n] = ["normal", base]
        else:
            raise ValueError(f"no rule for {n}")
    return plan
'''

HINTS = [
    'Check LayerNorm names before the generic bias and weight rules, because ln_1.weight also ends in "weight".',
    'Each block has two residual writes (attention and MLP), which is where the 2 in sqrt(2L) comes from.',
]

def _names(L):
    out = ["wte.weight", "wpe.weight"]
    for i in range(L):
        for p in ["ln_1", "attn.c_attn", "attn.c_proj", "ln_2", "mlp.c_fc", "mlp.c_proj"]:
            out += [f"h.{i}.{p}.weight", f"h.{i}.{p}.bias"]
    return out + ["ln_f.weight", "ln_f.bias"]

TESTS = [
    (_names(1), 1),
    (_names(12), 12),
    (_names(2), 2, 0.01),
    (["wte.weight", "h.0.attn.c_proj.weight"], 48),
    (["h.3.mlp.c_proj.bias", "ln_f.weight"], 4),
]

EXPLANATION = r"""
## Why scale the residual projections
Each block adds its output to the residual stream. With $2L$ independent additions of variance $\sigma^2$, the stream's variance grows like $2L\sigma^2$. Scaling those projections by $1/\sqrt{2L}$ keeps the total roughly constant at initialisation, so deep models start with activations and gradients of the same size as shallow ones.

## What is left at the defaults
Embeddings and the input-side projections use 0.02. LayerNorm starts as the identity (scale 1, shift 0) and biases at zero, so the network's first forward pass is driven only by the random projections.

## Interview follow-ups
- How does pre-LayerNorm interact with residual growth compared with post-LayerNorm?
- What do later schemes such as scaled embedding multipliers or μP change about this?
"""
