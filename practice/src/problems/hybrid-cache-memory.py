META = dict(title='Cache memory for a stack of mixed attention layers', tags=['Inference', 'Attention'], diff='Easy', runtime='python',
            related=['kv-cache-memory-for-mha-mqa-gqa-and-mla', 'sliding-window-causal-mask'])

STATEMENT = r"""
Hybrid models mix layer types. Write `hybrid_cache_bytes(pattern, seq, batch, kv_heads, head_dim, window, state_heads, d_k, d_v, bytes_per)`.

`pattern` is a string with one character per layer:

- `"F"` full attention: keys and values for every position, $2 \cdot b \cdot S \cdot h_{kv} \cdot d_h$ elements.
- `"S"` sliding-window attention: the same with $\min(S, W)$ positions.
- `"L"` linear attention or a delta-rule layer: a fixed state of $b \cdot h_s \cdot d_k \cdot d_v$ elements.

Multiply element counts by `bytes_per`. Return a dict with `F`, `S`, `L` (total bytes for each type, 0 if absent), `total`, `gib` (total / 2**30) and `full_equiv` (the bytes if every layer were full attention).
"""

STARTER = '''def hybrid_cache_bytes(pattern, seq, batch, kv_heads, head_dim, window, state_heads, d_k, d_v, bytes_per):
    """Cache bytes per layer type for a hybrid stack."""
    per = 2 * batch * seq * kv_heads * head_dim * bytes_per
    return {"total": per * len(pattern)}
'''

SOLUTION = '''def hybrid_cache_bytes(pattern, seq, batch, kv_heads, head_dim, window, state_heads, d_k, d_v, bytes_per):
    full = 2 * batch * seq * kv_heads * head_dim * bytes_per
    swa = 2 * batch * min(seq, window) * kv_heads * head_dim * bytes_per
    lin = batch * state_heads * d_k * d_v * bytes_per
    out = {"F": pattern.count("F") * full, "S": pattern.count("S") * swa, "L": pattern.count("L") * lin}
    out["total"] = out["F"] + out["S"] + out["L"]
    out["gib"] = out["total"] / 2 ** 30
    out["full_equiv"] = len(pattern) * full
    return out
'''

HINTS = [
    'Compute the cost of one layer of each type, then multiply by how many times it appears.',
    'Only full attention grows with the sequence length; the other two are bounded.',
]

TESTS = [
    ("F" * 32, 32768, 1, 8, 128, 4096, 16, 128, 128, 2),
    ("SSSF" * 8, 32768, 1, 8, 128, 4096, 16, 128, 128, 2),
    ("LLLF" * 12, 131072, 4, 2, 256, 4096, 32, 128, 128, 2),
    ("LLLF" * 12, 1024, 4, 2, 256, 4096, 32, 128, 128, 2),
    ("SL", 100, 2, 4, 64, 4096, 8, 64, 64, 4),
]

EXPLANATION = r"""
## Why hybrids
Full attention gives the best recall over long contexts, but its KV cache grows linearly with length and dominates serving memory. Sliding-window layers bound the cache; linear-attention and delta-rule layers replace it with a fixed-size state. Mixing a few full layers with many cheap ones keeps most of the quality at a fraction of the memory.

## Reading the numbers
At 128K tokens a 3:1 linear-to-full stack needs roughly a quarter of the cache of an all-full model, plus a small constant for the states. At 1K tokens the states can cost more than the caches they replace, so hybrids pay off only at long context.

## Interview follow-ups
- How does the cache saving translate into higher batch size and throughput on a fixed GPU?
"""
