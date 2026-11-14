import math

META = dict(title='Size a quantised KV cache and the batch it allows', tags=['Inference', 'Quantization'], diff='Easy', runtime='python',
            related=['kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Write `kv_cache_quant(layers, kv_heads, head_dim, seq, bits, group, asym, free_bytes)`.

- Elements per sequence: $2 \cdot$ layers $\cdot$ kv_heads $\cdot$ head_dim $\cdot$ seq (keys and values).
- 16-bit baseline: 2 bytes per element.
- Quantised: `bits / 8` bytes per element, plus per `group` consecutive elements one fp16 scale (2 bytes) and, if `asym`, one fp16 zero point (2 bytes). Use $\lceil \text{elements} / \text{group} \rceil$ groups.

Return a dict with `fp16_per_seq`, `quant_per_seq` (bytes, floats), `ratio` (fp16 / quant), `bits_per_elem` (effective bits including scales), and `max_batch_fp16`, `max_batch_quant`: how many full-length sequences fit in `free_bytes` (floor).
"""

STARTER = '''def kv_cache_quant(layers, kv_heads, head_dim, seq, bits, group, asym, free_bytes):
    """KV cache bytes per sequence, 16-bit versus quantised."""
    n = 2 * layers * kv_heads * head_dim * seq
    return {"fp16_per_seq": 2.0 * n, "quant_per_seq": n * bits / 8}
'''

SOLUTION = '''import math

def kv_cache_quant(layers, kv_heads, head_dim, seq, bits, group, asym, free_bytes):
    n = 2 * layers * kv_heads * head_dim * seq
    fp16 = 2.0 * n
    groups = math.ceil(n / group)
    q = n * bits / 8 + groups * (4 if asym else 2)
    return {"fp16_per_seq": fp16, "quant_per_seq": float(q), "ratio": fp16 / q, "bits_per_elem": 8 * q / n,
            "max_batch_fp16": int(free_bytes // fp16), "max_batch_quant": int(free_bytes // q)}
'''

HINTS = [
    'Count elements first; bytes follow from bits per element plus the per-group metadata.',
    'With a group of 32 and fp16 scale plus zero point, metadata adds 1 bit per element.',
]

TESTS = [
    (32, 8, 128, 8192, 8, 64, False, 40 * 2 ** 30),
    (32, 8, 128, 8192, 4, 32, True, 40 * 2 ** 30),
    (80, 8, 128, 32768, 4, 128, True, 60 * 2 ** 30),
    (80, 8, 128, 32768, 2, 32, True, 60 * 2 ** 30),
    (2, 1, 4, 10, 8, 3, False, 1000),
]

EXPLANATION = r"""
## Why quantise the cache
At long context and large batch, the KV cache is larger than the model. Halving or quartering it directly multiplies the batch a server can run, which is where most of the throughput comes from during decoding.

## The metadata cost
Small groups track the data better but carry more scales: at 2-bit with groups of 32 and a zero point, metadata is a third of the storage. That is why aggressive cache quantisation schemes use larger groups or quantise keys per channel and values per token.

## Quality
Keys are more sensitive than values because errors in keys change attention weights for every later query. 8-bit caches are usually lossless in practice; 4-bit needs care; 2-bit needs special schemes.

## Interview follow-ups
- Why might you quantise keys per channel but values per token?
"""
