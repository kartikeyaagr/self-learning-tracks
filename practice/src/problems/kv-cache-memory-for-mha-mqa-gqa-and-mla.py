META = dict(title='KV cache size for MHA, MQA, GQA and MLA', tags=['Inference', 'Attention'], diff='Easy', runtime='python',
            lessons=['ieng:03.06', 'aieng:06.04'],
            related=['multi-query-attention-mqa', 'multi-head-latent-attention-mla', 'estimate-model-memory-and-minimum-gpu-count'])

STATEMENT = r"""
The attention variant decides how many numbers each token leaves behind in the KV cache, and that number decides how many requests fit on a GPU. Compare the four common designs for one model shape.

Write `kv_cache_bytes(n_layers, n_heads, head_dim, n_kv_heads, kv_rank, rope_dim, seq_len, batch, bytes_per_elem)`.

## Per token, per layer, the cache holds
| Variant | Numbers cached |
|---|---|
| `'MHA'` | keys and values for all heads: $2 \cdot n_{\text{heads}} \cdot d_{\text{head}}$ |
| `'MQA'` | one shared key head and one value head: $2 \cdot d_{\text{head}}$ |
| `'GQA'` | $n_{\text{kv}}$ key/value heads: $2 \cdot n_{\text{kv}} \cdot d_{\text{head}}$ |
| `'MLA'` | one latent vector plus one shared RoPE key: $r_{\text{kv}} + d_{\text{rope}}$ |

Each number takes `bytes_per_elem` bytes (2 for fp16/bf16, 1 for fp8).

## Output
A dict with keys `'MHA'`, `'MQA'`, `'GQA'`, `'MLA'`. Each value is a tuple of Python ints `(bytes_per_token, total_bytes)`:
- `bytes_per_token`: summed over all layers, for one token.
- `total_bytes`: for `batch` sequences of `seq_len` tokens each.

All inputs are non-negative ints.
"""

STARTER = '''def kv_cache_bytes(n_layers, n_heads, head_dim, n_kv_heads, kv_rank, rope_dim, seq_len, batch, bytes_per_elem):
    """Return {'MHA': (bytes_per_token, total), 'MQA': ..., 'GQA': ..., 'MLA': ...}."""
    per_layer = {'MHA': 2 * n_heads * head_dim, 'MQA': head_dim, 'GQA': n_kv_heads * head_dim, 'MLA': kv_rank}
    return {k: (n * n_layers, n * n_layers * seq_len) for k, n in per_layer.items()}
'''

SOLUTION = '''def kv_cache_bytes(n_layers, n_heads, head_dim, n_kv_heads, kv_rank, rope_dim, seq_len, batch, bytes_per_elem):
    per_layer = {                                   # numbers cached per token in one layer
        'MHA': 2 * n_heads * head_dim,              # K and V for every head
        'MQA': 2 * head_dim,                        # one K head and one V head shared by all queries
        'GQA': 2 * n_kv_heads * head_dim,           # one K/V head per group
        'MLA': kv_rank + rope_dim,                  # latent c plus the decoupled RoPE key, no factor 2
    }
    out = {}
    for name, n in per_layer.items():
        per_token = n * n_layers * bytes_per_elem
        out[name] = (per_token, per_token * seq_len * batch)
    return out
'''

HINTS = [
    'Count numbers per token per layer first (the table), then multiply by layers and bytes per number.',
    'MHA, MQA and GQA store keys **and** values, hence the factor 2. MLA stores one latent that both are rebuilt from, plus the small RoPE key, so there is no factor 2.',
    'The total is bytes per token times `seq_len` times `batch`. Keep everything as ints.',
]

TESTS = [
    (1, 4, 8, 2, 16, 4, 10, 1, 2),
    (2, 2, 4, 1, 4, 2, 3, 2, 1),
    (32, 32, 128, 8, 512, 64, 4096, 1, 2),
    (80, 64, 128, 8, 512, 64, 8192, 16, 2),
    (61, 128, 128, 8, 512, 64, 32768, 4, 1),
    (12, 12, 64, 12, 256, 0, 1024, 8, 4),
    (0, 8, 64, 2, 128, 32, 100, 3, 2),
]

EXPLANATION = r"""
## Why it works
Each formula is "numbers kept per token per layer" times layers times bytes. Keys and values both scale with the number of KV heads, so MQA and GQA shrink the cache by $n_{\text{heads}} / n_{\text{kv}}$. MLA instead caches a compressed latent of width $r_{\text{kv}}$ from which every head's keys and values are re-expanded, plus a $d_{\text{rope}}$-wide key that carries position.

## A worked example
A Llama-style 70B shape (test 4: 80 layers, 64 heads of 128, 8 KV heads) in bf16: MHA needs 2.6 MB per token, GQA 0.33 MB. At 8k context, one sequence costs 21 GB with MHA and 2.7 GB with GQA. With 80 GB GPUs that is the difference between a handful of requests and dozens.

## Concurrency
Free memory after weights divided by `bytes_per_token * seq_len` is the maximum number of full-length sequences, which is the hard ceiling on batch size and therefore on decode throughput. That is why every recent open model uses GQA or MLA.

## Common bugs
- Forgetting the factor 2 for K and V, or adding it to MLA.
- Mixing up `n_heads` and `n_kv_heads` for GQA.
- Converting with $10^9$ in one place and $2^{30}$ in another when reporting GB.

## Interview follow-ups
- With an fp8 KV cache, how many more concurrent 8k-token sequences fit on an 80 GB GPU serving a 70B model in fp8? What quality risks come with it?
- MQA's cache is smaller than MLA's in this formula when $2 d_{\text{head}} < r_{\text{kv}} + d_{\text{rope}}$. Why might MLA still be preferred?
"""
