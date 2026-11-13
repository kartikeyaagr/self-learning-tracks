META = dict(title='Minimum tensor-parallel GPU count for a model and its KV cache', tags=['Inference', 'Distributed Computing'],
            diff='Easy', runtime='python', lessons=['ieng:03.05', 'ieng:18.01'],
            related=['kv-cache-memory-for-mha-mqa-gqa-and-mla', 'tensor-parallel-communication-cost', 'analyze-prefill-and-decode-with-roofline-model'])

STATEMENT = r"""
Before renting GPUs, size the deployment: the weights plus the KV cache for the target concurrency must fit, split across a tensor-parallel group, with headroom for activations and the CUDA context.

Write `min_gpus(n_params, bytes_per_param, n_layers, n_kv_heads, head_dim, kv_bytes, context_len, concurrency, gpu_gb, mem_util=0.9, overhead_gb=2.0)`.

## Memory model (GB = $10^9$ bytes)
- **Weights:** `n_params * bytes_per_param` bytes, split evenly over $n$ GPUs.
- **KV cache:** $2 \cdot \text{n\_layers} \cdot \text{n\_kv\_heads} \cdot \text{head\_dim} \cdot \text{kv\_bytes}$ bytes per token, for `context_len * concurrency` tokens.
- Tensor parallelism splits the KV heads across GPUs, but a head can't be split: with $n > \text{n\_kv\_heads}$ each KV head is **replicated**. So each GPU holds $\text{KV} / \min(n, \text{n\_kv\_heads})$.
- Each GPU may use `mem_util * gpu_gb` GB, of which `overhead_gb` is reserved for activations and runtime state.

## Rule
A tensor-parallel size $n \in \{1, 2, 4, 8\}$ **fits** if
$$\frac{\text{weights}}{n} + \frac{\text{KV}}{\min(n, \text{n\_kv\_heads})} + \text{overhead} \le \text{mem\_util} \cdot \text{gpu\_gb}.$$
Pick the smallest $n$ that fits.

## Output
A dict:
- `'weights_gb'`, `'kv_gb'`: totals in GB (floats).
- `'tp'`: the chosen $n$, or `None` if even 8 GPUs are not enough.
- `'per_gpu_gb'`: the left-hand side of the rule (including overhead) for the chosen $n$, or `None`.
"""

STARTER = '''def min_gpus(n_params, bytes_per_param, n_layers, n_kv_heads, head_dim, kv_bytes, context_len, concurrency,
             gpu_gb, mem_util=0.9, overhead_gb=2.0):
    """Return {'weights_gb', 'kv_gb', 'tp', 'per_gpu_gb'}."""
    w = n_params * bytes_per_param / 1e9
    for n in (1, 2, 4, 8):                         # weights only: forgets the KV cache
        if w / n <= gpu_gb:
            return {'weights_gb': w, 'kv_gb': 0.0, 'tp': n, 'per_gpu_gb': w / n}
    return {'weights_gb': w, 'kv_gb': 0.0, 'tp': None, 'per_gpu_gb': None}
'''

SOLUTION = '''def min_gpus(n_params, bytes_per_param, n_layers, n_kv_heads, head_dim, kv_bytes, context_len, concurrency,
             gpu_gb, mem_util=0.9, overhead_gb=2.0):
    weights = n_params * bytes_per_param / 1e9
    per_token = 2 * n_layers * n_kv_heads * head_dim * kv_bytes          # K and V, every layer
    kv = per_token * context_len * concurrency / 1e9
    budget = mem_util * gpu_gb
    for n in (1, 2, 4, 8):
        # weights shard perfectly; KV heads shard only up to n_kv_heads ways, then replicate
        per_gpu = weights / n + kv / min(n, n_kv_heads) + overhead_gb
        if per_gpu <= budget:
            return {'weights_gb': weights, 'kv_gb': kv, 'tp': n, 'per_gpu_gb': per_gpu}
    return {'weights_gb': weights, 'kv_gb': kv, 'tp': None, 'per_gpu_gb': None}
'''

HINTS = [
    'KV bytes per token = 2 (K and V) × layers × KV heads × head dim × bytes per element. Multiply by context length and concurrency.',
    'Try $n = 1, 2, 4, 8$ in order and return at the first fit. Weights divide by $n$; KV divides by $\\min(n, \\text{n\\_kv\\_heads})$.',
    'The overhead is per GPU and does not shrink with $n$; compare against `mem_util * gpu_gb`, not the full card.',
]

TESTS = [
    (1_000_000_000, 2, 10, 4, 64, 2, 1000, 10, 10),
    (1_000_000_000, 2, 10, 4, 64, 2, 1000, 10, 4, 1.0, 1.0),
    (8_000_000_000, 2, 32, 8, 128, 2, 8192, 32, 80),
    (70_000_000_000, 2, 80, 8, 128, 2, 8192, 64, 80),
    (70_000_000_000, 1, 80, 8, 128, 1, 32768, 16, 80),
    (405_000_000_000, 2, 126, 8, 128, 2, 4096, 8, 80),
    (70_000_000_000, 2, 80, 2, 128, 2, 131072, 4, 80),
    (8_000_000_000, 2, 32, 8, 128, 2, 4096, 4, 24, 0.9, 1.5),
]

EXPLANATION = r"""
## Why it works
Weight shards are exact: each GPU holds $1/n$ of every matrix. KV cache shards follow the attention heads, so beyond `n_kv_heads` ways the extra GPUs hold copies (test 7, with 2 KV heads, gets no KV saving past $n = 2$). The fixed overhead per GPU is why doubling the GPU count less than halves the per-GPU memory.

## Reading the numbers
A 70B model in bf16 is 140 GB of weights: it never fits on one 80 GB card, and at 64 concurrent 8K-token requests its KV cache (about 172 GB here) is larger than the weights. Long context and high concurrency make the KV cache, not the weights, the thing that decides the GPU count, which is the case for GQA, FP8 KV caches and paged allocation.

## Common bugs
- Forgetting the factor 2 for keys **and** values, or using query heads instead of KV heads.
- Mixing GB ($10^9$) and GiB ($2^{30}$): a 7% difference that flips borderline answers.
- Dividing KV by $n$ even when $n$ exceeds the number of KV heads.

## Interview follow-ups
- If the model fits with $n = 2$, why might you still deploy $n = 4$? (More KV room means more concurrency and higher throughput per replica, and lower per-token latency from splitting the matmuls; it costs more communication.)
- When is pipeline parallelism a better way to fit a model than tensor parallelism? (Across nodes with slow interconnect: pipeline stages exchange activations once per stage, not twice per layer.)
"""
