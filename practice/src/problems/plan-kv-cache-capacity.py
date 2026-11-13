META = dict(title='Plan KV-cache capacity in blocks', tags=['Inference'], diff='Medium', runtime='python',
            lessons=['ieng:03.06', 'aieng:06.04'],
            related=['decode-attention-with-kv-cache', 'simulate-continuous-decode-batching', 'training-compute-budget'])

STATEMENT = r"""
Before an inference server accepts traffic it has to decide how much GPU memory goes to the KV cache and how many requests that memory can hold. Paged engines allocate the cache in fixed-size **blocks** of tokens, so a request always occupies a whole number of blocks.

Write `plan_kv_capacity(cfg, requests)` that returns a dict.

## Config
`cfg` is a dict with int values except `mem_util` (a float in $(0, 1]$):

| key | meaning |
|---|---|
| `gpu_bytes` | total device memory |
| `mem_util` | fraction of it the engine may use |
| `weight_bytes` | memory taken by the weights |
| `reserve_bytes` | activations, workspace and CUDA graphs |
| `n_layers`, `n_kv_heads`, `head_dim` | model shape (KV heads, not query heads) |
| `dtype_bytes` | bytes per cached element (2 for bf16, 1 for fp8) |
| `block_size` | tokens per block |

## Steps
1. `bytes_per_token` $= 2 \cdot$ `n_layers` $\cdot$ `n_kv_heads` $\cdot$ `head_dim` $\cdot$ `dtype_bytes` (the 2 is one key plus one value).
2. `budget` $= \lfloor$ `gpu_bytes * mem_util` $\rfloor$ − `weight_bytes` − `reserve_bytes`. If it is not positive, there are no blocks.
3. `num_blocks` $= \lfloor$ `budget` / (`bytes_per_token * block_size`) $\rfloor$, and `max_tokens` = `num_blocks * block_size`.
4. `requests` is a list of `(prompt_len, max_new_tokens)` int pairs. Each reserves $\lceil (\text{prompt\_len} + \text{max\_new\_tokens}) / \text{block\_size} \rceil$ blocks for its whole lifetime. Admit them **in order** while they fit; stop at the first one that doesn't (later requests are not considered, even if they are smaller).

## Output
| key | value |
|---|---|
| `"bytes_per_token"` | int |
| `"num_blocks"` | int |
| `"max_tokens"` | int |
| `"admitted"` | int, number of requests admitted |
| `"free_blocks"` | int, blocks left after admission |
| `"wasted_tokens"` | int, reserved-but-unusable slots in the admitted requests' last blocks: $\sum(\text{blocks} \cdot \text{block\_size} - \text{prompt\_len} - \text{max\_new\_tokens})$ |

Use integer arithmetic everywhere except the single `gpu_bytes * mem_util` product.
"""

STARTER = '''def plan_kv_capacity(cfg, requests):
    """Return a dict: bytes_per_token, num_blocks, max_tokens, admitted, free_blocks, wasted_tokens."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def plan_kv_capacity(cfg, requests):
    bs = cfg["block_size"]
    bytes_per_token = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * cfg["dtype_bytes"]
    budget = math.floor(cfg["gpu_bytes"] * cfg["mem_util"]) - cfg["weight_bytes"] - cfg["reserve_bytes"]
    num_blocks = max(budget, 0) // (bytes_per_token * bs)

    free, admitted, wasted = num_blocks, 0, 0
    for prompt_len, max_new in requests:
        need = -(-(prompt_len + max_new) // bs)      # ceil division without floats
        if need > free:
            break                                    # FCFS: no skipping ahead of a blocked request
        free -= need
        admitted += 1
        wasted += need * bs - prompt_len - max_new
    return {"bytes_per_token": bytes_per_token, "num_blocks": num_blocks, "max_tokens": num_blocks * bs,
            "admitted": admitted, "free_blocks": free, "wasted_tokens": wasted}
'''

HINTS = [
    'The per-token KV size is independent of sequence length: one key and one value vector per KV head per layer.',
    'Do the floor of `gpu_bytes * mem_util` first, then subtract the weights and reserve, then integer-divide by the bytes of one **block**. Clamp a negative budget to zero.',
    'Admission is a loop with a `break`: ceil-divide each request into blocks (`-(-n // bs)`), stop at the first one that exceeds the free blocks, and track the slack in each last block.',
]

_GiB = 1 << 30
_llama8b = dict(gpu_bytes=80 * _GiB, mem_util=0.9, weight_bytes=16 * 10 ** 9, reserve_bytes=4 * _GiB,
                n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2, block_size=16)
TESTS = [
    (dict(gpu_bytes=10000, mem_util=1.0, weight_bytes=2000, reserve_bytes=0, n_layers=1, n_kv_heads=1, head_dim=4,
          dtype_bytes=2, block_size=4), [(5, 3), (10, 0), (100, 100)]),
    (dict(gpu_bytes=1000, mem_util=0.5, weight_bytes=600, reserve_bytes=0, n_layers=2, n_kv_heads=2, head_dim=2,
          dtype_bytes=2, block_size=2), [(1, 1)]),
    (_llama8b, [(2000, 500)] * 30),
    (_llama8b, [(4096, 4096)] * 200),
    ({**_llama8b, 'dtype_bytes': 1}, [(4096, 4096)] * 200),
    ({**_llama8b, 'block_size': 128}, [(10, 1)] * 5 + [(10 ** 7, 1), (1, 1)]),
    (dict(gpu_bytes=24 * _GiB, mem_util=0.92, weight_bytes=14 * 10 ** 9, reserve_bytes=2 * _GiB, n_layers=28,
          n_kv_heads=4, head_dim=128, dtype_bytes=2, block_size=32), [(300, 700), (17, 15), (31, 1), (8000, 192)] * 6),
    ({**_llama8b, 'gpu_bytes': 16 * _GiB}, [(1, 1)]),
]

EXPLANATION = r"""
## Where the formula comes from
Each layer caches one key vector and one value vector per KV head per token, so the cache costs $2 L H_{kv} d$ elements per token regardless of the model's query-head count. For an 8B Llama-style model ($L=32$, $H_{kv}=8$, $d=128$, bf16) that's 128 KiB per token. An 80 GB card with 16 GB of weights then holds roughly 450k tokens: about 50 requests of 8k tokens (test 4). Halving `dtype_bytes` with an fp8 cache doubles it (test 5).

## Why blocks
Contiguous per-request buffers must be sized for the worst case and fragment as requests come and go. Fixed-size blocks make every free block usable by anyone. The only waste is inside each request's last block, at most `block_size - 1` tokens. Test 6 shows that cost grows with the block size: tiny requests waste most of a 128-token block.

## Reservation vs on-demand
This planner reserves `prompt + max_new_tokens` up front, so admitted requests can never run out of memory, but most requests stop well before `max_new_tokens`, so the reservation is pessimistic. Engines like vLLM allocate blocks as tokens are generated and **preempt** (swap or recompute) a request when memory runs out, which admits far more requests at the cost of occasional rework.

## Common bugs
- Using the number of query heads instead of KV heads (overestimates GQA models' cache by $H/H_{kv}$).
- Floating-point ceil: `math.ceil(n / bs)` is fine for small numbers but integer `-(-n // bs)` never rounds wrongly.
- Skipping a request that doesn't fit and admitting a later, smaller one. That's a different (non-FCFS) policy, and it can starve large requests.
- Forgetting the 2 for keys and values.

## Interview follow-ups
- How does the answer change with tensor parallelism over 4 GPUs (weights and KV heads are both split)?
- Why does multi-head latent attention (MLA) change `bytes_per_token` so drastically?
"""
