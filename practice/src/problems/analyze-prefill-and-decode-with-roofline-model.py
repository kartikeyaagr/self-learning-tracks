META = dict(title='Roofline analysis of prefill and decode', tags=['Inference', 'GPU Programming'], diff='Medium', runtime='python',
            lessons=['ieng:12.07', 'ieng:12.05'],
            related=['kv-cache-memory-for-mha-mqa-gqa-and-mla', 'estimate-model-memory-and-minimum-gpu-count', 'tensor-parallel-communication-cost'])

STATEMENT = r"""
The roofline model says a kernel takes at least $\max(\text{FLOPs} / \text{peak},\ \text{bytes} / \text{bandwidth})$. Applied to a whole forward pass of an LLM it explains the most important fact in inference: prefill is compute-bound, decode is memory-bound, and batching is what moves decode toward the roof.

Write `roofline(n_params, bytes_per_param, kv_bytes_per_token, batch, prompt_len, ctx_len, peak_flops, mem_bw)`.

## Model (ignore attention FLOPs and activation traffic)
| Phase | FLOPs | Bytes moved |
|---|---|---|
| Prefill: `batch` prompts of `prompt_len` tokens | $2 \cdot P \cdot \text{batch} \cdot \text{prompt\_len}$ | weights once, $P \cdot \text{bpp}$, plus writing the new KV: $\text{batch} \cdot \text{prompt\_len} \cdot \text{kvb}$ |
| One decode step: `batch` sequences, each with `ctx_len` cached tokens | $2 \cdot P \cdot \text{batch}$ | weights once, plus reading the cache: $\text{batch} \cdot \text{ctx\_len} \cdot \text{kvb}$ |

Here $P$ = `n_params`, bpp = `bytes_per_param`, kvb = `kv_bytes_per_token` (all layers). `peak_flops` is in FLOP/s, `mem_bw` in bytes/s. All inputs are positive ints.

## Per phase report
A tuple `(flops, bytes, intensity, time_s, bound)`:
- `flops`, `bytes`: ints. `intensity` = flops / bytes (float, FLOP per byte).
- `time_s` = $\max(\text{flops}/\text{peak},\ \text{bytes}/\text{bw})$ as a float.
- `bound` = `'compute'` if $\text{flops} \cdot \text{bw} \ge \text{bytes} \cdot \text{peak}$ (compare with integers), else `'memory'`.

## Critical batch
The smallest integer batch $B \ge 1$ at which **decode** (with this `ctx_len`) becomes compute-bound, or `None` if no batch does. Each extra sequence adds $2P$ FLOPs and $\text{ctx\_len} \cdot \text{kvb}$ bytes, so it exists only when $2P \cdot \text{bw} > \text{ctx\_len} \cdot \text{kvb} \cdot \text{peak}$. Compute it exactly with integer arithmetic.

## Output
A dict `{'prefill': report, 'decode': report, 'ridge': peak_flops / mem_bw, 'critical_batch': int or None}`.
"""

STARTER = '''def roofline(n_params, bytes_per_param, kv_bytes_per_token, batch, prompt_len, ctx_len, peak_flops, mem_bw):
    """Return {'prefill': (flops, bytes, intensity, time_s, bound), 'decode': (...), 'ridge': float, 'critical_batch': int or None}."""
    flops = 2 * n_params * batch * prompt_len
    rep = (flops, 0, 0.0, flops / peak_flops, 'compute')
    return {'prefill': rep, 'decode': rep, 'ridge': peak_flops / mem_bw, 'critical_batch': None}
'''

SOLUTION = '''def _report(flops, nbytes, peak, bw):
    bound = 'compute' if flops * bw >= nbytes * peak else 'memory'      # exact: no float rounding at the ridge
    return (flops, nbytes, flops / nbytes, max(flops / peak, nbytes / bw), bound)

def roofline(n_params, bytes_per_param, kv_bytes_per_token, batch, prompt_len, ctx_len, peak_flops, mem_bw):
    weights = n_params * bytes_per_param
    prefill = _report(2 * n_params * batch * prompt_len,
                      weights + batch * prompt_len * kv_bytes_per_token, peak_flops, mem_bw)
    decode = _report(2 * n_params * batch,
                     weights + batch * ctx_len * kv_bytes_per_token, peak_flops, mem_bw)

    # compute-bound when 2PB*bw >= (W + B*c*kvb)*peak  <=>  B*(2P*bw - c*kvb*peak) >= W*peak
    gain = 2 * n_params * mem_bw - ctx_len * kv_bytes_per_token * peak_flops
    crit = max(1, -(-weights * peak_flops // gain)) if gain > 0 else None   # integer ceil division
    return {'prefill': prefill, 'decode': decode, 'ridge': peak_flops / mem_bw, 'critical_batch': crit}
'''

HINTS = [
    'Write one helper that turns (FLOPs, bytes) into the report tuple; both phases only differ in the two counts from the table.',
    'Compare `flops * bw >= bytes * peak` on integers to decide the bound: dividing first can put a value exactly at the ridge on the wrong side.',
    'Decode is compute-bound when $B(2P\\,bw - c\\,kvb\\,peak) \\ge P\\,bpp\\,peak$. If the bracket is $\\le 0$ return `None`; otherwise the answer is a ceiling division, done with `-(-a // b)`.',
]

_H100 = (989 * 10**12, 3350 * 10**9)            # dense BF16 FLOP/s and HBM bytes/s, rounded
TESTS = [
    (1000, 2, 10, 1, 4, 4, 100, 10),
    (1000, 2, 10, 8, 4, 50, 100, 10),
    (8 * 10**9, 2, 131072, 1, 2048, 2048, *_H100),
    (8 * 10**9, 2, 131072, 64, 512, 256, *_H100),
    (70 * 10**9, 2, 327680, 256, 1024, 1024, *_H100),
    (70 * 10**9, 1, 327680, 32, 8192, 32768, *_H100),
    (8 * 10**9, 2, 131072, 4, 128, 200000, *_H100),
    (1000, 2, 10, 1, 1, 5, 100, 10),
]

EXPLANATION = r"""
## Why prefill and decode differ
Arithmetic intensity is FLOPs per byte. Prefill multiplies the weights by `batch * prompt_len` tokens at once, so its intensity grows with the token count and is in the thousands for a single long prompt: compute-bound. A decode step uses each weight for only `batch` tokens, so with batch 1 the intensity is $2P / (P \cdot \text{bpp}) = 1$ FLOP per byte for fp16, far below an H100's ridge of about 300. The GPU spends the step waiting on memory.

## What the critical batch says
Batching amortises the weight read: intensity rises roughly linearly with batch until the KV-cache reads, which also grow with batch, take over. Long contexts make each extra sequence bring more bytes, and once a sequence's cache bytes outweigh its FLOPs ($c \cdot \text{kvb} \cdot \text{peak} \ge 2P \cdot \text{bw}$) no batch size makes decode compute-bound (test 7). That is why GQA, MLA and KV quantization matter so much for long-context serving.

## Common bugs
- Counting weight bytes once per token or per sequence: weights are read once per forward pass, which is the whole reason batching helps.
- Using `math.ceil(a / b)` on huge integers: float division loses precision above $2^{53}$ and can be off by one.
- Forgetting the KV bytes in decode, which makes long-context decode look compute-bound at realistic batch sizes.

## Interview follow-ups
- How does weight-only int4 change the decode report at batch 1? (It cuts weight bytes by 4, so the step is about 4 times faster, while prefill barely changes.)
- Why does real decode stay below the roofline even at the critical batch? (Attention FLOPs, kernel launch overhead, imperfect overlap and the fact that small matmuls don't reach peak.)
"""
