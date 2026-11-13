import numpy as np

META = dict(title='One decode step of grouped-query attention with a KV cache', tags=['Inference', 'Attention'],
            diff='Hard', runtime='torch', lessons=['ieng:14.04', 'ieng:03.04'],
            related=['scaled-dot-product-attention', 'scaled-dot-product-attention-2', 'plan-kv-cache-capacity'])

STATEMENT = r"""
During decoding, each sequence in the batch adds exactly one token. Its key and value are written into a preallocated cache, and its query attends over everything cached so far, including itself. Sequences in a batch have different lengths, so each one writes at its own position and must ignore the unused tail of its cache buffer.

Write `decode_step(q, k_new, v_new, k_cache, v_cache, lengths)` and return the tuple `(out, k_cache, v_cache, new_lengths)`.

## Inputs (all float64 except `lengths`)
- `q`: `(B, H, D)`, the new token's query for each of `H` query heads.
- `k_new`, `v_new`: `(B, Hkv, D)`, the new token's key and value for each of `Hkv` KV heads. `H` is a multiple of `Hkv`; query head `h` uses KV head `h // (H // Hkv)` (grouped-query attention: consecutive query heads share a KV head).
- `k_cache`, `v_cache`: `(B, Hkv, S, D)` buffers. For sequence `b`, positions `0 .. lengths[b] - 1` hold valid entries; everything after is garbage (arbitrary finite values) and must not affect the result.
- `lengths`: int64 `(B,)` with `0 <= lengths[b] < S`.

## What to compute
1. Write `k_new[b]` and `v_new[b]` into position `lengths[b]` of sequence `b`'s cache (for every KV head).
2. For each `b` and query head `h`, attend over positions `0 .. lengths[b]` inclusive:
   $$\text{out}[b,h] = \sum_{j \le \ell_b} \operatorname{softmax}_j\!\Big(\tfrac{q_{b,h} \cdot k_{b,g(h),j}}{\sqrt D}\Big)\, v_{b,g(h),j}$$
3. `new_lengths = lengths + 1` (int64).

## Output
`out` `(B, H, D)` float64, the two updated caches `(B, Hkv, S, D)` (garbage positions other than the newly written one are returned unchanged), and `new_lengths`. You may update the caches in place or on a copy. No Python loops over the batch or heads.

This problem runs in **Colab** with a CPU runtime (PyTorch isn't available in the browser).
"""

STARTER = '''import torch

def decode_step(q, k_new, v_new, k_cache, v_cache, lengths):
    """Return (out (B, H, D), k_cache, v_cache, new_lengths)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def decode_step(q, k_new, v_new, k_cache, v_cache, lengths):
    B, H, D = q.shape
    Hkv, S = k_cache.shape[1], k_cache.shape[2]
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    b = torch.arange(B)
    # Advanced indices on dims 0 and 2 (separated by a slice) select a (B, Hkv, D) view: one slot per sequence.
    k_cache[b, :, lengths] = k_new
    v_cache[b, :, lengths] = v_new

    group = H // Hkv
    k = k_cache.repeat_interleave(group, dim=1)          # (B, H, S, D): query head h reads KV head h // group
    v = v_cache.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhd,bhsd->bhs", q, k) / D ** 0.5
    valid = torch.arange(S)[None, :] <= lengths[:, None]  # (B, S): the old entries plus the one just written
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)                # every row has at least one valid slot, so no NaNs
    out = torch.einsum("bhs,bhsd->bhd", probs, v)
    return out, k_cache, v_cache, lengths + 1
'''

HINTS = [
    'Write the cache first, then attend: the new token must see its own key. `k_cache[torch.arange(B), :, lengths] = k_new` writes one slot per sequence in a single op.',
    'Expand the KV heads to query heads with `repeat_interleave(H // Hkv, dim=1)` (not `repeat`, which would tile them in the wrong order), then compute scores `(B, H, S)` with one `einsum`.',
    'Mask with `positions <= lengths[:, None]` (inclusive, because slot `lengths[b]` now holds the new token) using `-inf` before the softmax. Garbage values then get exactly zero weight.',
]


def _case(seed, B, H, Hkv, S, D, lengths):
    r = np.random.default_rng(seed)
    kc = r.normal(size=(B, Hkv, S, D))
    vc = r.normal(size=(B, Hkv, S, D))
    for i, n in enumerate(lengths):                      # fill the unused tail with loud garbage
        kc[i, :, n:] = 40.0
        vc[i, :, n:] = -30.0
    return (r.normal(size=(B, H, D)), r.normal(size=(B, Hkv, D)), r.normal(size=(B, Hkv, D)), kc, vc,
            np.array(lengths, dtype=np.int64))


TESTS = [
    (np.array([[[1.0, 0.0]]]), np.array([[[1.0, 0.0]]]), np.array([[[2.0, 2.0]]]),
     np.array([[[[0.0, 0.0], [9.0, 9.0], [9.0, 9.0]]]]), np.array([[[[4.0, 0.0], [9.0, 9.0], [9.0, 9.0]]]]),
     np.array([1])),
    (np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]]), np.ones((2, 1, 2)), np.array([[[1.0, -1.0]], [[3.0, 3.0]]]),
     np.zeros((2, 1, 3, 2)), np.zeros((2, 1, 3, 2)), np.array([0, 2])),
    _case(0, 3, 4, 2, 6, 4, [0, 3, 5]),
    _case(1, 2, 4, 4, 5, 3, [2, 4]),
    _case(2, 2, 6, 1, 7, 8, [6, 1]),
    _case(3, 4, 8, 2, 9, 16, [0, 0, 8, 4]),
    _case(4, 1, 2, 1, 4, 2, [3]),
]

EXPLANATION = r"""
## Why the cache works
Causal attention never lets an old token look at a newer one, so the keys and values of past tokens never change. Caching them turns each decode step from recomputing the whole prefix ($O(T^2)$ work for the step) into one new key/value plus a dot product against $T$ cached keys ($O(T)$). Prefill builds the cache for the prompt in one batched pass; decode then appends one slot per step, which is exactly this function.

## Ragged batches
In a real server, sequences in the same decode batch started at different times, so their lengths differ. A dense buffer with a per-sequence length (here) is the simplest layout; paged KV caches replace the per-sequence buffer with a block table but keep the same "write at length, mask past length" logic. The garbage in the tests (keys of 40, values of -30) makes any masking mistake obvious.

## GQA and memory
With `Hkv < H`, the cache is `H / Hkv` times smaller, and decode is memory-bound: every step reads the entire cache once. That's the main reason modern models use GQA or MQA. A production kernel never materialises the `repeat_interleave`d tensors; it maps query head `h` to KV head `h // group` inside the kernel.

## Complexity
$O(B \cdot H \cdot S \cdot D)$ for the scores and the weighted sum. A fused kernel would also skip positions past `lengths[b]` instead of masking them.

## Common bugs
- Masking with `<` instead of `<=`, so the new token doesn't attend to itself (and a sequence with `lengths[b] = 0` gets an all-`-inf` row and NaNs).
- Attending before writing the cache.
- `repeat` instead of `repeat_interleave`: KV head 0 would serve query heads 0, 2, 4, ... instead of 0, 1, ...
- Masking garbage by multiplying the probabilities by the mask after the softmax: the garbage keys still distort the normalisation.

## Interview follow-ups
- Why is decode attention memory-bound while prefill attention is compute-bound? Estimate the arithmetic intensity of each.
- How would you split a very long sequence's cache across several thread blocks and combine the partial softmax results (split-K / flash-decoding)?
"""
