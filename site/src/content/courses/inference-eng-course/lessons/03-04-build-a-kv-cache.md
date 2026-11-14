---
id: "03.04"
course: "inference-eng-course"
module: "03"
slug: "build-a-kv-cache"
title: "Build a KV Cache"
keyIdeas: ["Allocate K and V buffers per layer: [batch, kv_heads, max_len, head_dim].", "Prefill writes positions 0..prompt_len-1; each decode step writes one more.", "Decode feeds only the new token; attention uses the cached prefix.", "Position ids (for RoPE) must continue from the cache length.", "Check correctness: cached and uncached greedy outputs must match."]
question: "What tensors does a per-layer KV cache hold, with what shapes, and what changes in the attention call once you use it?"
answer: "Per layer, a key tensor and a value tensor of shape [batch, kv_heads, max_len, head_dim], plus a current length per sequence. In decode the model computes Q, K and V for only the new token, writes K and V at position len, then attends with Q against K[:, :, :len+1] and V[:, :, :len+1]. Input to the model becomes one token per sequence instead of the full sequence."
sources: [{"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/main/en/kv_cache", "note": "DynamicCache, StaticCache and others."}, {"title": "nanoGPT", "url": "https://github.com/karpathy/nanoGPT", "note": "A minimal model to add a cache to."}]
tags: ["kv-cache", "lab"]
---

## Goal

Build a KV cache by hand for a real model, check that it matches the reference implementation token for token, and see its effect on step time. This lab makes the abstract cache concrete: two buffers per layer and a length counter.

## The data structure

For each layer l, allocate two buffers big enough for the longest sequence you plan to support:

```
B, H_kv, T_max, hd = 1, 8, 4096, 128
K = [torch.empty(B, H_kv, T_max, hd, dtype=torch.bfloat16, device="cuda")
     for _ in range(n_layers)]
V = [torch.empty_like(k) for k in K]
length = 0
```

For Llama 3 8B, that is 32 layers × 2 buffers × 8 heads × 4,096 positions × 128 × 2 bytes = 512 MB for one sequence.

## Prefill

Run the whole prompt through the model. In each layer, after projecting K and V and applying RoPE to K, copy them into positions 0 to P − 1:

```
K[l][:, :, :P] = k      # k has shape [B, H_kv, P, hd]
V[l][:, :, :P] = v
```

Then set `length = P`. Attention in prefill uses the causal mask as usual.

## A decode step

```
x = embed(new_token)                         # [B, 1, d]
for l, layer in enumerate(layers):
    h = rmsnorm(x)
    q, k, v = layer.qkv(h)                   # q [B, 32, 1, hd], k and v [B, 8, 1, hd]
    q, k = rope(q, k, position=length)
    K[l][:, :, length] = k[:, :, 0]
    V[l][:, :, length] = v[:, :, 0]
    a = attention(q, K[l][:, :, :length + 1], V[l][:, :, :length + 1])
    x = x + layer.o_proj(a)
    x = x + layer.mlp(rmsnorm(x))
length += 1
```

Three details matter:

- **Position.** RoPE must use the token's absolute position, which is `length` before incrementing. Using 0 for every decode token is the most common bug, and it produces fluent but wrong text.
- **No mask needed.** The cache holds only past tokens, so the new query may attend to all of them.
- **GQA.** With 32 query heads and 8 KV heads, each KV head serves four query heads. Either repeat K and V along the head axis before attention, or use a kernel that supports grouped heads directly. Repeating costs memory traffic; the grouped kernel avoids it.

## Check correctness

1. Run greedy decoding with your cache and with the library's built-in cache on the same prompt.
2. Compare the generated token ids for at least 100 steps. They should match exactly, or diverge only after a near-tie in the logits.
3. Also compare logits at a few steps. The maximum absolute difference should be around 1e-2 or less in BF16.

If the outputs diverge in the first few tokens, check RoPE positions, the order of K and V heads, and whether you wrote the K vector before or after rotation. The cache must hold rotated keys.

## Check performance

Time each decode step with `torch.cuda.synchronize()` and plot step time against position for 1,000 steps.

<div class="ex">
<div class="lab">Worked example</div>

**what the plot shows**

Without the cache, step time on an 8B model climbed from 25 ms at position 128 to 140 ms at position 2,048.

With the hand-built cache, step time started at 23 ms and reached 25 ms at position 2,048. The slow rise is attention reading a longer cache.

The 23 ms is mostly launch overhead in eager PyTorch. The weight read alone would be about 5 ms. CUDA graphs close that gap later in the course.

</div>

## What this design gets wrong

Pre-allocating T_max positions for every sequence is simple but wasteful in a server:

- **Internal waste.** A sequence that uses 300 tokens still holds 4,096 slots. At 128 KB per slot, that is 490 MB wasted.
- **Fragmentation.** When sequences of different sizes finish at different times, free memory ends up in scattered pieces that cannot hold a new large request.
- **No sharing.** Two requests with the same system prompt store identical K and V twice.

Studies of early serving systems found that 60 to 80% of KV memory was wasted this way. Paged attention, the subject of Module 08, stores the cache in small fixed-size blocks allocated on demand, which fixes all three problems.

## Deep dive: the cache's shape and the cost of growing it

A contiguous cache stores, per layer, ⟦m322⟧. Writing token ⟦m323⟧ is an in-place copy into slot ⟦m324⟧. The alternative, `torch.cat` on every step, copies the whole cache each time:

⟦M321⟧

For 2,000 tokens at 128 KB per token, concatenation copies about 262 GB in total, against 256 MB for in-place writes. That is the most common bug in a first KV cache: correct output, step time that grows with position.

~~~dg-raw
%%chart Decode step time against position: correct in-place cache versus concatenation (illustrative).
x = 100, 500, 1000, 1500, 2000
xlab = Position in sequence
ylab = Step time (ms)
In-place writes: 7.0, 7.1, 7.2, 7.4, 7.5
torch.cat every step: 7.1, 7.6, 8.3, 9.0, 9.7
No cache at all: 9, 14, 24, 35, 47
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a KV cache is two buffers per layer and a length counter. Decode feeds one token, uses its correct position, and attends to the cached prefix, which keeps step cost nearly flat. Pre-allocating per sequence works but wastes most of the memory.

</div>
