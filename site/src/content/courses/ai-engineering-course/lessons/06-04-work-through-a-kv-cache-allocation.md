---
id: "06.04"
course: "ai-engineering-course"
module: "06"
slug: "work-through-a-kv-cache-allocation"
title: "Work through a KV-cache allocation"
keyIdeas: ["Step 1: per-token KV bytes = 2 x layers x kv_heads x head_dim x bytes.", "Step 2: per-block bytes = per-token bytes x block size.", "Step 3: number of blocks = free memory / per-block bytes.", "Step 4: blocks per request = ceil(tokens / block size). Waste is at most one partial block per sequence.", "Step 5: compare with contiguous allocation at the maximum length to see the batch-size gain.", "Shared prefixes: blocks of a common system prompt are counted once and shared by reference."]
question: "A GPU has 24 GB free for KV cache. Per-token KV size is 128 KB, block size is 16 tokens. How many blocks exist, and how many 1,000-token requests fit with paging vs with 4,096-token contiguous reservations?"
answer: "Block = 128 KB x 16 = 2 MB, so 24 GB / 2 MB = 12,288 blocks. A 1,000-token request needs ceil(1000/16) = 63 blocks, so paging fits 12,288 / 63 = 195 requests. Contiguous 4,096-token reservations cost 4,096 x 128 KB = 512 MB each, so only 24 GB / 512 MB = 48 fit: about 4x fewer, with ~76% of reserved memory unused."
sources: [{"title": "Kwon et al.: PagedAttention (2023)", "url": "https://arxiv.org/abs/2309.06180", "note": "Section 4: block tables, sharing and copy-on-write."}, {"title": "vLLM docs: Automatic prefix caching", "url": "https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/", "note": "How shared prompt blocks are reused across requests."}, {"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/kv_cache", "note": "Cache size and memory options."}]
tags: ["Exercise"]
---

## The setup

This exercise uses the numbers from lessons 04.04 and 06.01. Model: 32 layers, 8 KV heads, head dim 128, fp16. GPU memory left after weights and activations: 24 GB. Block size: 16 tokens.

## Step 1: bytes per token

```
2 (K and V) x 32 layers x 8 heads x 128 dims x 2 bytes = 131,072 bytes = 128 KB
```

## Step 2: bytes per block

```
128 KB x 16 tokens = 2 MB per block
```

## Step 3: how many blocks

```
24 GB / 2 MB = 12,288 blocks  (using 1 GB = 1,024 MB)
```

That's 12,288 x 16 = 196,608 tokens of total KV capacity.

## Step 4: paged allocation

A request with 1,000 tokens (prompt plus output so far) needs ceil(1,000 / 16) = 63 blocks. The last block holds 8 tokens and wastes 8 slots, under 1%.

```
12,288 blocks / 63 blocks per request = 195 requests
```

## Step 5: contiguous allocation

A pre-vLLM server reserves the maximum, 4,096 tokens, for every request:

```
4,096 x 128 KB = 512 MB per request
24 GB / 512 MB = 48 requests
```

Each request only uses 1,000 of its 4,096 reserved slots, so about 76% of the memory is wasted.

| Allocation | Requests in memory | Memory actually holding tokens |
| --- | --- | --- |
| Contiguous (max 4,096) | 48 | ~24% |
| Paged (16-token blocks) | 195 | ~99% |

Four times the batch means, roughly, up to four times the decode throughput, as long as compute isn't yet the limit. That's in line with the 2-4x the vLLM paper reports.

## Step 6: shared prefix

Now suppose every request starts with the same 512-token system prompt. That's 32 blocks, stored once and shared. Each request needs only its own ceil(488 / 16) = 31 blocks, plus the shared 32 counted once:

```
(12,288 - 32) / 31 = 395 requests
```

~~~dg-flow How the memory budget flows to batch size.
24 GB free -(/ 2 MB)-> 12,288 blocks -(/ 63 per request)-> *195 concurrent requests -(shared 512-token prefix)-> ~395
~~~

## Run the numbers for your own setup

The calculator below follows the six steps of this lesson. Change the free memory, model shape, block size and request lengths, and compare how many concurrent requests fit under contiguous, paged, and paged-with-shared-prefix allocation.

~~~dg-raw
%%widget pagecalc Concurrent requests that fit in GPU memory under three allocation schemes.
~~~

## Sensitivity: which input matters most

Try these in the calculator:

- **Double the max length** (from 4k to 8k) while actual requests stay at 1k. Contiguous capacity halves; paged capacity doesn't move. Paging decouples capacity from the worst case.
- **Double the actual tokens per request.** Both schemes lose capacity, because now the memory is really used.
- **Grow the shared prefix.** With a 2,000-token system prompt and 500-token user turns, sharing the prefix nearly quadruples capacity, because 80% of each request's tokens are stored once.
- **Change block size** from 16 to 128. Waste in the last block grows, but the effect is small next to the others. Engines choose 16 or 32 mostly for kernel efficiency.

<div class="ex">
<div class="lab">Worked example</div>

**the capacity plan for a chat product**

Llama 3 8B in bf16 on an 80 GB GPU: 16 GB weights, about 8 GB for activations and overhead, leaving about 56 GB for KV.

At 128 KiB per token, 56 GB holds about 430,000 tokens. With an average live conversation of 3,000 tokens, that is about 140 concurrent conversations with paging.

Contiguous allocation with an 8k reservation (1 GiB each) would fit about 50. Same GPU, same model, almost 3× the users.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

per-token KV bytes x block size gives block size in bytes; free memory over that gives the block pool. Paging packs requests by actual length instead of maximum length, and shared prefixes stretch it further.

</div>
