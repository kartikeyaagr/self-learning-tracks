---
id: "28.05"
course: "inference-eng-course"
module: "28"
slug: "add-a-paged-kv-allocator"
title: "Add a Paged KV Allocator"
keyIdeas: ["Replace per-sequence buffers with a shared block pool.", "Integrate the allocator from Module 08: block tables, refcounts, free list.", "Write new KV via slot mapping; read via block tables.", "Use a paged attention kernel or a gather step.", "Measure max concurrency before and after."]
question: "What changes in the attention path when you switch from contiguous to paged KV?"
answer: "New K and V are written to slots given by a slot mapping (physical block id x block size + offset) rather than a contiguous position, and attention gathers K and V through each sequence's block table. You need either a gather step before standard attention (simple, slower) or a paged attention kernel (FlashInfer, FlashAttention with block tables, or your own Triton kernel)."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Design."}, {"title": "FlashInfer", "url": "https://github.com/flashinfer-ai/flashinfer", "note": "Paged attention kernels to call."}]
tags: ["mini-engine", "paging"]
---

## What changes

| Component | Contiguous | Paged |
| --- | --- | --- |
| Storage | per-sequence buffers | one pool of [num_blocks, block_size, kv_heads, head_dim] |
| Write | K[seq][pos] | pool[slot] |
| Read | K[seq][:len] | gather through the block table |
| Admission | reserve the maximum length | allocate blocks as tokens arrive |

## Why it matters

Contiguous buffers reserve memory for the longest possible output. Most requests never use most of it.

<div class="ex">
<div class="lab">Worked example</div>

**concurrency at 2k context**

Say 40 GB of KV memory on an 8B model, at 128 KB per token. Contiguous allocation reserves max_len = 8,192 tokens per sequence, 1 GB each, so 40 sequences fit.

The real sequences average 2,000 tokens. Paged allocation with 16-token blocks uses about 250 MB each, with at most one partly empty block per sequence.

About 160 sequences fit, four times more, with the same memory.

</div>

~~~dg-raw
%%widget paged Allocate requests of different lengths and watch the block tables fill.
%%
~~~

## Kernel options

1. **Gather** each sequence's blocks into a temporary contiguous tensor, then run standard attention. Easiest, but the copy costs bandwidth.
2. **Call a library's paged attention kernel** with block tables, such as FlashInfer's.
3. **Write a Triton paged decode kernel** yourself, as a stretch goal.

## Measure

The maximum number of concurrent sequences at 2k context before out-of-memory or preemption, contiguous versus paged. Expect several times more with paging.

<div class="take">
<div class="lab">Key takeaway</div>

paging changes where KV is written and how it is read. With a paged kernel, the same memory serves many more sequences.

</div>
