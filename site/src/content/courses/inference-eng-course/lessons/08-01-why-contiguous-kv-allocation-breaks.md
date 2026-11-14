---
id: "08.01"
course: "inference-eng-course"
module: "08"
slug: "why-contiguous-kv-allocation-breaks"
title: "Why Contiguous KV Allocation Breaks"
keyIdeas: ["Contiguous allocation reserves one block of memory per sequence, sized for the maximum length.", "Most sequences end far short of the maximum: reserved space is wasted.", "Allocations of different sizes leave gaps: external fragmentation.", "The vLLM paper found earlier systems used only a fraction of KV memory for real tokens.", "Less usable memory means smaller batches and lower throughput."]
question: "Name the three kinds of memory waste in contiguous per-request KV allocation."
answer: "Reservation waste: space reserved up to max length that the sequence never uses. Internal fragmentation: unused space at the end of each allocation once the sequence finishes short. External fragmentation: free memory split into gaps between allocations, none large enough for a new sequence even when the total free is sufficient."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Quantifies KV waste in earlier systems."}, {"title": "vLLM blog: Easy, Fast, and Cheap LLM Serving with PagedAttention", "url": "https://blog.vllm.ai/2023/06/20/vllm.html", "note": "Accessible summary."}]
tags: ["memory"]
---

## The naive approach

The simplest way to store a sequence's KV cache is one contiguous buffer per sequence, as in lesson 03.04. Each buffer is sized for the longest the sequence could become: its prompt plus the maximum output, or simply the model's maximum context length. Attention kernels like it because finding token p is simple pointer arithmetic.

It works for one sequence. In a server running hundreds of sequences of unknown length, it wastes most of the memory.

## Three kinds of waste

~~~dg-layers Where contiguous allocation loses memory.
Reservation | space held for tokens the sequence has not generated yet, and may never
Internal fragmentation | the unused tail left when a sequence ends before its maximum
External fragmentation | free gaps between buffers, each too small for a new request
~~~

- **Reservation.** A sequence with a 4,096-token limit holds 4,096 slots from its first step, even though it has written only its prompt so far. Those slots cannot be used by anyone else while it runs.
- **Internal fragmentation.** Most sequences stop well before their limit. A chat answer of 300 tokens leaves thousands of reserved slots unused until the sequence is freed.
- **External fragmentation.** As sequences of different sizes finish at different times, free memory is scattered in pieces. A new request needing a large contiguous buffer may not fit even when the total free memory is enough.

<div class="ex">
<div class="lab">Worked example</div>

**how much is lost**

A server allows 4,096 tokens per sequence. Typical sequences use about 600 tokens.

Each sequence reserves 4,096 slots and uses 600. Utilization is about 15%.

A 40 GB KV pool at 128 KB per token holds about 312,000 tokens. With contiguous reservation, that is 76 sequences. With exact allocation it would be about 520.

The vLLM paper measured existing systems and found only 20 to 38% of KV memory held real token data.

</div>

~~~dg-raw
%%widget paged Add requests and decode. The contiguous side runs out of room while the paged side keeps admitting.
%%
~~~

## Why waste limits throughput

Decode throughput depends on how many sequences share each step, because every step reads the weights once regardless of batch size. If 85% of the KV pool is reserved but empty, the batch stays small and most of the GPU's bandwidth advantage goes unused. Memory waste becomes throughput loss directly.

## Can you just predict the output length?

One idea is to reserve only the expected output length. This fails in practice:

- Output lengths are hard to predict. The same prompt can produce 50 or 1,500 tokens.
- An underestimate means the sequence needs more space mid-generation. With contiguous buffers, that means copying the whole cache to a larger buffer, if one is free.
- An overestimate brings back the waste.

## What a better design needs

1. **Allocate as tokens are generated**, not up front, so reservation waste disappears.
2. **Make any free memory usable by any sequence**, so external fragmentation disappears.
3. **Let sequences share identical data**, such as a common system prompt, so it is stored once.

Operating systems solved the same problem for process memory decades ago with paging: virtual pages mapped to scattered physical frames through a page table. PagedAttention brings that idea to the KV cache.

## Deep dive: what contiguous reservation wastes, as a formula

If every sequence reserves room for ⟦m532⟧ tokens but ends at ⟦m533⟧, the share of reserved memory that ever holds data is

⟦M530⟧

and while the sequence is still running, the share actually filled is lower again, about ⟦m534⟧. With ⟦m535⟧, 1,500-token prompts and 500-token answers, that is ⟦m536⟧. Four fifths of the pool holds nothing.

~~~dg-raw
%%grid The same 24 slots of KV memory, two allocation schemes. Each cell is one block of 16 tokens.
cols = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
key A = request A | c1
key B = request B | c2
key C = request C | c3
key D = request D | wine
key R = reserved, never used | line
contiguous | A A A R R R R R B B R R R R R R C C C C R R R R
paged | A B C A C B A C D C D D D . . . . . . . . . . .
%%
~~~

Contiguous allocation fits three requests and leaves 15 reserved blocks empty, so request D cannot start. Paged allocation fits all four in 13 blocks and still has 11 free.

## Deep dive: external fragmentation

Even without over-reservation, variable-length contiguous regions leave holes. A new sequence needing ⟦m537⟧ contiguous slots may fail even when the total free space exceeds ⟦m538⟧. With paging, every free block is usable by any sequence, so the only test is

⟦M531⟧

which turns fragmentation into a counting problem.

<div class="take">
<div class="lab">Key takeaway</div>

contiguous KV allocation wastes most of the cache through reservation and fragmentation, which caps the batch size and throughput. Allocating on demand in fixed-size pages is the largest single capacity win in LLM serving.

</div>
