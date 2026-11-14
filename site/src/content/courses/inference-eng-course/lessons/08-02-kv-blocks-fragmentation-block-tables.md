---
id: "08.02"
course: "inference-eng-course"
module: "08"
slug: "kv-blocks-fragmentation-block-tables"
title: "KV Blocks, Fragmentation & Block Tables"
keyIdeas: ["Split KV memory into fixed-size blocks (e.g. 16 tokens each).", "Each sequence has a block table: logical block index to physical block id.", "Blocks need not be contiguous; any free block can serve any sequence.", "Waste is at most one partly filled block per sequence.", "Block size trades waste against per-block overhead and kernel efficiency."]
question: "With block size 16 and a sequence of 1,000 tokens, how many blocks are used and how much is wasted? How does a block table find token 517?"
answer: "ceil(1000/16) = 63 blocks; the last holds 8 tokens so 8 slots are wasted (under 1%). Token 517 is in logical block 517 // 16 = 32 at offset 517 % 16 = 5. The block table maps logical block 32 to some physical block id, and the kernel reads slot 5 of that physical block."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Block tables and paging design."}, {"title": "vLLM: Paged attention design", "url": "https://docs.vllm.ai/en/latest/design/paged_attention.html", "note": "Kernel-level view of blocks."}]
tags: ["memory", "paging"]
---

## Blocks

A paged KV cache divides the pool into fixed-size **blocks**, each holding the K and V vectors for a fixed number of tokens. A block size of 16 tokens is a common default. In vLLM, one block covers those 16 tokens for every layer and every KV head, allocated as one entry per layer in the per-layer cache tensors.

For Llama 3 8B with 16-token blocks, one block holds 16 × 128 KB = 2 MB across all layers. A 52 GB pool is about 26,000 blocks.

## Block tables

Each sequence owns a **block table**: an ordered list of physical block ids, exactly like a page table in an operating system.

| Logical block | Tokens | Physical block |
| --- | --- | --- |
| 0 | 0 to 15 | 812 |
| 1 | 16 to 31 | 37 |
| 2 | 32 to 47 | 1,504 |
| 3 | 48 to 54 (7 used) | 9 |

To find token p, compute its logical block p ÷ 16 (integer division) and its slot p mod 16, then look up the physical block in the table. Token 50 lives in logical block 3, slot 2, which is physical block 9.

The physical blocks can be anywhere in the pool and in any order. Only the table records the order.

## Growing a sequence

When a sequence's last block fills, the KV manager takes a free block from the pool and appends its id to the table. No data moves. A sequence grows one block at a time, so it never holds more than one partly filled block.

## How little is wasted

Only the last block of each sequence can be partly empty. On average it is half full, so the waste is about half a block, or 8 tokens with 16-token blocks. For a 600-token sequence that is about 1.3%, compared with 85% for contiguous reservation in the previous lesson.

~~~dg-raw
%%widget paged Watch the fill of each block. On the paged side, only the last block of each request is partly empty.
%%
~~~

## Choosing the block size

| Block size | Waste per sequence | Block table size | Kernel efficiency |
| --- | --- | --- | --- |
| 1 to 8 | minimal | large tables, more metadata | many small gathers |
| 16 | about 8 tokens | moderate | a good default |
| 32 to 64 | 16 to 32 tokens | small | longer contiguous reads |
| 128 to 256 | more | very small | used by some kernels and hardware |

The trade-off:

- **Small blocks** waste less memory and allow finer-grained prefix sharing, since sharing works in whole blocks.
- **Large blocks** give attention kernels longer contiguous reads and smaller tables to process.

Some kernel libraries and hardware prefer specific sizes. For example, some FlashAttention-based paths and some accelerators work best with 64 or more. Engines expose block size as a setting. Change it only if a kernel needs it or a profile shows block handling is a cost.

<div class="ex">
<div class="lab">Worked example</div>

**block size and prefix sharing**

A 1,000-token system prompt is shared by every request.

With 16-token blocks, 62 full blocks (992 tokens) can be shared. The last 8 tokens sit in a partly filled block that each sequence must own.

With 256-token blocks, only 3 full blocks (768 tokens) can be shared, and each sequence recomputes the other 232 tokens.

</div>

## The free list

The pool keeps a free list of block ids. Allocation pops an id, and freeing pushes it back. Both are constant-time operations, cheap enough to run on the CPU every step for hundreds of sequences. Because every block is the same size, any free block can serve any sequence, so external fragmentation cannot happen.

## Deep dive: the waste of a paged cache

Only each sequence's last block is partly empty. If lengths are spread evenly across a block, that block is on average half full:

⟦M539⟧

At block size 16 and 1,750 tokens per sequence, waste is ⟦m541⟧. Compare that with the 79% from contiguous reservation.

~~~dg-raw
%%chart Two costs of block size, at 1,750 tokens per sequence.
x = 1, 8, 16, 32, 64, 128
logx = 1
xlab = Block size (tokens)
ylab = % of KV memory or prefix tokens
Memory wasted in last blocks: 0, 0.2, 0.43, 0.89, 1.8, 3.6
Prefix tokens that cannot be shared (2,000-token prefix)*: 0, 0, 0, 0.8, 0.8, 4
%%
~~~

The second line shows the other cost: prefix caching shares whole blocks only. A 2,000-token prefix is 125 full blocks of 16, so all of it can be shared. At 32 or 64, the last 16 tokens cannot be. At block size 128, the last 80 tokens (2,000 = 15 × 128 + 80) are in a partial block and must be recomputed.

## Deep dive: what a block holds

In vLLM each layer has its own block pool. One block holds K and V for ⟦m542⟧ tokens across all KV heads of that layer:

⟦M540⟧

The block table maps a sequence's logical block ⟦m543⟧ to a physical block id. It is shared by all layers, so a sequence of 8,192 tokens at ⟦m544⟧ needs a 512-entry table: 2 KB of int32, tiny next to its 1 GB of KV.

<div class="take">
<div class="lab">Key takeaway</div>

fixed-size blocks plus a per-sequence block table let any free block serve any sequence. Sequences grow one block at a time, and waste drops to about half a block per sequence. Block size trades waste and sharing against kernel efficiency.

</div>
