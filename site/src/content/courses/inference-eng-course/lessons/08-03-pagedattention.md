---
id: "08.03"
course: "inference-eng-course"
module: "08"
slug: "pagedattention"
title: "PagedAttention"
keyIdeas: ["PagedAttention computes attention over K and V stored in non-contiguous blocks.", "Inputs: query, the KV block pool, block tables, context lengths.", "Each thread block walks a sequence's block table and reads its blocks.", "Small overhead per step, large gain in usable memory and batch size.", "FlashInfer and FlashAttention now support paged KV layouts too."]
question: "What does a PagedAttention kernel need as input that a standard attention kernel does not, and what does it cost?"
answer: "It needs the block table for each sequence (mapping logical to physical blocks) and the sequence lengths, and reads K and V from non-contiguous blocks. The cost is extra indexing and less perfectly contiguous memory access, which makes the kernel slightly slower per byte than ideal contiguous attention, but the memory saved allows much larger batches, which is a big net win."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Original paper."}, {"title": "vLLM: Paged attention design", "url": "https://docs.vllm.ai/en/latest/design/paged_attention.html", "note": "How the kernel walks blocks."}]
tags: ["attention", "paging"]
---

## The kernel's job

Decode attention computes, for each sequence, one query against all of that sequence's cached keys and values. With a contiguous cache, the keys for a sequence sit in one continuous range of memory. With paging, they are scattered across physical blocks. The **PagedAttention** kernel handles this: it follows each sequence's block table to find its keys and values wherever they are.

~~~dg-flow Paged decode attention for one sequence. Press Play to follow the lookup.
Query -> Block table -(physical ids)-> *Gather K and V blocks -> Scores and softmax -> Weighted sum of V
~~~

## Inputs

- `query`: shape [num_seqs, num_heads, head_dim], one row per sequence for decode.
- `key_cache` and `value_cache`: the whole block pool for this layer, shape [num_blocks, block_size, num_kv_heads, head_dim] or a similar layout.
- `block_tables`: shape [num_seqs, max_blocks_per_seq], padded with a sentinel.
- `context_lens`: shape [num_seqs], the number of valid tokens for each sequence.

## How it runs

A simplified version of the decode kernel:

1. Each GPU thread block takes one sequence and one KV head (serving all the query heads in its GQA group).
2. It loops over the sequence's logical blocks. For each, it reads the physical id from the block table and loads that block's 16 keys.
3. It computes the dot products of the query with those keys, and keeps a running maximum and running sum for the softmax. This **online softmax** means it never needs all scores at once.
4. It loads the matching values and accumulates the weighted sum, rescaling earlier partial results whenever the running maximum changes.
5. After the last block, it divides by the running sum and writes the output.

For long contexts, one thread block per sequence is too little parallelism. Kernels then split each sequence's blocks across several thread blocks, each computing a partial result with its own maximum and sum, and a second small step combines them. This is the idea behind split-KV and "flash decoding" approaches.

## What the indirection costs

- **Table lookups.** One small read per block. Negligible next to loading 16 × 128 × 2 bytes of keys per head.
- **Non-contiguous reads.** Within a block, data is contiguous, so each block load is an efficient burst. Across blocks, the kernel jumps around memory, which costs a little efficiency. With 16-token blocks, well-tuned kernels reach close to the bandwidth of contiguous attention.
- **Kernel complexity.** The kernel must handle variable lengths, padding in the block table and GQA grouping in one launch.

<div class="ex">
<div class="lab">Worked example</div>

**is the indirection worth it?**

Early measurements of PagedAttention showed the attention kernel itself running up to about 20 to 25% slower than a contiguous kernel in some configurations.

But paging let the engine fit 2 to 4 times as many sequences in the same memory. Attention is a minority of decode time at moderate context, and the weight read is shared across the larger batch.

Net result: 2 to 4 times higher throughput. Later kernels such as FlashInfer and FlashAttention's paged mode closed most of the kernel gap.

</div>

## Writing into the cache

Each layer also writes the new token's K and V into the right slot before attention runs. A small kernel takes each token's slot mapping, computed by the scheduler as physical block × block_size + offset, and scatters K and V there. Some engines fuse this write with the RoPE rotation or with the attention kernel.

## Beyond decode

Prefill also writes into blocks, and its attention must read keys from blocks too in two common cases:

- **Prefix caching**: the first part of the prompt is already cached, so the new tokens attend to cached blocks plus themselves.
- **Chunked prefill**: each chunk attends to the cached blocks from earlier chunks.

Modern kernel libraries support paged KV in prefill attention as well, with variable-length queries per sequence. This lets an engine run one attention call for a step that mixes decodes, prefill chunks and cached prefixes.

## Deep dive: the kernel's math, block by block

For one query head, decode attention over a sequence split into blocks ⟦m548⟧ can be computed one block at a time with a running maximum ⟦m549⟧, running sum ⟦m550⟧ and running output ⟦m551⟧, all starting at ⟦m552⟧, ⟦m553⟧, ⟦m554⟧. For block ⟦m555⟧ with scores ⟦m556⟧:

⟦M545⟧

⟦M546⟧

After the last block, the result is ⟦m557⟧. This is the same online softmax as FlashAttention. It means blocks can be visited in any physical order, which is exactly what paging needs.

~~~dg-raw
%%grid One sequence's 5 logical blocks (top) scattered across a 32-block physical pool (bottom). The table holds 3, 8, 17, 21, 29.
cols = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
key A = logical block 0 | c1
key B = logical block 1 | c2
key C = logical block 2 | c3
key D = logical block 3 | wine
key E = logical block 4 | flame
key X = other sequences | line
logical | A B C D E . . . . . . . . . . . . . . . . . . . . . . . . . . .
physical | X X X A X . . X B X X X . . . X X C X X X D X X X . . X X E . .
%%
~~~

## Deep dive: splitting long sequences across thread blocks

At batch 1 with a 32k context, one thread block per head would leave most of the GPU idle: 32 heads on 132 SMs. Split-K decoding (often called flash decoding) gives each of ⟦m558⟧ partitions a range of blocks and combines the partial ⟦m559⟧ at the end with the same rule:

⟦M547⟧

With ⟦m560⟧, the same step uses 256 thread blocks instead of 32, and long-context attention at small batch gets several times faster.

<div class="ex">
<div class="lab">Worked example</div>

**what the indirection costs**

Per layer, each 16-token block holds 32 KB of K and 32 KB of V in one contiguous run. Fetching its table entry costs 4 bytes.

The extra read is ⟦m561⟧ of the traffic. The real costs are elsewhere: less freedom to vectorize across block edges, and an extra dependent load before each block's data. Measured overheads are typically a few percent of attention time.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

PagedAttention follows each sequence's block table to gather scattered keys and values, using an online softmax so it never holds all scores at once. It costs a little kernel efficiency and buys a much larger batch, which is a large net win.

</div>
