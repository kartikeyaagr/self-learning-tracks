---
id: "06.01"
course: "ai-engineering-course"
module: "06"
slug: "paged-attention"
title: "Paged attention"
keyIdeas: ["Older servers reserved one contiguous KV cache block per request, sized for the maximum possible length.", "Most requests are shorter than the maximum, so most of that memory sat unused (internal fragmentation) and gaps between requests couldn't be reused (external fragmentation).", "The vLLM paper measured only 20-38% of KV cache memory actually holding tokens in existing systems.", "PagedAttention borrows virtual memory from operating systems: the cache is split into fixed-size blocks (e.g. 16 tokens), allocated on demand.", "A per-request block table maps logical blocks to physical blocks, so a sequence's cache need not be contiguous.", "Blocks can be shared across sequences (copy-on-write) for parallel sampling, beam search and shared prompt prefixes. Waste drops below 4%, and throughput rose 2-4x."]
question: "Why did pre-vLLM servers waste most of their KV cache memory, and how does paging fix it?"
answer: "They pre-allocated one contiguous KV region per request sized to the maximum sequence length, so space reserved for tokens never generated, early finishes (internal fragmentation) and unusable gaps between allocations (external fragmentation) left only ~20-38% of memory holding real tokens. PagedAttention splits the cache into fixed-size blocks allocated on demand and tracked by a per-sequence block table, so memory needn't be contiguous, waste is limited to the last partial block (<4%), and blocks can be shared copy-on-write across sequences."
sources: [{"title": "Kwon et al.: Efficient Memory Management for LLM Serving with PagedAttention (2023)", "url": "https://arxiv.org/abs/2309.06180", "note": "The vLLM paper."}, {"title": "vLLM blog: Easy, Fast, and Cheap LLM Serving with PagedAttention", "url": "https://blog.vllm.ai/2023/06/20/vllm.html", "note": "Plain-language intro with animations."}, {"title": "vLLM documentation", "url": "https://docs.vllm.ai/", "note": "The open-source serving engine built on it."}]
tools: ["vLLM"]
---

## The memory problem

Lesson 04.04 showed that the KV cache can take more GPU memory than the weights. The more requests whose caches fit at once, the larger the batch and the higher the throughput. So wasted cache memory directly means lost throughput.

Before vLLM, serving systems allocated each request a **contiguous** chunk of KV memory big enough for its maximum length (say 2,048 tokens), because they didn't know how long the output would be. Kwon et al. found three kinds of waste:

- **Reserved**: space held for tokens not generated yet.
- **Internal fragmentation**: space never used because the request finished early.
- **External fragmentation**: gaps between allocations too small to use.
Only 20-38% of KV cache memory held actual token states in the systems they measured.

## The OS idea

Operating systems solved the same problem decades ago with virtual memory and paging: give each process a contiguous **logical** address space, backed by fixed-size **physical** pages placed anywhere.

~~~dg-flow PagedAttention maps each sequence's logical KV blocks onto any free physical blocks.
Sequence tokens -(every 16 tokens)-> Logical blocks 0, 1, 2 -(block table)-> *Physical blocks 7, 1, 3 -(attention kernel reads)-> Output
~~~

PagedAttention:

1. Splits the KV cache into blocks of a fixed number of tokens (16 by default).
2. Allocates a new block only when a sequence fills its current one.
3. Keeps a **block table** per sequence mapping logical to physical blocks.
4. Uses an attention kernel that gathers K and V from non-contiguous blocks.

The only waste is the unfilled part of each sequence's last block: under 4% in the paper.

## Sharing blocks

Because blocks are indirect, two sequences can point at the same physical block:

- **Parallel sampling** (n completions for one prompt): all share the prompt's blocks.
- **Beam search**: beams share their common history.
- **Shared prefixes**: many requests with the same system prompt share its blocks (prefix caching).
A reference count tracks sharing; when a shared block must change, it is copied first (copy-on-write).

## Results

vLLM reported 2-4x higher throughput than prior systems at the same latency, with bigger gains for long sequences and parallel sampling. The block idea is now standard in serving engines (vLLM, SGLang, TensorRT-LLM, TGI).

## Watch the allocator

Requests arrive with different lengths and finish at different times. A contiguous allocator reserves the maximum length up front for each request; a paged allocator hands out fixed-size blocks as tokens are generated. Step through the simulation and compare wasted memory.

~~~dg-raw
%%widget paging Contiguous reservation versus paged blocks for a stream of requests.
~~~

The Kwon et al. (2023) vLLM paper measured that earlier serving systems wasted 60 to 80% of KV cache memory on reservation and fragmentation, and that paging cut waste to under 4%. More free memory means a bigger batch, which is where the reported 2 to 4× throughput gain came from.

## How the block table works

Each sequence has a block table, like a page table in an operating system:

```
sequence 17: logical block 0 -> physical block 812
             logical block 1 -> physical block 45
             logical block 2 -> physical block 1,203   (partly filled)
```

The attention kernel reads keys and values through this table, so the blocks can be anywhere in GPU memory. When a sequence finishes, its blocks go straight back to a free list. With 16-token blocks, the only waste per sequence is the unused part of its last block: at most 15 tokens.

## Sharing is the second win

Two requests with the same system prompt can point their first blocks at the same physical blocks. A copy is only made when one of them writes to a shared block (copy-on-write). This enables:

- **Parallel sampling**: n candidate answers from one prompt share the prompt's blocks.
- **Beam search**: beams share their common prefix.
- **Prefix caching across requests**: a long system prompt is stored once for every user who sends it. SGLang's RadixAttention generalises this with a tree of shared prefixes.

<div class="take">
<div class="lab">Key takeaway</div>

PagedAttention stores the KV cache in small fixed-size blocks mapped through a block table, like OS pages. Near-zero fragmentation means bigger batches, and shared blocks make prefix reuse cheap.

</div>
