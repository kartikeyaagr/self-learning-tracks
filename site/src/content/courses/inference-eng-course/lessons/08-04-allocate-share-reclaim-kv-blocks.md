---
id: "08.04"
course: "inference-eng-course"
module: "08"
slug: "allocate-share-reclaim-kv-blocks"
title: "Allocate, Share & Reclaim KV Blocks"
keyIdeas: ["A free list holds unused blocks; allocation pops, freeing pushes back.", "Reference counts let several sequences share the same physical block.", "Shared prefixes (system prompts, parallel samples, beam search) share blocks.", "Copy-on-write when a sequence must modify a shared block.", "Finished sequences' blocks can stay cached for future prefix hits, evicted LRU."]
question: "Two requests share a 1,000-token system prompt. How does block sharing work, and what happens when one of them writes a new token into a shared, partially filled block?"
answer: "Both block tables point to the same physical blocks for the shared prefix, with a reference count of 2 on each. Full blocks are read-only and can be shared freely. If the last shared block is partially filled and a request must append to it, copy-on-write duplicates that block for the writer so the other request's view is unchanged. Blocks are freed when their reference count drops to zero."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Sharing and copy-on-write."}, {"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Hash-based block reuse."}]
tags: ["paging", "prefix"]
---

## Allocation and freeing

The KV manager's basic job is bookkeeping:

- At admission, it allocates enough blocks for the prompt: ceiling(prompt tokens ÷ block size).
- During decode, it allocates one more block whenever the last one fills, which with 16-token blocks is every 16 steps.
- When a sequence finishes or is aborted, it returns all its blocks to the free list.

Before admitting a request, the scheduler checks that the free list holds enough blocks for its prompt, plus a small watermark of spare blocks so running sequences can keep growing.

## Sharing with reference counts

Paging makes sharing easy. Two sequences can list the same physical block in their tables. Each block carries a **reference count**: the number of sequences that use it. A block returns to the free list only when its count drops to zero.

Three situations produce sharing:

- **Parallel sampling.** A request asks for n completions of one prompt. All n sequences share the prompt's blocks and own only their generated tokens. With n = 8 and a 2,000-token prompt, that saves 7 × 2,000 × 128 KB ≈ 1.8 GB.
- **Beam search.** Beams share their common history and diverge only at the end.
- **Common prefixes.** Many requests begin with the same system prompt, tool definitions or few-shot examples. They can share those blocks.

~~~dg-flow Two sequences sharing a prefix.
Sequence A -> *Shared blocks 0 to 62 (refcount 2) -> A's own blocks
Sequence B -> *Shared blocks 0 to 62 (refcount 2) -> B's own blocks
~~~

## Copy-on-write

Sharing is safe as long as nobody modifies a shared block. Full blocks never change: once a token's K and V are written, they stay. The only risky case is a partly filled shared block, which can happen with parallel sampling, where all n sequences would next append to the same last block.

The fix is **copy-on-write**: before a sequence appends to a shared block, it copies the block to a new one it owns, and decrements the old block's count. Many engines avoid the problem for prefixes entirely by sharing only full blocks, so each sequence starts appending in a block of its own.

## From freeing to caching

Here is the step that turns sharing into **prefix caching**. When a sequence finishes, its full blocks still contain valid K and V for a particular token sequence. Instead of discarding that data, the manager keeps the blocks, indexed by their content, and marks them as free to evict.

The index key for a block is a hash of its tokens together with the hash of the block before it. Chaining the hashes means a block's key identifies the whole prefix up to and including that block, which matters because K and V depend on every earlier token.

When a new request arrives:

1. Hash its prompt block by block.
2. Look up each hash in the index, from the start, until one misses.
3. Reuse every matching block (increasing its count) and skip their prefill.
4. Allocate fresh blocks for the rest.

When the pool needs space, evict cached blocks with a count of zero, least recently used first.

<div class="ex">
<div class="lab">Worked example</div>

**the effect of prefix caching**

Every chat request starts with the same 1,500-token system prompt. Prompts average 2,000 tokens.

Without caching, every request prefills 2,000 tokens. With caching, after the first request, every request prefills about 500 tokens: the system prompt's 93 full blocks (1,488 tokens) are reused.

Prefill work falls by 75%, and TTFT for a typical request drops from about 90 ms to about 30 ms. The shared blocks are stored once instead of once per running request.

</div>

## Caching interacts with scheduling

- Hit rates depend on keeping popular prefixes in memory. If the pool is always full of active sequences, cached blocks are evicted quickly, and the hit rate falls under load.
- In a multi-replica deployment, a request benefits only if it reaches a replica that has its prefix. That is why cache-aware routing exists (Module 22).
- The order of content in prompts matters. Put shared content first and variable content last, so the shared prefix is as long as possible.

## Deep dive: hashing blocks for prefix caching

A cached block is identified by a hash that chains its own tokens with everything before it:

⟦M562⟧

Two requests share block ⟦m564⟧ only if they agree on every token up to the end of that block. Because of the chain, a single different token in block 3 changes ⟦m565⟧: everything after a difference is a miss.

<div class="ex">
<div class="lab">Worked example</div>

**what sharing saves**

100 conversations share a 2,000-token system prompt and each has 500 tokens of its own. Llama 3 8B, 128 KB per token.

Without sharing: ⟦m566⟧ tokens, 32.8 GB.

With sharing: ⟦m567⟧ tokens, 6.8 GB. The pool now fits almost five times as many conversations.

</div>

~~~dg-raw
%%bars KV memory for 100 conversations with a shared 2,000-token prompt (GB).
unit = GB
stack = Shared prefix, Per-conversation tokens
No sharing | 26.2+6.6
Prefix sharing | 0.26+6.6
%%
~~~

## Deep dive: when a cached block is reused

Freed blocks with a hash stay in an LRU list until they are needed for new data. A block with access rate ⟦m568⟧ survives if it is reused before the pool cycles through its free list. If the pool reuses ⟦m569⟧ free blocks per second and the LRU holds ⟦m570⟧, the block's lifetime is about ⟦m571⟧ seconds, and its hit probability is roughly

⟦M563⟧

A system prompt used 10 times per second never leaves the cache. A user's own chat history, revisited after 3 minutes of reading time, survives only if ⟦m572⟧ is longer than 3 minutes. That is why routing a conversation back to the same replica matters (module 22).

<div class="take">
<div class="lab">Key takeaway</div>

a free list, reference counts and copy-on-write let sequences share KV blocks safely. Keeping freed blocks indexed by a chained hash of their content turns sharing into prefix caching, which skips prefill for repeated prefixes.

</div>
