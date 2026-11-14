---
id: "08.06"
course: "inference-eng-course"
module: "08"
slug: "visual-lab-build-a-paged-kv-allocator"
title: "Visual Lab: Build a Paged KV Allocator"
keyIdeas: ["Implement a block pool with a free list, refcounts and per-sequence block tables.", "Support allocate, append token, fork (share), free, and copy-on-write.", "Add a content-hash index to reuse blocks for shared prefixes.", "Visualize the pool as a grid colored by owner.", "Test invariants under random workloads."]
question: "What invariants should your allocator's tests check?"
answer: "Every physical block is either free or owned with refcount at least 1, never both; the sum of free and used blocks equals the pool size; a sequence's block table covers exactly ceil(len / block_size) blocks; freeing a sequence decrements refcounts and returns blocks only at zero; and no two sequences write to the same block without copy-on-write."
sources: [{"title": "vLLM: Paged attention design", "url": "https://docs.vllm.ai/en/latest/design/paged_attention.html", "note": "Reference for layouts."}, {"title": "nano-vllm", "url": "https://github.com/GeeeekExplorer/nano-vllm", "note": "A small engine with a readable block manager."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Build a working paged KV allocator in Python, visualize it, and test it. After this lab, the KV manager in vLLM or SGLang will read like a more careful version of your own code.

## The API to implement

```
class BlockPool:
    def __init__(self, num_blocks: int, block_size: int): ...
    def allocate(self, seq_id, n_tokens) -> list[int]       # blocks for a new prompt
    def append(self, seq_id) -> int | None                  # call once per new token
    def fork(self, src_id, dst_id) -> None                  # share all blocks
    def free(self, seq_id) -> None
    def lookup_prefix(self, token_ids) -> list[int]         # cached full blocks
```

## Steps

1. **Pool state.** A free list of block ids, a reference count per block, and a dict from sequence id to block table and token count.
2. **allocate.** Pop ceiling(n_tokens ÷ block_size) blocks, set their counts to 1, and store the table. Fail cleanly if not enough blocks are free.
3. **append.** Increment the sequence's token count. If the last block is full, pop a new block. If the last block is shared (count above 1), copy-on-write: take a new block, copy the slot data conceptually, decrement the old count.
4. **fork.** Copy the source's table to the destination and increment every block's count.
5. **free.** Decrement every block's count. A block whose count reaches 0 goes back to the free list, or into the prefix cache if it is full.
6. **Prefix cache.** Hash each full block as hash(parent_hash, tuple of its tokens). Keep a dict from hash to block id and an LRU order. `lookup_prefix` walks the prompt block by block and returns the matching blocks until the first miss. When the free list is empty, evict the least recently used cached block with count 0.
7. **Driver.** Generate a random workload with arrivals, token appends, finishes and occasional forks for parallel sampling. Run it for a few thousand steps.

## Visualize it

Draw the pool as a grid, one square per block:

- Colour by owning sequence.
- Shade blocks with a count above 1 to show sharing.
- Grey blocks that are free but cached.
- White blocks that are truly free.

Animate the run. You should see the pool fill without ever leaving unusable gaps, shared prefixes appear as shaded regions, and cached blocks absorb repeat prompts.

~~~dg-raw
%%widget paged This is the picture to aim for. Compare contiguous reservation with your paged allocator on the same requests.
%%
~~~

## Tests

Write these as assertions that run after every operation:

- **Conservation.** Free blocks + cached-free blocks + blocks with a count above 0 = pool size.
- **Table length.** Each table has ceiling(tokens ÷ block_size) entries.
- **Counts.** Each block's count equals the number of tables that contain it.
- **Prefix lookup.** Two prompts with the same first 100 tokens return the same first 6 blocks (with 16-token blocks), and diverge after that.
- **Copy-on-write.** After forking and appending to both sequences, their last blocks differ and the shared blocks' counts are 2.

<div class="ex">
<div class="lab">Worked example</div>

**a bug the conservation test catches**

A first version freed a forked sequence by pushing all its blocks onto the free list, ignoring counts.

The other sequence still listed those blocks. A new request then received one of them, and two sequences wrote to the same block.

The conservation check failed immediately: used plus free exceeded the pool size. In a real engine, this bug would show up as garbled outputs under load, which is far harder to trace.

</div>

## Stretch goals

- **Plug it into your generation loop** from Module 03. Store K and V in a tensor of shape [num_blocks, block_size, kv_heads, head_dim] per layer, and before attention, gather each sequence's blocks using its table. Check that outputs match the contiguous cache.
- **Add preemption.** When allocation fails, free the newest sequence and requeue it, then measure how often it happens at different pool sizes.
- **Measure utilization.** Log the fraction of allocated slots that hold real tokens. It should stay above 95%.

## Deep dive: the invariants your allocator must keep

Write these as assertions and run them after every operation in a randomized test:

⟦M577⟧

⟦M578⟧

- The first is conservation: no block is lost or counted twice.
- The second is reference correctness: a shared block is freed only when its last user leaves.
- The third ties the table to the sequence length: a missing or extra block is a bug even if the output looks right.

A test that runs 10,000 random add, append, fork and free operations against these three assertions catches nearly every allocator bug before any GPU code exists.

<div class="take">
<div class="lab">Key takeaway</div>

a paged allocator is a free list, reference counts, block tables and a hash index. Building and testing one makes every engine's KV manager easy to read, and its invariants catch the bugs that would otherwise appear as corrupted outputs.

</div>
