META = dict(title='Paged KV block allocator with forking and copy-on-write', tags=['Inference'], diff='Hard', runtime='python',
            lessons=['ieng:08.04', 'ieng:08.02'],
            related=['prefix-cache-matching-and-reuse', 'kv-cache-memory-for-mha-mqa-gqa-and-mla', 'simulate-continuous-batching'])

STATEMENT = r"""
PagedAttention stores the KV cache in fixed-size **blocks** of `block_size` token slots. Each sequence has a **block table**: the list of physical block ids holding its tokens in order. Blocks can be shared between sequences (for example after forking a prompt for parallel sampling) and are reference-counted; a shared block that is about to be written is copied first.

Write `paged_alloc(num_blocks, block_size, ops)` that replays a list of operations on a pool of blocks `0 .. num_blocks - 1` (all free at the start).

## Allocation rule
Whenever a block is needed, take the **smallest-id** free block.

## Operations (3-tuples)
- `('append', sid, n)` with `n >= 1`: add `n` tokens to sequence `sid` (a new `sid` starts empty). Let its current length be $L$.
  1. If $L$ is not a multiple of `block_size` (the last block has free slots) **and** that last block is shared (reference count $> 1$), it must be copied: allocate a fresh block, replace the last entry of this sequence's table with it, and drop one reference from the old block.
  2. Then allocate $\lceil (L + n)/\text{bs} \rceil - \lceil L/\text{bs} \rceil$ new blocks, appended to the table in allocation order.
  - If the free pool cannot supply **all** blocks this op needs (copy plus new), the op fails and changes nothing, not even creating `sid`.
- `('fork', src, dst)`: `dst` (a new id) gets a copy of `src`'s table and length; every block in it gains one reference.
- `('free', sid, None)`: drop one reference from each of its blocks (blocks reaching 0 return to the pool) and forget the sequence.

Operations only refer to existing sequences where they must (fork's `src`, free's `sid`).

## Output
A tuple `(tables, n_free, failed)`:
- `tables`: list of `[sid, length, [block ids]]` for the live sequences, sorted by `sid`.
- `n_free`: number of free blocks at the end.
- `failed`: list of indices into `ops` of the appends that failed.
"""

STARTER = '''def paged_alloc(num_blocks, block_size, ops):
    """Return (tables [[sid, length, blocks], ...] sorted by sid, n_free, failed op indices)."""
    tables, failed = {}, []
    free = list(range(num_blocks))
    for i, (kind, a, b) in enumerate(ops):
        if kind == 'append':
            t = tables.setdefault(a, [0, []])
            t[0] += b
            while len(t[1]) * block_size < t[0]:
                t[1].append(free.pop(0))
    return [[s, t[0], t[1]] for s, t in sorted(tables.items())], len(free), failed
'''

SOLUTION = '''import heapq

def paged_alloc(num_blocks, block_size, ops):
    free = list(range(num_blocks))          # a min-heap: pop gives the smallest free id
    heapq.heapify(free)
    ref = [0] * num_blocks
    seqs = {}                               # sid -> [length, block table]
    failed = []
    ceil_div = lambda a: -(-a // block_size)

    def release(b):
        ref[b] -= 1
        if ref[b] == 0:
            heapq.heappush(free, b)

    def take():
        b = heapq.heappop(free)
        ref[b] = 1
        return b

    for i, (kind, a, b) in enumerate(ops):
        if kind == 'append':
            L, table = seqs.get(a, (0, []))
            cow = L % block_size != 0 and ref[table[-1]] > 1      # partially filled AND shared
            need = int(cow) + ceil_div(L + b) - ceil_div(L)
            if need > len(free):
                failed.append(i)                                   # atomic: nothing changes
                continue
            table = list(table)
            if cow:
                old, table[-1] = table[-1], take()                 # copy-on-write (data copy not modelled)
                release(old)
            table += [take() for _ in range(need - int(cow))]
            seqs[a] = (L + b, table)
        elif kind == 'fork':
            L, table = seqs[a]
            for blk in table:
                ref[blk] += 1
            seqs[b] = (L, list(table))
        else:                                                      # 'free'
            for blk in seqs.pop(a)[1]:
                release(blk)

    return [[s, L, t] for s, (L, t) in sorted(seqs.items())], len(free), failed
'''

HINTS = [
    'Keep a min-heap of free ids, a reference count per block, and per sequence its length plus block table. Every op is a 3-tuple, so `for kind, a, b in ops` unpacks all of them.',
    'Before mutating anything, compute how many blocks the append needs: one for copy-on-write (only if the last block is partly filled and its count is above 1) plus the change in `ceil(length / block_size)`. Fail the op if the pool is too small.',
    'On copy-on-write, take the new block first, then drop the reference to the old one. Freeing just decrements counts; a block goes back to the heap only when its count reaches 0.',
]

TESTS = [
    (4, 2, [('append', 0, 3), ('append', 1, 1)]),
    (4, 2, [('append', 0, 3), ('fork', 0, 1), ('append', 1, 1), ('append', 0, 2)]),
    (3, 4, [('append', 0, 5), ('append', 1, 5), ('append', 1, 3), ('free', 0, None), ('append', 1, 5)]),
    (6, 2, [('append', 7, 4), ('fork', 7, 8), ('fork', 7, 9), ('append', 8, 1), ('append', 9, 2), ('free', 7, None),
            ('append', 8, 1)]),
    (5, 3, [('append', 0, 2), ('fork', 0, 1), ('fork', 1, 2), ('append', 1, 1), ('append', 2, 4), ('free', 0, None),
            ('free', 1, None), ('append', 2, 2)]),
    (2, 1, [('append', 0, 3), ('append', 1, 2), ('free', 1, None), ('append', 0, 1)]),
    (8, 4, [('append', 3, 4), ('fork', 3, 1), ('append', 1, 1), ('append', 3, 1), ('free', 3, None), ('free', 1, None)]),
    (10, 2, [('append', i % 3, 1 + i % 4) for i in range(9)] + [('free', 1, None), ('fork', 2, 5), ('append', 5, 3),
                                                                  ('append', 2, 1)]),
]

EXPLANATION = r"""
## Why it works
Paging decouples a sequence's logical token order (its block table) from physical memory. Growing a sequence only ever needs whole new blocks, so there is no external fragmentation and at most `block_size - 1` wasted slots per sequence. Reference counts make sharing free: a forked sequence costs nothing until one branch writes into a shared, partially filled block, and only that one block is copied. Full shared blocks are never written again, so they stay shared forever.

## Complexity
Each append or free touches $O(\text{blocks changed} \cdot \log N)$ heap operations; a fork costs $O(\text{table length})$. The attention kernel reads K/V through the block table, so slot $j$ of a sequence is at physical slot `table[j // bs] * bs + j % bs`.

## Choosing the block size
Small blocks waste less memory at the end of each sequence and share prefixes at finer grain, but give the attention kernel shorter contiguous reads and longer tables. vLLM defaults to 16 tokens.

## Common bugs
- Allocating some blocks before discovering the pool is too small, leaving a half-applied op (real engines then preempt a request instead).
- Copying on write when the last block is full: the new token goes into a new block anyway.
- Returning a block to the pool when one of several owners frees it.

## Interview follow-ups
- When the pool runs out mid-generation, an engine can preempt by **swapping** a sequence's blocks to CPU or by **recomputing** them later. When is each cheaper?
- How does beam search use fork and copy-on-write, and what happens to the reference counts when a beam is pruned?
"""
