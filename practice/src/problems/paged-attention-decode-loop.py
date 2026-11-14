import numpy as np

META = dict(title='PagedAttention decode loop: append to blocks, allocate on demand, attend through block tables',
            tags=['Inference', 'Attention'], diff='Hard', runtime='numpy', lessons=['ieng:08.03', 'ieng:08.04', 'ieng:14.05'],
            related=['pagedattention-block-allocation', 'paged-kv-cache-gather', 'plan-kv-cache-capacity', 'simulate-continuous-batching'])

STATEMENT = r"""
PagedAttention stores each sequence's keys and values in fixed-size blocks scattered through one shared pool, with a per-sequence block table mapping logical blocks to physical ones. Run several decode steps for a batch of sequences.

Write `paged_decode(k_cache, v_cache, block_tables, seq_lens, free_blocks, q, k_new, v_new, block_size)` returning `(out, block_tables, k_cache, v_cache)`.

## State
- `k_cache`, `v_cache` `(num_blocks, block_size, d)`: the physical pool.
- `block_tables`: one list of physical block ids per sequence; `seq_lens` `(S,)`: tokens already stored per sequence.
- `free_blocks`: free physical ids, used from the front.

## Each step $i$, for each sequence $s = 0..S-1$ in order
1. Let $n$ = current length. If $n$ is a multiple of `block_size`, take the first free block and append it to the sequence's table.
2. Write `k_new[i, s]` and `v_new[i, s]` at physical block `table[n // block_size]`, offset `n % block_size`; the length becomes $n + 1$.
3. Gather the sequence's $n + 1$ keys and values through its table and compute `out[i, s]` $= \mathrm{softmax}(q_{i,s}K^\top/\sqrt d)\,V$ with `q[i, s]`.

`q`, `k_new`, `v_new` are `(steps, S, d)`. Return `out` `(steps, S, d)`, the final block tables, and the updated caches (new arrays). The free list always has enough blocks.
"""

STARTER = '''import numpy as np

def paged_decode(k_cache, v_cache, block_tables, seq_lens, free_blocks, q, k_new, v_new, block_size):
    """Return (out, block_tables, k_cache, v_cache)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def paged_decode(k_cache, v_cache, block_tables, seq_lens, free_blocks, q, k_new, v_new, block_size):
    K, V = np.array(k_cache, dtype=float), np.array(v_cache, dtype=float)
    tables = [list(map(int, t)) for t in block_tables]
    lens = [int(n) for n in seq_lens]
    free = list(map(int, free_blocks))
    q, kn, vn = (np.asarray(a, dtype=float) for a in (q, k_new, v_new))
    steps, S, d = q.shape
    out = np.zeros((steps, S, d))
    for i in range(steps):
        for s in range(S):
            n = lens[s]
            if n % block_size == 0:
                tables[s].append(free.pop(0))
            b, o = tables[s][n // block_size], n % block_size
            K[b, o], V[b, o] = kn[i, s], vn[i, s]
            lens[s] = n + 1
            idx = np.arange(n + 1)
            phys = np.array(tables[s])[idx // block_size]
            Ks, Vs = K[phys, idx % block_size], V[phys, idx % block_size]
            sc = Ks @ q[i, s] / np.sqrt(d)
            w = np.exp(sc - sc.max())
            out[i, s] = (w / w.sum()) @ Vs
    return out, tables, K, V
'''

HINTS = [
    'Logical position $j$ of a sequence lives at physical block `table[j // block_size]`, slot `j % block_size`. Build both index arrays for all $n + 1$ positions and gather in one step.',
    'Allocate a block only when the write position starts a new block. Blocks are never freed in this exercise.',
    'Process sequences in order within each step: allocation order decides which physical block each sequence receives.',
]

_r = np.random.default_rng(2302)
def _c(S, d, bs, steps, n_blocks, init_lens):
    K, V = _r.normal(size=(n_blocks, bs, d)), _r.normal(size=(n_blocks, bs, d))
    perm = [int(v) for v in _r.permutation(n_blocks)]
    tables = []
    for n in init_lens:
        need = -(-n // bs)
        tables.append(perm[:need]); perm = perm[need:]
    return (K, V, tables, np.array(init_lens), perm, _r.normal(size=(steps, S, d)), _r.normal(size=(steps, S, d)),
            _r.normal(size=(steps, S, d)), bs)
TESTS = [
    _c(1, 2, 2, 1, 4, [0]),
    _c(1, 2, 2, 3, 4, [1]),
    _c(2, 4, 4, 5, 10, [3, 4]),
    _c(3, 3, 2, 4, 20, [0, 5, 2]),
    _c(4, 8, 8, 10, 24, [7, 16, 1, 30]),
]

EXPLANATION = r"""
## The problem it solves
Serving systems used to reserve one contiguous KV region per request sized for the maximum output length. Most of that was never used, and fragmentation wasted more, so the KV cache, which limits batch size, was badly underused. Paging, borrowed from operating-system virtual memory, allocates blocks only as tokens arrive and lets blocks live anywhere in the pool, so waste is limited to the last partly filled block of each sequence.

## What it enables
- Larger batches from the same memory, which is where vLLM's throughput gains came from.
- Sharing: sequences with a common prefix (parallel samples, beam search, system prompts) can point at the same physical blocks, with copy-on-write when one diverges.
- Cheap preemption: a sequence's blocks can be freed or swapped and recomputed later.

## The kernel cost
Attention has to gather keys and values through an indirection instead of reading a contiguous array. Real kernels load whole blocks at a time to keep reads coalesced, so the overhead is small.

## Common bugs
- Allocating when $n$ is not a multiple of the block size, or one step late.
- Gathering with the logical block index instead of the physical one.
- Writing the new key after computing attention, so the current token can't attend to itself.

## Interview follow-ups
- How would you pick the block size, and what does it trade?
- What happens when the free list runs out mid-batch, and what are the scheduler's options?
"""
