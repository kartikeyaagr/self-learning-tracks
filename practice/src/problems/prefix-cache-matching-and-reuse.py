META = dict(title='Block-level prefix cache with LRU eviction', tags=['Inference'], diff='Medium', runtime='python',
            lessons=['ieng:10.04', 'ieng:10.06'],
            related=['pagedattention-block-allocation', 'simulate-chunked-prefill-scheduling', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
Many requests start with the same tokens: a system prompt, a few-shot template, the earlier turns of a chat. An engine with **automatic prefix caching** keeps the KV blocks of past prompts and, for a new prompt, reuses the longest run of leading blocks it already has, so prefill only computes the rest.

Write `prefix_cache(requests, block_size, capacity)` that replays a stream of prompts through such a cache.

## Blocks and keys
- A prompt of length $L$ has $n = \lfloor L / \text{block\_size} \rfloor$ **full** blocks; a trailing partial block is never cached.
- Full block $i$ is identified by the **entire prefix** up to its end, `tuple(tokens[: (i + 1) * block_size])`. So two prompts share block $i$ only if they agree on every token before it as well. (Real engines hash this chain; a tuple keeps the exercise exact.)

## For each request, in order
1. **Match.** $m$ = the number of leading full blocks found in the cache: check block 0, 1, ... and stop at the first miss.
2. **Keep one token to compute.** The model must run on at least one token to produce logits, so if the match covers the whole prompt ($m \cdot \text{block\_size} = L$), use $m - 1$ instead.
3. Record `hit = m * block_size` cached tokens for this request.
4. **Update.** Visit the request's full blocks in **reverse** order, $i = n-1, \dots, 0$. A block already in the cache becomes most recently used; a new block is inserted as most recently used, after evicting the least recently used block if the cache already holds `capacity` blocks. With `capacity == 0` nothing is ever stored.

Visiting in reverse leaves block 0 as the most recently used, so a shared prefix is evicted last (evicting it first would orphan every block after it).

## Inputs and output
- `requests`: list of non-empty lists of ints. `block_size` $\ge 1$, `capacity` $\ge 0$.
- Return `(hits, hit_rate)`: `hits` is a list of ints (one per request), `hit_rate` = `sum(hits) / total prompt tokens` as a float.
"""

STARTER = '''def prefix_cache(requests, block_size, capacity):
    """Return (hits: list of cached-token counts, hit_rate: float)."""
    hits = [0 for _ in requests]
    return hits, 0.0
'''

SOLUTION = '''from collections import OrderedDict

def prefix_cache(requests, block_size, capacity):
    cache = OrderedDict()                       # prefix tuple -> None, oldest first (LRU at the front)
    hits = []
    for toks in requests:
        n = len(toks) // block_size
        keys = [tuple(toks[: (i + 1) * block_size]) for i in range(n)]

        m = 0
        while m < n and keys[m] in cache:       # longest run of leading cached blocks
            m += 1
        if m * block_size == len(toks):         # fully cached: still compute the last block for logits
            m -= 1
        hits.append(max(m, 0) * block_size)

        for key in reversed(keys):              # reverse: block 0 ends up most recently used
            if key in cache:
                cache.move_to_end(key)
            elif capacity > 0:
                if len(cache) >= capacity:
                    cache.popitem(last=False)   # evict least recently used
                cache[key] = None
    total = sum(len(t) for t in requests)
    return hits, sum(hits) / total
'''

HINTS = [
    'Use an `OrderedDict` as the LRU: `move_to_end(key)` marks a block as just used and `popitem(last=False)` evicts the oldest.',
    'Matching stops at the first missing block, because block $i$\'s key contains every earlier token. Then apply the "at least one token to compute" rule only when the match covers the entire prompt.',
    'Update the cache **after** matching, visiting blocks from last to first, and evict only when inserting a block that is not already present.',
]

_sys = [7, 7, 1, 2, 3, 4, 5, 6]
TESTS = [
    ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 9, 9]], 2, 10),
    ([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4, 5]], 2, 10),
    ([_sys + [10, 11, 12], _sys + [20, 21], _sys + [10, 11, 12, 13], [9] + _sys], 4, 8),
    ([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], 2, 0),
    ([[1, 2, 3, 4, 5, 6, 7, 8], [9, 9, 9, 9], [1, 2, 3, 4, 5, 6, 7, 8, 0]], 2, 3),
    ([[1, 2, 3, 4, 5, 6, 7, 8, 1], [1, 2, 3, 4, 5, 6, 7, 8, 2]], 2, 2),
    ([[5], [5], [5, 5]], 1, 4),
    ([[(i * 7 + j) % 5 for j in range(3 + (i % 4) * 2)] for i in range(12)] + [[0, 2, 4, 1, 3, 0, 2]] * 3, 2, 5),
]

EXPLANATION = r"""
## Why it works
A block's KV values depend on every token before it (attention looks back), so a block is reusable only when the whole prefix matches. Keying each block by its full prefix encodes that, and it also means the cached blocks of a prompt always form a chain from block 0: the first miss ends the match. Reverse-order touching makes the root of a chain the last thing LRU removes, so popular system prompts survive even when their long tails churn.

## Cost
Tuple keys make each lookup $O(L)$ to hash, so a prompt costs $O(L^2 / \text{bs})$ here. Engines hash incrementally, $h_i = \mathrm{hash}(h_{i-1}, \text{block}_i)$, which is $O(L)$ per prompt; a radix tree (SGLang's RadixAttention) shares the prefix structure explicitly and matches at token rather than block granularity.

## Common bugs
- Keying blocks by their own tokens only: two prompts that differ early but share a later block would wrongly reuse KV computed under a different context.
- Returning a full hit for a fully cached prompt: the engine then has no token to run and no logits to sample from.
- Touching blocks front to back, so block 0 becomes the eviction victim and orphans the rest of the chain.

## Interview follow-ups
- How does block size trade off hit rate against bookkeeping? (Smaller blocks match more of a partially shared prefix, but mean more hashes, more table entries and smaller kernel tiles.)
- With several replicas, how should a router use prefix caches? (Send requests with the same prefix to the same replica, balanced against queue length: cache-aware routing.)
"""
