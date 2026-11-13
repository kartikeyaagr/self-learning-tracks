META = dict(title='Cache-aware routing of prefill requests across replicas', tags=['Inference'], diff='Hard',
            runtime='python', lessons=['ieng:23.03', 'ieng:10.04'],
            related=['find-reusable-prefixes-in-kv-trie', 'simulate-continuous-decode-batching', 'plan-kv-cache-capacity'])

STATEMENT = r"""
With several model replicas behind a router, the prefix cache only helps if a request lands on a replica that already holds its prefix. Pure load balancing (round-robin) scatters related requests and wastes the caches; pure cache affinity piles everything sharing a popular prompt onto one replica. A cache-aware router scores both.

Write `route(requests, n_workers, block_size, drain)` and return a dict.

## Worker state
Each worker has its own block-level prefix cache (unbounded, initially empty) and a `load`: prefill tokens queued but not yet processed (initially 0).

## Prefix rule
As in a paged prefix cache: block $k$ of a request is `tokens[k*B:(k+1)*B]` (full blocks only). On a worker, the request's **hit** is $B$ times the number of leading blocks found along that worker's trie path, matching only blocks that lie entirely within the first `len(tokens) - 1` tokens. Its **cost** there is `len(tokens) - hit`.

## Routing each request, in order
1. Every worker's load drops by `drain` (time passes between arrivals), but not below 0.
2. For every worker $w$, score $= \text{load}_w + \text{cost}_w$ (the time until this request's prefill would finish there).
3. Pick the lowest score; on a tie, the **larger hit**; then the lower worker index.
4. On the chosen worker: `load += cost`, and insert all of the request's full blocks into its cache.

## Baseline
Also replay the same requests with **round-robin** assignment (request $i$ goes to worker $i \bmod W$, caches updated the same way) and report its total computed tokens.

## Output
| key | value |
|---|---|
| `"worker"` | list, chosen worker per request |
| `"hit"` | list, reused tokens per request on its chosen worker |
| `"prefill_tokens"` | int, total cost over all requests (cache-aware) |
| `"prefill_tokens_rr"` | int, the same total under round-robin |
| `"final_load"` | list of `n_workers` numbers, loads after the last request |

`drain` is a non-negative int; requests are non-empty int lists.
"""

STARTER = '''def route(requests, n_workers, block_size, drain):
    """Return a dict: worker, hit, prefill_tokens, prefill_tokens_rr, final_load."""
    raise NotImplementedError
'''

SOLUTION = '''def _blocks(tokens, B):
    return [tuple(tokens[k * B:(k + 1) * B]) for k in range(len(tokens) // B)]

def _hit(trie, tokens, B):
    """Reusable tokens: leading cached blocks, never covering the last token."""
    blocks, node, n = _blocks(tokens, B), trie, 0
    while n < (len(tokens) - 1) // B and blocks[n] in node:
        node, n = node[blocks[n]], n + 1
    return n * B

def _insert(trie, tokens, B):
    node = trie
    for blk in _blocks(tokens, B):
        node = node.setdefault(blk, {})

def route(requests, n_workers, block_size, drain):
    B = block_size
    tries, load = [{} for _ in range(n_workers)], [0] * n_workers
    worker, hit, total = [], [], 0
    for tokens in requests:
        load = [max(0, l - drain) for l in load]
        hits = [_hit(t, tokens, B) for t in tries]
        # score = queue + own work; ties prefer more reuse, then the lower index
        w = min(range(n_workers), key=lambda i: (load[i] + len(tokens) - hits[i], -hits[i], i))
        cost = len(tokens) - hits[w]
        load[w] += cost
        total += cost
        _insert(tries[w], tokens, B)
        worker.append(w)
        hit.append(hits[w])

    rr_tries, rr_total = [{} for _ in range(n_workers)], 0
    for i, tokens in enumerate(requests):
        t = rr_tries[i % n_workers]
        rr_total += len(tokens) - _hit(t, tokens, B)
        _insert(t, tokens, B)
    return {"worker": worker, "hit": hit, "prefill_tokens": total, "prefill_tokens_rr": rr_total,
            "final_load": load}
'''

HINTS = [
    'Reuse the prefix-trie logic: one trie per worker (dicts of block tuples), a `hit` function with the "never the last token" cap, and an `insert` function.',
    'Apply the drain to all workers **before** scoring each request. Then compute every worker\'s hit and score, and pick with `min` over a key tuple `(score, -hit, index)`.',
    'Only the chosen worker\'s load and cache change. The round-robin baseline is a separate replay with fresh caches, so keep its state apart.',
]

_sysA = list(range(1000, 1032))
_sysB = list(range(2000, 2032))
TESTS = [
    ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], 2, 2, 0),
    ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [9, 9, 9]], 2, 2, 100),
    ([_sysA + [i] * 4 if i % 2 == 0 else _sysB + [i] * 4 for i in range(8)], 2, 8, 20),
    ([_sysA + [i] * 4 for i in range(10)], 3, 8, 0),
    ([_sysA + [i] * 4 for i in range(10)], 3, 8, 40),
    ([[7] * 9, [7] * 9, [7] * 9], 3, 4, 0),
    ([(_sysA if i % 3 else _sysB)[:8 + 8 * (i % 4)] + [i, i + 1, i + 2] for i in range(15)], 4, 8, 12),
]

EXPLANATION = r"""
## Why score by finish time
`load + cost` estimates when this request's prefill would finish on each worker. A cache hit lowers `cost`, a busy worker raises `load`, so one number trades affinity against balance with no tuning knob. With `drain = 0` (test 4: a burst sharing one system prompt), nothing ever leaves the queues, so balance dominates: the router spreads the burst exactly like round-robin and every worker builds its own copy of the shared prefix. With `drain = 40` (test 5), queues empty between arrivals and everything goes to the warm worker. Test 6 shows load beating affinity on a small scale: a hit of 8 tokens doesn't outweigh a 9-token queue.

## Why compare with round-robin
Round-robin is the default in most load balancers. With two prompt families and two workers (test 3) it happens to keep each family on one worker and ties the cache-aware router. When requests outnumber the families' natural split (tests 5 and 7), it recomputes the shared prefixes on every worker, and the cache-aware router computes roughly 40-50% fewer tokens.

## Complexity
For each request, one trie walk per worker: $O(W \cdot L)$ for a prompt of length $L$.

## Common bugs
- Breaking ties by load alone, so a cold worker beats a warm one at equal scores (the spec prefers more reuse).
- Draining loads after routing instead of before.
- Sharing one trie across workers: KV caches live in each replica's own GPU memory.
- Letting the round-robin replay reuse the cache-aware caches.

## Interview follow-ups
- A real router doesn't see each replica's cache. How would you approximate it (e.g. consistent hashing on the prefix, or replicas publishing block-hash summaries)?
- One prefix is so popular that a single replica can't serve it. How would you replicate it deliberately?
"""
