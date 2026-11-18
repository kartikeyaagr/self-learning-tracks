META = dict(title='LRU cache with per-key TTL and lazy expiry', tags=['Distributed Computing', 'Data Processing'],
            diff='Medium', runtime='python', lessons=['sdsd:05.03', 'sdsd:05.04'],
            related=['prefix-cache-matching-and-reuse', 'consistent-hash-ring', 'idempotency-key-store'])

STATEMENT = r"""
Write `simulate(capacity, ops)` that replays a trace against a cache holding at most `capacity` keys, with LRU
eviction and a TTL per key, and reports what happened.

Each op is a list starting with an integer time `t` (non-decreasing):
- `[t, "get", key]`
- `[t, "set", key, value, ttl]`, where `ttl` is seconds or `None` (the key never expires).

A key set at time `t` with TTL `ttl` expires at `t + ttl`: at any time `>= t + ttl` it is **logically expired**.
Expiry is **lazy**: an expired key stays in the cache until something touches it.

**get:** if the key is present but expired, delete it (count one `expired`) and treat it as absent. Absent: count
a miss and record `None`. Present: count a hit, record its value, and make it the most recently used. A hit does
not extend the TTL.

**set:** if the key is already present (expired or not), overwrite its value, give it the new expiry, and make it
the most recently used. Otherwise, if the cache is full, first delete **every** entry already expired at `t`
(each counts as `expired`); if it is still full, evict the least recently used entry (count one eviction). Then
insert the key as the most recently used.

Return `{"hits", "misses", "evictions", "expired", "values": [the result of each get, in order],
"keys": [keys still stored, from least to most recently used]}`. `keys` includes expired keys nothing has
touched yet.
"""

STARTER = '''from collections import OrderedDict

def simulate(capacity, ops):
    """LRU cache over a trace of get/set operations."""
    cache, values = OrderedDict(), []
    hits = misses = evictions = 0
    for op in ops:
        t, kind, key = op[0], op[1], op[2]
        if kind == "get":
            if key in cache:
                hits += 1
                values.append(cache[key])
                cache.move_to_end(key)
            else:
                misses += 1
                values.append(None)
        else:
            if key not in cache and len(cache) >= capacity:
                cache.popitem(last=False)
                evictions += 1
            cache[key] = op[3]
            cache.move_to_end(key)
    return {"hits": hits, "misses": misses, "evictions": evictions, "expired": 0,
            "values": values, "keys": list(cache)}
'''

SOLUTION = '''from collections import OrderedDict

def simulate(capacity, ops):
    cache = OrderedDict()                  # key -> (value, expires_at or None); first entry = least recently used
    s = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}
    values = []
    dead = lambda e, t: e[1] is not None and t >= e[1]
    for op in ops:
        t, kind, key = op[0], op[1], op[2]
        if kind == "get":
            e = cache.get(key)
            if e is not None and dead(e, t):      # passive expiry: checked only when touched
                del cache[key]
                s["expired"] += 1
                e = None
            if e is None:
                s["misses"] += 1
                values.append(None)
            else:
                s["hits"] += 1
                values.append(e[0])
                cache.move_to_end(key)
        else:
            value, ttl = op[3], op[4]
            if key not in cache and len(cache) >= capacity:
                for k in [k for k, e in cache.items() if dead(e, t)]:   # free dead entries before live ones
                    del cache[k]
                    s["expired"] += 1
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                    s["evictions"] += 1
            cache[key] = (value, None if ttl is None else t + ttl)
            cache.move_to_end(key)
    return {**s, "values": values, "keys": list(cache)}
'''

HINTS = [
    'An `OrderedDict` is the hash map plus linked list: `move_to_end(key)` marks a use, `popitem(last=False)` removes the least recently used.',
    'Store the absolute expiry time `t + ttl` with the value, not the TTL itself, so a check is one comparison: `t >= expires_at`.',
    'Assigning to an existing `OrderedDict` key keeps its old position; call `move_to_end` after every set.',
]

TESTS = [
    (2, [[0, "set", "a", 1, None], [1, "set", "b", 2, None], [2, "get", "a"], [3, "set", "c", 3, None],
         [4, "get", "b"], [5, "get", "a"], [6, "get", "c"]]),
    (3, [[0, "set", "s", "v1", 10], [9, "get", "s"], [10, "get", "s"], [11, "get", "s"]]),
    (3, [[0, "set", "s", "v1", 10], [8, "set", "s", "v2", 10], [12, "get", "s"], [18, "get", "s"]]),
    (2, [[0, "set", "a", 1, None], [1, "set", "b", 2, 5], [2, "get", "b"], [6, "set", "c", 3, None],
         [7, "get", "a"], [8, "get", "b"], [9, "get", "c"]]),
    (3, [[0, "set", "a", 1, 5], [1, "set", "b", 2, 5], [2, "set", "c", 3, None], [20, "get", "c"]]),
    (4, [[0, "set", "A", 1, None], [0, "set", "B", 2, None], [0, "set", "C", 3, None],
         [1, "get", "A"], [2, "get", "B"], [3, "get", "A"], [4, "get", "C"], [5, "get", "A"], [6, "get", "B"],
         [7, "set", "W", 0, None], [8, "set", "X", 0, None], [9, "set", "Y", 0, None], [10, "set", "Z", 0, None],
         [11, "get", "A"], [12, "get", "B"], [13, "get", "C"]]),
    (1, [[0, "set", "a", 1, 3], [1, "get", "a"], [2, "set", "b", 2, None], [3, "get", "a"], [4, "get", "b"]]),
    (2, [[0, "set", "a", 1, 4], [1, "get", "a"], [2, "get", "a"], [3, "get", "a"], [4, "get", "a"],
         [5, "set", "z", 9, 0], [5, "get", "z"]]),
    (2, [[0, "get", "x"], [1, "set", "x", 7, 2], [5, "set", "y", 8, None], [6, "set", "x", 9, None], [7, "get", "x"]]),
]

EXPLANATION = r"""
## Two ways a key leaves
**Expiry** removes a key because its TTL ran out; **eviction** removes a valid key because there is no room. A
healthy cache does both, and the counters tell them apart: many evictions with a low hit rate means the cache is
too small; many expirations with a low hit rate means TTLs are too short.

## LRU in O(1)
LRU is a hash map plus a doubly linked list ordered by last use. `OrderedDict` is exactly that: lookup is a hash
lookup, a use moves the entry to the end, and eviction pops the front. Test 1 is the basic trace: reading `a`
saves it, so `b` is evicted when `c` arrives.

## Lazy expiry, like Redis
Checking the expiry on every access is one integer comparison and guarantees nobody reads a stale value (test 2:
still a hit at 9, gone at exactly 10). Keys nobody touches are not deleted (test 5 ends with `a` and `b` still
stored, long expired): memory is reclaimed late, which is why Redis also runs a background sampler. Here a full
cache sweeps dead entries before evicting a live one (in test 4, `a` is least recently used but live, so the
expired `b` goes instead), so a dead key never costs a live one its place.

## TTL traps
- A set overwrites the TTL, as a plain Redis `SET` does: test 3 refreshes `s` at 8 so it lives until 18, and test 9
  turns `x` permanent by setting it with `None`.
- A hit does not extend the TTL (test 8): a TTL bounds staleness, so popular keys must still be refreshed.
- A TTL of 0 means the key is dead as soon as it is written.

## LRU and scans
Test 6 is the lesson's trace: `A`, `B` and `C` are hot, then a batch job reads `W`, `X`, `Y`, `Z` once each and LRU
evicts every hot key. LFU or a segmented LRU would keep them, because one read does not make a key look popular.

## Interview follow-ups
- Why does Redis approximate LRU by sampling? (Exact LRU costs two pointers per key and a list update on every
  read; sampling 5 keys into a 16-entry pool is nearly as good on skewed workloads.)
- Many keys written together expire together. What happens and how do you prevent it? (A miss storm on the
  database; add TTL jitter, and coalesce concurrent misses.)
"""
