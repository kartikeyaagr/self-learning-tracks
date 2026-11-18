META = dict(title='Consistent hash ring with virtual nodes: which keys move', tags=['Distributed Computing'],
            diff='Medium', runtime='python', lessons=['sdsd:04.07'],
            related=['route-cache-aware-prefill-requests', 'lru-cache-with-ttl'])

STATEMENT = r"""
Write `move_keys(nodes, vnodes, keys, change)` for a cache cluster that maps keys to servers with a consistent
hash ring, and report what happens when one server joins or leaves.

**Hash.** Use `ring_hash(s) = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)`, a position in
$[0, 2^{32})$. It is given in the starter.

**Ring.** Each node `n` gets `vnodes` virtual nodes named `f"{n}#{i}"` for `i` in `range(vnodes)`, placed at
`ring_hash` of that name. Sort the virtual nodes by (position, node name).

**Lookup.** A key belongs to the node of the first virtual node clockwise from `ring_hash(key)`: the first one
whose position is **greater than or equal to** the key's. If there is none, wrap around to the first virtual node
on the ring.

**Change.** `change` is `["add", node]` or `["remove", node]`. Adding a node that is already present, or
removing one that is absent, changes nothing. The cluster never becomes empty.

Return a dict with:
- `"owners"`: the owner of each key **after** the change, in the order of `keys`,
- `"moved"`: the keys whose owner changed, in the order of `keys`,
- `"load"`: `{node: number of keys it owns after the change}` for every node after the change.
"""

STARTER = '''import hashlib

def move_keys(nodes, vnodes, keys, change):
    """Map keys to nodes and report which keys moved."""
    op, node = change
    after = nodes + [node] if op == "add" else [n for n in nodes if n != node]
    owners = lambda ns: [ns[ring_hash(k) % len(ns)] for k in keys]
    old, new = owners(nodes), owners(after)
    return {"owners": new, "moved": [k for k, a, b in zip(keys, old, new) if a != b],
            "load": {n: new.count(n) for n in after}}

def ring_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
'''

SOLUTION = '''import bisect
import hashlib

def move_keys(nodes, vnodes, keys, change):
    op, node = change
    after = [n for n in nodes if n != node]
    if op == "add":
        after = nodes if node in nodes else nodes + [node]

    def owners(ns):
        ring = sorted((ring_hash(f"{n}#{i}"), n) for n in ns for i in range(vnodes))
        pos = [p for p, _ in ring]
        # first position >= the key's; index len(ring) wraps to 0
        return [ring[bisect.bisect_left(pos, ring_hash(k)) % len(ring)][1] for k in keys]

    old, new = owners(nodes), owners(after)
    return {"owners": new, "moved": [k for k, a, b in zip(keys, old, new) if a != b],
            "load": {n: new.count(n) for n in after}}

def ring_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
'''

HINTS = [
    'Build the ring as a sorted list of `(position, node)` pairs. A lookup is a binary search over the positions (`bisect.bisect_left`).',
    'Wrap-around is one modulo: `bisect_left` returns `len(ring)` for a key past the last position, and `% len(ring)` turns that into 0.',
    'Compute owners for the old and the new node list with the same function, then compare key by key.',
]

_KEYS = [f"user:{i}" for i in range(200)]
TESTS = [
    (["A", "B", "C"], 1, [f"k{i}" for i in range(12)], ["add", "D"]),
    (["A", "B", "C"], 1, [f"k{i}" for i in range(12)], ["remove", "B"]),
    (["A", "B", "C"], 1, ["A#0", "B#0", "C#0", "k3"], ["add", "D"]),
    ([f"cache-{i}" for i in range(1, 5)], 100, _KEYS, ["add", "cache-5"]),
    ([f"cache-{i}" for i in range(1, 11)], 100, _KEYS, ["add", "cache-11"]),
    ([f"cache-{i}" for i in range(1, 6)], 100, _KEYS, ["remove", "cache-3"]),
    (["A", "B"], 20, _KEYS[:30], ["add", "B"]),
    (["A", "B"], 20, _KEYS[:30], ["remove", "Z"]),
]

EXPLANATION = r"""
## Why not `hash % N`
The starter is modulo hashing. Going from $N$ to $N+1$ servers changes `hash % N` for about $N/(N+1)$ of keys,
so almost every key lands on a server that does not have it, the cache hit rate collapses and the full read load
falls on the database. Run it on test 5 and roughly nine keys in ten move. (It also appends a node that is
already present, which test 7 catches.)

## What the ring guarantees
On a ring, a new node only takes the keys in the arcs just counter-clockwise of its virtual nodes, so every moved
key moves **to** the new node (check `owners` in tests 4 and 5); on removal, every moved key comes **from** the
removed node (test 6). The share that moves is about $1/(N+1)$ on an add and $1/N$ on a remove: 42 of 200 keys
going from 4 to 5 nodes (20% expected), 24 of 200 going from 10 to 11 (9% expected; 200 keys is a small sample).
Modulo hashing moves 166 and 181 of the same keys.

## Why virtual nodes
With one point per node (tests 1 to 3) the arcs are uneven, so loads are uneven, and a removed node's whole arc
goes to a single neighbour. With 100 points per node, each node owns many small arcs, the load evens out, and a
change takes small slices from, or hands them to, every other node.

## The `>=` and the wrap
Test 3 looks up keys that are exactly a virtual node's name, so they hash to that virtual node's position: the
key belongs to that node, not the next one. A key hashing past the last position belongs to the first virtual
node, which is the point of joining the two ends of the hash range into a ring.

## Interview follow-ups
- How do all clients agree on the ring? (Same node list from a configuration service or gossip, same hash, same
  vnode naming; clients with different lists disagree about ownership.)
- How do you place replicas? (Walk clockwise and take the next $R-1$ **distinct physical** nodes, skipping more
  vnodes of nodes already chosen.)
- Does the ring balance traffic? (It balances key counts, not hot keys; replicate or salt hot keys.)
"""
