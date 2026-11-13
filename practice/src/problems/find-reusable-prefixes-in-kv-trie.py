META = dict(title='Block-level prefix reuse in a KV-cache trie', tags=['Inference'], diff='Medium', runtime='python',
            lessons=['ieng:10.04', 'ieng:10.06'],
            related=['route-cache-aware-prefill-requests', 'simulate-continuous-decode-batching', 'decode-attention-with-kv-cache'])

STATEMENT = r"""
Chat and agent traffic repeats long prefixes: the same system prompt, the same few-shot examples, the conversation so far. A prefix cache keeps the KV entries of earlier prompts so a new request only has to prefill the part that differs. Paged engines cache at the granularity of **full KV blocks**, and organise them as a trie: a block is reusable only if every block before it matched too, because a token's key and value depend on everything before it.

Write `prefix_reuse(requests, block_size)` that processes requests in order and returns a dict.

## Blocks
Block $k$ of a request is the tuple of tokens `tokens[k*B:(k+1)*B]` with $B$ = `block_size`; only **full** blocks exist (a trailing partial block is never cached).

## Per request (in order)
1. **Match.** Walk the trie from the root along the request's blocks $0, 1, 2, \dots$ while the next block is present as a child. Only blocks lying entirely within the first `len(tokens) - 1` tokens may be matched: the last prompt token must always be computed, because its forward pass produces the logits for the first output token. The number of reused tokens is $B \times$ (matched blocks).
2. **Insert.** Add every full block of the request (all of `len(tokens) // B` of them) along its path, creating the nodes that don't exist yet. The cache is unbounded.

## Output
| key | value |
|---|---|
| `"reused"` | list of ints, reused tokens per request |
| `"computed"` | list of ints, `len(tokens) - reused` per request |
| `"cached_blocks"` | int, number of blocks (trie nodes, excluding the root) after the last request |
| `"hit_rate"` | float, total reused tokens / total prompt tokens |

`requests` is a non-empty list of non-empty token lists (ints).
"""

STARTER = '''def prefix_reuse(requests, block_size):
    """Return a dict: reused, computed, cached_blocks, hit_rate."""
    raise NotImplementedError
'''

SOLUTION = '''def prefix_reuse(requests, block_size):
    B = block_size
    root, n_nodes = {}, 0                               # node: dict mapping block tuple -> child node
    reused, computed = [], []
    for tokens in requests:
        blocks = [tuple(tokens[k * B:(k + 1) * B]) for k in range(len(tokens) // B)]
        matchable = (len(tokens) - 1) // B              # never reuse the final token
        node, hits = root, 0
        while hits < matchable and blocks[hits] in node:
            node, hits = node[blocks[hits]], hits + 1
        reused.append(hits * B)
        computed.append(len(tokens) - hits * B)

        node = root
        for blk in blocks:                              # insert the whole path of full blocks
            if blk not in node:
                node[blk] = {}
                n_nodes += 1
            node = node[blk]
    return {"reused": reused, "computed": computed, "cached_blocks": n_nodes,
            "hit_rate": sum(reused) / sum(len(t) for t in requests)}
'''

HINTS = [
    'Represent each trie node as a dict from a block (a tuple of `B` tokens) to its child node. The root is an empty dict.',
    'Matching stops at the first missing block *or* after `(len(tokens) - 1) // B` blocks, whichever comes first. That cap is what keeps at least the last token computed.',
    'Insert after matching, walking from the root again and creating children as needed; count each newly created node for `cached_blocks`.',
]

_sys = list(range(100, 112))                            # a 12-token shared system prompt
TESTS = [
    ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 9]], 2),
    ([[1, 2, 3, 4], [1, 2, 3, 4]], 2),
    ([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 5, 6], [1, 2, 3, 4, 5, 6, 7]], 2),
    ([_sys + [1, 2, 3], _sys + [4, 5], _sys + [1, 2, 3, 6, 7, 8, 9], _sys[:7]], 4),
    ([[5], [5], [5, 5]], 1),
    ([[1, 2, 3], [4, 5, 6]], 8),
    ([_sys + [i % 3] * (2 + i % 5) for i in range(12)], 3),
    ([list(range(i, i + 10)) for i in range(0, 4)] + [list(range(0, 10))], 5),
]

EXPLANATION = r"""
## Why a trie of blocks
The key and value of token $t$ depend on all tokens before it, so a cached block is only valid after an *identical* prefix. That's exactly a trie: a path from the root spells out a prefix, and a node stores one block of KV for that prefix. Two requests that share the system prompt share its path; at the first differing block they branch. Caching whole blocks matches the paged allocator, so a hit is just a pointer into an existing block table with no copying. (Radix trees compress chains of single-child nodes, but they store the same information.)

## Why the last token is never reused
Even if a request is fully cached, the engine needs that request's final hidden state to produce logits for the first output token, so at least the last token goes through the model. Test 2 is the same 4-token prompt twice with $B = 2$: only the first block can be reused, not both.

## Complexity
$O(L)$ per request of length $L$ with hashing (tuple hashing is linear in the block size). Memory is one node per distinct cached block.

## Common bugs
- Matching blocks by content regardless of position (a "set of blocks" cache): block 2 of one prompt can equal block 2 of another after different prefixes, and their KV differs.
- Continuing to match after the first miss.
- Caching the partial last block, which then can't be extended in place.

## Interview follow-ups
- The cache has a fixed number of blocks. Which blocks should LRU eviction be allowed to remove, and why only leaves?
- How do you hash blocks so that lookups are $O(1)$ per block without walking the trie (chain the parent's hash into the child's)?
"""
