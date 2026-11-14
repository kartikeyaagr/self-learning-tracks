import math

META = dict(title='Which causal attention tiles can be skipped', tags=['Attention', 'GPU Programming'], diff='Easy', runtime='python')

STATEMENT = r"""
A tiled attention kernel splits queries into blocks of `Bq` rows and keys into blocks of `Bk` columns. Query row $i$ has absolute position $i + $ `q_offset` (non-zero when queries are the last part of a cached sequence); key column $j$ has position $j$. Causal attention lets a query see keys with position $\le$ its own.

Write `classify_tiles(n_q, n_k, Bq, Bk, q_offset)` returning a list of strings, one per query block, with one character per key block:

- `"."` skip: no query in the block can see any key in the block.
- `"F"` full: every query in the block sees every key in the block (no mask needed).
- `"M"` masked: some but not all pairs are visible.

Edge blocks are shorter when the sizes do not divide evenly; use the actual first and last positions in each block. Also return counts: `{"tiles": [...], "skip": int, "full": int, "masked": int}`.
"""

STARTER = '''import math

def classify_tiles(n_q, n_k, Bq, Bk, q_offset):
    """Skip, full or masked for each (query block, key block) tile."""
    rows = ["M" * math.ceil(n_k / Bk) for _ in range(math.ceil(n_q / Bq))]
    return {"tiles": rows, "skip": 0, "full": 0, "masked": len(rows) * len(rows[0])}
'''

SOLUTION = '''import math

def classify_tiles(n_q, n_k, Bq, Bk, q_offset):
    rows, cnt = [], {".": 0, "F": 0, "M": 0}
    for qb in range(math.ceil(n_q / Bq)):
        q0 = qb * Bq + q_offset
        q1 = min((qb + 1) * Bq, n_q) - 1 + q_offset
        row = ""
        for kb in range(math.ceil(n_k / Bk)):
            k0, k1 = kb * Bk, min((kb + 1) * Bk, n_k) - 1
            c = "." if k0 > q1 else ("F" if k1 <= q0 else "M")
            row += c; cnt[c] += 1
        rows.append(row)
    return {"tiles": rows, "skip": cnt["."], "full": cnt["F"], "masked": cnt["M"]}
'''

HINTS = [
    'Only the corners of a tile matter: the first query against the last key, and the last query against the first key.',
    'A tile is skippable when its first key comes after its last query.',
    'With q_offset = n_k - n_q (decoding with a cache) almost every tile is full.',
]

TESTS = [(8, 8, 2, 2, 0), (8, 8, 4, 2, 0), (7, 9, 3, 4, 2), (1, 16, 1, 4, 15), (4, 12, 2, 4, 8), (6, 6, 6, 6, 0)]

EXPLANATION = r"""
## Half the work is free
For square causal attention about half of the tiles lie entirely above the diagonal and can be skipped without loading anything, which halves the FLOPs. Tiles entirely below the diagonal need no mask at all, so only the thin band of diagonal tiles pays for masking.

## Why it matters for kernel design
Branching on tile type per block is cheap, and splitting the inner loop into "full" tiles followed by "masked" tiles keeps the common path free of mask arithmetic. Block sizes that do not divide the sequence length create ragged edge tiles that need bounds masks as well as causal ones.

## Interview follow-ups
- How does load imbalance arise when each thread block handles one query block of a causal attention?
"""
