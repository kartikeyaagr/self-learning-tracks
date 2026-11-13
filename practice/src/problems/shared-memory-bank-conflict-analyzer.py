META = dict(title='Shared-memory bank conflicts for one warp', tags=['GPU Programming'], diff='Easy', runtime='python',
            lessons=['ieng:13.03', 'ieng:13.01'],
            related=['global-memory-coalescing-counter', 'gpu-occupancy-calculator', 'tiled-transpose'])

STATEMENT = r"""
Shared memory is split into 32 **banks**; successive 4-byte words go to successive banks. A warp's access is served in one pass if no two threads need **different words in the same bank**. Several threads reading the same word is free (a broadcast). Otherwise the access is replayed once per extra word: an $n$-way conflict costs $n$ passes.

Write `bank_conflicts(addresses, n_banks=32, word_bytes=4)`.

## Input
- `addresses`: a list of up to 32 entries, one per thread of the warp: a non-negative int **byte** address of a 4-byte access (aligned to 4), or `None` for an inactive thread.

## Rules
- Word index $= \text{address} \,//\, \text{word\_bytes}$, bank $=$ word index $\bmod$ `n_banks`.
- For each bank, count the **distinct** word indices requested in it.

## Output
A dict:
- `'bank_words'`: list of `n_banks` ints, distinct words per bank;
- `'passes'`: int, the maximum of `bank_words` (0 if no thread is active);
- `'conflict_free'`: bool, `passes <= 1`.
"""

STARTER = '''def bank_conflicts(addresses, n_banks=32, word_bytes=4):
    """Return {'bank_words', 'passes', 'conflict_free'}."""
    counts = [0] * n_banks
    for a in addresses:
        counts[(a // word_bytes) % n_banks] += 1      # TODO: counts broadcasts as conflicts, crashes on None
    return {'bank_words': counts, 'passes': max(counts), 'conflict_free': max(counts) <= 1}
'''

SOLUTION = '''def bank_conflicts(addresses, n_banks=32, word_bytes=4):
    words = [set() for _ in range(n_banks)]
    for a in addresses:
        if a is None:                            # inactive lane: requests nothing
            continue
        w = a // word_bytes
        words[w % n_banks].add(w)                # a set: same word twice is a broadcast, not a conflict
    counts = [len(s) for s in words]
    passes = max(counts)
    return {'bank_words': counts, 'passes': passes, 'conflict_free': passes <= 1}
'''

HINTS = [
    'Keep a **set** of word indices per bank. Two threads reading the same word land in the same set entry, so broadcasts are free automatically.',
    'Skip `None` lanes. The number of passes is the largest set size; with no active lanes it is 0.',
    'Try the classic case: a float32 column read of a 32x32 tile, `addr = 4 * (32 * t)`. Every thread hits bank 0: a 32-way conflict. Padding rows to 33 words spreads them over all banks.',
]

TESTS = [
    ([4 * t for t in range(32)],),
    ([4 * 32 * t for t in range(32)],),
    ([4 * 33 * t for t in range(32)],),
    ([0] * 32,),
    ([8 * t for t in range(32)],),
    ([None] * 32,),
    ([4 * (t % 4) * 32 + 4 * (t // 4) for t in range(32)],),
    ([4 * t if t % 2 == 0 else None for t in range(32)],),
    ([4 * 32 * (t // 16) + 4 * (t % 16) for t in range(32)],),
]

EXPLANATION = r"""
## Why it works
Each bank delivers one 4-byte word per cycle. A warp request is split into as many passes as the most-loaded bank needs distinct words; banks with fewer words idle during the extra passes. Equal addresses are merged by the hardware into a single read that is broadcast to all requesters, so only **distinct** words count.

## Reading the tests
- Unit stride (test 1): thread $t$ hits bank $t$, one pass.
- Stride 32 words (test 2): a column of a row-major `float[32][32]` tile. All threads hit bank 0: 32 passes, a 32x slowdown for that instruction.
- Stride 33 (test 3): padding the tile to `float[32][33]` makes thread $t$ hit bank $33t \bmod 32 = t$: conflict-free. This one-word padding is the standard fix in transpose kernels.
- Stride 2 words (test 5): threads $t$ and $t + 16$ share a bank with different words: 2-way.
- Same word everywhere (test 4): a broadcast, one pass.

## Complexity
$O(\text{threads})$ with sets. Real tools (Nsight Compute's `l1tex__data_bank_conflicts` counters) report the same quantity per instruction.

## Common bugs
- Counting threads per bank instead of distinct words, which reports broadcasts as conflicts.
- Using the byte address modulo 32 instead of the word index modulo 32.

## Interview follow-ups
- How do 8-byte and 16-byte (vectorised) accesses change the analysis? (The warp is split into phases of 16 or 8 threads, each phase checked separately.)
- Swizzling (XOR-ing the column index with the row) avoids conflicts without padding. Why do tensor-core matmul kernels prefer it?
"""
