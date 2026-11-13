META = dict(title='Count sectors for a warp memory access', tags=['GPU Programming'], diff='Easy', runtime='python',
            lessons=['ieng:13.03', 'ieng:12.06'],
            related=['shared-memory-bank-conflict-analyzer', 'gpu-occupancy-calculator', 'tiled-transpose'])

STATEMENT = r"""
Global memory moves data in 32-byte **sectors** (four of them make a 128-byte cache line). When a warp issues a load, the hardware fetches every sector that any active thread touches. A coalesced access touches the minimum number; a strided or misaligned one fetches bytes nobody asked for.

Write `count_sectors(addresses, access_bytes, sector_bytes=32, line_bytes=128)`.

## Input
- `addresses`: a list of up to 32 entries, one per thread: a non-negative int byte address, or `None` for an inactive thread.
- `access_bytes`: bytes each active thread reads, starting at its address (1, 2, 4, 8 or 16). Addresses need **not** be aligned, so one access can straddle two sectors.

## Output
A dict:
- `'sectors'`: number of distinct sectors touched, a sector being $\lfloor \text{byte} / \text{sector\_bytes} \rfloor$ for any byte of any access;
- `'lines'`: number of distinct cache lines touched (same with `line_bytes`);
- `'requested'`: number of distinct bytes requested by active threads (overlapping requests count once);
- `'efficiency'`: float, `requested / (sectors * sector_bytes)`, or `0.0` if no thread is active.
"""

STARTER = '''def count_sectors(addresses, access_bytes, sector_bytes=32, line_bytes=128):
    """Return {'sectors', 'lines', 'requested', 'efficiency'}."""
    sectors = {a // sector_bytes for a in addresses}          # TODO: ignores straddling, None and overlap
    lines = {a // line_bytes for a in addresses}
    requested = len(addresses) * access_bytes
    return {'sectors': len(sectors), 'lines': len(lines), 'requested': requested,
            'efficiency': requested / (len(sectors) * sector_bytes)}
'''

SOLUTION = '''def count_sectors(addresses, access_bytes, sector_bytes=32, line_bytes=128):
    touched = set()
    for a in addresses:
        if a is not None:
            touched.update(range(a, a + access_bytes))     # every byte this thread reads
    sectors = {b // sector_bytes for b in touched}
    lines = {b // line_bytes for b in touched}
    eff = len(touched) / (len(sectors) * sector_bytes) if touched else 0.0
    return {'sectors': len(sectors), 'lines': len(lines), 'requested': len(touched), 'efficiency': eff}
'''

HINTS = [
    'Work with the set of bytes requested. Each active thread contributes `range(addr, addr + access_bytes)`; overlaps and broadcasts collapse automatically.',
    'Sectors and lines are then just `{b // sector_bytes}` and `{b // line_bytes}` over that set. A misaligned 4-byte read at address 30 touches sectors 0 and 1.',
    'Efficiency is useful bytes over fetched bytes. Guard the no-active-thread case before dividing.',
]

TESTS = [
    ([4 * t for t in range(32)], 4),
    ([4 * 32 * t for t in range(32)], 4),
    ([4 + 4 * t for t in range(32)], 4),
    ([0] * 32, 4),
    ([16 * t for t in range(32)], 16),
    ([2 * t for t in range(32)], 2),
    ([None] * 32, 8),
    ([30, 62, None, 94], 4),
    ([8 * t for t in range(32)], 4),
    ([4 * ((7 * t) % 32) for t in range(32)], 4),
]

EXPLANATION = r"""
## Why it works
The memory system's unit is the sector, so cost is proportional to distinct sectors, not to threads or bytes. Collecting the requested bytes into a set and mapping them to sectors counts exactly what the hardware would fetch for one load instruction (ignoring cache hits from earlier instructions).

## Reading the tests
- 32 consecutive floats (test 1): 128 bytes, 4 sectors, 1 line, 100% efficient.
- Stride 128 bytes (test 2): each thread its own sector: 32 sectors for 128 useful bytes, 12.5%. This is reading a column of a row-major matrix.
- Shifted by 4 bytes (test 3): 5 sectors, 80%: misalignment costs one extra sector.
- A permutation within 128 bytes (test 10) is still perfectly coalesced: order within the warp does not matter, only which sectors are touched.
- 16-byte vector loads (test 5): 512 bytes in 16 sectors, 100%, with a quarter of the instructions.

## Complexity
$O(\text{threads} \cdot \text{access\_bytes})$, at most 512 bytes per warp.

## Common bugs
- Using only the start address: a straddling access (test 8) touches two sectors.
- Counting each thread's bytes separately, so a broadcast (test 4) looks like 128 useful bytes instead of 4.

## Interview follow-ups
- Transposing a matrix reads rows and writes columns. How does staging a tile through shared memory make both sides coalesced, and what new problem appears there? (Bank conflicts, fixed by padding.)
- Why does `x[:, ::2]` in PyTorch make a following elementwise kernel slower even though it reads half the data?
"""
