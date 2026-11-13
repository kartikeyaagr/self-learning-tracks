META = dict(title='Occupancy from threads, registers and shared memory', tags=['GPU Programming'], diff='Medium',
            runtime='python', lessons=['ieng:13.01', 'ieng:12.06'],
            related=['shared-memory-bank-conflict-analyzer', 'global-memory-coalescing-counter', 'blockwise-online-softmax'])

STATEMENT = r"""
An SM runs as many thread blocks at once as its scarcest resource allows. **Occupancy** is the fraction of the SM's warp slots those blocks fill. Kernel authors compute it to see whether registers or shared memory, not threads, are the limit.

Write `occupancy(threads_per_block, regs_per_thread, smem_per_block, device=None)`.

## Device
`device` is `None` or a dict overriding any of these defaults (an Ampere-class SM):

| key | default | meaning |
|---|---|---|
| `'max_warps'` | 64 | warp slots per SM |
| `'max_blocks'` | 32 | resident blocks per SM |
| `'regs'` | 65536 | 32-bit registers per SM |
| `'reg_unit'` | 256 | registers are allocated per warp in multiples of this |
| `'smem'` | 167936 | shared memory per SM available to blocks, bytes |
| `'smem_unit'` | 128 | shared memory is allocated per block in multiples of this |
| `'smem_reserved'` | 1024 | bytes the system reserves per resident block |

## Rules
- Warps per block: $w = \lceil \text{threads} / 32 \rceil$.
- Limits on resident blocks:
  - `'blocks'`: `max_blocks`;
  - `'warps'`: $\lfloor \text{max\_warps} / w \rfloor$;
  - `'registers'`: registers per warp $r_w = \lceil 32 \cdot \text{regs\_per\_thread} / \text{reg\_unit} \rceil \cdot \text{reg\_unit}$, limit $\lfloor \text{regs} / (r_w\, w) \rfloor$;
  - `'shared_memory'`: bytes per block $s_b = \lceil (\text{smem\_per\_block} + \text{smem\_reserved}) / \text{smem\_unit} \rceil \cdot \text{smem\_unit}$, limit $\lfloor \text{smem} / s_b \rfloor$.
- Resident blocks = the minimum of the four limits. The **limiter** is the resource that attains the minimum; if several do, take the first in the order `'blocks'`, `'warps'`, `'registers'`, `'shared_memory'`.

## Output
A dict `{'blocks_per_sm': int, 'active_warps': int, 'occupancy': float, 'limiter': str}` with `active_warps` $=$ blocks $\cdot\, w$ and `occupancy` $=$ `active_warps / max_warps`. Inputs: $1 \le$ `threads_per_block` $\le 1024$, $1 \le$ `regs_per_thread` $\le 255$, `smem_per_block` $\ge 0$ bytes (it may be too large for even one block, giving 0 blocks).
"""

STARTER = '''def occupancy(threads_per_block, regs_per_thread, smem_per_block, device=None):
    """Return {'blocks_per_sm', 'active_warps', 'occupancy', 'limiter'}."""
    warps = threads_per_block // 32                  # TODO: rounds down; registers and smem ignored
    blocks = min(32, 64 // warps)
    return {'blocks_per_sm': blocks, 'active_warps': blocks * warps, 'occupancy': blocks * warps / 64, 'limiter': 'warps'}
'''

SOLUTION = '''DEFAULT = dict(max_warps=64, max_blocks=32, regs=65536, reg_unit=256,
               smem=167936, smem_unit=128, smem_reserved=1024)

def _round_up(x, unit):
    return -(-x // unit) * unit

def occupancy(threads_per_block, regs_per_thread, smem_per_block, device=None):
    d = {**DEFAULT, **(device or {})}
    w = -(-threads_per_block // 32)                                   # a partial warp still takes a slot
    regs_per_warp = _round_up(32 * regs_per_thread, d['reg_unit'])
    smem_block = _round_up(smem_per_block + d['smem_reserved'], d['smem_unit'])
    limits = {                                                        # dict order = tie-break order
        'blocks': d['max_blocks'],
        'warps': d['max_warps'] // w,
        'registers': d['regs'] // (regs_per_warp * w),
        'shared_memory': d['smem'] // smem_block,
    }
    limiter = min(limits, key=limits.get)                            # first key with the minimum value
    blocks = limits[limiter]
    return {'blocks_per_sm': blocks, 'active_warps': blocks * w,
            'occupancy': blocks * w / d['max_warps'], 'limiter': limiter}
'''

HINTS = [
    'Round up twice: warps per block (`ceil(threads / 32)`) and the allocation granularities (registers per warp, shared memory per block). Integer ceil: `-(-x // unit)`.',
    'Compute all four block limits, then take the minimum. A dict in tie-break order plus `min(d, key=d.get)` returns the first key with the smallest value.',
    'Shared memory per block includes the reserved bytes even when the kernel asks for 0. Occupancy is in warps, so a 48-thread block wastes half of its second warp.',
]

TESTS = [
    (256, 32, 0),
    (256, 64, 0),
    (128, 40, 49152),
    (48, 16, 0),
    (1024, 255, 0),
    (256, 32, 200000),
    (64, 72, 16384, {'smem': 233472, 'max_blocks': 32}),
    (512, 128, 0),
    (32, 8, 0),
    (96, 30, 5000, {'max_warps': 48, 'max_blocks': 16, 'regs': 65536, 'smem': 101376}),
]

EXPLANATION = r"""
## Why it works
Every resident block holds its warps' registers and its shared memory for its whole lifetime, so the SM fits only as many blocks as each resource allows; the minimum wins. Allocation granularity matters: 40 registers per thread is $1280$ per warp, already a multiple of 256, but 33 would round $1056$ up to $1280$ and cost as much as 40.

Examples from the tests:
- 256 threads, 32 registers: 8 warps per block, 8 blocks fill all 64 warps (100%).
- 64 registers per thread: $2048 \cdot 8 = 16384$ registers per block, so only 4 blocks fit: 50%, limited by registers.
- 255 registers with 1024 threads: one block would need $8192 \cdot 32 = 262144$ registers, more than the SM has, so 0 blocks. The compiler would refuse or spill; this is why `__launch_bounds__` exists.

## Is higher occupancy always better?
No. Occupancy hides latency by giving the scheduler other warps to run while one waits on memory. A kernel with high instruction-level parallelism (e.g. a tensor-core matmul with large tiles in registers) often runs fastest at 25 to 50% occupancy. The calculator tells you what limits you, not what is optimal.

## Common bugs
- Rounding warps per block down: a 48-thread block uses 2 warp slots, not 1.
- Forgetting the per-block reserved shared memory. A block asking for exactly 48 KB really costs 49 KB, so with a 96 KB carveout only one fits, not two.
- Computing register limits per thread rather than per warp with the allocation unit.

## Interview follow-ups
- Your kernel uses 96 registers per thread at 256 threads and is latency-bound. What two knobs would raise occupancy, and what can each cost? (Cap registers with launch bounds, which may spill to local memory; or shrink the block.)
- Why can shared memory per SM be configured (carveout), and how does it trade against L1 cache?
"""
