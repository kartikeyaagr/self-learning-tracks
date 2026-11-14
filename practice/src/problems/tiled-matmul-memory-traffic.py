import math

META = dict(title='Global memory traffic and arithmetic intensity of a tiled matmul', tags=['GPU Programming'], diff='Medium', runtime='python',
            related=['analyze-prefill-and-decode-with-roofline-model', 'masked-edge-block-load'])

STATEMENT = r"""
A tiled matmul $C = AB$ with `A (M, K)` and `B (K, N)` launches one program per `BM x BN` output tile. Each program loops over K in steps of `BK`, loading an A block and a B block each step, and writes its C tile once. With no cache reuse between programs, write `matmul_traffic(M, N, K, BM, BN, BK, bytes_per)`:

- Programs: $\lceil M/BM \rceil \cdot \lceil N/BN \rceil$.
- Elements read from A: every element of A is read once per column of tiles, $MK \lceil N/BN \rceil$ (count only valid elements; masked padding costs nothing). B: $KN \lceil M/BM \rceil$. C written: $MN$.
- Bytes = elements × `bytes_per`. FLOPs = $2MNK$. Arithmetic intensity = FLOPs / bytes.
- Ideal (each matrix touched once): bytes $(MK + KN + MN) \cdot$ `bytes_per`, and its intensity.

Return `{"programs", "bytes", "intensity", "ideal_bytes", "ideal_intensity", "reuse_factor": bytes / ideal_bytes}`.
"""

STARTER = '''import math

def matmul_traffic(M, N, K, BM, BN, BK, bytes_per):
    """HBM bytes and arithmetic intensity of a tiled matmul."""
    b = (M * K + K * N + M * N) * bytes_per
    return {"programs": 1, "bytes": b, "intensity": 2 * M * N * K / b}
'''

SOLUTION = '''import math

def matmul_traffic(M, N, K, BM, BN, BK, bytes_per):
    tm, tn = math.ceil(M / BM), math.ceil(N / BN)
    elems = M * K * tn + K * N * tm + M * N
    b = elems * bytes_per
    ideal = (M * K + K * N + M * N) * bytes_per
    flops = 2 * M * N * K
    return {"programs": tm * tn, "bytes": b, "intensity": flops / b, "ideal_bytes": ideal,
            "ideal_intensity": flops / ideal, "reuse_factor": b / ideal}
'''

HINTS = [
    'Each output tile needs a full BM x K strip of A and a K x BN strip of B; BK only changes how the strip is chunked, not its size.',
    'Larger tiles mean fewer programs re-reading the same strips: traffic falls roughly as 1/BN for A and 1/BM for B.',
    'Compare the intensity with the GPU ridge point (peak FLOP/s divided by bandwidth) to see if the kernel can be compute bound.',
]

TESTS = [
    (4096, 4096, 4096, 128, 128, 32, 2),
    (4096, 4096, 4096, 16, 16, 16, 2),
    (1, 4096, 4096, 16, 128, 64, 2),
    (1000, 1000, 1000, 128, 64, 32, 4),
    (64, 64, 64, 64, 64, 64, 2),
]

EXPLANATION = r"""
## Why tile size decides speed
Every output tile re-reads strips of A and B. With 16 x 16 tiles a 4096 cube matmul moves about 256 times more data than the ideal; with 128 x 128 tiles about 32 times. Arithmetic intensity rises in proportion, and a kernel above the GPU's ridge point (for example about 300 FLOP/byte for a modern data-centre GPU in 16-bit) can run at full compute speed.

## What this model leaves out
L2 cache absorbs some re-reads, which is what grouped program ordering exploits: programs that run at the same time share strips. Tile size is bounded by registers and shared memory, so the best tile is the largest that still fits with enough programs to fill the GPU.

## Decode is different
With $M = 1$ (one token) the intensity is about 1 FLOP/byte whatever the tiling: matrix-vector products are memory bound, which is why decoding speed is set by bandwidth.

## Interview follow-ups
- Why does BK not appear in the traffic formula, and what does it affect instead?
"""
