import math

META = dict(title='Count memory reads and writes: standard attention versus FlashAttention', tags=['Attention', 'GPU Programming'], diff='Medium', runtime='python',
            related=['flashattention-with-online-softmax'])

STATEMENT = r"""
Count elements moved to and from GPU main memory (HBM) for one head with sequence length $N$ and head dim $d$, forward pass only. Write `attention_traffic(N, d, M, bytes_per)` where `M` is the on-chip SRAM size in elements.

- **Standard**: read Q, K ($2Nd$); write $S = QK^\top$ ($N^2$); read S, write $P = \mathrm{softmax}(S)$ ($2N^2$); read P and V ($N^2 + Nd$); write O ($Nd$). Total $4Nd + 4N^2$.
- **FlashAttention-1** (outer loop over K/V blocks): $B_c = \lceil M / 4d \rceil$, $B_r = \min(B_c, d)$, $T_c = \lceil N / B_c \rceil$. K and V are read once ($2Nd$); for every K/V block, Q, O, and the row statistics $\ell, m$ are read and written back: $T_c(Nd + 2Nd + 4N)$.
- **FlashAttention-2** (outer loop over Q blocks): $T_r = \lceil N / B_r \rceil$. Q read once and O written once ($2Nd$); K and V read once per Q block ($2NdT_r$); one log-sum-exp written per row ($N$).

Return a dict from `"standard"`, `"fa1"`, `"fa2"` to `[elements, bytes]`, plus `"Bc"`, `"Br"` and `"speedup_fa2"` = standard elements / fa2 elements.
"""

STARTER = '''import math

def attention_traffic(N, d, M, bytes_per):
    """HBM traffic for three attention implementations."""
    std = 4 * N * d + 4 * N * N
    return {"standard": [std, std * bytes_per]}
'''

SOLUTION = '''import math

def attention_traffic(N, d, M, bytes_per):
    Bc = math.ceil(M / (4 * d))
    Br = min(Bc, d)
    Tc, Tr = math.ceil(N / Bc), math.ceil(N / Br)
    std = 4 * N * d + 4 * N * N
    fa1 = 2 * N * d + Tc * (3 * N * d + 4 * N)
    fa2 = 2 * N * d + 2 * N * d * Tr + N
    return {"standard": [std, std * bytes_per], "fa1": [fa1, fa1 * bytes_per], "fa2": [fa2, fa2 * bytes_per],
            "Bc": Bc, "Br": Br, "speedup_fa2": std / fa2}
'''

HINTS = [
    'The standard version is dominated by the N^2 score and probability matrices going to HBM and back.',
    'FlashAttention trades N^2 traffic for re-reading inputs once per tile; with large SRAM the number of tiles is small.',
    'In FA-1, O and the statistics are read and written for every K/V block, which is the inefficiency FA-2 removes.',
]

TESTS = [(1024, 64, 24576, 2), (4096, 64, 24576, 2), (16384, 128, 49152, 2), (512, 128, 8192, 2), (8192, 64, 100000, 2)]

EXPLANATION = r"""
## Why attention is memory bound
The arithmetic of attention is $O(N^2 d)$, but the standard implementation also moves $O(N^2)$ elements through HBM, and with $d$ around 64 to 128 the GPU spends most of its time waiting for memory. FlashAttention never writes $S$ or $P$: each tile's scores live in SRAM, are consumed by the online softmax, and are discarded.

## Reading the counts
FlashAttention's traffic is $O(N^2 d^2 / M)$: K and V are re-read once per query block, so the $N^2$ term is divided by the block size $B_r$. With the paper's block sizes and $d = 64$ that works out to about half the elements of the standard version; larger on-chip memory or larger query blocks cut it further. With a tiny SRAM (the fourth test) re-reading inputs costs more than writing $S$ and $P$, which is why block sizes are tuned per GPU.

The counts also hide two other wins: the standard version needs several kernel launches and writes its $N^2$ intermediates, while the fused kernel only streams reads, and its extra memory is $O(N)$ instead of $O(N^2)$. FA-2's loop order removes the repeated reads and writes of the output and statistics, visible in the third test.

## Interview follow-ups
- Why does the backward pass recompute $S$ and $P$ instead of storing them, and what does that cost?
"""
