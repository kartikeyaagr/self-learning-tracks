import math
import numpy as np

META = dict(title='Capstone: simulate a fused matmul, bias and ReLU kernel and place it on the roofline', tags=['GPU Programming'], diff='Hard', runtime='numpy',
            related=['grouped-program-id-matmul', 'masked-edge-block-load', 'tiled-matmul-memory-traffic', 'analyze-prefill-and-decode-with-roofline-model', 'fused-matmul-bias-relu'])

STATEMENT = r"""
Simulate a Triton-style kernel for $C = \mathrm{relu}(AB + \text{bias})$ with `A (M, K)`, `B (K, N)`, `bias (N,)`. Write `run_kernel(A, B, bias, BM, BN, BK, group_m, peak_flops, bandwidth, bytes_per)`.

1. **Program ids.** `num_m = ceil(M/BM)`, `num_n = ceil(N/BN)`, one program per output tile, `pid = 0 .. num_m*num_n - 1`, mapped with grouped ordering: `in_group = group_m * num_n`, `first_m = (pid // in_group) * group_m`, `size_m = min(num_m - first_m, group_m)`, `pid_m = first_m + (pid % in_group) % size_m`, `pid_n = (pid % in_group) // size_m`.
2. **Main loop.** For `k0` in steps of `BK`, load the `BM x BK` block of A and the `BK x BN` block of B with bounds masks (padding 0), and accumulate their product.
3. **Epilogue.** Add `bias` for the tile's columns, apply ReLU, and store only the valid part of the tile.
4. **Accounting.** Count valid elements loaded from A, B and bias, plus valid elements stored to C. Bytes = count × `bytes_per`. FLOPs = $2MNK$. Intensity = FLOPs / bytes. Attainable = $\min(\text{peak\_flops}, \text{intensity} \times \text{bandwidth})$; the kernel is `"memory"` bound if intensity × bandwidth < peak, else `"compute"`.

Return `{"C", "order": [[pid_m, pid_n] for each pid], "bytes", "intensity", "attainable", "bound"}`.
"""

STARTER = '''import math
import numpy as np

def run_kernel(A, B, bias, BM, BN, BK, group_m, peak_flops, bandwidth, bytes_per):
    """Simulate a tiled fused matmul kernel and its memory traffic."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return {"C": np.maximum(A @ B + bias, 0), "order": [], "bytes": 0, "intensity": 0.0, "attainable": 0.0, "bound": "memory"}
'''

SOLUTION = '''import math
import numpy as np

def run_kernel(A, B, bias, BM, BN, BK, group_m, peak_flops, bandwidth, bytes_per):
    A, B, bias = (np.asarray(x, float) for x in (A, B, bias))
    M, K = A.shape; N = B.shape[1]
    num_m, num_n = math.ceil(M / BM), math.ceil(N / BN)
    C = np.zeros((M, N)); order = []; loaded = stored = 0
    in_group = group_m * num_n
    for pid in range(num_m * num_n):
        first_m = (pid // in_group) * group_m
        size_m = min(num_m - first_m, group_m)
        pm = first_m + (pid % in_group) % size_m
        pn = (pid % in_group) // size_m
        order.append([pm, pn])
        rm, rn = pm * BM + np.arange(BM), pn * BN + np.arange(BN)
        vm, vn = rm < M, rn < N
        acc = np.zeros((BM, BN))
        for k0 in range(0, K, BK):
            rk = k0 + np.arange(BK); vk = rk < K
            a = np.zeros((BM, BK)); ma = vm[:, None] & vk[None, :]
            a[ma] = A[np.ix_(rm[vm], rk[vk])].ravel()
            b = np.zeros((BK, BN)); mb = vk[:, None] & vn[None, :]
            b[mb] = B[np.ix_(rk[vk], rn[vn])].ravel()
            loaded += int(ma.sum() + mb.sum())
            acc += a @ b
        bb = np.zeros(BN); bb[vn] = bias[rn[vn]]; loaded += int(vn.sum())
        out = np.maximum(acc + bb, 0)
        C[np.ix_(rm[vm], rn[vn])] = out[np.ix_(vm, vn)]
        stored += int(vm.sum() * vn.sum())
    nbytes = (loaded + stored) * bytes_per
    inten = 2 * M * N * K / nbytes
    att = min(peak_flops, inten * bandwidth)
    return {"C": C, "order": order, "bytes": nbytes, "intensity": inten, "attainable": att,
            "bound": "memory" if inten * bandwidth < peak_flops else "compute"}
'''

HINTS = [
    'Compute pid_m and pid_n first and check the order on a small grid by hand: with group_m = 1 it is plain row-major.',
    'Masked loads should place real values in the top-left of the block and leave zeros elsewhere.',
    'The bias is loaded once per program, not once per K step.',
    'Check C against np.maximum(A @ B + bias, 0) before looking at the accounting.',
]

def _case(seed, M, K, N):
    r = np.random.default_rng(seed)
    return r.normal(size=(M, K)), r.normal(size=(K, N)), r.normal(size=N)

TESTS = [
    _case(1, 8, 8, 8) + (4, 4, 4, 1, 1e12, 1e11, 2),
    _case(2, 10, 7, 9) + (4, 4, 4, 2, 1e12, 1e11, 2),
    _case(3, 16, 32, 16) + (8, 8, 16, 2, 1e12, 1e12, 2),
    _case(4, 1, 64, 32) + (16, 16, 32, 1, 1e12, 1e11, 2),
    _case(5, 12, 12, 20) + (4, 8, 8, 3, 5e11, 1e11, 4),
]

EXPLANATION = r"""
## What the simulation teaches
A GPU matmul kernel is three ideas: a mapping from program ids to output tiles, a K loop of masked block loads feeding a block product, and an epilogue fused before the store. Fusing bias and ReLU into the epilogue saves a full round trip of C through memory, which for small K is a large share of the total traffic.

## Grouped ordering
Programs run in waves. Row-major order sweeps across all of B for each row of tiles, so B rarely stays in L2. Grouping `group_m` rows of tiles together makes neighbouring programs share both A and B strips, raising the effective cache hit rate without changing the arithmetic.

## The roofline verdict
Intensity times bandwidth below peak means the kernel waits on memory; above it, on arithmetic. Small and skinny matmuls (the $M = 1$ case) are memory bound whatever you do; large square ones become compute bound once tiles are big enough.

## Writing it for real
In Triton the same structure is about 40 lines: `tl.program_id`, `tl.arange` offsets, masked `tl.load`, `tl.dot` in the loop, and a masked `tl.store`. Time it against `torch.matmul` across sizes and plot achieved TFLOP/s against the roofline.

## Interview follow-ups
- Why accumulate in fp32 even when A and B are fp16?
"""
