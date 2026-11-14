---
id: "13.05"
course: "inference-eng-course"
module: "13"
slug: "write-softmax-rmsnorm"
title: "Write Softmax & RMSNorm"
keyIdeas: ["Softmax and RMSNorm are row-wise reductions followed by elementwise work.", "Fusing them means one read and one write per element.", "Compute in FP32 inside the kernel for stability, store in BF16.", "Subtract the row max in softmax.", "Fuse residual add into RMSNorm for an extra saving."]
question: "Why is a fused RMSNorm kernel faster than PyTorch's sequence of x.pow(2).mean(), rsqrt, multiply, multiply?"
answer: "The unfused version launches several kernels and writes and reads intermediate tensors to HBM between them, moving the data several times. A fused kernel loads each row once, computes the mean square, scale and gain multiply in registers, and writes the output once, so it moves about the minimum bytes and launches one kernel."
sources: [{"title": "Triton tutorial: Fused softmax", "url": "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html", "note": "Fused row-wise softmax."}, {"title": "Triton tutorial: Layer normalization", "url": "https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html", "note": "Norm kernel reference."}]
tags: ["triton", "lab"]
---

## Goal

Write two kernels that every inference engine needs: a numerically stable softmax and a fused residual add plus RMSNorm. Both are row-wise and memory bound, and both gain a lot from fusion.

## Fused softmax

```
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)          # stability: largest value becomes 0
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + offs, y.to(tl.bfloat16), mask=mask)
```

Subtracting the maximum keeps `exp` from overflowing: without it, a logit of 100 gives exp(100) ≈ 2.7 × 10⁴³, which overflows FP16 and loses precision in FP32. Masked positions load as minus infinity, so they contribute exp(−∞) = 0.

PyTorch's eager softmax makes several passes over the data in some cases. This kernel reads each element once and writes it once.

## Fused residual add and RMSNorm

In a pre-norm transformer, every sublayer is followed by a residual add and the next sublayer's norm. Fusing them saves a full pass over the hidden state:

```
@triton.jit
def add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + row * n + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(res_ptr + row * n + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(res_ptr + row * n + offs, x.to(tl.bfloat16), mask=mask)  # new residual
    rms = tl.sqrt(tl.sum(x * x, axis=0) / n + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    tl.store(out_ptr + row * n + offs, (x / rms * w).to(tl.bfloat16), mask=mask)
```

The kernel reads the sublayer output and the residual, writes the updated residual for the next layer, and writes the normalized output for the next sublayer. The sum of squares is computed in FP32 even though inputs and outputs are BF16.

## Bytes moved

| Version | HBM passes over the row |
| --- | --- |
| Unfused PyTorch: add, then square, mean, rsqrt, multiply, multiply | about 6 to 8 |
| Fused add and RMSNorm | about 3: read x, read and write the residual, write the output |

<div class="ex">
<div class="lab">Worked example</div>

**speedup on real shapes**

For 256 rows of 4,096 BF16 values on an H100, the unfused PyTorch version took about 38 µs in eager mode across several kernels.

The fused Triton kernel took about 9 µs, near the bandwidth limit for its 6 MB of traffic.

In one decode step of a 32-layer model, this pattern runs 64 times. Saving about 29 µs each time is about 1.9 ms per step, more than 10% of a 15 ms step.

</div>

## Verify against a reference

- Compute a reference in FP32 with PyTorch from the same BF16 inputs.
- Compare with `torch.testing.assert_close` at tolerances suited to BF16 output, such as `rtol=1e-2, atol=1e-2`.
- Check that the residual buffer was updated exactly, since the next layer depends on it.
- Test with large input values, for example scaled by 100, to catch overflow in the sum of squares if you accidentally accumulate in BF16.

## Measure

Compute achieved bandwidth from the bytes in the table. Compare against PyTorch eager, torch.compile and your engine's existing kernel at batch sizes 1, 32, 256 and 4,096 rows. Expect 2 to 4 times speedups over eager PyTorch on the norm itself, and a smaller margin over torch.compile, which also fuses these operations.

## Deep dive: bytes moved by an unfused softmax

Softmax over rows of length ⟦m739⟧ in eager PyTorch runs as separate kernels:

| Step | Reads | Writes |
| --- | --- | --- |
| max over the row | N | small |
| subtract max and exponentiate | N | N |
| sum | N | small |
| divide | N | N |
| Total | 4N | 2N |

A fused kernel reads each row once into registers or shared memory, does all four steps there, and writes once: ⟦m740⟧. The fused version moves a third of the bytes, so for a memory-bound op it should be close to 3 times faster, plus three fewer launches.

<div class="take">
<div class="lab">Key takeaway</div>

row-wise kernels gain by doing every step in one pass. Accumulate in FP32, store in BF16, subtract the maximum in softmax, and fuse neighbouring operations such as the residual add.

</div>
