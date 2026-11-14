---
id: "13.04"
course: "inference-eng-course"
module: "13"
slug: "write-vector-reduction-kernels"
title: "Write Vector & Reduction Kernels"
keyIdeas: ["Start with a vector add and a row-sum to learn load, store, masks and grids.", "Measure achieved bandwidth = bytes moved / time and compare with peak.", "Check correctness against PyTorch with torch.testing.assert_close.", "Use triton.testing.do_bench for stable timings.", "Elementwise and reduction kernels are memory bound; bandwidth is the goal."]
question: "For a vector add kernel on 100 million BF16 elements, what is the best possible time on a 3.35 TB/s GPU and how do you check whether your kernel achieves it?"
answer: "It reads two arrays and writes one: 3 x 100M x 2 bytes = 600 MB. At 3.35 TB/s the minimum is about 0.18 ms. Time the kernel with CUDA events or triton.testing.do_bench and compute achieved bandwidth as bytes moved divided by time; a good kernel reaches 80 to 90% of peak."
sources: [{"title": "Triton tutorial: Vector addition", "url": "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html", "note": "First kernel."}, {"title": "Triton tutorial: Fused softmax", "url": "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html", "note": "Row reduction kernel."}]
tags: ["triton", "lab"]
---

## Goal

Write your first two Triton kernels, check them against PyTorch, and measure how close they come to peak memory bandwidth. Both are memory bound, so bandwidth is the only number that matters.

## Kernel 1: vector add

Use the `add_kernel` from lesson 13.02. Benchmark it:

```
n = 1 << 26                                # 67M elements
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
y = torch.randn_like(x)
out = torch.empty_like(x)
grid = (triton.cdiv(n, 1024),)
ms = triton.testing.do_bench(lambda: add_kernel[grid](x, y, out, n, BLOCK=1024))
gbps = 3 * n * x.element_size() / (ms * 1e-3) / 1e9    # 2 reads + 1 write
print(f"{ms:.3f} ms, {gbps:.0f} GB/s")
```

`triton.testing.do_bench` handles warm-up, repeats and synchronization, and returns a robust estimate. Count bytes carefully: every input read and every output write.

## Kernel 2: row sum

```
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offs, mask=offs < n_cols, other=0.0)
    tl.store(out_ptr + row, tl.sum(x.to(tl.float32), axis=0))

def row_sum(x):
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(cols)
    row_sum_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK=BLOCK)
    return out
```

Note the `other=0.0` in the masked load: the padded positions beyond `n_cols` must not change the sum. For a max reduction, use `other=-float("inf")`.

## Correctness checks

- Compare with PyTorch on random inputs: `torch.testing.assert_close(row_sum(x), x.float().sum(-1), rtol=1e-3, atol=1e-2)`.
- Test edge cases: a column count that is not a power of two, a single row, a single column, and very large rows.
- Test with real data ranges, including large values, not only standard normal noise.

## Sweep sizes

Run each kernel over a range of sizes and plot achieved bandwidth against size.

| Size | What to expect |
| --- | --- |
| 1k elements | a few microseconds, launch bound, tiny bandwidth |
| 1M | rising quickly |
| 10M | approaching peak bandwidth |
| 100M and more | 80 to 90% of peak |

On an H100 with 3.35 TB/s peak, a good vector add reaches about 2.8 to 3.0 TB/s at large sizes.

<div class="ex">
<div class="lab">Worked example</div>

**reading the sweep**

At 4,096 elements the add takes 4.1 µs: 24 KB of traffic in 4.1 µs is about 6 GB/s. The time is all launch overhead.

At 64M elements it takes 0.135 ms: 384 MB in 0.135 ms is about 2.85 TB/s, 85% of peak.

Decode-time elementwise ops on one token's hidden state (4,096 elements) live at the left end of this curve. That is why they are fused or captured in CUDA graphs, not optimized individually.

</div>

## Variations to try

- Change `BLOCK` from 256 to 4,096 and `num_warps` from 2 to 16. Note which settings reach peak at large sizes.
- Process two rows per program in the row sum. Does it help small row counts?
- Add autotuning and compare its choice with your manual best.

## Deep dive: why small vectors never reach peak bandwidth

A vector add moves ⟦m736⟧ bytes (two reads, one write). With a fixed launch cost ⟦m737⟧:

⟦M735⟧

⟦m738⟧ is the size that reaches half of peak. For FP32, 5 µs and 3 TB/s, it is 1.25 million elements.

~~~dg-raw
%%chart Predicted achieved bandwidth of an FP32 vector add (5 µs launch cost, 3 TB/s peak).
x = 65536, 1048576, 4194304, 16777216, 67108864
logx = 1
xlab = Elements
ylab = TB/s
Achieved: 0.15, 1.37, 2.31, 2.8, 2.95
%%
~~~

Your measured sweep should follow this shape. If the plateau is well below peak, the kernel has a problem. If the left side is lower than predicted, launch cost is higher than you assumed.

<div class="take">
<div class="lab">Key takeaway</div>

your first kernels are about bandwidth. Count bytes carefully, compute achieved GB/s, compare with peak across sizes, and check correctness against PyTorch every time, including edge cases.

</div>
