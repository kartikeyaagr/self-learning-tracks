---
id: "13.07"
course: "inference-eng-course"
module: "13"
slug: "benchmark-profile-against-pytorch"
title: "Benchmark & Profile Against PyTorch"
keyIdeas: ["Benchmark at the shapes the model actually uses: decode batch sizes and prefill lengths.", "Compare against the strongest baseline: eager PyTorch, torch.compile, and the engine's current kernel.", "Report time, achieved bandwidth or FLOPs, and percent of peak.", "Check end-to-end effect; small ops have small total impact.", "Test numerics across shapes, including edge cases."]
question: "Your Triton kernel is 3 times faster than PyTorch in a micro-benchmark but the end-to-end step improves only 2%. Why?"
answer: "The op was only a small fraction of step time, so even a large speedup on it barely moves the total (Amdahl's law). The micro-benchmark may also have used shapes or conditions (warm caches, large sizes) unlike the real model, or the real engine already used a fused or optimized version, so the baseline in the step was faster than eager PyTorch."
sources: [{"title": "Triton: testing utilities (do_bench)", "url": "https://triton-lang.org/main/python-api/triton.testing.html", "note": "Benchmark helpers."}, {"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Measure the op inside the model."}]
tags: ["benchmarking", "triton"]
---

## Micro-benchmarks at real shapes

A kernel benchmark is only useful at the shapes your workload actually produces. Use your model's real hidden sizes and a range of token counts that covers both decode and prefill:

```
shapes = [1, 8, 32, 128, 512, 4096]          # tokens per step
for M in shapes:
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16)
    t_ref = triton.testing.do_bench(lambda: reference(x))
    t_new = triton.testing.do_bench(lambda: my_kernel(x))
    print(M, t_ref * 1e3, t_new * 1e3, "µs")
```

Keep inputs realistic. Some kernels behave differently with real activation ranges, with long contexts, or with particular head counts.

## Compare against the right baselines

| Baseline | Why it matters |
| --- | --- |
| Eager PyTorch | the naive reference, easy to beat |
| torch.compile | what the compiler gives you for free |
| The engine's existing kernel | what you actually need to beat |

Beating eager PyTorch is not an achievement on its own. torch.compile often fuses the same operations, and engines ship well-tuned kernels. A new kernel is worth integrating only if it beats the strongest baseline at the shapes that matter.

## Report percent of peak

Raw times do not show how much room is left. For each shape, compute achieved bandwidth or FLOP/s and express it as a percentage of the hardware peak.

| Tokens | PyTorch (µs) | compile (µs) | engine (µs) | mine (µs) | Mine, % of peak bandwidth |
| --- | --- | --- | --- | --- | --- |
| 1 | 18 | 9 | 7 | 6 | low, launch bound |
| 128 | 25 | 12 | 10 | 8 | 70% |
| 4,096 | 210 | 110 | 95 | 80 | 88% |

At small shapes, all versions are launch bound, and percent of peak is meaningless. There, compare times and remember that CUDA graphs will hide much of the launch cost in production.

## Profile your kernel

Run Nsight Compute on your kernel at a large shape and check:

- **Memory throughput** near peak for a memory-bound kernel.
- **No uncoalesced access warnings** in the memory analysis.
- **Enough thread blocks** to fill all SMs at the shapes you care about.
- **Register spills**: if the compiler had to spill registers to local memory, the kernel is doing extra memory traffic.

## End to end

Plug the kernel into the engine behind a flag and measure step time and the system metrics on a real workload. Estimate first with Amdahl's law: if the operation is 4% of step time and your kernel is 2 times faster, expect the step to improve by about 2%.

<div class="ex">
<div class="lab">Worked example</div>

**a micro-benchmark that did not translate**

A new sampling kernel was 5 times faster than the engine's at batch 256 in isolation: 400 µs down to 80 µs.

End to end, TPOT at batch 256 improved by only 1.1%, from 27.4 ms to 27.1 ms.

The profile explained it: the old sampler ran on a separate stream and overlapped with the next step's preparation, so only part of its time was on the critical path. Always confirm the critical path before predicting the gain.

</div>

## Measure noise

Rerun end-to-end comparisons several times and report the spread. A 2% gain is real only if it exceeds run-to-run variance, which is often 1 to 3% for serving benchmarks.

## Deep dive: when the benchmark measures L2, not HBM

The H100's L2 cache is 50 MB. A kernel whose inputs fit in L2 and are reused across benchmark iterations reads them from cache:

⟦M753⟧

<div class="ex">
<div class="lab">Worked example</div>

**a 6 TB/s RMSNorm**

A benchmark runs RMSNorm on a [2048, 4096] BF16 tensor (16 MB) 100 times in a loop and reports 6 TB/s.

That is above the HBM peak, so it cannot be HBM. The 16 MB input stays in L2 between iterations.

In the real model, the previous kernel's output may or may not still be in L2. Flushing L2 between iterations (by writing a 64 MB buffer) gives 2.9 TB/s, the number to use for end-to-end estimates.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

benchmark at your real shapes against the strongest baseline, report percent of peak, profile the kernel, and then measure end to end. A large micro-benchmark win on a small operation is a small win overall.

</div>
