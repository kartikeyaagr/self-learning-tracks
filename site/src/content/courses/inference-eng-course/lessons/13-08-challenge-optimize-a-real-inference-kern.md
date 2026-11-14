---
id: "13.08"
course: "inference-eng-course"
module: "13"
slug: "challenge-optimize-a-real-inference-kernel"
title: "Challenge: Optimize a Real Inference Kernel"
keyIdeas: ["Pick a kernel from a profile, not from intuition.", "Score candidates by step share x headroom to the roofline.", "Write a Triton version, verify numerics, benchmark at real shapes.", "Integrate and measure end-to-end TPOT or TTFT.", "Write up the result, including what did not work."]
question: "How would you pick which kernel in a real engine to optimize for this challenge?"
answer: "Profile a representative workload, list kernels by share of step time, and for each estimate headroom from its roofline position (how far below its memory or compute ceiling it runs). Choose the kernel with the largest share times headroom that you can realistically improve, such as a fused norm, a sampling kernel, or a decode attention variant at your shapes."
sources: [{"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Kernels in csrc and Triton ops."}, {"title": "Triton tutorials", "url": "https://triton-lang.org/main/getting-started/tutorials/index.html", "note": "Techniques to apply."}]
tags: ["project"]
tools: ["Challenge"]
---

## The challenge

Find a kernel in a real engine that has headroom, write a faster one, and prove that the whole system got faster. This is what kernel work looks like in practice: it starts from a profile and ends with an end-to-end number.

## Step 1: profile a real workload

Use your workload and engine from earlier modules. Capture a steady-state timeline at the expected load with Nsight Systems, and group kernel time by kernel type.

## Step 2: score the candidates

For each of the top kernels, measure its share of step time and how close it runs to its roofline, using Nsight Compute.

| Kernel | Share of step | % of its roof | Headroom |
| --- | --- | --- | --- |
| Fused MoE or dense matmul | 45% | 75% | medium: large share, some room |
| Decode attention | 20% | 60% | high: good share, lots of room |
| Sampling | 4% | 20% | low: lots of room but small share |
| Add + RMSNorm | 5% | 40% | medium |

A useful score is share × (1 − fraction of roof). Decode attention at 20% × 0.4 = 0.08 beats sampling at 4% × 0.8 = 0.032, despite sampling's lower efficiency.

## Step 3: write and tune

Write the kernel in Triton. Autotune over block sizes and warp counts at the exact shapes your workload produces: real batch sizes, head counts, head dimensions and context lengths.

## Step 4: check numerics

- Compare with the reference kernel on random inputs and on activations captured from the real model.
- Check edge cases: length 1, lengths that are not multiples of the block size, the maximum context length, empty batches.
- Run greedy generation on 50 prompts with both kernels. Outputs should match, or diverge only after near-ties in the logits.

## Step 5: integrate and measure

Add the kernel behind a flag so you can switch between old and new without rebuilding. Then measure:

- Kernel time at production shapes, from a profile of the running engine.
- Step time at the expected load.
- System metrics: TPOT, TTFT and SLO-safe capacity, with several runs each.

## Rules

- Beat the engine's current kernel at the shapes your workload uses, not at a hand-picked shape.
- Report percent of peak before and after.
- Report the end-to-end change with its run-to-run spread.
- Do not change anything else between the before and after runs.

<div class="ex">
<div class="lab">Worked example</div>

**a successful entry**

Target: decode attention for a 70B model at batch 16 and 16k context, 24% of step time at 52% of peak bandwidth.

The engine's kernel used one thread block per (sequence, KV head): 16 × 8 = 128 blocks on 132 SMs, with long serial loops over each sequence.

The new kernel split each sequence's KV into 8 chunks and loaded each KV head once for its 8 query heads. It reached 86% of peak bandwidth.

Attention time fell from 5.3 ms to 3.2 ms per step. TPOT fell from 22.0 ms to 19.9 ms, 9.5%, with a run-to-run spread of 0.6%. SLO-safe capacity at the long-context workload rose 11%.

</div>

## Deliverable

A short report with the scored candidate table, the kernel code, numerical checks, kernel-level results with percent of peak, end-to-end results with spread, and a sentence on what limits the kernel now.

## Deep dive: scoring candidates before writing any code

Rank each candidate by its expected saving on the real workload, using the profile from step 1:

⟦M754⟧

| Candidate | Share of step ⟦m755⟧ | Realistic speed-up ⟦m756⟧ | Expected saving |
| --- | --- | --- | --- |
| Fuse residual add + RMSNorm | 4% | 1.5 | 1.3% |
| Fuse RoPE into QKV epilogue | 2% | 2.0 | 1.0% |
| FP8 KV attention kernel | 22% | 1.6 | 8.3% |
| Custom MLP matmul | 55% | 1.05 | 2.6% |

The attention kernel wins, even though the MLP is the largest share. The MLP already runs near peak in a vendor library, so its realistic ⟦m757⟧ is close to 1.

<div class="take">
<div class="lab">Key takeaway</div>

real kernel work starts from a profile, picks the kernel with the most share times headroom, and ends with an end-to-end number. The challenge is to move both the kernel and the step, and to prove it.

</div>
