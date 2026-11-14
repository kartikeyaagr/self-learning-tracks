---
id: "25.03"
course: "inference-eng-course"
module: "25"
slug: "compare-gpu-price-performance"
title: "Compare GPU Price/Performance"
keyIdeas: ["Specs that matter: HBM capacity, HBM bandwidth, tensor FLOPs per precision, interconnect.", "Decode-heavy: bandwidth and capacity per dollar.", "Prefill-heavy: FLOPs per dollar in supported low precisions.", "Large models: capacity and NVLink decide how many GPUs per replica.", "Benchmark your workload on candidates; spec sheets are a starting filter only."]
question: "Which GPU specs predict decode performance and which predict prefill performance, and why does the answer to \"which GPU is cheapest\" depend on the workload?"
answer: "Decode is predicted by memory bandwidth and memory capacity (for KV and batch size); prefill by dense tensor throughput in your precision (BF16, FP8, FP4). A decode-heavy workload favors GPUs with the best bandwidth per dollar and enough memory; a prefill-heavy RAG workload favors the best FLOPs per dollar. So the cheapest GPU changes with prompt and output lengths."
sources: [{"title": "NVIDIA H100 product page", "url": "https://www.nvidia.com/en-us/data-center/h100/", "note": "Specs for a common datacenter GPU."}, {"title": "Modal GPU Glossary", "url": "https://modal.com/gpu-glossary", "note": "Explanations of GPU specs."}]
tags: ["hardware", "cost"]
---

## Map specs to workload

| Spec | What it drives |
| --- | --- |
| HBM capacity | which models fit, KV capacity, batch size |
| HBM bandwidth | decode speed |
| Tensor FLOPs (FP8, FP4) | prefill speed, large-batch decode |
| NVLink or other interconnect | tensor-parallel efficiency |
| Price per hour | the denominator for all of the above |

## Ratios to compute

```
bandwidth_per_dollar = TB/s ÷ ($/hour)
flops_per_dollar     = TFLOP/s ÷ ($/hour)
memory_per_dollar    = GB ÷ ($/hour)
```

~~~dg-raw
%%widget gpuval Switch between the three ratios and watch the ranking change. Then apply a discount to H100 and H200, as a long-term commitment might.
%%
~~~

## The workload decides

| Workload | Favored ratio |
| --- | --- |
| Chat (short input, long output) | bandwidth and memory per dollar |
| RAG (long input, short output) | FLOPs per dollar |
| 70B and larger models | capacity and NVLink |
| Edge or small models | cheaper GPUs, including consumer cards |

<div class="ex">
<div class="lab">Worked example</div>

**shortlisting for an 8B chat service**

Decode dominates, so bandwidth per dollar matters most. At the illustrative prices, the RTX 4090 leads, followed by the MI300X, H200, A100 and H100. The L40S comes last, because its GDDR memory is much slower than HBM.

But a 24 GB card holds an 8B BF16 model with only a few GB left for KV, which limits batch size, and consumer cards are often not allowed in data centers.

A reasonable shortlist is A100, H100 and MI300X, if your engine supports it well. Benchmark all three with your harness before deciding, since real bandwidth and software maturity differ from the spec sheet.

</div>

## Then benchmark

Ratios only shortlist. Run your harness on the two or three best candidates and compare goodput per dollar under your SLO.

## Deep dive: ratios that pick the GPU

Decode is priced by bandwidth, prefill by FLOPs, and capacity by memory. Divide each by the hourly price:

| GPU | BW (TB/s) | BF16 dense (TFLOP/s) | Memory (GB) | Example \$/h | GB/s per \$ | TFLOP/s per \$ |
| --- | --- | --- | --- | --- | --- | --- |
| H100 SXM | 3.35 | 989 | 80 | 2.50 | 1,340 | 396 |
| H200 | 4.8 | 989 | 141 | 3.50 | 1,371 | 283 |
| A100 80 GB | 2.0 | 312 | 80 | 1.50 | 1,333 | 208 |
| L40S | 0.86 | 362 | 48 | 1.00 | 864 | 362 |

The prices are placeholders: use your own quotes. With these, H100, H200 and A100 are close per dollar for decode, H200 adds memory for long contexts or bigger models, and L40S is a prefill-per-dollar option that is weak at decode.

<div class="take">
<div class="lab">Key takeaway</div>

decode buys bandwidth, prefill buys FLOPs, and big models buy capacity. Compute per-dollar ratios to shortlist, then benchmark goodput per dollar.

</div>
