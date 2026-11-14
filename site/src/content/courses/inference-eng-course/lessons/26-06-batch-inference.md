---
id: "26.06"
course: "inference-eng-course"
module: "26"
slug: "batch-inference"
title: "Batch Inference"
keyIdeas: ["Batch jobs care about throughput and cost, not per-request latency.", "Use the offline API; skip HTTP and streaming overhead.", "Max out batch size and memory; sort or group by length and shared prefix.", "Use preemptible or idle capacity; checkpoint results.", "Measure tokens per dollar for the whole job."]
question: "How should you configure an engine differently for offline batch inference compared with interactive serving?"
answer: "Drop latency SLOs and maximize throughput: very large batches (high max_num_seqs and max_num_batched_tokens), no chunked prefill concerns, sort or group inputs by length to reduce waste, maximize prefix sharing by grouping requests with common prefixes, use the offline engine API instead of HTTP, and run at full GPU memory utilization. Checkpoint progress for restartability."
sources: [{"title": "vLLM: Offline inference", "url": "https://docs.vllm.ai/en/latest/serving/offline_inference.html", "note": "Python API for batch jobs."}, {"title": "OpenAI Batch API guide", "url": "https://platform.openai.com/docs/guides/batch", "note": "Asynchronous batch interface design."}]
tags: ["batch"]
---

## Configuration

| Setting | Interactive | Batch |
| --- | --- | --- |
| SLO | TTFT and TPOT | job completion time |
| max_num_seqs | moderate | high |
| Token budget per step | bounded to protect ITL | large |
| Interface | HTTP streaming | offline Python API |
| GPU memory utilization | safe margin | as high as it is stable |

With no latency SLO, batch inference can run the largest batch the KV cache allows. Throughput per GPU is often 2 to 4 times higher than in interactive serving of the same model.

~~~dg-raw
%%widget batch Push the batch size up. In batch inference you can go much further along this curve than an interactive SLO allows.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
ctx = 10
b = 7
%%
~~~

## Ordering inputs

- **Group by shared prefix** to maximize cache hits: for example, all questions about the same document together.
- **Mix long and short** inputs to keep batches full, or **sort by length** to reduce the tail at the end of a job, when only a few long requests are still running.

## Operations

- Shard input files and process shards independently.
- Write results incrementally, so a preempted job resumes instead of restarting.
- Run on off-peak or spot capacity.

```
from vllm import LLM, SamplingParams
llm = LLM(model=path, max_num_seqs=512, gpu_memory_utilization=0.95)
outs = llm.generate(prompts, SamplingParams(max_tokens=256))
```

<div class="ex">
<div class="lab">Worked example</div>

**cost of a labeling job**

10 million documents, 800 prompt tokens and 50 output tokens each.

An 8B model on one H100 in offline mode processes about 25,000 prompt tokens and 4,000 output tokens per second together, about 30 documents per second.

10 million ÷ 30 ≈ 93 GPU-hours. At \$3 per hour that is about \$280, or about \$140 on spot capacity at half the price.

</div>

## Deep dive: GPUs for a batch job with a deadline

A job of ⟦m1120⟧ output tokens, run at the throughput-optimal batch with per-GPU throughput ⟦m1121⟧, needs

⟦M1119⟧

GPUs to finish within deadline ⟦m1122⟧. For 2 billion output tokens on an 8B model at 13,000 tokens/s per GPU and a 24-hour deadline: ⟦m1123⟧, so 2 GPUs. Any spare capacity in the serving fleet's off-peak hours (25.04) can often absorb such a job at no extra cost.

<div class="take">
<div class="lab">Key takeaway</div>

batch inference trades latency for throughput: big batches, prefix grouping, offline APIs and cheap capacity. Measure cost per job.

</div>
