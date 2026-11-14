---
id: "01.04"
course: "inference-eng-course"
module: "01"
slug: "watch-gpu-memory-utilization-power"
title: "Watch GPU Memory, Utilization & Power"
keyIdeas: ["Memory used in nvidia-smi includes the PyTorch caching allocator's reserve and the CUDA context, not just tensors.", "\"GPU-Util\" means a kernel was running, not that the GPU was busy. It is a weak signal.", "Power draw and clock speed are better hints of real load. A busy GPU runs near its power limit.", "DCGM metrics like SM activity and DRAM activity give a truer picture.", "Log these alongside every benchmark so you can explain results later."]
question: "nvidia-smi shows 100% GPU utilization during decode at batch size 1. Does this mean the GPU is fully used? Explain."
answer: "No. nvidia-smi's utilization counts the fraction of time at least one kernel was running, not how much of the GPU's compute or bandwidth it used. A batch-1 decode can keep a tiny kernel running constantly and show 100% while using a small fraction of the SMs and tensor cores. Use SM occupancy, achieved bandwidth or a profiler to judge real usage."
sources: [{"title": "nvidia-smi documentation", "url": "https://docs.nvidia.com/deploy/nvidia-smi/index.html", "note": "Fields and query options."}, {"title": "NVIDIA DCGM Exporter", "url": "https://github.com/NVIDIA/dcgm-exporter", "note": "GPU metrics for Prometheus, including SM activity."}]
tags: ["gpu", "monitoring"]
---

## What nvidia-smi shows

`nvidia-smi` is the first tool everyone reaches for. Used well it answers useful questions; used badly it misleads.

```
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,\
power.draw,power.limit,clocks.sm,temperature.gpu --format=csv -l 1
```

- **memory.used**: everything allocated on the GPU. That includes the CUDA context (300 to 600 MB per process), tensors, memory the PyTorch allocator has reserved but not handed out, and any pool an engine grabbed at startup.
- **utilization.gpu**: the percent of time in the sample window during which at least one kernel was running. Read that definition twice.
- **power.draw** and **power.limit**: watts now and the cap. An H100 SXM is capped around 700 W.
- **clocks.sm**: the current core clock. A falling clock under steady load points to thermal or power throttling.
- **temperature.gpu**: degrees Celsius. Sustained readings in the high 80s are a warning.

## Why GPU-Util misleads

An H100 has 132 streaming multiprocessors (SMs). A kernel that keeps 4 of them busy counts as 100% "utilization" for as long as it runs. So:

- Batch-1 decode often shows 90 to 100% GPU-Util while using a small fraction of the tensor cores.
- A server that alternates between bursts of work and idle gaps shows a middling number that hides both.

Use utilization only to answer "is the GPU idle?". It cannot answer "is the GPU busy enough?".

## Reading signals together

| What you see | Likely meaning | Next step |
| --- | --- | --- |
| High util, low power (under 50% of cap) | Small kernels, memory bound or launch bound | Profile kernel gaps, try CUDA graphs |
| High util, high power | Heavy matmul work | Check achieved TFLOP/s against the peak |
| Low util | GPU starved by CPU, Python, network or an empty queue | Check the request rate and CPU usage |
| Clocks dropping under load | Thermal or power throttling | Check cooling and the power limit |
| Memory nearly full when idle | An engine pre-allocated its KV pool | Expected, not a leak |

<div class="ex">
<div class="lab">Worked example</div>

**two servers with the same GPU-Util**

Server A runs an 8B model at batch 1. nvidia-smi shows 97% util and 260 W.

Server B runs the same model at batch 64. nvidia-smi shows 99% util and 640 W.

Both look "fully utilized". In fact server B produces about 30 times as many tokens per second. Power draw tells the real story far better than util.

</div>

## Better metrics from DCGM

NVIDIA's Data Center GPU Manager (DCGM) exposes hardware counters that measure how busy different units are. Its Prometheus exporter makes them easy to graph.

- `DCGM_FI_PROF_SM_ACTIVE`: fraction of time SMs had at least one warp resident, averaged across SMs. Unlike GPU-Util, it drops if only a few SMs are busy.
- `DCGM_FI_PROF_SM_OCCUPANCY`: how many warps were resident relative to the maximum.
- `DCGM_FI_PROF_DRAM_ACTIVE`: fraction of cycles the memory interface was busy. Near 80 to 90% during decode means you are bandwidth bound, as expected.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`: fraction of cycles the tensor cores were working. Low during decode at small batch, high during prefill.

These counters are sampled, so they give averages over time. For per-kernel detail you need a profiler such as Nsight Systems or the PyTorch profiler, covered in Module 12.

## Memory, three ways

PyTorch has a caching allocator. When you free a tensor, PyTorch keeps the memory for reuse instead of returning it to the driver. So three numbers differ:

```
torch.cuda.memory_allocated()   # bytes in live tensors
torch.cuda.memory_reserved()    # bytes held by the allocator pool
torch.cuda.max_memory_allocated()  # peak since the last reset
```

nvidia-smi shows reserved memory plus the context and any non-PyTorch allocations. A large gap between allocated and reserved is fragmentation or a pool sized for a past peak.

Inference engines go further. vLLM, for example, measures the model's peak activation use at startup, then claims a set fraction of the remaining memory for KV cache blocks (controlled by `--gpu-memory-utilization`, 0.9 by default). nvidia-smi then shows the GPU 90% full before any request arrives. That is by design.

## Power and cost

Power draw is a cost signal too. A cluster of GPUs drawing 650 W each under load costs real money in electricity and cooling. If an optimization doubles throughput while raising power by 20%, energy per token still falls by about 40%. Recording joules per token next to tokens per second is a good habit for capacity reports.

## Deep dive: what "GPU-Util" actually samples

nvidia-smi's utilization is the share of sample periods (around 1/6 to 1 second) in which at least one kernel was running on the GPU. It says nothing about how many SMs were busy or how hard:

⟦M203⟧

A single tiny kernel running all the time reads 100%. A better picture comes from DCGM's SM activity and tensor-core activity:

⟦M204⟧

~~~dg-raw
%%bars Two servers both reporting 100% GPU-Util (DCGM fields, %).
unit = %
stack = SM activity, Tensor-core activity
Decode, batch 1 | 38+4
Decode, batch 128 | 92+35
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**reading the two servers**

Both show GPU-Util 100% and similar power. Server A, at batch 1, has SM activity of 38% and tensor-core activity near 4%: it is waiting on memory, and most of the chip idles. Server B, at batch 128, has 92% and 35%.

Server B produces about 60 times more tokens per second. GPU-Util could not tell them apart.

</div>

## Deep dive: power as a sanity check

An H100 SXM has a 700 W limit. A memory-bound decode at batch 1 typically draws 250 to 350 W. A large prefill or big-batch decode pushes toward 600 to 700 W. If your "compute-bound" workload draws 300 W, something is starving the GPU. Energy per token is a cost metric of its own:

⟦M205⟧

At 650 W and 3,000 tokens per second that is about 0.22 J per token. At batch 1, 300 W and 150 tokens per second, it is 2 J per token, nine times worse.

<div class="take">
<div class="lab">Key takeaway</div>

GPU-Util only tells you whether a kernel was running. Read power, clocks and memory together, use DCGM counters for real utilization, and use a profiler when you need detail.

</div>
