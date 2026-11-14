---
id: "20.04"
course: "inference-eng-course"
module: "20"
slug: "cpu-ssd-remote-kv-offload"
title: "CPU, SSD & Remote KV Offload"
keyIdeas: ["GPU HBM is small; CPU RAM is 5 to 20 times larger; SSD larger still.", "Offload KV that may be reused: past turns, popular documents, preempted requests.", "Reloading must beat recomputation: compare transfer time with prefill time.", "Bandwidth: PCIe Gen5 ~50 GB/s per GPU; NVLink-C2C on Grace systems much higher; SSD ~5 to 10 GB/s per drive.", "Remote KV stores share caches across replicas over the network."]
question: "When does offloading KV to CPU memory make sense, and what limits it?"
answer: "When KV could be reused later (multi-turn chats, repeated documents, preempted requests) but there is not enough GPU memory to keep it; recomputing a long prefix costs more than reloading it. It is limited by PCIe (or NVLink-C2C) bandwidth for moving KV back, by CPU memory capacity, and by the latency it adds to TTFT if loading is not overlapped with compute."
sources: [{"title": "LMCache", "url": "https://github.com/LMCache/LMCache", "note": "KV cache offload and sharing layer for engines."}, {"title": "Mooncake", "url": "https://arxiv.org/abs/2407.00079", "note": "KV cache pooled across CPU memory and SSD."}]
tags: ["kv-offload"]
---

## The tiers

| Tier | Capacity (rough) | Bandwidth to GPU |
| --- | --- | --- |
| HBM | 80 to 192 GB per GPU | about 3 to 8 TB/s |
| CPU DRAM | 0.5 to 2 TB per node | about 50 GB/s per GPU over PCIe Gen5 |
| Local NVMe SSD | several TB per node | about 5 to 25 GB/s |
| Remote store | very large | limited by the network, about 10 to 50 GB/s |

Each step down offers about 10 times more capacity at 10 to 100 times less bandwidth.

## Reload or recompute

When a request needs a prefix whose KV has been evicted from the GPU, there are two ways to get it back:

```
reload_time    = kv_bytes / link_bandwidth + fixed_overhead
recompute_time ≈ prefill time for that prefix
```

~~~dg-raw
%%widget offload Move the prefix length and the tier. Find where reloading starts to beat recomputing, and see how far apart they get at long prefixes.
model = Llama 3 8B
t = 15
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 32k-token prefix on an 8B model**

KV = 32k × 128 KB = 4 GB. Over PCIe at 50 GB/s, reloading takes about 80 ms.

Recomputing 32k tokens of prefill on an H100 takes about 1.6 s.

Reloading wins by about 20 times. Even from NVMe at 12 GB/s it takes about 0.35 s, still several times faster.

</div>

For short prefixes the two are within a few milliseconds of each other. Many engines simply recompute them: a short prefill batches cheaply with other work, while a reload uses link bandwidth and adds bookkeeping. Larger models make reloading more attractive, because their prefill costs more FLOPs per token while their KV per token grows more slowly.

## Integration

Engines add offload through connectors, for example LMCache with vLLM:

1. **On eviction,** the KV blocks are written to CPU memory or disk instead of being discarded.
2. **On a new request,** the engine looks up its prefix in every tier.
3. **On a hit in a lower tier,** the blocks are loaded back to HBM before or during the step that needs them.

## Deep dive: restore time by tier

For a cached prefix of ⟦m972⟧ tokens, restoring its KV from a tier with bandwidth ⟦m973⟧ costs ⟦m974⟧. Recomputing it costs prefill:

⟦M971⟧

~~~dg-raw
%%chart Time to restore a prefix for Llama 3 8B (128 KB per token), by source.
x = 4096, 16384, 32768, 131072
logx = 1
logy = 1
xlab = Prefix length (tokens)
ylab = ms
Recompute on the GPU: 110, 527, 1288, 10782
Network storage, 2 GB/s: 268, 1074, 2147, 8590
Local NVMe, 7 GB/s: 77, 307, 614, 2454
CPU memory over PCIe, 50 GB/s: 10.7, 43, 86, 344
%%
~~~

Recompute grows with ⟦m975⟧, reloads with ⟦m976⟧. For short prefixes, recomputing can beat slow storage. For long ones, every tier wins, and CPU memory wins by 30 times at 128k.

<div class="take">
<div class="lab">Key takeaway</div>

offload KV to bigger, slower tiers when reuse is likely and reloading beats recomputing. Long prefixes almost always favor reloading.

</div>
