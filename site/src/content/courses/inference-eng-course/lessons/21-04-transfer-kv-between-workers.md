---
id: "21.04"
course: "inference-eng-course"
module: "21"
slug: "transfer-kv-between-workers"
title: "Transfer KV Between Workers"
keyIdeas: ["KV size = tokens x KV bytes per token: often hundreds of MB to GBs.", "Transports: NVLink inside a node, RDMA (InfiniBand/RoCE) across nodes, TCP as fallback.", "Overlap transfer with prefill by sending each layer when done.", "Layout mismatches (different TP or block sizes) require reshaping during transfer.", "Transfer time adds directly to TTFT if not overlapped."]
question: "Estimate the KV transfer time for a 4,000-token prompt of a 70B-class model (320 KB per token) over 400 Gb/s RDMA, and compare with its prefill time."
answer: "KV size is 4,000 x 320 KB = 1.28 GB. At 50 GB/s, about 26 ms. Prefill of 4,000 tokens for a 70B model on a TP-8 H100 node takes roughly 300 to 600 ms. Transfer is small relative to prefill, especially if overlapped layer by layer."
sources: [{"title": "Mooncake", "url": "https://arxiv.org/abs/2407.00079", "note": "RDMA-based KV transfer engine."}, {"title": "NVIDIA NIXL", "url": "https://github.com/ai-dynamo/nixl", "note": "Inference transfer library used for KV movement."}]
tags: ["disaggregation", "kv-transfer"]
---

## Size

```
kv_bytes = prompt_tokens × 2 × layers × kv_heads × head_dim × bytes_per_value
```

| Model | Per token | 4k prompt | 32k prompt |
| --- | --- | --- | --- |
| 8B (GQA, BF16) | 128 KB | 0.5 GB | 4 GB |
| 70B (GQA, BF16) | 320 KB | 1.3 GB | 10 GB |
| 70B, FP8 KV | 160 KB | 0.65 GB | 5 GB |

## Links

| Link | Bandwidth | 1.3 GB takes |
| --- | --- | --- |
| NVLink inside a node | about 400 GB/s or more | about 3 ms |
| RDMA at 400 Gb/s | about 50 GB/s | about 26 ms |
| TCP at 100 Gb/s | about 10 GB/s | about 130 ms |

~~~dg-raw
%%widget offload The same arithmetic applies to moving KV between workers. Pick the 70B model, the remote tier and a 4k prefix to see the size and transfer time.
model = Llama 3 70B
tier = Remote (100 Gb/s)
t = 12
%%
~~~

## Overlap

Send layer l's KV as soon as prefill finishes layer l. Most of the transfer then happens during prefill, and only the last layer's transfer is exposed.

<div class="ex">
<div class="lab">Worked example</div>

**how much transfer is exposed**

A 70B model prefills a 4k prompt in about 600 ms on TP 4, so each of its 80 layers takes about 7.5 ms.

Each layer's KV is 1.3 GB ÷ 80 ≈ 16 MB. Over RDMA at 50 GB/s, that takes about 0.3 ms, well under the 7.5 ms of compute.

Only the last layer's 0.3 ms is exposed. Without layer-wise sending, the whole 26 ms would be added to TTFT.

</div>

## Layout differences

If prefill uses TP 4 and decode uses TP 2, the KV heads are split differently on each side. Each decode GPU needs heads from two prefill GPUs. Connectors handle the reshaping, at some cost in transfer count and complexity. Matching the layouts, when possible, keeps transfers simple.

## Deep dive: transfer time against prefill time

KV to move is ⟦m1002⟧. With layer-wise sending, each layer's KV leaves while the next layer computes, so the exposed part is roughly

⟦M1001⟧

~~~dg-raw
%%chart KV transfer time versus prefill time for Llama 3 8B (128 KB per token).
x = 1024, 8192, 32768, 131072
logx = 1
logy = 1
xlab = Prompt tokens
ylab = ms
Prefill on one H100: 26, 234, 1288, 10782
Transfer, 100 Gb Ethernet: 10.7, 86, 344, 1374
Transfer, 400 Gb InfiniBand: 2.7, 21, 86, 344
Transfer, NVLink: 0.3, 2.4, 9.5, 38
%%
~~~

Even over 100 Gb Ethernet, transfer is shorter than prefill at every length, and the gap grows with context because prefill has the quadratic attention term. Only the ⟦m1003⟧ tail is exposed: about 0.3 ms at 1k over Ethernet, 43 ms at 128k.

<div class="take">
<div class="lab">Key takeaway</div>

KV transfer is gigabytes per long request but fast over RDMA or NVLink. Send it layer by layer to hide it behind prefill, and use FP8 KV to halve it.

</div>
