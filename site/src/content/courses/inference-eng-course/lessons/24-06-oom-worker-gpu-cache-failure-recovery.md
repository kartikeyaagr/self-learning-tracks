---
id: "24.06"
course: "inference-eng-course"
module: "24"
slug: "oom-worker-gpu-cache-failure-recovery"
title: "OOM, Worker, GPU & Cache Failure Recovery"
keyIdeas: ["Failure types: OOM, engine hang, GPU hardware fault, NCCL failure, node loss, cache tier failure.", "Detect fast: liveness, deep health checks, GPU error metrics.", "Contain: remove from routing, fail in-flight requests as retryable.", "Recover: restart, reschedule on healthy GPUs, cordon bad nodes.", "Prevent: bound memory use, test long inputs, track recurring faults."]
question: "A replica crashes with CUDA OOM during a long prompt. Describe immediate and long-term responses."
answer: "Immediate: the orchestrator restarts the replica, the router stops sending to it while it is down, and in-flight requests fail with a retryable error so clients retry elsewhere. Long-term: find the triggering request (prompt length, batch composition), cap max prompt length or max batched tokens to bound activation memory, lower gpu memory utilization if needed, and add a test that reproduces the case."
sources: [{"title": "NVIDIA Xid errors", "url": "https://docs.nvidia.com/deploy/xid-errors/index.html", "note": "GPU error codes."}, {"title": "PyTorch: Understanding CUDA Memory Usage", "url": "https://pytorch.org/docs/stable/torch_cuda_memory.html", "note": "Diagnosing OOMs."}]
tags: ["reliability", "failures"]
---

## Failure catalogue

| Failure | How you detect it | How you recover |
| --- | --- | --- |
| CUDA out of memory | crash, error log | restart; cap prompt and batch tokens |
| Engine hang | deep health check times out | kill and restart |
| GPU Xid or ECC fault | DCGM, kernel log | cordon the node, move the replica |
| NCCL timeout (multi-GPU) | engine error | restart the whole TP group |
| Node lost | orchestrator | reschedule |
| KV tier unavailable | connector errors | fall back to recomputing |

## Blast radius

- **TP groups fail together.** One bad GPU takes down a replica spread over 8 GPUs.
- **Routers must drop failed replicas quickly,** within seconds, or they keep sending traffic into errors.
- **Cache tiers should fail open.** If the KV tier goes down, you lose cache hits, not requests.

<div class="ex">
<div class="lab">Worked example</div>

**one GPU, eight GPUs' worth of impact**

A 70B model runs as TP 8 replicas on 4 nodes. One GPU starts reporting Xid 79 at 03:10.

The NCCL all-reduce in that replica hangs, and after a 10-minute NCCL timeout the engine exits. All 8 GPUs of that replica are out, a quarter of the fleet.

If the router took 60 s to notice, about 25% of requests in that minute failed. A deep health check every 10 s and a shorter NCCL timeout cut detection to seconds. Cordoning the node stops the replica restarting on the bad GPU.

</div>

## Postmortem habits

Record the trigger, the timeline, the user impact and the fix. Repeated GPU faults on one node mean the hardware should be replaced, not restarted again.

## Deep dive: blast radius in numbers

If one GPU fails with probability ⟦m1081⟧ per day, a group of ⟦m1082⟧ GPUs that must all work fails with probability

⟦M1080⟧

With ⟦m1083⟧ per day, a TP 8 replica is down on about 0.8% of days, eight times as often as a single-GPU replica, and each failure removes 8 GPUs of capacity at once. Ten TP 8 replicas lose a replica about every 13 days on average. Spare capacity should cover at least one whole replica, not one GPU.

<div class="take">
<div class="lab">Key takeaway</div>

list your failure modes, detect each quickly, fail requests in a retryable way, and remove the root cause so it does not recur.

</div>
