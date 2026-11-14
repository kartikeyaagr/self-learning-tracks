---
id: "22.02"
course: "inference-eng-course"
module: "22"
slug: "model-weights-caches-cold-starts"
title: "Model Weights, Caches & Cold Starts"
keyIdeas: ["Cold start = pull image + fetch weights + load to GPU + compile/capture + warm up.", "Weight fetch usually dominates for large models.", "Local NVMe caches and streaming loaders cut it substantially.", "Compilation caches avoid recompiling on every start.", "Cold start time sets how fast you can scale up or recover."]
question: "Break down the cold start of a new 70B replica and name the biggest lever to shorten it."
answer: "Container pull (seconds to minutes), weight download (140 GB at 1 to 2 GB/s from object storage is over a minute, often several), loading weights into GPU memory (tens of seconds), compilation and CUDA graph capture (tens of seconds to minutes), and warm-up. The biggest lever is usually weight movement: keep weights on fast local NVMe or a nearby cache, use streaming loaders, and avoid re-downloading."
sources: [{"title": "Safetensors documentation", "url": "https://huggingface.co/docs/safetensors/index", "note": "Fast, memory-mapped weight loading."}, {"title": "Run:ai Model Streamer", "url": "https://github.com/run-ai/runai-model-streamer", "note": "Streams weights from storage to GPU concurrently."}]
tags: ["cold-start"]
---

## Where cold start time goes

| Phase | 8B model | 70B model |
| --- | --- | --- |
| Image pull (image cached on node) | seconds | seconds |
| Weights from object store | about 15 to 60 s | several minutes |
| Load into GPU memory | about 5 to 15 s | about 30 to 90 s |
| Compile and CUDA graph capture | about 20 to 120 s | about 30 to 180 s |
| Warm-up | seconds | seconds |

These are rough ranges. Measure your own.

~~~dg-raw
%%widget cold Start from a single object-store stream with no caches, then turn on each lever one at a time. Try the 70B model too.
model = Llama 3 8B
%%
~~~

## Levers

- **Weights:** pre-stage them on local NVMe. Use parallel or streaming downloads, which can be 5 to 10 times faster than a single stream. Load safetensors with memory mapping so the file is not copied twice.
- **Compile:** cache compile artifacts, keyed on the engine version, model and shapes. A warm compile cache can turn a minute into seconds.
- **Graph capture:** capture only the batch sizes you actually run. Capturing 67 sizes when traffic only uses 8 wastes time on every start.
- **Image:** pre-pull images on GPU nodes so a new replica does not wait for a 10 GB download.
- **Keep warm:** hold a small pool of idle but loaded replicas for bursts (Module 23).

<div class="ex">
<div class="lab">Worked example</div>

**one cold start, before and after**

A 70B model starts with weights from a single object-store stream, no compile cache and full graph capture. Download takes about 6 minutes, load about 1 minute, compile about 1.2 minutes and capture about 45 seconds. Time to ready is about 9 minutes.

With weights on local NVMe, a compile cache and 8 graph sizes: download 24 s, load 1 minute, compile 8 s, capture 5 s. Time to ready is under 2 minutes.

An autoscaler that needs 9 minutes to add capacity cannot react to a 5-minute burst. One that needs 2 minutes often can.

</div>

## Measure every start

Log a timestamp at the end of each phase on every start, and chart them over time. Cold-start regressions are common after engine upgrades, and nobody notices until an autoscaling event goes badly.

## Deep dive: a cold start as a sum

⟦M1011⟧

~~~dg-raw
%%bars One cold start for an 8B model, before and after the usual fixes (s).
unit = s
stack = Node provisioning, Image pull, Weight load, Compile and graph capture, Warm-up
Before | 90+120+60+75+15
After | 90+35+6+20+10
%%
~~~

With a warm node pool, the first term goes to zero too. Weight load is ⟦m1012⟧ over the slowest link: 60 s from object storage at 270 MB/s, 6 s from local NVMe. Compile time falls when the compiled artifacts are cached between starts. The order of fixes follows the bar: largest first.

<div class="take">
<div class="lab">Key takeaway</div>

cold start is mostly moving weights and compiling. Cache both close to the GPU, and measure each phase on every start.

</div>
