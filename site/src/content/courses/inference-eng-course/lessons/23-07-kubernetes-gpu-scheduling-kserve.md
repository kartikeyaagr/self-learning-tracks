---
id: "23.07"
course: "inference-eng-course"
module: "23"
slug: "kubernetes-gpu-scheduling-kserve"
title: "Kubernetes, GPU Scheduling & KServe"
keyIdeas: ["GPUs are extended resources exposed by the NVIDIA device plugin.", "Pods request whole GPUs; sharing needs MIG, time slicing or MPS.", "Node labels and affinity select GPU types; topology matters for multi-GPU pods.", "KServe: InferenceService CRD, runtimes (including vLLM), autoscaling, canaries.", "LLM-specific operators (e.g. llm-d, production stacks) add cache-aware routing and disaggregation."]
question: "How does Kubernetes give a pod a GPU, and what does KServe add for model serving?"
answer: "The NVIDIA device plugin advertises GPUs as an extended resource (nvidia.com/gpu); a pod requests a whole number of them and the scheduler places it on a node with free GPUs, with the container runtime exposing those devices. KServe adds a model-serving abstraction (InferenceService) with standard runtimes, autoscaling (including scale to zero), canary rollouts and a consistent inference API on top."
sources: [{"title": "Kubernetes: Schedule GPUs", "url": "https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/", "note": "Device plugins and GPU requests."}, {"title": "KServe documentation", "url": "https://kserve.github.io/website/", "note": "InferenceService and runtimes."}]
tags: ["kubernetes"]
---

## GPUs in Kubernetes

```
resources:
  limits:
    nvidia.com/gpu: 2
nodeSelector:
  gpu-type: h100
```

- **The device plugin** reports how many GPUs each node has.
- **The scheduler** places the pod on a node with enough free GPUs.
- **The container runtime** maps the chosen GPU devices into the container.

## Sharing GPUs

| Method | Isolation | Use |
| --- | --- | --- |
| Whole GPU | full | most LLM serving |
| MIG (A100, H100) | hardware partitions with their own memory and SMs | small models, many tenants |
| Time slicing | none | development and testing |

An H100 can be split into up to 7 MIG slices. A 1g.10gb slice has 10 GB of memory, enough for a small quantized model, but each slice gets only a fraction of the bandwidth.

## Multi-GPU pods

Request all GPUs of a tensor-parallel group in one pod, so they land on one node with NVLink between them. Replicas that span several nodes need an operator that starts and coordinates a group of pods together, such as LeaderWorkerSet.

~~~dg-flow Multi-node replica with LeaderWorkerSet.
Leader pod (8 GPUs) -> *Worker pod (8 GPUs) -> Started and restarted as one group -> One replica, TP 8 and PP 2
~~~

## KServe

An InferenceService resource declares the model, the runtime and the scaling. KServe handles deployment, routing, autoscaling and canary traffic splitting. A vLLM-based runtime serves LLMs with an OpenAI-style API.

## Deep dive: what fits in a GPU slice

MIG splits an H100 80 GB into up to seven slices of about 10 GB, each with its own share of compute and bandwidth. A model fits in ⟦m1066⟧ slices when

⟦M1065⟧

An 8B model in 4-bit (about 5 GB) plus 1.5 GB of runtime leaves about 3.5 GB of KV in one slice: about 27,000 tokens in BF16. Fine for a low-traffic model; each slice also gets only about a seventh of the bandwidth, so its TPOT is about seven times what the same model would see with the whole GPU.

<div class="take">
<div class="lab">Key takeaway</div>

Kubernetes schedules whole GPUs through device plugins. KServe and LLM-specific operators add serving workflows on top.

</div>
