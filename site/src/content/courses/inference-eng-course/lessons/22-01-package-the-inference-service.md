---
id: "22.01"
course: "inference-eng-course"
module: "22"
slug: "package-the-inference-service"
title: "Package the Inference Service"
keyIdeas: ["Package the engine, config and service code in a pinned container image.", "Keep weights outside the image; load from a model store or cached volume.", "Configuration (model, flags) comes from environment or config files per deployment.", "Expose health, readiness and metrics endpoints.", "Build reproducibly: same image hash means same behavior."]
question: "What goes into a production inference container image, and what should stay out of it?"
answer: "In: a pinned engine version with its CUDA and PyTorch stack, your server wrapper or config, health and metrics endpoints, and startup scripts. Out: model weights (pull from a model store or mount a cache volume, so images stay small and models can change independently), secrets (inject at runtime), and environment-specific settings (pass as config)."
sources: [{"title": "NVIDIA Container Toolkit install guide", "url": "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html", "note": "GPU access from containers."}, {"title": "vLLM: Using Docker", "url": "https://docs.vllm.ai/en/latest/deployment/docker.html", "note": "Official images and usage."}]
tags: ["production", "containers"]
---

## What goes in the image

~~~dg-layers A production inference image.
Base | CUDA runtime (drivers come from the host)
Framework | pinned PyTorch build
*Engine | pinned vLLM, SGLang or TensorRT-LLM version
Service | config loader, auth, request logging, health and metrics
Entrypoint | starts the engine with flags from config
~~~

Each layer is pinned to an exact version. "Latest" anywhere in the image means two builds a week apart can behave differently, and you will not know why.

- **Base:** the CUDA runtime must be compatible with the host driver. A newer runtime on an older driver fails at start, often with an unclear error.
- **Framework and engine:** pin exact versions and record the image digest (sha256), not only the tag. Tags can be moved.
- **Service layer:** the code you own. Keep it thin: config, auth, logging, health and metrics.

## Keep weights outside the image

| Option | Pros | Cons |
| --- | --- | --- |
| Bake into the image | simple, immutable | 16 to 140 GB images, slow pulls, a rebuild per model |
| Pull from object store at start | flexible, one image for many models | slow cold start |
| Shared cache volume or local NVMe | fast restarts | cache management, disk space |

Most teams use one engine image and keep weights on fast local or shared storage, pulled once per node.

## Config

One image serves many deployments. Model id, revision, dtype, parallelism and scheduler flags all come from a config file or environment variables, stored in version control.

```
model: meta-llama/Meta-Llama-3-8B-Instruct
revision: 5f0b02c
dtype: bfloat16
tensor_parallel_size: 1
max_num_seqs: 256
max_model_len: 16384
kv_cache_dtype: fp8
enable_prefix_caching: true
```

## Endpoints

- **/health:** the process is alive.
- **/ready** (or equivalent): the model is loaded and warmed up.
- **/metrics:** Prometheus metrics from the engine and the service.

<div class="ex">
<div class="lab">Worked example</div>

**what goes wrong without pinning**

A team rebuilds its image to patch an OS package. The Dockerfile installs the engine without a version, so the rebuild picks up a new minor release.

The new release changes the default chunked-prefill token budget. TTFT p95 rises 40% at the same load. Nothing in the team's own code changed, and the cause takes a day to find.

With the engine version and image digest pinned, the patch would have changed only the OS package.

</div>

## Deep dive: why weights stay out of the image

Start time is dominated by moving bytes. Each path is limited by its slowest link:

⟦M1010⟧

<div class="ex">
<div class="lab">Worked example</div>

**a 22 GB image versus a 6 GB image plus cached weights**

Baking 16 GB of weights into a 22 GB image, pulled from a registry at about 250 MB/s and then decompressed: about 90 s to pull, plus 30 to 60 s to unpack.

A 6 GB image (about 35 s) plus weights read from a node-local NVMe cache at 3 GB/s (about 5 s): about 40 s, and the weights download only once per node, not once per image version.

The second layout also lets you change code without re-shipping 16 GB.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ship a pinned, reproducible engine image and keep weights and config outside it. Expose health, readiness and metrics from day one.

</div>
