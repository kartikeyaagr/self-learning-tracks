---
id: "23.06"
course: "inference-eng-course"
module: "23"
slug: "ray-serve-llm"
title: "Ray Serve LLM"
keyIdeas: ["Ray Serve is a Python serving framework on Ray clusters.", "Its LLM APIs wrap engines like vLLM as autoscaled deployments.", "Handles replica placement, routing, autoscaling and multi-model apps.", "Useful when you already run Ray or need Python-level composition.", "Still need engine-level tuning; Ray manages replicas, not kernels."]
question: "What does Ray Serve add on top of an inference engine like vLLM?"
answer: "It provides the serving layer: deployments with replicas and autoscaling, request routing between them, composition of multiple models or steps in one application, multi-node placement of engine replicas (including tensor and pipeline parallel across nodes), and an OpenAI-compatible front end, all in Python on a Ray cluster."
sources: [{"title": "Ray Serve LLM documentation", "url": "https://docs.ray.io/en/latest/serve/llm/index.html", "note": "Serving LLMs with Ray Serve."}, {"title": "Ray Serve documentation", "url": "https://docs.ray.io/en/latest/serve/index.html", "note": "Core serving concepts."}]
tags: ["ray", "orchestration"]
---

## What Ray Serve provides

| Layer | Ray Serve's role |
| --- | --- |
| API | OpenAI-compatible ingress |
| Routing | requests to replicas, model multiplexing |
| Replicas | engine processes placed on GPUs across nodes |
| Autoscaling | based on ongoing requests per replica |
| Composition | chains of models and Python logic |

## Typical setup

1. Define an LLM config: model, engine arguments and autoscaling bounds.
2. Build an application from it.
3. Deploy it to a Ray cluster. Ray places each replica on GPUs, including replicas that span several GPUs or nodes.

```
llm_config = LLMConfig(
    model_loading_config={"model_id": "llama-3-8b", "model_source": "meta-llama/Meta-Llama-3-8B-Instruct"},
    engine_kwargs={"max_model_len": 16384, "kv_cache_dtype": "fp8"},
    deployment_config={"autoscaling_config": {"min_replicas": 2, "max_replicas": 8}},
)
app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app)
```

The exact API changes between Ray versions, so check the version you pin.

## When to choose it

- Your team already uses Ray for training or data processing.
- You need Python-native pipelines, such as retrieve, rerank and generate, in one application.
- You want autoscaling and multi-node placement without building them on Kubernetes yourself.

## What it does not do

It does not make the engine faster. The scheduler, quantization and kernels underneath still set each replica's capacity. Ray Serve decides how many replicas run and where requests go.

<div class="take">
<div class="lab">Key takeaway</div>

Ray Serve is an orchestration layer for engines: replicas, routing, autoscaling and composition. Tune the engine underneath as usual.

</div>
