---
id: "27.01"
course: "inference-eng-course"
module: "27"
slug: "map-the-vllm-codebase"
title: "Map the vLLM Codebase"
keyIdeas: ["Read the architecture docs first, then the code in request order.", "Key areas: entrypoints, engine/engine core, scheduler, KV cache manager, worker/model runner, attention backends, models, csrc kernels.", "Use a debugger or logging to follow one request.", "Note version: file layout changes between releases.", "Keep a personal map of files and their roles."]
question: "Which top-level areas of the vLLM repository would you read first to understand request handling, and why?"
answer: "The entrypoints (OpenAI-compatible API server) to see how requests arrive; the engine and engine core (the main loop and process structure); the core scheduler and KV cache manager (which requests run and where their KV goes); the worker and model runner (input preparation, forward pass, CUDA graphs); and the attention backends and model definitions. That order follows a request."
sources: [{"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Source code."}, {"title": "vLLM: Architecture overview", "url": "https://docs.vllm.ai/en/latest/design/arch_overview.html", "note": "Official map of components."}]
tags: ["vllm", "internals"]
---

## Reading order

Read an engine in the order a request travels through it. For vLLM, the areas are roughly these, from the outside in:

~~~dg-layers vLLM code areas in request order (names change between versions).
Entrypoints | API server, OpenAI schema, chat templates
Engine and engine core | main loop, process boundaries, input and output processing
*Core scheduler | waiting and running queues, token budget, preemption
KV cache manager | block pool, prefix cache, allocation
Worker and model runner | batch inputs, forward pass, sampling, CUDA graphs
Attention backends | kernel dispatch per backend
Model executor and models | model definitions, weight loading, layers
csrc | custom CUDA kernels
~~~

The scheduler and the KV cache manager are the heart of the engine. They are plain Python, a few thousand lines together, and they explain most of the latency behavior you see in benchmarks.

## How to read

1. **Pin a release tag** and check it out. Engines change fast; a map of main from three months ago may be wrong.
2. **Read the architecture and design docs** first. They name the components and say which process each runs in.
3. **Start a server with debug logging** and send one request.
4. **Set breakpoints** in the scheduler's schedule function and in the model runner's execute function. Step through one prefill and one decode.
5. **Write your own map:** file, main class, what it owns, what it calls.

<div class="ex">
<div class="lab">Worked example</div>

**a map entry**

File: the core scheduler module. Main class: Scheduler.

Owns: the waiting queue, the running list, the token budget for each step.

Calls: the KV cache manager to allocate and free blocks.

Called by: the engine core loop, once per step.

Produces: a scheduler output listing which requests run this step and how many tokens each.

</div>

## Tips

- **Read the tests.** They show intended behavior in small, runnable examples.
- **Search for metric names** to find where a number on your dashboard is computed.
- **Follow config objects** to see which flags affect which component. A flag in the CLI often passes through three or four config classes before it is used.

<div class="take">
<div class="lab">Key takeaway</div>

read an engine in the order a request travels through it. Pin a version, map files to roles, and confirm the map with a debugger.

</div>
