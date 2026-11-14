---
id: "01.02"
course: "ai-engineering-course"
module: "01"
slug: "the-ai-engineering-stack"
title: "The AI engineering stack"
keyIdeas: ["Chip Huyen splits the stack into three layers: application development, model development and infrastructure.", "Application layer: prompts, context construction, retrieval, tools, guardrails, evaluation, the user interface.", "Model layer: choosing, fine-tuning, distilling and quantizing models; datasets for adaptation.", "Infrastructure layer: inference serving, GPUs, vector databases, caching, monitoring.", "Most teams spend most of their time in the application layer and buy the other two as services.", "A production platform grows in steps: plain model call, then context, then guardrails, router, cache, and finally agent loops with write actions."]
question: "Name the three layers of the AI stack and give one component you'd find in each."
answer: "Application layer: prompts, retrieval/context construction, guardrails, evaluation (e.g. a RAG pipeline). Model layer: model selection, fine-tuning, distillation, quantization (e.g. a LoRA fine-tune). Infrastructure layer: inference serving, GPUs, vector databases, caching and monitoring (e.g. vLLM or a vector DB)."
sources: [{"title": "Chip Huyen: Building A Generative AI Platform", "url": "https://huyenchip.com/2024/07/25/genai-platform.html", "note": "Layer-by-layer walk through a production GenAI platform."}, {"title": "a16z: Emerging Architectures for LLM Applications", "url": "https://a16z.com/emerging-architectures-for-llm-applications/", "note": "Reference stack: data pipelines, embeddings, vector DBs, orchestration."}, {"title": "Chip Huyen: AI Engineering (O'Reilly, 2025)", "url": "https://www.oreilly.com/library/view/ai-engineering/9781098166298/", "note": "Chapter 1 splits the stack into application, model and infrastructure layers."}]
---

## Three layers

Chip Huyen's book describes the AI stack as three layers. Almost every tool you will hear about fits in one of them.

~~~dg-layers The AI engineering stack, top to bottom. Most product work happens at the top.
*Application | prompts, context and retrieval, tools, guardrails, evals, UI
Model | model choice, fine-tuning, distillation, quantization, datasets
Infrastructure | inference serving, GPUs, vector stores, caches, monitoring
~~~

The application layer changed the most when foundation models arrived. Before, a team needed its own model and infrastructure before it could build anything. Now a team can call a hosted model and spend all its time on the top layer.

## Growing a platform step by step

In "Building A Generative AI Platform", Huyen describes how a system grows from a single API call:

1. **Plain call**: user query goes to the model, response comes back.
2. **Enhance context**: retrieve documents or call tools and put the results in the prompt (RAG).
3. **Guardrails**: check inputs for prompt injection and personal data, check outputs for format errors, toxicity and hallucination.
4. **Router and gateway**: send easy queries to a cheap model and hard ones to a strong model; one gateway handles keys, fallbacks and rate limits.
5. **Cache**: reuse answers for repeated or similar queries, and reuse the processed prompt prefix (prompt caching).
6. **Agent patterns**: loops, branching, and tools that write (send an email, change a database).

Observability (logs, traces, metrics) and orchestration run across all of these steps.

~~~dg-flow A typical request path once a few layers are in place.
Query -(check)-> Input guardrail -(route)-> *Router -(context)-> Retriever -(prompt)-> Model -(check)-> Output guardrail -(cache)-> Answer
~~~

## Where the a16z reference fits

a16z's "Emerging Architectures for LLM Applications" draws the same picture from the data side: data pipelines feed an embedding model, embeddings go into a vector database, an orchestration layer assembles prompts, and an LLM API or hosted open model answers. Their main point is that in-context learning (putting the right data in the prompt) was the default pattern, with fine-tuning used much less often.

## Buy or build at each layer

- **Model**: closed API (fast to start, less control) vs open weights (control, privacy, more ops work).
- **Vector store**: a Postgres extension is often enough; a dedicated vector DB pays off at large scale or with heavy filtering.
- **Serving**: managed endpoints vs your own vLLM or TensorRT-LLM cluster.

## Put numbers on the stack

Each layer you add costs latency and money, and each one should earn its place. The figure below models a typical request mix. Move the cache hit rate and the share of traffic you can route to a small model, and watch which layer pays for itself.

~~~dg-raw
%%widget stack Monthly cost of a feature as you add a cache, a router, retrieval and a guardrail.
~~~

Three patterns show up again and again:

- **Caching wins first.** If 30% of questions repeat (common for FAQs and internal tools), a cache in front of the model removes 30% of the spend with almost no quality risk.
- **Routing wins big, but needs an eval.** Sending easy requests to a model that costs a tenth as much can cut the bill in half. You need a classifier or rule for "easy", and an eval that proves the small model is good enough on that slice.
- **Context is the silent cost.** Retrieval that stuffs 10 chunks of 500 tokens into every call adds 5,000 input tokens per request. At scale that often costs more than the model's answer.

## What each layer owns

| Layer | Owns | Typical failure if you skip it |
| --- | --- | --- |
| Application | Prompts, context assembly, evals, UX, guardrails | Plausible but wrong answers reach users unchecked |
| Model | Choice of model, fine-tunes, adapters | Paying for a frontier model where a small one would do |
| Infrastructure | Serving, batching, caching, rate limits, monitoring | Latency spikes and runaway bills under load |

Most teams start at the top and move down only when a number forces them to. Self-hosting a model (the infrastructure layer) makes sense when you have steady high volume, strict data rules, or a fine-tuned open model; below a few million tokens a day an API is usually cheaper once you count engineer time.

<div class="take">
<div class="lab">Key takeaway</div>

think of the stack as application, model and infrastructure. Start with a plain model call and add context, guardrails, routing, caching and agents only when a requirement asks for them.

</div>
