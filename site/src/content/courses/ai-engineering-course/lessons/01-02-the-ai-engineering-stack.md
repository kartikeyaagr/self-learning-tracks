---
id: "01.02"
course: "ai-engineering-course"
module: "01"
slug: "the-ai-engineering-stack"
title: "The AI engineering stack"
keyIdeas: ["Chip Huyen splits the stack into three layers: application development, model development and infrastructure. Most teams work in the application layer and buy the other two.", "A current request path runs through a gateway (keys, routing, rate limits, fallbacks), retrieval, the model with tools (often over MCP), and guardrails, with traces feeding evals and observability.", "A platform grows in steps: plain call, then context, guardrails, a router and gateway, caches, and finally agents with write actions. Add each only when a number asks for it.", "Each layer costs latency and money. Caching usually pays first, routing pays most but needs an eval, and retrieved context is the silent cost.", "Semantic caches and provider fallbacks are the two layers that most often hurt quality without anyone noticing."]
question: "Name the three layers of the AI stack with one component each, then say where a model gateway sits and what can go wrong when it fails over to another provider."
answer: "Application layer: prompts, context assembly and retrieval, tools, guardrails and evals (for example a RAG pipeline). Model layer: choosing, fine-tuning, distilling and quantizing models (for example a LoRA fine-tune). Infrastructure layer: inference serving, GPUs, vector stores, caches and monitoring (for example vLLM or a vector database). A gateway sits between the application and the model providers. It holds keys, enforces rate limits, routes requests to cheap or strong models and fails over when a provider errors or returns 429s. The risk is that failover silently changes behaviour: the fallback model was never tuned for your prompts, may format output differently or refuse more, and has its own limits. So the fallback path needs its own eval run and its own alerts, not just a health check."
rubric: ["Names application, model and infrastructure layers with a plausible component in each.", "Places the gateway between the app and the model providers, owning keys, rate limits, routing or fallbacks.", "Explains that a fallback model can change output quality or format because prompts were tuned for the primary model.", "Says the fallback path must be evaluated or monitored separately."]
sources: [{"title": "Chip Huyen: Building A Generative AI Platform", "url": "https://huyenchip.com/2024/07/25/genai-platform.html", "note": "Layer-by-layer walk through a production GenAI platform."}, {"title": "a16z: Emerging Architectures for LLM Applications", "url": "https://a16z.com/emerging-architectures-for-llm-applications/", "note": "Reference stack: data pipelines, embeddings, vector DBs, orchestration."}, {"title": "Model Context Protocol: introduction", "url": "https://modelcontextprotocol.io/docs/getting-started/intro", "note": "The open protocol most agent stacks now use to connect models to tools and data."}, {"title": "OpenTelemetry: semantic conventions for generative AI", "url": "https://opentelemetry.io/docs/specs/semconv/gen-ai/", "note": "Standard span and attribute names for tracing model and tool calls."}]
practice: ["optimize-throughput-and-cost-under-latency-slo", "prefix-cache-matching-and-reuse", "mcp-tool-descriptor", "authorize-tool-call"]
tools: ["MCP", "OpenTelemetry"]
---

## Three layers

Chip Huyen's book describes the AI stack as three layers. Almost every tool you will hear about fits in one of them.

~~~dg-layers The AI engineering stack, top to bottom. Most product work happens at the top.
*Application | prompts, context and retrieval, tools and agents, guardrails, evals, UI
Model | model choice, fine-tuning, distillation, quantization, datasets
Infrastructure | gateways, inference serving, GPUs, vector stores, caches, observability
~~~

| Layer | Owns | Typical failure if you skip it |
| --- | --- | --- |
| Application | Prompts, context assembly, tools, evals, UX, guardrails | Plausible but wrong answers reach users unchecked |
| Model | Choice of model, fine-tunes, adapters | Paying for a frontier model where a small one would do |
| Infrastructure | Gateway, serving, caching, rate limits, monitoring | Latency spikes, outages and runaway bills under load |

The application layer changed the most when foundation models arrived. Before, a team needed its own model and infrastructure before it could build anything. Now a team can call a hosted model and spend all its time on the top layer, moving down only when a number forces it to. a16z's reference architecture draws the same picture from the data side (pipelines, embeddings, a vector database, an orchestration layer) and makes the same point: putting the right data in the prompt is the default, and fine-tuning is the exception.

## A request path in 2026

In "Building A Generative AI Platform", Huyen describes how a system grows from a single API call. Each step adds one component:

1. **Plain call**: the query goes to the model, the response comes back.
2. **Enhance context**: retrieve documents or call tools and put the results in the prompt (RAG).
3. **Guardrails**: check inputs for prompt injection and personal data, and outputs for format errors, policy violations and unsupported claims.
4. **Router and gateway**: send easy queries to a cheap model and hard ones to a strong model; one gateway holds keys, rate limits and fallbacks.
5. **Cache**: reuse answers for repeated queries, and reuse the processed prompt prefix (prompt caching).
6. **Agent patterns**: loops, branching, and tools that write (send an email, change a database).

A mature request path looks like this, with every hop emitting a trace span:

~~~dg-flow A typical request path once the layers are in place. Every hop is traced.
Query -(check)-> Input guardrail -(route)-> *Gateway -(context)-> Retriever -(prompt)-> Model + tools -(check)-> Output guardrail -(log)-> Answer
~~~

What each newer component does:

- **Gateway and router.** One service in front of all model providers: API keys, per-team budgets, request and token rate limits, routing by task or difficulty, retries and provider failover. Products such as LiteLLM, Portkey, OpenRouter and the cloud providers' AI gateways do this; many teams start with a thin wrapper of their own. Retries, backoff and fallbacks are in [Reliability: gateways, retries and fallbacks](lesson:aieng:12.06).
- **Retrieval.** Embeddings and a vector or hybrid index (the rest of module 2 and module 11).
- **Agents and MCP.** The model calls tools in a loop ([The agent loop](lesson:aieng:09.02)). The Model Context Protocol has become the common way to expose tools and data sources, so one MCP server can serve any client that speaks it ([Model Context Protocol](lesson:aieng:09.03)).
- **Evals and observability.** Every model and tool call is a span in a trace, increasingly using the OpenTelemetry generative-AI conventions, so you can see cost, latency and errors per step and turn bad traces into eval cases ([Observability for LLM apps](lesson:aieng:12.07), [Why LLM evals are different](lesson:aieng:10.01)).

## Put numbers on the stack

Each layer costs latency and money, and each one should earn its place. The figure models a typical request mix. Move the cache hit rate and the share of traffic you can route to a small model, and watch which layer pays for itself.

~~~dg-raw
%%widget stack Monthly cost of a feature as you add a cache, a router, retrieval and a guardrail.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**One million requests a month (prices illustrative)**

Each request sends 2,000 input tokens and gets 300 output tokens from a strong model priced at \$3 per million input and \$15 per million output tokens. Per request: 2,000 × \$3/M = \$0.0060 plus 300 × \$15/M = \$0.0045, so \$0.0105. The month costs \$10,500.

- **Exact-match cache, 30% hit rate**: 700,000 calls remain, \$7,350.
- **Route half of those to a model at one tenth the price**: 350,000 × \$0.0105 + 350,000 × \$0.00105 = \$3,675 + \$367.50 = \$4,042.50.
- **Add retrieval that puts 5,000 more input tokens in every call**: +\$0.015 per strong call and +\$0.0015 per small call, so +\$5,250 + \$525. The bill is back to \$9,817.50.

Caching and routing cut the bill by 62%; careless context gave almost all of it back. Trimming retrieval to 2,000 tokens would keep the bill at about \$6,350. [Cost budgeting](lesson:aieng:12.04) works through prompt caching, batch APIs and cascades in detail.

</div>

## Where this breaks

- **Semantic caches return confident wrong answers.** An exact-match cache is safe. A semantic cache serves a stored answer when a new query's embedding is close to an old one, and embeddings blur exactly the details that matter: "cancel my order" vs "don't cancel my order", order 4512 vs order 4513. Cached answers can also be stale or personal. Scope cache keys by tenant and user, exclude anything time-sensitive or account-specific, tune the threshold on labelled pairs, and track the false-hit rate. Provider prompt caching is different: it reuses an exact token prefix and never changes the answer.
- **Fallbacks change behaviour.** When the primary provider returns errors or 429s, the gateway sends traffic to a fallback model your prompts were never tuned for. It may format JSON differently, refuse more, or be slower. Run the eval set against the fallback too, and alert when fallback traffic rises.
- **Rate limits are in tokens as well as requests.** A long-context feature can hit a tokens-per-minute limit at a modest request rate. Retry with backoff and jitter, and honour the provider's retry-after hint, instead of retrying immediately and making the overload worse.
- **Agents with write tools multiply risk.** A read-only RAG bug gives a wrong answer; an agent bug can send the wrong email. Put write tools behind permissions and confirmations.

## Buy or build at each layer

- **Model**: closed API (fast to start, less control) vs open weights (control, privacy, more ops work). [Choosing a model](lesson:aieng:01.05) works through the trade-off.
- **Vector store**: a Postgres extension is often enough; a dedicated vector database pays off at large scale or with heavy filtering ([Matching storage to your workload](lesson:aieng:02.03)).
- **Serving**: managed endpoints vs your own vLLM or SGLang cluster. Self-hosting makes sense with steady high volume, strict data rules, or a fine-tuned open model; at low volume an API is usually cheaper once you count engineer time.

<div class="take">
<div class="lab">Key takeaway</div>

Think of the stack as application, model and infrastructure. Start with a plain model call and add context, guardrails, a gateway, caches and agents one at a time, each justified by a measured number and each with its own way of failing.

</div>
