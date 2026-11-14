---
id: "21.03"
course: "inference-eng-course"
module: "21"
slug: "build-prefill-decode-workers"
title: "Build Prefill & Decode Workers"
keyIdeas: ["Prefill worker: runs prompt, produces KV and first token.", "Decode worker: allocates blocks, receives KV, continues generation.", "A coordinator or router pairs prefill and decode workers per request.", "KV connectors abstract the transfer method (NVLink, RDMA, TCP, storage).", "Handle failures: if transfer fails, fall back to local prefill on the decode worker."]
question: "What must a decode worker receive from the prefill worker to continue a request, and in what order should things happen?"
answer: "The KV cache for all prompt tokens (every layer), the first generated token (or the logits to sample it), the request's sampling parameters and state, and position information. Order: router sends request to prefill; prefill computes KV and first token; KV is transferred to blocks the decode worker allocated; decode worker adds the request to its running batch and continues generating; the first token can be streamed by prefill or decode."
sources: [{"title": "vLLM: Disaggregated prefilling", "url": "https://docs.vllm.ai/en/latest/features/disagg_prefill.html", "note": "Experimental support and connectors."}, {"title": "NVIDIA Dynamo", "url": "https://github.com/ai-dynamo/dynamo", "note": "Framework for disaggregated serving."}]
tags: ["disaggregation"]
---

## Request flow

~~~dg-seq Disaggregated request flow.
Router > Prefill : request
Router > Decode : reserve KV blocks
Prefill > Prefill : prefill prompt
Prefill > Decode : KV + first token
Decode > Decode : join batch
Decode -> Router : stream tokens
~~~

1. The router picks a prefill worker and a decode worker.
2. The decode worker reserves KV blocks for the prompt, so the transfer can land in place.
3. The prefill worker computes the prompt's KV and samples the first token.
4. KV blocks and the first token travel to the decode worker.
5. The decode worker adds the sequence to its running batch and streams tokens back.

## Components

- **Router or coordinator:** picks a prefill and a decode worker, usually by load and cache locality.
- **Prefill worker:** an engine configured for large token budgets and prefill-friendly parallelism, for example TP 4.
- **Decode worker:** an engine configured for large decode batches and high KV capacity.
- **KV connector:** moves KV between them over NVLink, RDMA or TCP. NIXL, Mooncake and LMCache are examples.

## Design points

- **Reserve decode-side blocks before the transfer**, so KV lands directly in its final place with no extra copy.
- **Transfer layer by layer** to overlap with prefill compute.
- **Backpressure:** if the decode pool is full, hold the request at the prefill worker or queue it before prefill. Never prefill into nowhere.
- **Failure handling:** timeouts on transfers, and a fallback path that re-runs prefill on the decode side if a transfer fails.

<div class="ex">
<div class="lab">Worked example</div>

**what the first token costs**

TTFT in a disaggregated system = router time + prefill queue + prefill compute + exposed transfer time + the first decode step's scheduling delay.

For a 4k prompt on an 8B model: about 1 ms + 20 ms + 180 ms + 5 ms + 10 ms ≈ 216 ms.

The prefill worker already produced the first token, so the user can see it before the decode worker starts. Most systems send it straight away.

</div>

## Deep dive: the path of the first token

In a disaggregated system, TTFT has four parts:

⟦M1000⟧

~~~dg-raw
%%timeline One 4,000-token request, colocated under load versus disaggregated (ms, illustrative).
unit = ms
tick = 50
Colocated | 0-60 queue c4 ; 60-240 prefill, shared with decodes c1 ; 240-255 first decode c3
Prefill pool | 0-10 queue c4 ; 10-130 prefill c1
Link | 30-134 KV transfer flame
Decode pool | 134-146 first decode c3
%%
~~~

The disaggregated prefill is faster because nothing else shares its steps, and its transfer runs while later layers compute. Only the last layer's transfer and the decode step are exposed.

<div class="take">
<div class="lab">Key takeaway</div>

a disaggregated system is two engine pools plus a coordinator and a KV transfer path. The decode worker needs the prompt's KV and the first token to carry on.

</div>
