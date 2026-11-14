---
id: "01.05"
course: "inference-eng-course"
module: "01"
slug: "serve-the-model-over-http"
title: "Serve the Model Over HTTP"
keyIdeas: ["A minimal server: an HTTP endpoint that takes a prompt, runs generate, returns text.", "Streaming responses usually use server-sent events (SSE): \"data: ...\" lines over one HTTP response.", "OpenAI-compatible request and response shapes let existing clients work with your server.", "A naive server processes one request at a time. Concurrency queues up and the GPU sits mostly idle.", "This limitation is what motivates real inference engines with batching."]
question: "Your FastAPI server wraps model.generate. Two users send requests at the same time. What happens and why is it a problem?"
answer: "With a single model and a blocking generate call, requests run one at a time: the second user waits for the first request to finish, so latency grows linearly with queue length. The GPU also runs at batch size 1, wasting most of its capacity. Real engines batch concurrent requests together at every step."
sources: [{"title": "FastAPI documentation", "url": "https://fastapi.tiangolo.com/", "note": "Building HTTP APIs in Python."}, {"title": "MDN: Using server-sent events", "url": "https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events", "note": "The streaming protocol most LLM APIs use."}]
tags: ["serving", "api"]
---

## The smallest useful server

A model that only runs in a notebook cannot serve anyone. The first step to a service is an HTTP endpoint that takes a prompt and streams tokens back. FastAPI and a streaming response are enough:

```
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/completions")
def complete(req: dict):
    def events():
        for text in stream_tokens(req["prompt"], req.get("max_tokens", 128)):
            yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
```

Here `stream_tokens` wraps the streamer from lesson 01.03.

## Server-sent events

Server-sent events (SSE) is a simple protocol for streaming. The server keeps one HTTP response open and writes lines that start with `data:`, each followed by a blank line. The client reads them as they arrive. SSE works through most proxies and load balancers, and it is what OpenAI-compatible APIs use when a request sets `"stream": true`.

~~~dg-seq One streamed request.
Client > Server : POST, stream=true
Server > GPU : prefill prompt
GPU -> Server : first token
Server -> Client : data: chunk 1
Server > GPU : decode step
Server -> Client : data: chunk 2
Server -> Client : data: [DONE]
~~~

Details that matter in production:

- **Disable proxy buffering.** Nginx and some load balancers buffer responses by default, so the client receives everything at the end. Set `X-Accel-Buffering: no` or the equivalent.
- **Send keep-alive comments during long prefills.** A line that starts with `:` is ignored by clients but keeps idle-timeout proxies from closing the connection.
- **Handle client disconnects.** If the user closes the tab, stop generating. Otherwise the GPU keeps working for nobody.
- **Report usage at the end.** OpenAI-style APIs can send prompt and completion token counts in a final chunk, which billing and benchmarking tools rely on.

## The OpenAI-compatible shape

Most clients and tools speak the OpenAI chat completions format, so most engines expose it. A request looks like this:

```
POST /v1/chat/completions
{"model": "qwen2.5-7b", "stream": true, "max_tokens": 256,
 "messages": [{"role": "user", "content": "Hi"}]}
```

Each streamed chunk carries a `choices[0].delta.content` piece. Matching this format means your server works with existing SDKs, benchmarks and gateways for free.

## What breaks under load

This server runs one generation at a time per model copy. That is fine for one user and wrong for many.

<div class="ex">
<div class="lab">Worked example</div>

**ten users arrive together**

Each request takes 4 s to generate 200 tokens at 50 tokens per second.

Served one at a time, user 1 waits 4 s, user 2 waits 8 s, and user 10 waits 40 s before finishing. Their TTFT ranges from 0.1 s to 36 s.

Throughput stays at 50 tokens per second the whole time, because the GPU always runs at batch 1.

An engine with continuous batching would run all ten together at maybe 40 tokens per second each, 400 in total. Everyone finishes in about 5 s.

</div>

You could try simple batching: wait for a few requests, pad them to the same length and generate together. Two problems appear at once:

- **Ragged lengths.** Prompts differ in length, so padding wastes compute and memory.
- **Different finish times.** One request wants 10 tokens and another wants 1,000. In a static batch, the short one either waits for the long one or leaves an idle slot.

Solving both properly needs continuous batching (Module 07) and a KV cache manager that can grow and free memory per request (Module 08). That is what dedicated engines such as vLLM, SGLang and TensorRT-LLM provide, and why Module 06 switches to one.

## Other things a real server does

Even before performance, a production endpoint needs:

- **Input limits**: reject prompts longer than the model's context, and cap `max_tokens`.
- **Timeouts**: separate limits for the time to first token and for total time.
- **Health checks**: a cheap endpoint that says whether the model is loaded and the GPU responds.
- **Metrics**: request counts, TTFT, TPOT, tokens in and out, errors.
- **Concurrency control**: a queue with a maximum depth, so overload produces quick errors instead of unbounded waits.

## Deep dive: why a one-request-at-a-time server queues

A server that handles one request at a time is a single-server queue. If requests arrive at rate ⟦m207⟧ and each takes ⟦m208⟧ seconds, utilization is ⟦m209⟧. For random (Poisson) arrivals and random service times, the average time waiting before service starts is

⟦M206⟧

This blows up as ⟦m210⟧.

~~~dg-raw
%%widget queue Set the service time to a plain-generate request (about 8 s) and watch wait times as load rises.
s = 4000
r = 0.5
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**ten users at once**

A plain Hugging Face server takes 8 s per request. Ten users arrive together.

User 1 waits 0 s, user 2 waits 8 s, ..., user 10 waits 72 s. Average wait 36 s, and TTFT for the last user is 72 s.

A batching engine could serve all ten in one batch with TPOT rising perhaps from 6 to 7 ms, so all ten get their first token within about a second.

</div>

~~~dg-raw
%%timeline Ten simultaneous requests: one at a time (top lanes) versus batched (bottom lane).
unit = s
User 1 | 0-8 serve c3
User 2 | 0-8 wait flame ; 8-16 serve c3
User 3 | 0-16 wait flame ; 16-24 serve c3
User 10 | 0-72 wait flame ; 72-80 serve c3
Batched, all 10 | 0-2.4 serve together go
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a FastAPI wrapper around generate is a good demo and a poor service. SSE streaming is easy, but one-at-a-time serving queues users and leaves the GPU mostly idle.

</div>
