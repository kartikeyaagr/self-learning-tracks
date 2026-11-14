---
id: "28.06"
course: "inference-eng-course"
module: "28"
slug: "add-streaming-an-api"
title: "Add Streaming & an API"
keyIdeas: ["Separate the engine loop from HTTP handling.", "Use queues: requests in, token outputs out per request.", "Detokenize incrementally on the server side.", "Implement the OpenAI chat completions shape with SSE streaming.", "Propagate client disconnects to abort requests."]
question: "How should the engine loop and the HTTP server communicate so streaming does not slow the GPU loop?"
answer: "Run the engine loop in its own thread or process and the HTTP server in an async event loop. The server puts new requests on a queue to the engine and receives generated tokens via per-request output queues (or a shared output channel); detokenization and SSE writes happen on the server side, so the engine loop never waits on network I/O."
sources: [{"title": "FastAPI documentation", "url": "https://fastapi.tiangolo.com/", "note": "Async HTTP server."}, {"title": "OpenAI API reference: Chat completions", "url": "https://platform.openai.com/docs/api-reference/chat", "note": "Request and response format."}]
tags: ["mini-engine", "api"]
---

## Architecture

~~~dg-flow Server and engine separation.
HTTP handlers (async) -(request queue)-> *Engine loop (thread or process) -(output queues)-> SSE writers
~~~

The engine loop should never wait on the network. If it does, one slow client slows every request in the batch. Queues decouple the two sides.

## API

- POST /v1/chat/completions, with stream: true support.
- Apply the chat template on the server side.
- Return usage (prompt and completion token counts) at the end.

```
async def chat(req):
    rid = new_id()
    out_q = asyncio.Queue()
    engine.submit(rid, tokenize(apply_template(req.messages)), req.params, out_q)
    async def stream():
        try:
            while (item := await out_q.get()) is not None:
                yield f"data: {json.dumps(item)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            engine.abort(rid)        # runs on disconnect too
    return StreamingResponse(stream(), media_type="text/event-stream")
```

## Disconnects

When a client disconnects, the handler sends an abort for that request id. The engine frees its KV on the next step. Without this, a disconnected request keeps generating and holding memory until it reaches its token limit.

## Test

Run your benchmark harness from Module 04 against your mini engine. It should work unchanged if your API matches the OpenAI format.

<div class="take">
<div class="lab">Key takeaway</div>

keep the GPU loop free of network work. Queues between an async server and the engine loop give clean streaming and cancellation.

</div>
