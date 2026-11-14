---
id: "06.03"
course: "inference-eng-course"
module: "06"
slug: "openai-compatible-apis-streaming"
title: "OpenAI-Compatible APIs & Streaming"
keyIdeas: ["Most engines expose /v1/chat/completions and /v1/completions in the OpenAI format.", "stream: true returns SSE chunks with delta content; the stream ends with data: [DONE].", "Set stream_options.include_usage to get exact token counts at the end.", "The chat endpoint applies the model's chat template server side.", "Engines add extra parameters (top_k, min_p, guided decoding) beyond the OpenAI spec."]
question: "What fields in a streamed chat completion response would you use to compute TTFT and token counts, and what is the stream_options usage flag for?"
answer: "Each SSE chunk carries choices[0].delta.content; the time of the first chunk with non-empty content gives TTFT. Token counts come from the usage object; with stream_options {\"include_usage\": true} the server sends a final chunk containing prompt_tokens and completion_tokens, so you do not have to re-tokenize to count."
sources: [{"title": "OpenAI API reference: Chat completions", "url": "https://platform.openai.com/docs/api-reference/chat", "note": "The request and response schema."}, {"title": "vLLM: OpenAI-compatible server", "url": "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html", "note": "Supported endpoints and extra parameters."}]
tags: ["api"]
---

## Why one API shape matters

Most LLM clients, SDKs, agent frameworks and benchmark tools speak the OpenAI chat completions format. vLLM, SGLang, TensorRT-LLM's server, TGI and most hosted providers expose it too. If your server speaks it, every tool works against it, and switching engines becomes a configuration change.

## The request

```
POST /v1/chat/completions
{
  "model": "qwen2.5-7b",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is a KV cache?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

The server applies the model's chat template to `messages`, so the client never handles special tokens. The older `/v1/completions` endpoint takes a raw `prompt` string instead and applies no template, which is useful for base models and for exact control in benchmarks.

## The streamed response

```
data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}
data: {"choices":[{"index":0,"delta":{"content":"A"}}]}
data: {"choices":[{"index":0,"delta":{"content":" KV cache"}}]}
...
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: {"choices":[],"usage":{"prompt_tokens":31,"completion_tokens":87}}
data: [DONE]
```

~~~dg-seq A streamed chat completion.
Client > Engine : POST, stream=true
Engine -> Client : role chunk
Engine -> Client : content chunk
Engine -> Client : content chunk
Engine -> Client : finish_reason
Engine -> Client : usage chunk
Engine -> Client : [DONE]
~~~

## Measuring correctly against this format

- **TTFT is the first chunk with non-empty content.** The first chunk often carries only the role. Timing that chunk understates TTFT, sometimes by the whole prefill.
- **Chunks are not tokens.** An engine may send several tokens in one chunk, especially under load or with speculative decoding. Use `usage.completion_tokens` for counts.
- **ITL needs care.** If one chunk carries three tokens, the gap before it covers three tokens. Either divide the gap by the token count or report chunk-level gaps and say so.
- **finish_reason tells you why it stopped.** "stop" means a stop token or string, "length" means max_tokens. Many "length" results in a benchmark that expected natural stops point to a template or stop-token bug.

## Engine extensions

Engines accept extra parameters in the same body. Clients that only know the OpenAI spec ignore them; clients that know them get more control:

- Sampling: `top_k`, `min_p`, `repetition_penalty`, `seed`.
- Benchmarking: `ignore_eos`, `min_tokens`.
- Structured output: a JSON schema, a regex or a grammar that constrains generation.
- Logging: `logprobs` and `top_logprobs` for evaluation.

## Behaviour to check before going live

- **Client disconnects.** Close the stream from the client mid-generation and check that the engine aborts the request. The running-requests metric should drop at once.
- **Errors in streams.** A failure after streaming starts cannot change the HTTP status. Check how your engine reports it, usually as an error object in a final chunk.
- **Long silences.** During a long prefill under load, no chunk may arrive for seconds. Make sure proxies and clients do not time out.
- **Tool calls.** If you use function calling, check that the engine's tool-call parser matches the model's format, which differs between model families.

<div class="ex">
<div class="lab">Worked example</div>

**a TTFT bug from the role chunk**

A team's dashboard showed p95 TTFT of 35 ms for a model whose prefill alone took 120 ms.

Their client timed the first SSE chunk, which the engine sent as soon as the request was accepted, with only the role in it.

Timing the first chunk with content gave p95 TTFT of 210 ms. The dashboard had been wrong for months.

</div>

## Deep dive: why streamed text lags tokens

Text cannot always be sent the moment a token is sampled:

- **Partial characters.** A token can end in the middle of a multi-byte UTF-8 character. The detokenizer must hold it until the character is complete, so a chunk can carry no text at all.
- **Stop strings.** If a stop string has length ⟦m472⟧ characters, the last ⟦m473⟧ characters might be its beginning. The engine must hold them back until it can rule that out.

⟦M471⟧

A stop string like `"\n\nUser:"` (7 characters) can delay up to 6 characters, about 1 to 2 tokens. Clients that time "first text" instead of "first token" see this as extra TTFT.

## Deep dive: the cost of streaming itself

Each SSE chunk is a small JSON object, about 150 to 250 bytes for one token. At 6,000 output tokens per second that is about 1.2 MB/s of JSON to build and send, all on the API server's CPU. When that process is single-threaded and also detokenizes, it can become the limit before the GPU does. Engines send several tokens per chunk under load for this reason, which is why a harness must count tokens from usage, not chunks (04.08).

<div class="take">
<div class="lab">Key takeaway</div>

speak the OpenAI format, stream with SSE, time the first chunk with real content, and use the usage chunk for token counts. Your clients and tools then work against any engine.

</div>
