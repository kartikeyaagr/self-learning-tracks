---
id: "01.03"
course: "ai-engineering-course"
module: "01"
slug: "one-model-call-end-to-end"
title: "One model call, end to end"
keyIdeas: ["Your list of messages is rendered through the model's chat template into one token sequence with special tokens; pricing, limits and latency are all counted in those tokens.", "Prefill processes the whole prompt in one parallel pass and fills the KV cache; it sets time to first token (TTFT). Decode then produces one token per forward pass and sets time per output token.", "At each step the model outputs a distribution over the vocabulary; a sampler picks the next token, and logprobs expose how confident that choice was.", "Streaming sends tokens as server-sent events as they are decoded, so users feel TTFT instead of total time; cancelling the stream should stop generation and the bill.", "Your request shares the GPU with others through continuous batching, so per-token speed depends on load."]
question: "A chat request has a 2,000-token prompt and streams a 300-token answer. Describe what the server does, which phase sets time to first token, and what should happen if the user closes the tab after 100 tokens."
answer: "The server renders the messages through the chat template, tokenizes them, and queues the request into a running batch. Prefill then pushes all 2,000 prompt tokens through the model in one parallel pass, filling the KV cache and producing the first token, so prefill (plus queueing and network) sets time to first token. Decode follows: one forward pass per token, each reading the weights and the growing KV cache, sampling the next token and sending it to the client as a server-sent event. Three hundred sequential steps make decode dominate total time. If the user closes the tab after 100 tokens, the client should abort the request; the connection closes, the server stops decoding and frees the KV cache, and you pay for roughly the tokens produced so far instead of all 300. If the app keeps the stream open in the background, it pays for the full answer nobody reads."
rubric: ["Mentions templating and tokenizing the prompt before the model runs.", "Says prefill processes the whole prompt in parallel, fills the KV cache and dominates TTFT.", "Says decode is one token per forward pass and dominates total latency for a 300-token answer.", "Describes streaming tokens to the client as they are produced (for example as SSE events).", "Explains that aborting the request stops generation and saves output tokens."]
sources: [{"title": "Hugging Face: LLM inference optimization", "url": "https://huggingface.co/docs/transformers/llm_optims", "note": "Prefill vs decode, KV cache and static caches."}, {"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "What happens on the server while your request waits."}, {"title": "Anthropic: Streaming messages", "url": "https://docs.claude.com/en/docs/build-with-claude/streaming", "note": "A full SSE event flow, including deltas, usage and mid-stream errors."}, {"title": "Hugging Face: Chat templates", "url": "https://huggingface.co/docs/transformers/chat_templating", "note": "How message lists become the token sequence a model was trained on."}]
practice: ["ttft-tpot-itl-and-token-throughput", "chat-template-format", "temperature-top-k-and-top-p-sampling", "score-multiple-choice-continuations", "autoregressive-decoding-with-kv-cache"]
tools: ["vLLM"]
---

## The request path

~~~dg-seq One streamed chat completion from the client's point of view.
Client > API server: POST messages, stream=true
API server > Tokenizer: render chat template
Tokenizer -> API server: token ids
API server > GPU: prefill (whole prompt)
GPU -> Client: first token (SSE event)
API server > GPU: decode step, repeat
GPU -> Client: one event per few tokens, then stop
~~~

Every step on this path is measured in tokens. A tokenizer splits text into pieces from a fixed vocabulary, usually built with byte-pair encoding; in English a token averages about four characters, while code, numbers and many other languages need more. How tokenizers are trained and why the same text costs different amounts on different models is the subject of [Tokenization in practice](lesson:aieng:01.04). Type below to see the split.

~~~dg-raw
%%widget tokenizer Approximate tokenization of any text you type.
~~~

## Messages become one sequence: chat templates

An API takes a list of messages with roles (system, user, assistant, tool). The model never sees that list. It sees one flat token sequence in the exact format it was trained on, with special tokens marking where each turn starts and ends. The chat template does this rendering. A ChatML-style template, used by Qwen models among others, produces:

```text
<|im_start|>system
You are a support assistant.<|im_end|>
<|im_start|>user
How do I reset my password?<|im_end|>
<|im_start|>assistant
```

The final open `assistant` header is the generation prompt: it tells the model it is its turn. Other families use different markers (Llama 3 uses header and end-of-turn tokens; gpt-oss uses the "harmony" format with separate channels for reasoning and the final answer), but the idea is the same. Hosted APIs apply the template on their side. You meet it directly when you self-host or fine-tune, through calls such as Hugging Face's `apply_chat_template`. Use the wrong template and the model still answers, just worse: it may ignore the system prompt, run past its turn, or never emit the end-of-turn token that stops generation.

## Prefill and decode

Inference has two phases with very different costs.

- **Prefill**: all prompt tokens go through the model at once. It is a large parallel matrix multiplication, so it is compute-bound and uses the GPU well. Along the way it stores each layer's keys and values in the **KV cache** ([KV cache](lesson:aieng:04.04)). Prefill is most of your **time to first token (TTFT)**.
- **Decode**: one token per forward pass. Each pass reads all the weights and the whole KV cache from GPU memory but does little arithmetic, so it is memory-bandwidth bound. Its speed is the **time per output token (TPOT)**.

Total latency is roughly network and queueing + TTFT + output tokens × TPOT. Rough rules of thumb for hosted models: prefill runs at thousands of tokens per second, decode at roughly 30 to 150 tokens per second per user.

~~~dg-raw
%%widget latency Where the time goes in one model call: network, queue, prefill, decode.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**Why answer length dominates latency**

A 3,000-token prompt and a 500-token answer, on a server doing prefill at 5,000 tokens per second and decode at 60 tokens per second per user, plus 0.2 s of network and queueing.

Prefill: 3,000 / 5,000 = 0.6 s. Decode: 500 / 60 = 8.3 s.

Time to first token is about 0.8 s; total time about 9.1 s. Halving the prompt saves 0.3 s; halving the answer saves 4.2 s. Ask for shorter outputs before you optimize prompts, and stream so users see text after 0.8 s instead of 9.

</div>

## Choosing the next token, and how sure it was

After each forward pass the model produces logits, one score per vocabulary entry. Softmax turns them into probabilities and a sampler picks one: greedy takes the top token, temperature sharpens (below 1) or flattens (above 1) the distribution, and top-p samples from the smallest set whose probability adds up to p. The chosen token is appended and the loop continues until an end-of-turn token, a stop sequence or the `max_tokens` limit. The full order of sampling operations is in [LLM text generation](lesson:aieng:03.03).

Many APIs can also return **logprobs**: the log probability of each generated token, and optionally the top few alternatives at each position. They are cheap signals you can use. Ask a classifier prompt to answer "yes" or "no" with `max_tokens` of 1, and the logprob of "yes" is a confidence score: a logprob of −0.105 means $e^{-0.105} \approx 0.90$. Low-probability tokens inside an answer often mark the spans where the model was guessing (a date, a figure). Support varies: some providers and many reasoning models do not expose logprobs, so check before you design around them.

## Streaming over server-sent events

With `stream: true`, the server keeps the HTTP response open and sends tokens as they are decoded, using server-sent events (SSE): plain text lines of `event:` and `data:`, separated by blank lines. OpenAI-style chat streams send `data:` chunks whose `delta` holds the new text, ending with `data: [DONE]`. Anthropic's stream has typed events: `message_start`, then per content block a `content_block_start`, many `content_block_delta` events and a `content_block_stop`, then `message_delta` (stop reason and usage) and `message_stop`, with `ping` events in between.

```text
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"To reset"}}
```

Three things follow from this design:

- **Users feel TTFT, not total time.** A 9-second answer that starts after 0.8 s feels responsive; the same answer delivered all at once after 9 s does not. That is why TTFT is the latency number product teams watch.
- **Partial output is not final output.** Text arrives in fragments, tool-call arguments arrive as partial JSON you cannot parse until the block ends, and usage counts arrive at the end (OpenAI-style APIs only include them if you ask). Render text progressively, but validate structured output only when it is complete.
- **Cancellation is your cost control.** When the user hits stop or leaves, abort the request so the connection closes. Self-hosted engines such as vLLM then stop decoding and free the KV cache, and hosted APIs stop generating; expect to pay for the tokens produced before the server noticed.

## What the server does around you

Your request shares a GPU with others. Servers use **continuous batching**: new requests join the running batch at every decode step instead of waiting for the whole batch to finish, and finished ones leave immediately. Anyscale measured up to 23x throughput gains from this. The cost is that your TPOT depends on how busy the server is: the same request can stream at 80 tokens per second at night and 30 at peak.

## Where this breaks

- **Errors arrive after a 200.** Once streaming starts, the HTTP status is already sent, so an overload or failure mid-answer comes as an error event. Handle it, and decide whether to retry the whole request or show what you have.
- **Proxies buffer the stream.** A reverse proxy or serverless platform that buffers responses turns your stream back into one blob delivered at the end. Disable response buffering on every hop and test TTFT from the client, not the server.
- **Unaborted streams keep billing.** Navigating away in a single-page app does not always close the fetch. Wire the stop button and page unload to an abort.
- **Template mismatches look like bad models.** A self-hosted model that rambles past its turn or ignores the system prompt is often being fed the wrong chat template.

<div class="take">
<div class="lab">Key takeaway</div>

A model call is: render the chat template, tokenize, prefill the prompt into the KV cache, then decode one token at a time with a sampler, streaming each token as an event. Prefill sets time to first token, output length sets total time, and cancelling the stream is how you stop paying for answers nobody reads.

</div>
