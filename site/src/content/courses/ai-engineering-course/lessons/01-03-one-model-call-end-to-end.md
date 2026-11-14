---
id: "01.03"
course: "ai-engineering-course"
module: "01"
slug: "one-model-call-end-to-end"
title: "One model call, end to end"
keyIdeas: ["Text is split into tokens by a tokenizer (e.g. BPE). Pricing, context limits and latency are all measured in tokens.", "Tokens become embedding vectors and flow through the transformer layers.", "Prefill: the whole prompt is processed in one parallel pass, filling the KV cache. This sets time-to-first-token (TTFT).", "Decode: one new token per forward pass, each reading the KV cache. This sets time-per-output-token and is memory-bandwidth bound.", "At every step the model outputs a probability distribution over the vocabulary; a sampler (temperature, top-p) picks the next token.", "Generation stops at an end-of-sequence token, a stop sequence or max_tokens. The server batches many users' requests together to keep GPUs busy."]
question: "Your chat request has a 2,000-token prompt and gets a 300-token answer. Describe what the server does, and which part dominates time-to-first-token."
answer: "The server tokenizes the prompt, then runs prefill: all 2,000 tokens go through the model in one parallel pass, filling the KV cache and producing the first token, so prefill dominates TTFT. Then decode runs 300 sequential forward passes, each sampling one token from the output distribution and appending to the KV cache, streaming tokens back until EOS or max_tokens. Decode dominates total latency."
sources: [{"title": "Hugging Face: LLM inference optimization", "url": "https://huggingface.co/docs/transformers/llm_optims", "note": "Prefill vs decode, KV cache and static caches."}, {"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "What happens on the server while your request waits."}, {"title": "OpenAI tiktoken", "url": "https://github.com/openai/tiktoken", "note": "A fast BPE tokenizer you can try on your own prompts."}]
---

## From text to tokens

A model never sees characters. A tokenizer splits text into tokens using a fixed vocabulary, typically built with byte-pair encoding (BPE). Common words are one token; rare words split into pieces. In English a token averages around four characters, or about three quarters of a word. Everything is counted in tokens: the context window, the price, and the latency. Try OpenAI's tiktoken on your own prompt and you will see that code, numbers and non-English text often use more tokens than you expect.

## The request path

~~~dg-seq One chat completion from the client's point of view.
Client > API server: POST prompt
API server > Tokenizer: text
Tokenizer -> API server: token ids
API server > GPU: prefill (whole prompt)
GPU -> API server: first token
API server > GPU: decode step, repeat
GPU -> Client: stream tokens
~~~

## Prefill and decode

Inference has two phases with very different costs.

- **Prefill**: the model processes all prompt tokens at once. It is a big parallel matrix multiplication, so it is compute-bound and uses the GPU well. While doing this it stores each layer's keys and values in the **KV cache**. Prefill time is most of your **time to first token (TTFT)**.
- **Decode**: the model generates one token per forward pass. Each pass reads all the weights and the whole KV cache from GPU memory but does little maths, so it is memory-bandwidth bound. Its speed is the **time per output token (TPOT)**.

Total latency is roughly TTFT + output tokens x TPOT. That's why long outputs are slow even when prompts are short, and why long prompts hurt TTFT.

## Choosing the next token

After each forward pass the model produces logits, one score per vocabulary entry. Softmax turns them into probabilities, and a sampler picks one:

- **Greedy** takes the top token.
- **Temperature** sharpens (below 1) or flattens (above 1) the distribution.
- **Top-p** keeps the smallest set of tokens whose probability adds up to p and samples from that.
The chosen token is appended and the loop continues until an end-of-sequence token, a stop string, or the max_tokens limit.

## What the server does around you

Your request shares a GPU with others. Servers use **continuous batching**: new requests join the batch at every decode step instead of waiting for the whole batch to finish. Anyscale measured up to 23x throughput gains from this. The cost is that your TPOT depends on how busy the server is.

<div class="ex">
<div class="lab">Worked example</div>

**2,000-token prompt, 300-token answer**

Prefill handles 2,000 tokens in one pass, maybe 100-300 ms on a modern GPU. That is your TTFT.

Decode then runs 300 sequential steps. At 30 ms per token that is 9 seconds, far longer than prefill.

So for this request, output length drives total latency; prompt length drives the wait before streaming starts.

</div>

## Try it: tokens and time

Type into the box to see how text breaks into tokens. Notice that common English words are one token, while rare words, numbers, dates and non-English text split into several pieces. You pay per token and the model reads per token, so this split matters for both cost and behavior (it's why models used to miscount letters in "strawberry").

~~~dg-raw
%%widget tokenizer Approximate tokenization of any text you type.
~~~

The second figure splits one request into its phases. Prefill processes the whole prompt in parallel and is fast per token. Decode generates one token at a time and is slow per token. Raise the prompt length and the time to first token grows; raise the output length and total time grows much faster.

~~~dg-raw
%%widget latency Where the time goes in one model call: network, queue, prefill, decode.
~~~

## Worked numbers

Rules of thumb that hold for current hosted models:

- English text averages about 4 characters, or 0.75 words, per token. A 1,000-word document is about 1,300 tokens.
- Prefill runs at thousands of tokens per second on a GPU, because every prompt token is processed at once.
- Decode runs at roughly 30 to 150 tokens per second per user, because each new token needs a full pass through the model.

<div class="ex">
<div class="lab">Worked example</div>

**why the answer length dominates latency**

A request with a 3,000-token prompt and a 500-token answer, on a server doing prefill at 5,000 tokens per second and decode at 60 tokens per second per user.

Prefill: 3,000 / 5,000 = 0.6 s. Decode: 500 / 60 = 8.3 s. Add 0.2 s of network and queueing.

Time to first token is about 0.8 s, total time about 9.1 s. Halving the prompt saves 0.3 s; halving the answer saves 4.2 s. Ask for shorter outputs before you optimize prompts, and stream tokens so users see progress after 0.8 s instead of 9.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a model call is tokenize, prefill the prompt into the KV cache, then decode one token at a time with a sampler. Prefill sets time to first token; output length sets total time.

</div>
