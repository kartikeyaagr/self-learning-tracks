---
id: "01.02"
course: "inference-eng-course"
module: "01"
slug: "load-tokenize-inspect-the-model"
title: "Load, Tokenize & Inspect the Model"
keyIdeas: ["A tokenizer maps text to integer ids from a fixed vocabulary, usually with byte-pair encoding.", "Chat models expect a chat template that wraps messages in special tokens. Skipping it degrades output.", "Inspect config.json: layers, hidden size, attention heads, KV heads, vocab size, max positions.", "Count parameters and check dtype after loading to confirm what you are actually running.", "Tokens, not characters or words, are the unit of cost, latency and context length."]
question: "Why does the same sentence produce a different number of tokens in two models, and why does that matter for cost and latency comparisons?"
answer: "Each model has its own tokenizer and vocabulary, so it splits text into different pieces; larger vocabularies tend to use fewer tokens per word. Since compute, memory and pricing scale with tokens, comparing models by requests or words without counting tokens per model gives misleading cost and speed comparisons."
sources: [{"title": "Hugging Face Tokenizers documentation", "url": "https://huggingface.co/docs/tokenizers/index", "note": "How BPE and other tokenizers work."}, {"title": "Hugging Face: Chat templates", "url": "https://huggingface.co/docs/transformers/main/en/chat_templating", "note": "Formatting conversations for chat models."}]
tags: ["tokenizer"]
---

## Models read numbers, not text

A language model never sees characters. It sees a sequence of integer ids, each pointing to one row of an embedding table. The tokenizer converts text into those ids and back. It is a separate piece of software from the network, and it shapes cost, latency and even output quality.

## How byte-pair encoding works

Most LLM tokenizers use **byte-pair encoding (BPE)** or a close relative. Training a BPE tokenizer works like this:

1. Start with every byte (or character) as its own token.
2. Count every adjacent pair of tokens in a large text corpus.
3. Merge the most common pair into a new token and add it to the vocabulary.
4. Repeat until the vocabulary reaches the target size, often 32k to 200k entries.

Common words end up as single tokens. Rare words, names and code identifiers get split into pieces. Byte-level BPE can represent any input, because in the worst case it falls back to single bytes.

<div class="ex">
<div class="lab">Worked example</div>

**one sentence, different tokenizers**

The text "Inference engineering is fun" might become 5 tokens in a 128k-vocabulary tokenizer: "In", "ference", " engineering", " is", " fun".

A 32k-vocabulary tokenizer might need 6 or 7 tokens for the same text.

For English prose, 1 token is roughly 0.75 words. Python code is often 1 token per 3 to 4 characters, and many non-Latin scripts use more tokens per word.

</div>

## Why token counts matter

Every cost in serving is counted in tokens:

- **Prefill compute** grows with prompt tokens.
- **KV cache memory** grows with prompt plus output tokens.
- **Decode time** grows with output tokens.
- **Price** for API customers is per token.

So when you compare two models, count tokens with each model's own tokenizer. A model with a larger vocabulary may need 15% fewer tokens for the same text. That is 15% less work, which can matter more than a small difference in per-token speed.

```
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/models/qwen2.5-7b")
ids = tok("Inference engineering is fun")["input_ids"]
print(len(ids), tok.convert_ids_to_tokens(ids))
```

## Chat templates

Chat models are trained on conversations wrapped in special marker tokens. A template might turn a list of messages into something like this:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain the KV cache in one line.<|im_end|>
<|im_start|>assistant
```

The template is stored in `tokenizer_config.json` as a Jinja string. Use `apply_chat_template` instead of building the string yourself:

```
msgs = [{"role": "user", "content": "Explain the KV cache in one line."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
```

`add_generation_prompt=True` appends the opening of the assistant turn, so the model starts answering instead of continuing the user's message.

Getting the template wrong is one of the most common serving bugs. The model still produces text, so nothing crashes, but quality drops: it rambles, answers as the user, or never emits its stop token. If a model seems worse in your server than in its demo, check the template first.

## Special tokens and stopping

The tokenizer defines special tokens such as beginning of sequence (BOS), end of sequence (EOS) and the chat markers. Some models have more than one end token, for example one that ends a turn and one that ends the document. The `generation_config.json` file lists which ids should stop generation. If your server stops only on one of them, generations can run on to `max_tokens`, which wastes GPU time and confuses users.

## Inspecting the model itself

Load the model and check that what you are running matches what you think you are running:

```
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="cuda")
print(model.config)
print(sum(p.numel() for p in model.parameters()) / 1e9, "B params")
print(next(model.parameters()).dtype)
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
```

## The config fields you will use most

| Field | Llama 3 8B | Used for |
| --- | --- | --- |
| num_hidden_layers | 32 | KV cache size, FLOPs, pipeline splits |
| hidden_size | 4096 | Matmul shapes, activation memory |
| num_attention_heads | 32 | Attention shapes, tensor-parallel splits |
| num_key_value_heads | 8 | KV cache size per token |
| head_dim | 128 | KV cache size, attention kernels |
| intermediate_size | 14336 | MLP weights and FLOPs |
| vocab_size | 128256 | LM head size, sampling cost |
| max_position_embeddings | 8192 (later 128k) | Longest supported context |

From these you can compute KV bytes per token: 2 (K and V) × layers × KV heads × head size × bytes per element. For Llama 3 8B in BF16 that is 2 × 32 × 8 × 128 × 2 = 131,072 bytes, or 128 KB. You will use this number in nearly every capacity estimate.

## Deep dive: how BPE builds its vocabulary

Byte-pair encoding starts from single bytes and repeatedly merges the most frequent adjacent pair in a training corpus:

1. Count every adjacent pair of symbols.
2. Merge the most frequent pair into a new symbol and record the merge.
3. Repeat until the vocabulary reaches the target size (128,256 for Llama 3).

At inference time, the tokenizer replays the recorded merges in order. Frequent words become one token; rare words split into pieces.

~~~dg-raw
%%grid "unbelievably" through successive merges. Each cell is one character, colored by the token it belongs to.
key A = un | c1
key B = believ | c3
key C = ably | c2
key D = single byte or short piece | c4
key E = believably | go
bytes | D D D D D D D D D D D D
early merges | A A D D D D D D C C C C
later merges | A A B B B B B B C C C C
final | A A E E E E E E E E E E
%%
~~~

## Deep dive: tokens per character changes cost

The cost of a request scales with tokens, but users write characters. Define the compression ratio

⟦M190⟧

For English prose with a modern 128k vocabulary, ⟦m191⟧ to 4.5. For code it is often 3 to 3.5. For many non-Latin scripts it can fall below 2.

<div class="ex">
<div class="lab">Worked example</div>

**the same document in two languages**

A 12,000-character support article. In English with ⟦m192⟧: about 2,800 tokens. In a language where ⟦m193⟧: about 6,700 tokens.

Prefill cost, KV memory and TTFT for the second version are 2.4 times higher, for the same content.

If you price per token, the second user pays 2.4 times more. If you price per request, you absorb it. Either way, capacity planning must use token counts measured on your real traffic.

</div>

## Deep dive: the KV cost of a chat template

Every turn wraps messages in special tokens. For a template that adds about 5 tokens per message, a 40-turn conversation carries about 200 extra tokens on every request. At 128 KB per token that is 25 MB of KV per conversation. Small per request, but it is paid on every turn and every user.

<div class="take">
<div class="lab">Key takeaway</div>

measure everything in tokens using the model's own tokenizer, always apply the chat template, check every stop token, and read config.json before doing capacity math.

</div>
