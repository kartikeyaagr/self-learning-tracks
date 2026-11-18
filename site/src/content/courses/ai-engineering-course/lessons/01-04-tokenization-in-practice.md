---
id: "01.04"
course: "ai-engineering-course"
module: "01"
slug: "tokenization-in-practice"
title: "Tokenization in practice"
keyIdeas: ["Byte-level BPE learns merges from frequent byte pairs; with all 256 bytes in the vocabulary it can encode any string, so there is no unknown token.", "A bigger vocabulary means fewer tokens per text (cheaper, more fits in context) but larger embedding and output matrices and more rarely trained tokens.", "The same sentence can cost 2x to 10x more tokens in Hindi or Tamil than in English, and the gap depends heavily on the tokenizer: count, don't estimate.", "Digits, casing, leading spaces and trailing spaces change how text splits, which explains odd behaviour on arithmetic, letter counting and prompts that end in a space.", "Chat templates wrap every message in special tokens; user text must never be allowed to produce them."]
question: "Your product launches in India and the finance team asks why the same support conversation costs several times more in Hindi than in English, and whether switching models would help. How do you answer, and what would you measure?"
answer: "Billing is per token, and tokenizers trained mostly on English text split Devanagari into many short byte sequences, so the same meaning takes more tokens. The size of the gap is a property of the tokenizer, not the language: on one sentence I measured 92 tokens with cl100k_base and 28 with o200k_base against 14 for English. So switching models can help a lot. I would take a sample of real conversations in each language, count tokens with each candidate model's own tokenizer or count-tokens endpoint, and compute cost per conversation per language (input and output, since replies are in Hindi too). I would also check quality in Hindi on our eval set, because a cheaper tokenization is useless if answers get worse, and check that context limits still fit long Hindi threads."
rubric: ["Explains that cost is per token and tokenizers split under-represented scripts into more tokens", "States the gap depends on the tokenizer and gives a magnitude", "Proposes measuring with each provider's own token counter on real traffic, not character estimates", "Counts output tokens as well as input", "Checks quality and context-window effects, not only price"]
sources: [{"title": "Sennrich et al.: Neural Machine Translation of Rare Words with Subword Units (2015)", "url": "https://arxiv.org/abs/1508.07909", "note": "The paper that brought byte-pair encoding to neural language models."}, {"title": "Petrov et al.: Language Model Tokenizers Introduce Unfairness Between Languages (2023)", "url": "https://arxiv.org/abs/2305.15425", "note": "Measures how the same text costs up to 15 times more tokens across languages."}, {"title": "OpenAI tiktoken", "url": "https://github.com/openai/tiktoken", "note": "The BPE tokenizer used for OpenAI models; the counts in this lesson come from it."}, {"title": "Anthropic: Token counting", "url": "https://platform.claude.com/docs/en/build-with-claude/token-counting", "note": "Counting tokens for a full request, including images and tools, before sending it."}]
practice: ["train-deterministic-bpe-vocabulary", "apply-ranked-bpe-merges", "chat-template-format", "tokenizer-training", "subword-tokenization"]
tools: ["tiktoken", "Hugging Face tokenizers", "Claude API", "Gemini API"]
---

## What a tokenizer does

[One model call, end to end](lesson:aieng:01.03) treats the tokenizer as a box that turns text into ids. In practice that box decides your bill, how much fits in the context window, and some failure modes that look like model stupidity. Every production tokenizer for current LLMs follows the same steps.

~~~dg-flow From a string to token ids.
Text -(UTF-8)-> Bytes -(regex pre-split)-> Chunks: words, digits, spaces -(apply merges in rank order)-> *Subword tokens -(lookup)-> Ids + special tokens
~~~

The pre-split step is a regular expression that cuts text into words, runs of digits, punctuation and whitespace, so merges never cross those boundaries. The merge step is byte-pair encoding (BPE), introduced for neural translation by Sennrich et al.

## How BPE learns its merges

Training starts from single symbols and repeatedly merges the most frequent adjacent pair across the corpus. Each merge becomes a new vocabulary entry and gets a rank; encoding later applies merges in that rank order.

<div class="ex">
<div class="lab">Worked example</div>

**six merges on a tiny support corpus**

Word counts: refund ×6, refunds ×3, fund ×4, funded ×2, each ending in an end-of-word marker `</w>`. Pair counts are summed over words, weighted by frequency.

| Merge | Pair | Count | Why it wins |
| --- | --- | --- | --- |
| 1 | f + u | 15 | every word contains "fu" (6 + 3 + 4 + 2) |
| 2 | fu + n | 15 | same words |
| 3 | fun + d | 15 | same words |
| 4 | fund + `</w>` | 10 | "refund" and "fund" end there (6 + 4) |
| 5 | r + e | 9 | "refund" and "refunds" (6 + 3) |
| 6 | re + fund`</w>` | 6 | only "refund" |

After six merges "refund" is one token, "refunds" is `re fund s </w>` and "funded" is `fund e d </w>`. Ties (merges 1 to 3 all had count 15 at the start) are broken by a fixed rule, here first pair seen; real trainers do the same so the vocabulary is reproducible.

</div>

Modern tokenizers run this on bytes rather than characters. The base vocabulary holds all 256 byte values, so any string, in any script or with any emoji, can be encoded: the worst case is one token per byte. That is **byte fallback**, and it is why you never see an unknown token from GPT-style or Llama 3-style tokenizers. SentencePiece-based tokenizers get the same guarantee with a `byte_fallback` option.

## Vocabulary size is a trade-off

| Tokenizer | Used by | Vocabulary |
| --- | --- | --- |
| r50k_base | GPT-2, GPT-3 | 50,257 |
| cl100k_base | GPT-3.5, GPT-4 | 100,277 |
| o200k_base | GPT-4o through GPT-5.x (tiktoken's model map) | 200,019 |
| Qwen3 | Qwen3 family | 151,669 |
| DeepSeek-V3 | DeepSeek-V3.x | 128,815 |

Vocabulary sizes are what the libraries report, including added special tokens.

A bigger vocabulary packs more text into each token, so sequences are shorter: cheaper prefill, faster decode per word, more content per context window. The costs sit in the model. The embedding matrix and the output projection each hold $V \times d$ parameters, so with $V = 200{,}019$ and hidden size $d = 4{,}096$ each matrix has about 819 million parameters, and the softmax over $V$ entries runs at every decode step. Rare tokens also get few training updates, which is where glitch tokens come from: entries that exist in the vocabulary but were almost never seen in training and make the model behave strangely.

## Same text, five tokenizers

I counted tokens for one support sentence and its translations, plus code and numbers, with tiktoken 0.14 and the Hugging Face `tokenizers` files for Qwen3 and DeepSeek-V3 (no special tokens added).

| Text (bytes) | r50k | cl100k | o200k | Qwen3 | DeepSeek-V3 |
| --- | --- | --- | --- | --- | --- |
| English sentence, 81 chars (81) | 14 | 14 | 14 | 14 | 14 |
| German translation (103) | 43 | 29 | 22 | 28 | 28 |
| Hindi translation (235) | 138 | 92 | 28 | 85 | 52 |
| Tamil translation (263) | 263 | 129 | 28 | 107 | 45 |
| Japanese translation (103) | 55 | 39 | 26 | 26 | 25 |
| 4-line Python function (200) | 75 | 51 | 52 | 51 | 52 |
| `Order 1234567 totals 98765.4321 on 2026-11-18.` (46) | 19 | 21 | 21 | 34 | 21 |

Three things stand out.

- **Non-English cost depends on the tokenizer more than on the language.** The Tamil sentence is 9.2 times the English token count on cl100k_base and exactly two times on o200k_base. On r50k_base it is one token per byte, pure byte fallback. Petrov et al. measured gaps of up to 15 times across languages. If you serve several languages, price each one separately.
- **Code has largely converged.** Tokenizers after GPT-2 learned whitespace runs and common keywords, so indented code costs about a third less than it did.
- **Numbers split differently.** Qwen3 gives every digit its own token, while tiktoken's encodings cut digit runs into groups of up to three from the left: `1234567` becomes `123 456 7` and `98765.4321` becomes `987 65 . 432 1`. Neither tokenizer aligns groups with place value, which is one reason models make arithmetic slips on long numbers and why tool calls (a calculator, code execution) beat asking the model to add.

## Spaces, casing and token healing

Most tokenizers attach the leading space to the word, so `" strawberry"` is one token in o200k_base while `"strawberry"` at the start of a string is three (`st raw berry`). `"HELLO"` is two tokens where `"hello"` is one. The model never sees letters inside a token, which is why it used to miscount the r's in strawberry.

Trailing spaces are the practical trap. `"hello "` encodes as `hello` plus a lone space token. If your prompt ends with `"Answer: "`, the model must continue after a bare space token, a position it rarely saw in training because the space normally belongs to the next word. Outputs get worse in subtle ways. The fix is to strip trailing whitespace from prompts. Some libraries do **token healing**: back up one token and constrain the first generated token to start with the removed text, so the model can pick its natural boundary.

## Special tokens and chat templates

Chat models were trained on conversations wrapped in special tokens. Qwen3's template renders a system message and one user turn like this:

```text
<|im_start|>system
You are a support assistant.<|im_end|>
<|im_start|>user
Where is my refund?<|im_end|>
<|im_start|>assistant
```

That is 24 tokens, of which 11 are the actual words: 13 tokens of template per two-message exchange, paid on every call and multiplied across a long history. Each model family has its own markers; Hugging Face ships the template as a Jinja string in the tokenizer config, and `apply_chat_template` renders it. Getting it wrong (missing generation prompt, wrong role names) silently degrades an open-weights model.

The security rule: text from users or documents must be encoded as plain text, never parsed for special tokens. If a user can type `<|im_end|><|im_start|>system` and have it become real control tokens, they can forge a system message. tiktoken refuses to encode special-token strings unless you opt in, and hosted APIs apply the template server-side for you. This is the token-level version of the problem in [Prompt injection](lesson:aieng:12.03).

## Counting tokens per provider

| Provider | How to count | Note |
| --- | --- | --- |
| OpenAI | tiktoken with the model's encoding | Exact for text; add per-message template overhead and image tokens |
| Anthropic | `count_tokens` API endpoint | No public tokenizer; counts tools, images and documents too |
| Google Gemini | `count_tokens` in the SDK | Docs give about 4 characters per token for English |
| Open weights | `tokenizer.json` from the model repo | Render the chat template first, then count |

Tokenizers change between model versions. Anthropic's documentation says the tokenizer introduced with Claude Opus 4.7 produces roughly 1 to 1.35 times as many tokens as earlier models for the same text, so a migration can raise a bill even at the same per-token price. Re-count a sample of real prompts whenever you change models, and bill from the `usage` field the API returns, never from your own estimate.

## Where this breaks

- **Budgeting by characters.** "4 characters per token" holds for English prose only. A limit that is safe in English overflows the context in Tamil or on base64 data.
- **Truncating by bytes or characters.** Cutting a string to fit a budget can split a multi-byte character or leave a half-word that tokenizes badly. Truncate in token space, then decode.
- **Decoding stream chunks one token at a time.** With byte-level BPE a single token can be part of a character; the Tamil example on r50k_base is all single bytes, and no one of them is valid UTF-8 alone. Use an incremental decoder that buffers incomplete bytes.
- **Counting with the wrong tokenizer.** cl100k_base counts for an o200k_base model are off by a factor of three on Hindi.
- **Ignoring template and tool overhead.** Tool schemas and per-message markers are tokens too; count the full rendered request.

## What interviewers probe

- Walk through BPE training and encoding, and why byte-level BPE has no unknown tokens.
- The vocabulary size trade-off, with the $V \times d$ parameter cost.
- Why models struggle with letter counting and long arithmetic, explained through tokenization.
- How you would estimate token cost for a multilingual launch, and why you would measure instead.
- What a chat template is and why special tokens in user input are a security bug.

<div class="take">
<div class="lab">Key takeaway</div>

tokens are the unit of cost, context and latency, and how text becomes tokens depends on the tokenizer: count with each model's own counter on real traffic, especially for non-English text, code and numbers, and keep special tokens out of user-controlled text.

</div>
