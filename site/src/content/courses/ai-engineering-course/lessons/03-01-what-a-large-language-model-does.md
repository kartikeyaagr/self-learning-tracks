---
id: "03.01"
course: "ai-engineering-course"
module: "03"
slug: "what-a-large-language-model-does"
title: "What a large language model does"
keyIdeas: ["An LLM takes a sequence of tokens and outputs a probability for every possible next token; generating text is that step in a loop.", "Training on trillions of tokens forces the model to learn grammar, facts, style and some reasoning, because all of these help predict the next token.", "The same pretrained network ships in different families: base, instruct, reasoning and multimodal, each post-trained for a different job.", "The weights are fixed. The context window (prompt plus output) is the only working memory, and the usable part is often smaller than the advertised limit.", "Failures follow from prediction too: confident but false text, a knowledge cut-off, and no built-in way to say 'I don't know'."]
question: "An LLM is 'just predicting the next token'. Explain what that means and give one capability and one failure that follow from it."
answer: "Given the tokens so far, the model outputs a probability distribution over its vocabulary for the next token; generation samples one, appends it, and runs the model again on the longer sequence until a stop token or limit. Capability: predicting text well across trillions of tokens requires modelling the structure behind it, so the model can translate, summarise or write code without task-specific training. Failure: it produces the most plausible continuation, not a verified one, so it can state false facts fluently, and it knows nothing after its training cut-off unless the facts are put into the context window. Because the weights never change between calls, everything it should know about your conversation has to be sent again each time."
rubric: ["Describes the output as a probability distribution over the vocabulary for the next token", "Explains generation as sample, append, repeat until a stop condition", "Names a capability and links it to what good prediction requires", "Names a failure (hallucination, cut-off, no abstention) and links it to plausibility rather than truth", "Mentions that the context window is the only memory between calls"]
sources: [{"title": "Andrej Karpathy: Intro to Large Language Models (video)", "url": "https://www.youtube.com/watch?v=zjkBMFhNj_g", "note": "One-hour talk: LLMs as next-token predictors, base vs assistant models."}, {"title": "Stephen Wolfram: What Is ChatGPT Doing ... and Why Does It Work?", "url": "https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/", "note": "Long, clear essay on next-token probabilities."}, {"title": "Hsieh et al.: RULER, What's the Real Context Size of Your Long-Context Language Models? (2024)", "url": "https://arxiv.org/abs/2404.06654", "note": "Effective context is often far shorter than the advertised window."}]
practice: ["text-generation-sampling", "gpt2-greedy-generate", "perplexity-computation", "tokenization"]
---

## One job: the next token

A large language model is a function. Its input is a sequence of tokens; its output is a probability distribution over the whole vocabulary (often 100k to 200k entries) for the next token. Karpathy's intro talk frames it as a compressed model of the internet's text: a few hundred gigabytes of weights distilled from many terabytes of text.

To generate, you sample a token from that distribution, append it, and run the model again on the longer sequence. Everything an LLM product does is a loop around this one step.

~~~dg-flow Generation is next-token prediction in a loop.
"The capital of France is" -(model)-> *P(next token) -(sample)-> "Paris" -(append, repeat)-> "Paris."
~~~

Click a candidate below to append it and watch the model run again on the longer text.

~~~dg-raw
%%widget nexttoken The generation loop with toy next-token probabilities.
~~~

The probabilities in the widget are hand-written, but the shape is realistic. After "The capital of France is", a real model puts almost all its probability on " Paris". After "My favourite food is", the probability is spread across hundreds of tokens. The first case looks like knowledge, the second like choice; to the model they are the same operation. How a token is chosen from the distribution is the subject of [LLM text generation](lesson:aieng:03.03).

## Why one objective produces so much

To predict the next token well across trillions of tokens, the model has to compress the regularities that produced the text. Predicting the next line of a proof needs some logic. Predicting a translation needs both languages. Predicting the reply in a dialogue needs a model of the speakers. Predicting how a Python function ends needs the language's syntax and common idioms. Nobody programs these in; they are simply what lowers the prediction loss, as Wolfram's essay argues at length. [How LLMs learn](lesson:aieng:03.02) covers the training itself.

## Model families

The pretrained network is rarely what you call. Labs post-train it into several products, and choosing the right family matters as much as choosing a size.

| Family | What it was trained to do | Typical use | 2025–26 examples |
| --- | --- | --- | --- |
| Base | Continue raw text | Fine-tuning starting point, research, few-shot completion | Llama 3.1/4 base, Qwen3 base checkpoints |
| Instruct (chat) | Follow instructions in a chat template, refuse unsafe requests | Most product features | GPT-4.1, Claude Sonnet 4.x, Gemini 2.5 Flash, Qwen3 instruct |
| Reasoning | Produce a long hidden or visible chain of thought before answering, trained with reinforcement learning | Maths, code, multi-step planning | o3, GPT-5 thinking, Claude with extended thinking, DeepSeek-R1, Qwen3 thinking mode |
| Multimodal | Read images, audio or video as tokens alongside text | Documents, screenshots, voice | GPT-4o, Gemini 2.5, Claude 4.x (image input), Qwen2.5-VL |

The lines blur: many current models are one checkpoint with a switch (Qwen3's thinking mode, Claude's extended thinking, GPT-5's reasoning effort), and most frontier chat models accept images. A base model is the odd one out in practice: ask it "What is the capital of France?" and it may continue with three more quiz questions, because that is a plausible continuation of a quiz on the web. Instruct training is what makes the most likely continuation of a question a helpful answer; [Making a model useful](lesson:aieng:03.04) covers that ladder, [Reasoning models and human feedback](lesson:aieng:08.01) covers reasoning training, and [Multimodal inputs](lesson:aieng:03.07) covers non-text tokens.

## What a model is and isn't

- **Weights**: billions of numbers fixed after training. They store patterns and facts, imperfectly, as of a training cut-off date.
- **Context window**: the only working memory. The model knows about your conversation only because the whole conversation is sent on each call.
- **No lookup**: unless you give it tools or retrieved documents, it cannot check a fact. It produces the most plausible continuation.

## Context windows

The context window is the maximum number of tokens the model can attend to in one call, and it is shared: system prompt, tool definitions, retrieved documents, conversation history and the generated output all count against it. Providers also cap output tokens separately, usually far below the window.

Windows grew fast. As of 2025–26, frontier APIs commonly offer 128k to 1M tokens (illustrative; for example Gemini 2.5 Pro at 1M, GPT-5 at 400k, Claude Sonnet 4.x at 200k with a 1M option), and open-weights models such as Qwen3 and Llama 4 ship long-context variants. Check the provider's model page; these numbers change with each release.

Three cautions apply to any advertised window:

- **Effective context is smaller.** RULER (Hsieh et al., 2024) found many models claiming 32k+ tokens fell well below their baseline quality long before the limit on tasks harder than finding one planted fact.
- **Cost and latency grow with it.** Every input token is billed and must be prefilled; attention memory grows with length ([The KV cache](lesson:aieng:04.04)).
- **Position matters.** Facts in the middle of a long prompt are used less reliably than those at the start or end ([Retrieval-augmented generation](lesson:aieng:03.05)).

<div class="ex">
<div class="lab">Worked example</div>

**how long a chat fits in a 200k window**

A support assistant sends a 3,000-token system prompt and 5,000 tokens of tool definitions on every call, and reserves 16,000 tokens for the reply. That leaves 200,000 − 3,000 − 5,000 − 16,000 = 176,000 tokens for history.

Each turn (user message, tool results, answer) adds about 1,500 tokens, so the conversation hits the limit after 176,000 / 1,500 ≈ 117 turns. Long before that it gets expensive: turn 50 re-sends 8,000 + 50 × 1,500 = 83,000 input tokens, because the model has no memory of turn 49.

The answer itself is 200 tokens, so 200 forward passes. If one token in fifty is a slightly risky choice, a 200-token answer has about four. Most are harmless word choices; one wrong number in a table is not. That is why long factual outputs need checks.

</div>

## Where this breaks

| Property | Consequence for apps |
| --- | --- |
| Predicts plausible text | States false things fluently (hallucination); a plausible citation with the wrong year is a likely string |
| No built-in "I don't know" | The loop always emits a next token; abstaining has to be trained in or prompted for |
| Fixed training data | Knowledge cut-off; needs retrieval or tools for fresh or private data |
| Context is the only memory | You must choose what goes into each call, trim history, and pay for re-sending it |
| Sampling is random | The same prompt can give different answers |

<div class="take">
<div class="lab">Key takeaway</div>

An LLM predicts the next token, over and over. Its abilities and its failures both come from that one objective, so pick the family that was post-trained for your job, treat the context window as a budget rather than a promise, and plan for plausible-but-wrong output.

</div>
