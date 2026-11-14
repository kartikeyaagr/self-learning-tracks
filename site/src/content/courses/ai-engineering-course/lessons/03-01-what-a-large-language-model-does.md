---
id: "03.01"
course: "ai-engineering-course"
module: "03"
slug: "what-a-large-language-model-does"
title: "What a large language model does"
keyIdeas: ["An LLM takes a sequence of tokens and outputs a probability for every possible next token.", "Generating text is that step in a loop: pick a token, append it, predict again.", "Training on trillions of tokens of text forces the model to learn grammar, facts, style and some reasoning, because all of these help predict the next token.", "The model is a fixed set of weights (billions of parameters). It has no memory between calls except what is in the context window.", "Capabilities follow from prediction: translation, summarising, coding, following instructions after fine-tuning.", "Failures follow too: confident but false text (hallucination), knowledge cut-off, sensitivity to prompt wording."]
question: "An LLM is 'just predicting the next token'. Explain what that means and give one capability and one failure that follow from it."
answer: "Given the tokens so far, the model outputs a probability distribution over its vocabulary for the next token; generation samples one, appends it, and repeats. Capability: because predicting text well requires modelling its structure, it can translate, summarise or write code without task-specific training. Failure: it produces the most plausible continuation, not a verified one, so it can hallucinate confident false facts (and knows nothing after its training cut-off unless given context)."
sources: [{"title": "Andrej Karpathy: Intro to Large Language Models (video)", "url": "https://www.youtube.com/watch?v=zjkBMFhNj_g", "note": "One-hour talk: LLMs as next-token predictors and what that implies."}, {"title": "Stephen Wolfram: What Is ChatGPT Doing ... and Why Does It Work?", "url": "https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/", "note": "Long, clear essay on next-token probabilities."}, {"title": "3Blue1Brown: Large Language Models explained briefly", "url": "https://www.youtube.com/watch?v=LPZh9BOjkQs", "note": "Eight-minute visual overview."}]
---

## One job: the next token

A large language model is a function. Its input is a sequence of tokens; its output is a probability distribution over the whole vocabulary for the next token. Karpathy's intro talk frames it as a compressed model of the internet's text: a few hundred gigabytes of weights distilled from many terabytes of text.

To generate, you sample a token from that distribution, append it, and run the model again.

~~~dg-flow Generation is next-token prediction in a loop.
"The capital of France is" -(model)-> *P(next token) -(sample)-> "Paris" -(append, repeat)-> "Paris."
~~~

## Why prediction leads to ability

Wolfram's essay makes the point at length: to predict text well, the model has to capture the structure that produced the text. Predicting the next line of a proof needs some logic. Predicting a translation needs both languages. Predicting the reply in a dialogue needs a model of the speakers. None of these were trained directly; they emerge because they reduce prediction error.

## What a model is and isn't

- **Weights**: billions of numbers fixed after training. They store patterns and facts, imperfectly.
- **Context window**: the only working memory. The model knows about your conversation only because the whole conversation is sent on each call.
- **No lookup**: unless you give it tools or retrieved documents, it can't check a fact. It produces the most plausible continuation.

## Base model vs assistant

A base model trained only on raw text will continue a question with more questions, because that's plausible on the web. Assistant models are further trained on instruction-following examples and human feedback so that the most likely continuation of a question is a helpful answer. Lesson 03.04 and module 8 cover this.

## Consequences you will design around

| Property | Consequence for apps |
| --- | --- |
| Predicts plausible text | Can state false things fluently (hallucination) |
| Fixed training data | Knowledge cut-off; needs retrieval for fresh or private data |
| Context is the only memory | You must manage what goes in each prompt |
| Sampling is random | Same prompt can give different answers |

## Run the loop yourself

A language model does one thing: given the tokens so far, output a probability for every token in its vocabulary. Everything else is a loop around that. Click a candidate to append it and see the model run again on the longer text.

~~~dg-raw
%%widget nexttoken The generation loop with toy next-token probabilities.
~~~

The probabilities in the figure are hand-written, but the shape is realistic. After "The capital of France is", a real model puts almost all its probability on " Paris". After "My favourite food is", the probability is spread across hundreds of tokens. The first case looks like knowledge, the second like choice; to the model they are the same operation.

## Why one objective produces so much

To predict the next token well across trillions of tokens of text, the model has to compress the regularities behind the text: grammar, facts, the style of a legal contract, the way a Python function usually ends, the steps of a proof. Nobody programs these in. They are what lowers the prediction loss.

That also explains the failure modes you will design around:

- **Fluent errors.** The model outputs what is likely, not what is true. A plausible citation with the wrong year is a likely string.
- **No built-in "I don't know".** The loop always produces a next token. Abstaining has to be trained in or prompted for.
- **No memory between calls.** The model only sees what is in the context window this time. Chat history, user preferences and facts must be sent again on every call.
- **Knowledge cut-off.** Anything after the training data ends is unknown unless you put it in the prompt.

<div class="ex">
<div class="lab">Worked example</div>

**one token at a time, with real counts**

A 200-token answer means the model runs 200 forward passes, each reading the prompt plus everything generated so far (with a KV cache, module 4, it only computes the new token's part).

If one in fifty tokens is a slightly risky choice, a 200-token answer has about four. Most are harmless word choices; one wrong number in a table is not. That is why long, factual outputs need checks.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

an LLM predicts the next token, over and over. Its abilities and its failures both come from that one objective, so plan for plausible-but-wrong output and give it the context it needs.

</div>
