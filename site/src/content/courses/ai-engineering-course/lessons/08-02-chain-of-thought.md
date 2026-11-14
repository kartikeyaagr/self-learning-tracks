---
id: "08.02"
course: "ai-engineering-course"
module: "08"
slug: "chain-of-thought"
title: "Chain of thought"
keyIdeas: ["Chain of thought (CoT): have the model write intermediate reasoning before the final answer.", "Few-shot CoT (Wei et al.): include worked examples with reasoning in the prompt. Zero-shot CoT (Kojima et al.): add 'Let's think step by step'.", "Mechanism: each generated token gets a full forward pass, so more tokens means more computation, and intermediate results are written to the context where later tokens can attend to them.", "Gains are large for multi-step maths and logic, small for simple lookup tasks, and appeared mainly in larger models.", "Self-consistency: sample several reasoning paths at temperature > 0 and take the majority answer. Often a big accuracy boost.", "Caveats: the written reasoning isn't always faithful to how the model reached its answer, and CoT costs tokens and latency. Reasoning models do CoT internally by default."]
question: "Why does writing out intermediate steps improve accuracy for a next-token predictor? Give a mechanical explanation, not 'it thinks harder'."
answer: "A transformer does a fixed amount of computation per output token. Writing intermediate steps gives the model more forward passes (more total compute) before the answer, and stores partial results in the context where later tokens can attend to them instead of having to compute everything implicitly in one pass. So multi-step problems get decomposed into steps that each fit in one pass."
sources: [{"title": "Wei et al.: Chain-of-Thought Prompting Elicits Reasoning (2022)", "url": "https://arxiv.org/abs/2201.11903", "note": "Few-shot examples with worked reasoning."}, {"title": "Kojima et al.: Large Language Models are Zero-Shot Reasoners (2022)", "url": "https://arxiv.org/abs/2205.11916", "note": "'Let's think step by step' alone."}, {"title": "Wang et al.: Self-Consistency (2022)", "url": "https://arxiv.org/abs/2203.11171", "note": "Sample many reasoning paths and take the majority answer."}]
---

## The technique

Wei et al. (2022) showed that adding a few examples with worked reasoning to a prompt sharply improved large models on maths word problems. Kojima et al. found an even simpler trigger: append "Let's think step by step." to the question.

```
Q: A cafe had 23 apples. They used 20 for lunch and bought 6 more. How many now?
A: They started with 23. After lunch: 23 - 20 = 3. After buying: 3 + 6 = 9. The answer is 9.
```

## Why it works mechanically

A transformer does a fixed amount of computation per generated token: one forward pass through all layers. If it must output the answer immediately, all the reasoning has to fit inside that one pass. Writing steps out helps in two ways:

- **More computation**: every intermediate token is another full forward pass.
- **External working memory**: intermediate results ("3") are written into the context, where later tokens can attend to them directly instead of recomputing.

~~~dg-flow Direct answer vs chain of thought.
[Direct] Question -(1 forward pass)-> Answer token
[Chain of thought] Question -(pass)-> Step 1 -(pass, reads step 1)-> Step 2 -(pass, reads both)-> *Answer
~~~

That's also why CoT helps most on multi-step problems and barely at all on single-fact recall.

## Self-consistency

A single reasoning chain can go wrong at one step. Wang et al. sampled many chains at temperature > 0 and took a majority vote on the final answer. On GSM8K maths this added large gains over a single greedy chain. It's an early example of spending more inference compute to get better answers.

## Where it falls short

- **Cost**: reasoning tokens are output tokens; they add latency and price.
- **Faithfulness**: research has found the written reasoning can be a rationalisation: models sometimes change the answer because of a hint in the prompt without mentioning the hint. Don't treat CoT as a reliable explanation.
- **Errors compound**: one wrong step early poisons everything after it, which motivates verification (lesson 08.05) and tree search (08.04).

## CoT in reasoning models

Models like o1, o3, DeepSeek-R1 and Claude with extended thinking produce long chains of thought by default, trained with RL (lesson 08.01). For these, adding "think step by step" adds little; the useful controls are the reasoning budget and a clear statement of the goal and constraints.

## Why writing steps helps, mechanically

A transformer does a fixed amount of computation per token. A hard multi-step problem may need more computation than one forward pass provides. Writing intermediate steps gives the model more tokens, and so more passes, and each step's result becomes part of the context the next step can read. The reasoning text is working memory.

Measured effect, from Wei et al. (2022): PaLM 540B on the GSM8K grade-school maths benchmark went from 17.9% with standard prompting to 56.9% with eight chain-of-thought examples. The gain only appeared in large models; small models produced fluent but wrong chains.

Zero-shot chain of thought (Kojima et al., 2022) needs no examples, just "Let's think step by step." It raised one model's accuracy on the MultiArith benchmark from 17.7% to 78.7%.

## Self-consistency: vote over paths

Sample several reasoning chains at a temperature around 0.7, take each one's final answer, and return the most common one. On GSM8K, Wang et al. (2023) raised PaLM 540B from 56.5% (one greedy chain) to 74.4% with 40 samples.

The figure simulates voting. Set how often one sample is right and how scattered the wrong answers are.

~~~dg-raw
%%widget selfconsistency Majority-vote accuracy as you add samples.
~~~

Voting works because correct reasoning paths tend to converge on the same answer, while errors scatter across many different wrong answers. When errors agree (a common misconception, an ambiguous question), voting can lock in the wrong answer. Try p = 30% with a single wrong option in the figure.

## Practical notes

- Ask for the answer in a fixed, parseable form at the end ("Final answer: 42"), so votes can be counted.
- Voting needs answers that can be compared: numbers, labels, short strings. For open-ended writing, use a judge model to pick the best candidate instead.
- Cost is linear in samples. Five to ten samples get most of the benefit on many tasks.
- With reasoning models (o1, R1 and successors), the model already produces long internal chains. Adding "think step by step" helps less, and per-call reasoning budgets replace manual prompting.

<div class="take">
<div class="lab">Key takeaway</div>

chain of thought gives the model more forward passes and a scratchpad in its own context. It's strongest on multi-step problems, gets better with self-consistency voting, and isn't a faithful explanation.

</div>
