---
id: "08.02"
course: "ai-engineering-course"
module: "08"
slug: "chain-of-thought"
title: "Chain of thought"
keyIdeas: ["Chain of thought (CoT): have the model write intermediate reasoning before the final answer.", "Few-shot CoT (Wei et al.) puts worked examples with reasoning in the prompt; zero-shot CoT (Kojima et al.) adds 'Let's think step by step'.", "Mechanism: each generated token is another full forward pass, and intermediate results written to the context can be attended to by later tokens. The text is working memory.", "Self-consistency: sample several chains at temperature above 0 and take the majority answer. It helps when errors scatter and hurts when they agree.", "Reasoning models do CoT internally by default; for them, budgets and clear goals matter more than 'think step by step'. Written reasoning is not a faithful explanation."]
question: "Why does writing out intermediate steps improve accuracy for a next-token predictor? Give a mechanical explanation, not 'it thinks harder'."
answer: "A transformer does a fixed amount of computation per output token: one forward pass through every layer. If it must emit the answer immediately, the whole multi-step computation has to fit in that one pass. Writing intermediate steps gives it more forward passes before the answer, so more total compute, and stores partial results in the context, where later tokens can attend to them directly instead of recomputing them implicitly. A multi-step problem is decomposed into steps that each fit in one pass. That is why the gain is large for multi-step maths and logic, near zero for single-fact recall, and why each written step can also introduce an error that later steps build on."
rubric: ["States that compute per generated token is fixed (one forward pass), so a direct answer must fit the reasoning in one pass.", "Explains that each extra token adds another forward pass, increasing total computation before the answer.", "Explains that intermediate results become context that later tokens attend to (external working memory).", "Connects the mechanism to where CoT helps (multi-step problems) and where it does not (single-fact lookup)."]
sources: [{"title": "Wei et al.: Chain-of-Thought Prompting Elicits Reasoning (2022)", "url": "https://arxiv.org/abs/2201.11903", "note": "Few-shot examples with worked reasoning."}, {"title": "Kojima et al.: Large Language Models are Zero-Shot Reasoners (2022)", "url": "https://arxiv.org/abs/2205.11916", "note": "'Let's think step by step' alone."}, {"title": "Wang et al.: Self-Consistency (2022)", "url": "https://arxiv.org/abs/2203.11171", "note": "Sample many reasoning paths and take the majority answer."}]
practice: ["binomial-distribution", "pass-at-k-estimator"]
---

## The technique

Wei et al. (2022) put a few examples with worked reasoning into the prompt and asked the model to answer in the same style:

```text
Q: A cafe had 23 apples. They used 20 for lunch and bought 6 more. How many now?
A: They started with 23. After lunch: 23 - 20 = 3. After buying: 3 + 6 = 9. The answer is 9.
```

PaLM 540B on the GSM8K grade-school maths benchmark went from 17.9% with standard prompting to 56.9% with eight chain-of-thought examples. The gain appeared only in large models; small models wrote fluent but wrong chains. Kojima et al. (2022) found a trigger that needs no examples at all: append "Let's think step by step." It raised one model's accuracy on the MultiArith benchmark from 17.7% to 78.7%.

How to write the rest of the prompt (structure, examples, delimiters) is covered in [Prompting that holds up](lesson:aieng:03.06). This lesson is about why the reasoning text helps and how to spend more of it.

## Why it works, mechanically

A transformer does a fixed amount of computation per generated token: one forward pass through all its layers. If it must output the answer immediately, all the reasoning has to fit inside that one pass. Writing steps out helps in two ways:

- **More computation**: every intermediate token is another full forward pass.
- **External working memory**: intermediate results ("3") are written into the context, where later tokens can attend to them directly instead of recomputing them.

~~~dg-flow Direct answer vs chain of thought.
[Direct] Question -(1 forward pass)-> Answer token
[Chain of thought] Question -(pass)-> Step 1 -(pass, reads step 1)-> Step 2 -(pass, reads both)-> *Answer
~~~

This is also why CoT helps most on multi-step problems and barely at all on single-fact recall: a lookup already fits in one pass.

## Self-consistency: vote over paths

A single chain can go wrong at one step. Wang et al. sampled many chains at a temperature around 0.7, took each one's final answer and returned the most common. On GSM8K this raised PaLM 540B from 56.5% (one greedy chain) to 74.4% with 40 samples. It is an early example of spending more inference compute to get a better answer, the idea that reasoning models later built in ([Reasoning models and human feedback](lesson:aieng:08.01)).

Voting works because correct paths tend to converge on the same answer while errors scatter across many different wrong answers. When errors agree (a common misconception, an ambiguous question), voting locks in the wrong answer.

~~~dg-raw
%%widget selfconsistency Majority-vote accuracy as you add samples.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**when five votes help and when they hurt**

Take the worst case for voting: every wrong chain gives the same wrong answer, so it is a two-way race. With five samples, the vote is right when at least three chains are right. The count of right chains is binomial.

If one chain is right with $p = 0.6$:

$$
P(\ge 3 \text{ of } 5) = 10(0.6)^3(0.4)^2 + 5(0.6)^4(0.4) + (0.6)^5 = 0.346 + 0.259 + 0.078 = 0.683
$$

Voting lifts accuracy from 60% to 68%. If wrong answers scatter across several values, the right answer only needs a plurality and the gain is larger.

If one chain is right with $p = 0.4$ and the wrong chains agree, the same sum gives 0.230 + 0.077 + 0.010 = 0.317. Voting drops accuracy from 40% to 32%: the shared wrong answer now wins most votes.

The price is five chains of output. At 300 tokens a chain that is 1,500 output tokens instead of 300. Latency stays roughly the same if the five calls run in parallel.

</div>

Practical notes:

- Ask for the answer in a fixed, parseable form at the end ("Final answer: 42") so votes can be counted.
- Voting needs answers that compare cleanly: numbers, labels, short strings. For open-ended writing, use a verifier or judge to pick the best candidate instead ([Tree of thought](lesson:aieng:08.04) covers best-of-n).
- Cost is linear in samples. Five to ten samples get most of the benefit on many tasks.

## Chain of thought in reasoning models

Reasoning models (OpenAI's o-series and GPT-5 thinking modes, DeepSeek-R1, Claude with extended thinking, Gemini thinking models, Qwen3 in thinking mode) produce long chains by default, learned through RL. Adding "think step by step" to their prompts adds little. The useful controls are the reasoning effort or thinking budget, and a clear statement of the goal, constraints and output format. Several providers return only a summary of the reasoning or keep it hidden, so you cannot always read or reuse it. Budgets, cost of thinking tokens and when to switch thinking off are in [Using reasoning models in apps](lesson:aieng:08.06).

## Common mistakes

- **Treating the chain as an explanation.** The written steps can be a rationalisation. Models change their answer because of a hint in the prompt and often do not mention the hint ([Diagnose four reasoning failures](lesson:aieng:08.05)). Judge outputs, not the story told about them.
- **Ignoring compounding errors.** One wrong step early poisons everything after it. Search over steps and external checks address this ([Tree of thought](lesson:aieng:08.04)).
- **Paying for reasoning nobody needs.** Reasoning tokens are output tokens: they add latency and cost. For lookups, extraction and classification, ask for the answer directly.
- **Parsing the answer out of free text.** Without a fixed final-answer format, a vote or an eval silently miscounts.

<div class="take">
<div class="lab">Key takeaway</div>

Chain of thought gives the model more forward passes and a scratchpad in its own context. It is strongest on multi-step problems, gets better with self-consistency voting when errors scatter, and is not a faithful explanation of how the model reached its answer.

</div>
