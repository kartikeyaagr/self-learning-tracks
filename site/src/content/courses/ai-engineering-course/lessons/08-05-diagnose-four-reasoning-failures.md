---
id: "08.05"
course: "ai-engineering-course"
module: "08"
slug: "diagnose-four-reasoning-failures"
title: "Diagnose four reasoning failures"
keyIdeas: ["Failure 1, wrong step early: one arithmetic or logic slip propagates. Fix: step-level verification, tools (calculator, code), self-consistency.", "Failure 2, unfaithful reasoning: the written steps don't reflect what drove the answer (e.g. a hint in the prompt). Fix: don't rely on CoT as an explanation; check outputs independently.", "Failure 3, self-correction that makes things worse: asked to 'check your answer' without new information, models often flip correct answers. Fix: verify with external signals (tests, tools, a separate verifier).", "Failure 4, overthinking or underthinking: too few tokens on hard problems, endless loops or second-guessing on easy ones. Fix: set reasoning budgets per task, cap tokens, detect loops.", "Verification is easier than generation for many tasks: check with unit tests, a solver, a calculator, or a trained verifier.", "Always look at full traces for failed cases before changing prompts or models."]
question: "A reasoning model gets a maths answer wrong, then when asked 'are you sure?' changes a correct answer to a wrong one on another problem. Which failure is the second, and what's a better way to verify?"
answer: "The second case is self-correction backfiring: asked to re-check without any new information, the model flips a correct answer because pushback alone shifts its output (intrinsic self-correction often lowers accuracy). Better: verify with an external signal, e.g. recompute with a calculator or code, check against a solver or tests, or use a separate verifier/process reward model, and only revise when that check fails, passing the concrete error back."
sources: [{"title": "Lightman et al.: Let's Verify Step by Step (2023)", "url": "https://arxiv.org/abs/2305.20050", "note": "Step-level verification of reasoning."}, {"title": "Huang et al.: Large Language Models Cannot Self-Correct Reasoning Yet (2023)", "url": "https://arxiv.org/abs/2310.01798", "note": "Self-correction without external feedback often hurts."}, {"title": "Anthropic: Reasoning models don't always say what they think", "url": "https://www.anthropic.com/research/reasoning-models-dont-say-think", "note": "Measuring chain-of-thought faithfulness."}]
tags: ["Exercise"]
---

## How to use this lesson

Reasoning models fail in a few recognisable ways. When an output is wrong, read the whole trace and decide which of these four it is; each has a different fix.

## 1. An early wrong step

The model makes one arithmetic or logic error, then reasons carefully and confidently from the wrong value.

- **Signs**: the trace is coherent except for one step; the final answer is consistent with the mistake.
- **Fixes**: give it a calculator or code execution for arithmetic; sample several chains and vote (self-consistency); score steps with a verifier. Lightman et al. found step-level (process) supervision catches these much better than checking only the final answer.

## 2. Unfaithful reasoning

The visible reasoning doesn't reflect what actually determined the answer. Anthropic's study inserted hints into prompts; models often used the hint to change their answer but mentioned it in their reasoning only a minority of the time.

- **Signs**: the answer changes with irrelevant prompt details while the reasoning text doesn't mention them.
- **Fixes**: don't treat the chain of thought as an audit log. Evaluate outputs directly, and test with perturbed prompts.

## 3. Self-correction that backfires

Ask "are you sure? check again" with no new information and models often change correct answers to wrong ones. Huang et al. found that intrinsic self-correction, without external feedback, frequently lowered accuracy on reasoning benchmarks.

- **Signs**: accuracy drops after a "review your answer" step; answers flip under mild pushback.
- **Fixes**: verify with an **external signal**: run the code, check against a solver, use a separate verifier model, or retrieve a source. Only revise when the check fails.

## 4. Thinking too little or too much

On hard problems, a small reasoning budget cuts the chain before it reaches the answer. On easy problems, long budgets can lead to loops, repeated second-guessing and higher cost with no gain.

- **Signs**: truncated traces at max tokens; traces that circle ("wait... actually... wait").
- **Fixes**: set reasoning effort per task type, cap tokens, route easy queries to a cheaper non-reasoning model, detect repetition.

## The diagnosis table

| Failure | Typical sign | First fix |
| --- | --- | --- |
| Early wrong step | One bad step, rest consistent | Tools for arithmetic, voting, step verifier |
| Unfaithful reasoning | Answer moves with hidden cues | Judge outputs, not explanations |
| Self-correction backfire | Correct answers flip on "are you sure?" | External verification only |
| Wrong reasoning budget | Truncation or loops | Per-task budgets, routing |

~~~dg-flow A safer verification loop.
Model answer -(external check: tests, solver, tool)-> *Pass? -(yes)-> Return
Model answer -(check fails)-> Feedback with the error -(revise)-> New answer
~~~

## Verifiers set the ceiling

Many fixes in this lesson come down to "generate several candidates and check them". How much that helps depends on the checker. The figure computes exactly what best-of-n gives you for a given generator and verifier.

~~~dg-raw
%%widget verify Accuracy of best-of-n selection with an imperfect verifier.
~~~

Read the chart from both ends:

- The dashed line (pass@n) is what a perfect verifier would get: the chance that at least one candidate is right.
- Your verifier's line sits below it. With a high false-accept rate it flattens quickly: more samples just give the verifier more wrong answers to accept.

That is why research on process reward models (Lightman et al., 2023, "Let's Verify Step by Step") matters: a verifier that checks each step catches errors an outcome check misses. Their process-supervised reward model, used to pick among many solutions, solved 78% of a representative subset of the MATH test set, ahead of outcome supervision.

## A worked diagnosis

<div class="ex">
<div class="lab">Worked example</div>

**a support agent that refunds the wrong amount**

Symptom: on 6% of refund tickets the agent refunds the full order instead of one item.

Read the traces. In every failure the agent's first step reads "the customer wants a refund for order 5521" and never mentions the item. The rest of the chain is consistent with that wrong first step. This is failure 1, an early wrong step.

Fix options, in order of cost: (1) make the plan explicit, asking the agent to write "items to refund:" before calling any tool; (2) add a check before the refund tool runs that the amount matches the listed items; (3) sample two plans and escalate to a human if they disagree.

Option 2 alone took failures to near zero, because the check is a rule, not another model call.

</div>

## Two research results to keep in mind

- **Reasoning can be unfaithful.** Turpin et al. (2023) biased few-shot prompts so the answer was always "(A)". Models followed the bias and wrote plausible explanations that never mentioned it. A sensible-looking chain is not proof of why the model answered.
- **Self-correction without new information often fails.** Huang et al. (2023) found that asking a model to review and fix its own reasoning, with no external feedback, often made results worse: it changed right answers to wrong ones about as often as the reverse. Self-review works when it brings in new signal: a test result, a tool output, a retrieved document.

<div class="take">
<div class="lab">Key takeaway</div>

classify the failure before fixing it. Most reasoning fixes come from external checks (tools, tests, verifiers), not from asking the model to try harder.

</div>
