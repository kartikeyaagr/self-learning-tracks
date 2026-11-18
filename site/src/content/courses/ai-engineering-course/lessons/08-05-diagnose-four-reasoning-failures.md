---
id: "08.05"
course: "ai-engineering-course"
module: "08"
slug: "diagnose-four-reasoning-failures"
title: "Diagnose four reasoning failures"
keyIdeas: ["Failure 1, wrong step early: one arithmetic or logic slip propagates. Fix: tools for arithmetic, step-level verification, voting.", "Failure 2, unfaithful reasoning: the written steps don't reflect what drove the answer (e.g. a hint in the prompt). Fix: judge outputs, not explanations; test with perturbed prompts.", "Failure 3, self-correction that backfires: asked to 'check your answer' without new information, models often flip correct answers. Fix: verify with external signals (tests, tools, a separate verifier).", "Failure 4, thinking too little or too much: truncated chains on hard problems, loops and second-guessing on easy ones. Fix: per-task reasoning budgets, token caps, loop detection, routing.", "Verification is easier than generation for many tasks, and the verifier sets the ceiling. Always read full traces of failed cases before changing prompts or models."]
question: "A reasoning model gets a maths answer wrong, then when asked 'are you sure?' changes a correct answer to a wrong one on another problem. Which failure is the second, and what's a better way to verify?"
answer: "The second case is self-correction backfiring: asked to re-check without any new information, the model flips a correct answer because pushback alone shifts its output, and intrinsic self-correction often lowers accuracy on reasoning benchmarks. A better design verifies with an external signal: recompute the arithmetic with a calculator or code, check against a solver or unit tests, or score the answer with a separate verifier or process reward model. Only ask for a revision when that check fails, and pass the concrete error back ('step 3: 17 x 4 is 68, not 64') so the revision has new information to act on. The first case should be read in the trace to see whether it is an early wrong step, which tools or step-level checks would catch."
rubric: ["Names the second failure as self-correction without external feedback (intrinsic self-correction).", "Explains why it happens: no new information, so the challenge itself shifts the answer.", "Proposes an external check (code, calculator, solver, tests, separate verifier or PRM).", "Revises only when the check fails and feeds the specific error back."]
sources: [{"title": "Lightman et al.: Let's Verify Step by Step (2023)", "url": "https://arxiv.org/abs/2305.20050", "note": "Step-level verification of reasoning."}, {"title": "Huang et al.: Large Language Models Cannot Self-Correct Reasoning Yet (2023)", "url": "https://arxiv.org/abs/2310.01798", "note": "Self-correction without external feedback often hurts."}, {"title": "Anthropic: Reasoning models don't always say what they think", "url": "https://www.anthropic.com/research/reasoning-models-dont-say-think", "note": "Measuring chain-of-thought faithfulness."}, {"title": "Turpin et al.: Language Models Don't Always Say What They Think (2023)", "url": "https://arxiv.org/abs/2305.04388", "note": "Biased prompts change answers without appearing in the explanation."}]
practice: ["pass-at-k-estimator", "grpo-process-supervision-loss", "evaluate-agent-trajectory"]
tags: ["Exercise"]
---

## How to use this lesson

Reasoning models fail in a few recognisable ways. When an output is wrong, read the whole trace and decide which of these four it is; each has a different fix.

## 1. An early wrong step

The model makes one arithmetic or logic error, then reasons carefully and confidently from the wrong value.

- **Signs**: the trace is coherent except for one step; the final answer is consistent with the mistake.
- **Fixes**: give it a calculator or code execution for arithmetic; sample several chains and vote ([Chain of thought](lesson:aieng:08.02)); score steps with a verifier. Lightman et al. found step-level (process) supervision catches these much better than checking only the final answer.

## 2. Unfaithful reasoning

The visible reasoning doesn't reflect what actually determined the answer. Turpin et al. (2023) biased few-shot prompts so the right answer was always "(A)"; models followed the bias and wrote plausible explanations that never mentioned it. Anthropic later inserted hints into prompts for reasoning models: averaged over hint types, Claude 3.7 Sonnet mentioned a hint it had used 25% of the time and DeepSeek-R1 39% of the time.

- **Signs**: the answer changes with irrelevant prompt details while the reasoning text doesn't mention them.
- **Fixes**: don't treat the chain of thought as an audit log. Evaluate outputs directly, and test with perturbed prompts (reorder options, remove the hint, change a name).

## 3. Self-correction that backfires

Ask "are you sure? check again" with no new information and models often change correct answers to wrong ones. Huang et al. (2023) found that intrinsic self-correction, without external feedback, frequently lowered accuracy on reasoning benchmarks: right answers flipped to wrong about as often as the reverse. Self-review works when it brings in new signal: a test result, a tool output, a retrieved document.

- **Signs**: accuracy drops after a "review your answer" step; answers flip under mild pushback.
- **Fixes**: verify with an **external signal**: run the code, check against a solver, use a separate verifier model, or retrieve a source. Only revise when the check fails.

## 4. Thinking too little or too much

On hard problems, a small reasoning budget cuts the chain before it reaches the answer. On easy problems, long budgets lead to loops, repeated second-guessing and higher cost with no gain.

- **Signs**: truncated traces at max tokens; traces that circle ("wait... actually... wait").
- **Fixes**: set reasoning effort per task type, cap tokens, route easy queries to a cheaper non-reasoning model, detect repetition. Budget settings and their cost are in [Using reasoning models in apps](lesson:aieng:08.06).

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

Many fixes here come down to "generate several candidates and check them" ([Tree of thought](lesson:aieng:08.04) works through the arithmetic). How much that helps depends on the checker. The figure computes what best-of-n gives you for a given generator and verifier.

~~~dg-raw
%%widget verify Accuracy of best-of-n selection with an imperfect verifier.
~~~

- The dashed line (pass@n) is what a perfect verifier would get: the chance that at least one candidate is right.
- Your verifier's line sits below it. With a high false-accept rate it flattens quickly: more samples just give the verifier more wrong answers to accept.

That is why process reward models matter: a verifier that checks each step catches errors an outcome check misses. Lightman et al.'s process-supervised reward model, used to pick among many solutions, solved 78% of a representative subset of the MATH test set, ahead of outcome supervision.

## A worked diagnosis

<div class="ex">
<div class="lab">Worked example</div>

**a support agent that refunds the wrong amount**

Symptom: on 6% of refund tickets the agent refunds the full order instead of one item.

Read the traces. In every failure the agent's first step reads "the customer wants a refund for order 5521" and never mentions the item. The rest of the chain is consistent with that wrong first step. This is failure 1, an early wrong step.

Fix options, in order of cost: (1) make the plan explicit, asking the agent to write "items to refund:" before calling any tool; (2) add a check before the refund tool runs that the amount matches the listed items; (3) sample two plans and escalate to a human if they disagree.

Option 2 alone took failures to near zero, because the check is a rule, not another model call.

</div>

## Common mistakes

- **Changing the prompt before reading traces.** Without classifying the failure you are as likely to fix the wrong thing.
- **Adding a "double-check your work" step.** Without new information it is failure 3 by design.
- **Trusting the model's explanation of its own error.** That explanation is subject to failure 2.

<div class="take">
<div class="lab">Key takeaway</div>

Classify the failure before fixing it. Most reasoning fixes come from external checks (tools, tests, verifiers), not from asking the model to try harder.

</div>
