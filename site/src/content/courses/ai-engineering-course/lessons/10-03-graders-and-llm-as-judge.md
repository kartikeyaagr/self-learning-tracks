---
id: "10.03"
course: "ai-engineering-course"
module: "10"
slug: "graders-and-llm-as-judge"
title: "Graders and LLM-as-judge"
keyIdeas: ["Use the cheapest grader that is valid: code checks, then reference-based comparison, then execution, and an LLM judge only for criteria code cannot check.", "A reliable judge grades one criterion with a binary verdict, sees the reference facts, writes a short critique before its verdict, and learns from labelled pass and fail examples.", "Pointwise judging gives absolute pass rates for gating and monitoring; pairwise judging is more sensitive for choosing between two variants but must be run in both orders.", "Judges have known biases (position, length, self-preference, deference to confident tone), and each has a specific mitigation you can measure.", "Trust a judge only after measuring it against expert labels on held-out cases: report Cohen's kappa, and the true positive and true negative rates separately, because missed failures are the costly error."]
question: "You built an LLM judge for 'the answer is faithful to the retrieved documents'. On 200 expert-labelled outputs it agrees with the expert 85% of the time, and your team wants to start using it to gate releases. What do you check before agreeing, and what would make you reject it?"
answer: "Raw agreement hides the base rate. If 80% of outputs are faithful, a judge that always says pass agrees 80% of the time. I would split the 200 into the cases used to write the judge prompt and untouched test cases, then report on the test cases the confusion matrix, Cohen's kappa, and the true positive and true negative rates. For example, 150 true passes and 20 true failures caught, with 10 false fails and 20 failures waved through, is 85% agreement but kappa 0.48 and a true negative rate of 50%: the judge misses half the unfaithful answers, so I would reject it for gating. Fixes: add the expert's critiques of failing cases as few-shot examples, ask for the unsupported claim to be quoted before the verdict, and check for length bias and self-preference. Re-measure on fresh labelled cases, require a kappa around 0.7 or more with a high true negative rate, pin the judge model snapshot, and re-validate whenever the judge or the generator changes."
rubric: ["Notes that raw agreement is inflated by class imbalance and asks for kappa or chance-corrected agreement.", "Separates true positive and true negative rates and identifies missed failures as the critical error.", "Insists the judge is evaluated on labelled cases not used to tune its prompt.", "Proposes concrete improvements to the judge (few-shot critiques, quoting evidence, binary criterion, bias checks).", "Mentions pinning the judge version and re-validating on changes."]
sources: [{"title": "Zheng et al.: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (2023)", "url": "https://arxiv.org/abs/2306.05685", "note": "Strong judges reach human-level agreement; documents position, verbosity and self-enhancement bias and reference-guided grading."}, {"title": "Wang et al.: Large Language Models are not Fair Evaluators (2023)", "url": "https://arxiv.org/abs/2305.17926", "note": "Swapping answer order flips judge rankings; balanced position calibration."}, {"title": "Panickssery et al.: LLM Evaluators Recognize and Favor Their Own Generations (2024)", "url": "https://arxiv.org/abs/2404.13076", "note": "Self-preference bias and its link to self-recognition."}, {"title": "Hamel Husain: Creating a LLM-as-a-Judge That Drives Business Results", "url": "https://hamel.dev/blog/posts/llm-judge/", "note": "Binary verdicts with expert critiques, iterating the judge until it agrees with the expert."}]
practice: ["judge-human-agreement-kappa", "pairwise-judge-position-swap", "remove-response-length-bias-from-judge-scores", "fit-elo-ratings-from-pairwise-votes", "cohens-kappa"]
tags: ["LLM evaluation", "LLM-as-judge"]
---

## The grader ladder

Each failure mode found in error analysis ([Why LLM evals are different](lesson:aieng:10.01)) needs a grader. Climb the ladder only as far as the criterion requires: lower rungs are cheaper, faster, deterministic and never need validating.

| Grader | Examples | Cost per case | Use when |
| --- | --- | --- | --- |
| Code assertion | Valid JSON against a schema, required tool called, no URL outside an allowlist, length under a limit, contains a ticket id | ~0 | The property is structural or checkable from the output and the trace |
| Reference-based | Exact or normalised match, numeric tolerance, field-level F1 for extraction, set overlap of cited ids | ~0 | There is one right answer, or a small set of required facts |
| Execution-based | Run generated code against unit tests, run SQL and compare result sets, render a chart and check its data | Compute, seconds | The output is a program; running it is the truest check |
| LLM judge | Faithfulness to sources, policy correctness, tone, whether a question was answered | One model call per case per criterion | The criterion needs reading comprehension |
| Human expert | Final validation, ambiguous domains, calibrating the judge | Minutes per case | Ground truth, and spot checks forever |

Many judged criteria can be partly pushed down the ladder. "Cites its sources" becomes a code check that every citation id exists in the retrieved set, with a judge only for whether the cited text supports the claim. Text similarity (BLEU, ROUGE, embedding cosine) is cheap but punishes valid paraphrases, so treat it as a weak signal.

## Writing a judge that holds up

A judge is a prompt, and the rules for reliable prompts ([Prompting that holds up](lesson:aieng:03.06)) apply, plus a few specific to grading:

1. **One criterion per judge**, with a binary verdict. "Rate the answer 1 to 10" mixes correctness, tone and length into a number the judge assigns inconsistently; PASS/FAIL is easier to validate and is already a pass rate.
2. **Give it the reference.** A judge cannot know your refund policy. Zheng et al. found judges graded maths questions far better when shown a reference solution. Pass the required facts from the eval case.
3. **Critique before verdict.** Ask for a short justification that quotes the deciding sentence, then the verdict. The quote makes errors auditable and tends to improve accuracy. G-Eval (Liu et al., 2023) used the same idea with generated evaluation steps.
4. **Few-shot with real critiques.** Include two or three passing and failing examples with the expert's own explanations, taken from the dev split of the labelled data, never from the test split.
5. **Constrain the output.** Use schema-constrained output for `{critique, verdict}` ([Structured outputs](lesson:aieng:12.01)), temperature 0 where the model allows it, and a pinned model snapshot.

```text
You grade one reply from a retail support assistant on ONE criterion.

Criterion: policy_correct
PASS if every policy statement in the reply agrees with the policy facts.
FAIL if the reply contradicts a policy fact, or states a time limit, fee or
exception that the facts do not contain. Ignore tone, length and formatting.

Policy facts: {reference.must}
The reply must not claim: {reference.must_not}
Conversation: {messages}
Reply: {reply}

Examples (expert-graded):
{two PASS and two FAIL examples, each with the expert's critique}

Write a critique of at most three sentences that quotes the sentence
deciding the verdict, then return JSON: {"critique": "...", "verdict": "PASS" or "FAIL"}
```

## Pointwise or pairwise

| | Pointwise | Pairwise |
| --- | --- | --- |
| Question | "Does this output pass criterion X?" | "Which of these two outputs is better on X?" |
| Output | Absolute pass rate per criterion | Win rate of A over B |
| Good for | Release gates, monitoring, per-slice reports | Choosing between two prompts or models; arena-style rankings |
| Sensitivity | Misses small improvements | Detects small differences; easier for raters too |
| Main risk | Score drift and leniency | Position bias; needs both orders, doubling cost |

Use pointwise as the default: it gives numbers that mean the same thing next month. Use pairwise when two variants both pass most checks and you need to know which is better. With more than two systems, fit Bradley-Terry or Elo ratings to the resolved comparisons.

## Biases and their fixes

| Bias | What you see | Mitigation |
| --- | --- | --- |
| Position | The response in one slot wins more often than chance | Judge both orders; count only consistent verdicts |
| Length (verbosity) | Longer answers score higher at equal quality | Rubric that penalises padding; regress scores on length with a per-system intercept |
| Self-preference | A judge rates outputs from its own model family higher | Judge from a different family, or a panel of judges |
| Confident tone | Assertive, well-formatted wrong answers pass | Reference facts in the prompt; require quoting evidence |
| Rubric sensitivity | Small wording changes move scores | Binary criteria with examples; version the judge prompt |
| Leniency drift | Pass rate shifts after a provider update | Pin the snapshot; re-validate against labels on every judge change |

Wang et al. showed that changing only the order of two answers was enough to make a weaker model appear to beat ChatGPT on 66 of 80 questions, with ChatGPT as the judge. Panickssery et al. found that models recognise their own outputs better than chance and that self-preference grows with that ability. Length-Controlled AlpacaEval (Dubois et al., 2024) removes length effects from a leaderboard by regression, the same idea as the `remove-response-length-bias-from-judge-scores` problem.

<div class="ex">
<div class="lab">Worked example</div>

**Position bias in a prompt comparison**

A new prompt (A) is compared with the current one (B) on 100 cases, A shown first. The judge prefers A in 64, B in 30, ties 6: a naive win rate of $(64 + 3)/100 = 67\%$.

Re-running with B first gives, per pair:

| Original / swapped | Pairs | Resolved |
| --- | --- | --- |
| A wins both orders | 40 | A |
| B wins both orders | 22 | B |
| Same slot both times ("1"/"1" or "2"/"2") | 24 | tie |
| Tie in both | 6 | tie |
| Winner in one order, tie in the other | 8 | tie |

Resolved win rate: $(40 + 0.5 \times 38)/100 = 59\%$. Position inflated A's lead by 8 points. Across all 180 non-tie verdicts the judge chose slot 1 in 106 (59%), about 2.4 standard deviations above 50%. A is probably still better, but 38 pairs are close calls this judge cannot separate; read them.

</div>

## Calibrating the judge against humans

A judge is a pass/fail classifier; validate it like one.

~~~dg-flow Aligning a judge with a domain expert. Tune on the dev split, report on the test split.
Traces -(expert labels pass/fail + critiques)-> Labelled set -(split)-> Dev cases -(write and revise judge prompt)-> *Judge v1, v2, ... -(score untouched test cases)-> Kappa, TPR, TNR -(good enough)-> Judge in the eval suite
~~~

1. Have one domain expert label 100 to 200 outputs pass/fail with a one-line critique each. Include enough failures to measure the judge on them, oversampling flagged traces if failures are rare.
2. Split into dev and test. Iterate the judge prompt on dev until it stops improving.
3. Score the test cases and compute the confusion matrix, Cohen's $\kappa$, the true positive rate (TPR, passes it passes) and the true negative rate (TNR, failures it catches).
4. Use the judge's error rates to correct the pass rates it reports.

<div class="ex">
<div class="lab">Worked example</div>

**Agreement that looked fine**

200 test cases; the expert passed 160 and failed 40.

| | Judge PASS | Judge FAIL |
| --- | --- | --- |
| Expert PASS | 150 | 10 |
| Expert FAIL | 20 | 20 |

- Agreement $p_o = 170/200 = 0.85$.
- Chance agreement from the marginals: $p_e = 0.80 \times 0.85 + 0.20 \times 0.15 = 0.71$.
- $\kappa = (p_o - p_e)/(1 - p_e) = 0.14/0.29 = 0.48$, only moderate on the common Landis and Koch scale.
- TPR $= 150/160 = 0.94$, TNR $= 20/40 = 0.50$. The judge waves through half the failures.

On 2,000 unlabelled production traces it reports 88% passing. The judge passes a true pass with probability TPR and a true failure with probability $1 - \text{TNR}$, so the true rate $\theta$ satisfies $p_{obs} = \theta\,\text{TPR} + (1-\theta)(1-\text{TNR})$:

$$
\theta = \frac{p_{obs} + \text{TNR} - 1}{\text{TPR} + \text{TNR} - 1} = \frac{0.88 + 0.50 - 1}{0.9375 + 0.50 - 1} = 0.869
$$

After adding the expert's critiques of failing cases as examples and requiring the unsupported claim to be quoted, the test split gives 152, 8, 6, 34: $\kappa = 0.79$, TNR $= 0.85$. That judge can gate releases.

</div>

Two cautions. TNR here rests on only 40 failures, so its own interval is wide (about ±11 points), which is why step 1 oversamples failures. And the target is bounded by how well humans agree: if two experts reach only $\kappa = 0.6$ with each other ([Building an eval set](lesson:aieng:10.02)), fix the labelling guide before chasing a better judge.

## Where this breaks

- **Judging what code could check.** A judge asked "is this valid JSON?" is slower, dearer and sometimes wrong.
- **Overfitting the judge.** Tuning the judge prompt on the cases you report agreement on gives a flattering number. Keep a test split, and refresh it as outputs change.
- **Optimising against the judge.** When the judge's score becomes the target (prompt search, preference tuning in [Preference tuning for products](lesson:aieng:07.09)), the system finds the judge's blind spots: padding, confident hedging, rubric keywords. Keep human spot checks and a second, independent grader.
- **Unpinned judges.** A judge alias that updates silently changes every historical comparison. Version the judge model, prompt and examples together.
- **Cost.** 400 cases × 4 criteria × 3 repeats is 4,800 judge calls per run, doubled for pairwise swaps. Run code checks on every commit and judges on release candidates ([Regression testing in CI](lesson:aieng:10.06)).

## What interviewers probe

- "How do you know your judge is right?" Expert labels, a held-out split, kappa with TPR and TNR, re-validation on changes.
- "Why not 1 to 10?" Inconsistent use of the scale and no clear pass line.
- "Can GPT-5 judge GPT-5?" Measure self-preference, or use another family.
- "Your judge and users disagree." Relabel a sample; the judge is what gets fixed.

<div class="take">
<div class="lab">Key takeaway</div>

Grade with code wherever possible and use an LLM judge only for criteria that need reading. Make each judge one binary criterion with reference facts, critique before verdict and expert examples, run pairwise comparisons in both orders, and trust a judge only after measuring kappa, TPR and TNR on expert labels it was not tuned on.

</div>
