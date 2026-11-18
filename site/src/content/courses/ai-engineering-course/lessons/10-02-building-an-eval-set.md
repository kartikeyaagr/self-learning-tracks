---
id: "10.02"
course: "ai-engineering-course"
module: "10"
slug: "building-an-eval-set"
title: "Building an eval set"
keyIdeas: ["An eval case is an input plus everything needed to grade it: context and tool fixtures, a reference or list of required facts, the criteria that apply, and slice tags.", "Draw cases from production logs (stratified, not uniform), past failures, domain experts, dimension-driven synthetic generation and adversarial probes; each source covers a gap the others leave.", "Size the set from confidence intervals: the 95% margin on a pass rate is about 1.96 times the square root of p(1-p)/n, so 50 cases at 80% gives plus or minus 11 points and 400 gives plus or minus 4.", "Comparing two variants needs far fewer cases when both run on the same cases (paired), and far more effective cases when cases come in correlated clusters.", "Version the set like code, keep a dev split you iterate on apart from a held-out split you report on, and guard against leakage into prompts, few-shot examples and training data."]
question: "You have 300 eval cases for a support assistant, stratified across six intents, with the rare 'escalation' intent deliberately oversampled. The raw pass rate across the set is 78%. Your manager wants one number for the launch review and asks whether 300 cases is enough to compare the current prompt with a candidate that might be 5 points better. What do you report and what do you recommend?"
answer: "The 78% is not the number to report, because oversampling rare intents makes the set harder than real traffic. Reweight each intent's pass rate by its share of traffic (a stratified estimate) and report that with its interval, plus the per-intent rates, flagging small slices whose intervals are wide (40 escalation cases at 50% is roughly plus or minus 15 points). For the comparison, two independent samples would need about 1,000 cases per variant to detect a 5-point change at 95% confidence and 80% power, so 300 is not enough that way. But both prompts can run on the same 300 cases. Then only the cases where they disagree carry information, and with about 15% of cases flipping, roughly 460 cases detect 5 points. I would run both variants on the same cases with several repeats, test the paired difference, read every flipped case, and grow the set toward 500 from production logs, prioritising the slices with the widest intervals."
rubric: ["Explains why the raw average of an oversampled set misstates production quality and reweights by traffic share.", "Reports intervals overall and per slice, and flags slices too small to support conclusions.", "Computes or correctly estimates that independent comparison of a 5-point change needs around 1,000 cases per variant.", "Recommends a paired comparison on the same cases and explains that discordant cases drive its precision.", "Proposes concrete next steps: repeats, reading flipped cases, targeted growth of weak slices."]
sources: [{"title": "Miller: Adding Error Bars to Evals (2024)", "url": "https://arxiv.org/abs/2411.00640", "note": "Standard errors, clustered errors, paired differences and power analysis for LLM evals."}, {"title": "Hamel Husain and Shreya Shankar: Frequently asked questions about AI evals", "url": "https://hamel.dev/blog/posts/evals-faq/", "note": "Sampling traces, dimension-based synthetic data, stale gold sets, sensitive data."}, {"title": "Biderman et al.: Lessons from the Trenches on Reproducible Evaluation of Language Models (2024)", "url": "https://arxiv.org/abs/2405.14782", "note": "Why small setup details change scores, and what to record so results reproduce."}, {"title": "OpenAI docs: Evaluation best practices", "url": "https://platform.openai.com/docs/guides/evaluation-best-practices", "note": "Building eval datasets from production data, synthetic data and expert labels."}]
practice: ["stratified-sampling", "bootstrap-confidence-interval", "binomial-distribution", "text-deduplication-minhash"]
tags: ["LLM evaluation"]
---

## What an eval case contains

An eval case is not just a prompt. It is everything needed to reproduce the system's behaviour on one input and grade it. A case for a support assistant, stored as one JSONL line:

```json
{"id": "ret-0142", "version": 3, "source": "prod-log", "added": "2026-09-12",
 "slices": ["returns", "no-receipt", "multi-turn"],
 "input": {"messages": [{"role": "user", "content": "it was a gift, no receipt, can I still swap it?"}],
           "user": {"tier": "standard", "country": "UK"}},
 "fixtures": {"lookup_order": {"error": "not_found"}},
 "reference": {"must": ["exchange or store credit is possible without a receipt", "30-day window"],
               "must_not": ["promises a cash refund"]},
 "criteria": ["policy_correct", "asks_for_gift_details", "tone"]}
```

Four parts matter. **Fixtures** pin what tools and retrieval return, so the case tests the model's behaviour rather than whether the order database was up. **References as required facts** ("must mention", "must not claim") grade more reliably than a full reference answer, because there are many good phrasings and few required facts. **Criteria** say which checks apply, since not every check is relevant to every case. **Slices and source** let you report per segment and trace where a case came from.

## Where cases come from

| Source | What it covers | Watch out for |
| --- | --- | --- |
| Production logs | Real phrasing, real distribution, messy inputs | Uniform samples are dominated by easy, common intents; personal data must be scrubbed or consented |
| Past failures (bugs, thumbs-down, escalations) | Exactly what broke before | Over-represents yesterday's problems if never pruned |
| Domain experts | Correct references, hard policy edge cases | Expensive; experts write cleaner inputs than users |
| Synthetic generation | Coverage before launch, rare combinations | Homogeneous, too polite and too fluent unless structured |
| Adversarial probes | Injection, jailbreaks, off-topic, abuse | Can crowd out ordinary quality cases if unbounded |

Two techniques make these sources work. For **logs**, stratify: sample within each intent, feature or user segment, and deliberately take more from rare but important strata. Also sample by signal: traces with negative feedback, long conversations, tool errors or a low judge score are richer in failures than random ones. For **synthetic** data, do not ask a model for "100 diverse customer questions"; you get 100 variations of the same polite question. Define dimensions first (intent × persona × complication, for example "returns × first-time buyer × no receipt"), enumerate or sample combinations, generate one input per combination, and have a person discard unrealistic ones. Synthetic cases are best for launching and for filling slices that logs leave thin; replace them with real traffic once you have it.

## Labelling and rubrics

Whoever labels defines what "correct" means, so choose one accountable domain expert per area, not a crowd. Write a labelling guide with each criterion's definition and two or three passing and failing examples. Have a second person double-label a random 50 to 100 cases and measure agreement with Cohen's $\kappa$ ([Graders and LLM-as-judge](lesson:aieng:10.03) works through the computation). Low agreement means the guide is ambiguous, and no grader, human or model, can be more reliable than the labels it is checked against. Resolve disagreements by editing the guide, not by voting.

Prefer binary labels per criterion (passes "policy_correct" or not) to a 1-to-5 score. Binary labels are faster to assign, agree better between people, and map directly onto pass rates you can put intervals on.

## How many cases

Every number an eval produces is an estimate from a sample of possible inputs. For a pass rate $\hat p$ on $n$ independent cases, the 95% margin of error is

$$
m = 1.96\sqrt{\frac{\hat p(1-\hat p)}{n}}
$$

Inverting it gives the cases needed for a target margin: $n = 1.96^2\,p(1-p)/m^2$. Two adjustments matter in practice. When $\hat p$ is near 0 or 1 or $n$ is small, use the Wilson interval, which stays inside $[0, 1]$:

$$
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

And when cases come in clusters that share context (ten questions about one document, several turns of one conversation), they are not independent. With $c$ cases per cluster and intra-cluster correlation $\rho$, the variance grows by the design effect $1 + (c-1)\rho$, and the effective sample size shrinks by the same factor. Miller's paper on eval error bars gives the clustered standard error directly.

Comparing two variants is a different question from measuring one. Detecting a difference $\delta$ at 95% confidence with 80% power between two **independent** samples needs about

$$
n = (1.96 + 0.84)^2 \cdot \frac{2p(1-p)}{\delta^2}
$$

cases per variant. If instead both variants run on the **same** cases, only the cases where they disagree carry information. With $q$ the fraction of discordant cases, the paired requirement is about $(1.96+0.84)^2 (q - \delta^2)/\delta^2$.

<div class="ex">
<div class="lab">Worked example</div>

**Sizing a pass-rate eval at around 80%**

| Question | Calculation | Answer |
| --- | --- | --- |
| Margin with 50 cases | $1.96\sqrt{0.8 \times 0.2 / 50}$ | ±11.1 points |
| Margin with 400 cases | $1.96\sqrt{0.8 \times 0.2 / 400}$ | ±3.9 points |
| Cases for ±3 points | $1.96^2 \times 0.16 / 0.03^2$ | 683 |
| Cases for ±5 points, unknown $p$ (worst case 0.5) | $1.96^2 \times 0.25 / 0.05^2$ | 385 |
| 49 of 50 pass: naive interval | $0.98 \pm 0.039$ | 94.1% to 101.9% (impossible) |
| 49 of 50 pass: Wilson | formula above | 89.5% to 99.6% |
| Detect +5 points, independent samples | $7.85 \times 0.32 / 0.0025$ | 1,005 per variant |
| Detect +5 points, same cases, 15% discordant | $7.85 \times (0.15 - 0.0025) / 0.0025$ | 464 cases |
| 400 cases from 40 conversations of 10 turns, $\rho = 0.3$ | deff $= 1 + 9 \times 0.3 = 3.7$ | effective $n = 108$, margin ±7.5 |

Two lessons from the table. A 50-case set catches a prompt that collapses, and nothing finer. And running both variants on the same cases halves the cases needed, which is why regression testing is always paired ([Regression testing in CI](lesson:aieng:10.06) does the paired bootstrap). For a refresher on what a 95% interval does and does not promise, see [What a confidence interval means](lesson:mlmaths:12.17).

</div>

## Slices, coverage and the number you report

Plan coverage as a grid of slices (intent, user segment, language, input length, difficulty) and give every slice you will make decisions about enough cases to be readable. Then remember what oversampling does to the average.

<div class="ex">
<div class="lab">Worked example</div>

**A 300-case set where the raw average misleads**

| Intent | Traffic share | Cases | Passed | Pass rate | 95% margin |
| --- | --- | --- | --- | --- | --- |
| Billing | 40% | 70 | 63 | 90.0% | ±7.0 |
| Shipping | 25% | 55 | 47 | 85.5% | ±9.3 |
| Returns | 15% | 50 | 40 | 80.0% | ±11.1 |
| Account | 10% | 45 | 36 | 80.0% | ±11.7 |
| Product questions | 7% | 40 | 28 | 70.0% | ±14.2 |
| Escalation | 3% | 40 | 20 | 50.0% | ±15.5 |

Raw pass rate: 234/300 = **78.0%**. But rare, hard intents were deliberately oversampled, so the set is harder than real traffic. The stratified estimate weights each slice by traffic, $\hat p = \sum_h w_h \hat p_h$:

$0.40(0.900) + 0.25(0.855) + 0.15(0.80) + 0.10(0.80) + 0.07(0.70) + 0.03(0.50) =$ **83.8%**, with a margin of about ±4.3 points from $1.96\sqrt{\sum_h w_h^2 \hat p_h(1-\hat p_h)/n_h}$.

Report 83.8% as the expected production pass rate and the escalation slice as the risk: even at ±15 points, 50% is clearly bad.

</div>

## Splits, versions and leakage

~~~dg-flow How cases move through a maintained eval set.
Logs, failures, experts, synthetic -(dedupe, scrub PII, label)-> Candidate cases -(random split)-> *Dev split: iterate freely -(gate releases)-> Held-out split: report rarely -(freeze a stable core)-> Golden set for CI
~~~

- **Dev versus held-out.** Every prompt tweak tested against the set fits the prompt a little more to those cases. Iterate on a dev split you read freely, and score the held-out split only for decisions. When dev scores rise and held-out scores do not, you are tuning to the set.
- **Version it like code.** Never edit a case in place silently: bump its version, keep a changelog, and record the set version with every result. Scores on v7 and v8 of a set are not comparable. A frozen core (the golden set used for CI gating in 10.06) keeps a long-running baseline while the rest of the set grows.
- **Leakage.** Eval cases must not appear as few-shot examples in the prompt, in fine-tuning data ([Data for fine-tuning](lesson:aieng:07.08) covers decontamination), or as answer documents in the retrieval index when the case is meant to test reasoning. Near-duplicates across splits count too; MinHash dedup catches paraphrased copies. Public benchmarks may already be in a model's pretraining data, one more reason they are a poor product eval.
- **Record the setup.** Biderman et al. show scores moving with prompt formatting, few-shot count and answer extraction. Store the exact rendered prompt, model snapshot, decoding settings and grader version alongside each result.

## Where this breaks

- **Stale sets.** The product changes, users change, and a set built at launch slowly stops covering what goes wrong. Re-run error analysis on fresh traces regularly and retire cases for features that no longer exist.
- **A set that only grows.** Hundreds of near-identical regression cases for one old bug inflate cost and the bug's weight. Cap cases per failure mode and reweight.
- **Synthetic-only sets.** They pass easily because they read like model output. Treat a high synthetic pass rate as a smoke test, not evidence.
- **Unlabelled sensitive data.** Production logs carry names, addresses and account numbers. Scrub or synthesise them before cases reach labellers, and respect retention rules.
- **Ignoring the interval.** A slice at 70% on 20 cases is "somewhere between 48% and 85%". Say so rather than ranking slices by point estimates.

<div class="take">
<div class="lab">Key takeaway</div>

An eval set is a versioned sample of inputs with everything needed to grade them, drawn from several sources and stratified by slice. Size it from intervals (±11 points at 50 cases, ±4 at 400), compare variants on the same cases, reweight oversampled slices before reporting, and keep it free of leakage.

</div>
