---
id: "10.06"
course: "ai-engineering-course"
module: "10"
slug: "regression-testing-prompts-and-agents"
title: "Regression testing prompts and agents in CI"
keyIdeas: ["Version everything that changes outputs (prompt, model snapshot, tools, retrieval config, judge, eval set) and re-run the eval whenever any of it changes.", "Use two kinds of gate: hard assertions on must-pass cases that block on any failure, and a statistical gate on the aggregate score.", "Compare candidate and baseline on the same cases. Paired differences cancel per-case difficulty and usually give confidence intervals two or more times narrower than comparing two means.", "Bootstrap the per-case differences (resampling cases, not repeats) and gate on a non-inferiority margin: pass only if the lower bound of the interval is above minus the margin.", "Handle flakiness with repeats averaged per case, pinned judges, cached baselines and an A/A run that measures the noise floor; budget the cost per PR and per night."]
question: "A pull request rewrites the system prompt. On your 200-case golden set the baseline passes 160 cases and the candidate 156; 9 cases went from pass to fail and 5 from fail to pass. Your team's rule is 'block if we cannot rule out a regression of more than 3 points'. Does this PR merge, and what do you do next?"
answer: "Compare on paired per-case differences, not two separate pass rates. The mean difference is (5 - 9) / 200 = -2 points. Only the 14 discordant cases vary, so the standard error of the paired difference is about 1.9 points and the 95% interval is roughly -5.7 to +1.7 points (a paired bootstrap gives about the same). The lower bound is below -3, so the regression is not ruled out: the gate says inconclusive and the PR does not merge automatically. The unpaired interval, about -10 to +6, would have been useless. Next: read all 14 flipped cases, since that is cheap and shows whether the 9 losses share a cause; check whether they sit in a protected slice; then either fix the prompt or grow the set. At a 7% discordance rate, about 300 cases are needed for a ±3-point interval."
rubric: ["Uses per-case paired differences rather than comparing two independent pass rates", "Computes the mean difference of -2 points and a confidence interval of roughly -5.7 to +1.7", "Applies the non-inferiority rule correctly: lower bound below -3 means the gate does not pass", "Proposes reading the discordant cases before deciding", "Mentions growing the eval set or a sample-size estimate as the remedy for an inconclusive gate"]
sources: [{"title": "Miller: Adding Error Bars to Evals (2024)", "url": "https://arxiv.org/abs/2411.00640", "note": "Clustered standard errors, resampling answers, paired differences and power analysis for LLM evals."}, {"title": "Koehn: Statistical Significance Tests for Machine Translation Evaluation (EMNLP 2004)", "url": "https://aclanthology.org/W04-3250/", "note": "The paired bootstrap for comparing two systems on the same test set."}, {"title": "Dror et al.: The Hitchhiker's Guide to Testing Statistical Significance in NLP (ACL 2018)", "url": "https://aclanthology.org/P18-1128/", "note": "Choosing a significance test for paired system comparisons."}, {"title": "Promptfoo: Testing prompts with GitHub Actions", "url": "https://www.promptfoo.dev/docs/integrations/github-action/", "note": "Running a before-and-after eval on every pull request that touches prompts."}]
practice: ["paired-bootstrap-regression-gate", "bootstrap-confidence-interval", "pass-at-k-and-pass-hat-k", "standard-error-comparison"]
tools: ["promptfoo", "GitHub Actions"]
tags: ["regression testing", "CI", "paired bootstrap"]
---

## What counts as a change

In ordinary software a regression test runs when the code changes. In an LLM app the output can change when no line of application code does: someone edits a prompt, the provider retires a model snapshot, a tool schema gains a field, the retriever's top $k$ moves from 5 to 8, or the judge model is upgraded. Each of these is a deploy. So the first job is to make all of them visible to CI.

| Component | Versioned as | Notes |
| --- | --- | --- |
| Prompts | files in the repo, reviewed like code | template plus few-shot examples |
| Model | dated snapshot id, never a floating alias | an alias can change under you ([reliability](lesson:aieng:12.06)) |
| Tools and retrieval config | schemas and parameters in the repo | $k$, reranker, chunking version |
| Judge | model snapshot plus judge prompt | a judge change moves scores with no system change |
| Eval set | content hash and version tag | scores on different set versions are not comparable |

Every eval result is stored with the hash of all five. Comparing two runs is only valid when everything except the thing under test matches.

## Tiers and gates

Running the full suite on every PR is too slow and too expensive, so most teams split it.

~~~dg-layers Eval tiers, from every commit to production.
Per PR | deterministic checks on all golden cases + paired gate on 100 to 300 cases; minutes; baseline cached from main
*Nightly | full suite, agent tasks with repeats, main versus last release; batch API; catches provider-side drift
Pre-release | full suite + human read of flipped cases + judge re-calibration
Production | shadow, canary, online evals on sampled traces
~~~

Within a tier there are two kinds of gate, and mixing them up is a common mistake.

- **Hard assertions** are for cases that must never fail: the output parses, a known jailbreak is refused, the bug fixed last month stays fixed. Any failure blocks. These are unit tests and need no statistics.
- **Statistical gates** are for the aggregate quality score, where some cases flip by chance on every run. Here "the score went down" is not evidence of anything until you know how much it moves by chance.

## Pair the comparison

Both prompts run on the same cases, so compare them case by case. With $d_i = \text{candidate}_i - \text{baseline}_i$,

$$
\operatorname{Var}(\bar d) = \frac{\sigma_b^2 + \sigma_c^2 - 2\,\operatorname{Cov}(b, c)}{n}
$$

Easy cases pass under both prompts and hard ones fail under both, so the scores are strongly correlated and the covariance term cancels most of the variance. Miller's error-bars paper for LLM evals makes this its fourth recommendation: do inference on question-level paired differences.

<div class="ex">
<div class="lab">Worked example</div>

**the same 2-point drop, two ways**

200 cases. Baseline passes 160 (80%), candidate 156 (78%). 186 cases agree; 9 flipped pass to fail and 5 fail to pass.

Unpaired: $\text{SE} = \sqrt{0.80 \cdot 0.20/200 + 0.78 \cdot 0.22/200} = 0.041$, so the 95% interval is $-2 \pm 8.0$ points, about $(-10.0, +6.0)$. That cannot distinguish a disaster from an improvement.

Paired: $d_i$ is $-1$ nine times, $+1$ five times and 0 otherwise. Its sample standard deviation is 0.264, so $\text{SE} = 0.264/\sqrt{200} = 0.0187$ and the interval is $(-5.7, +1.7)$ points, 2.2 times narrower. A paired bootstrap with 10,000 resamples gives $(-5.5, +1.5)$.

For binary scores the exact alternative is McNemar's test on the discordant pairs: 5 against 9 out of 14 gives a two-sided $p = 0.42$. Either way, only the 14 flipped cases carry information, and reading 14 cases takes ten minutes.

</div>

## The bootstrap gate

The bootstrap works for any per-case score (pass rate, judge score from 1 to 5, cost) without a formula for its variance; the mechanics are in [resampling](lesson:mlmaths:13.05). For a regression gate:

1. Average each case's repeats within each arm, so a case contributes one number per arm.
2. Compute per-case differences $d_i$.
3. Resample **cases** with replacement $B$ times (10,000 is plenty) and take the mean of each resample.
4. Take the 2.5th and 97.5th percentiles as the interval $(lo, hi)$.
5. Decide against a margin $\delta$ the team picks in advance (the largest drop it will accept):

```python
if lo > 0:            verdict = "improved"
elif lo > -delta:     verdict = "pass"          # non-inferior: any drop is smaller than delta
elif hi < 0:          verdict = "regressed"
else:                 verdict = "inconclusive"  # cannot rule out a drop bigger than delta
block = verdict in ("regressed", "inconclusive")
```

Resampling cases rather than individual trials matters: repeats of one case are correlated, and treating them as independent makes the interval far too narrow (the clustered-error point in Miller's paper).

In the worked example, with $\delta = 3$ points, $lo = -5.7$, so the PR is inconclusive and blocked. The two honest exits are fixing whatever the nine losses share, or growing the set. With 7% of cases discordant, the half-width is $1.96\sqrt{0.07/n}$, which reaches 3 points at about $n = 300$ and 2 points at about $n = 670$. The sample-size reasoning from [Building an eval set](lesson:aieng:10.02) applies, but with the discordance rate in place of $p(1-p)$, which is why paired sets can be much smaller.

## Flakiness

Four sources of noise reach a CI eval: sampling temperature, provider-side non-determinism (batched serving is not bit-reproducible even at temperature 0), judge noise, and environment failures in agent harnesses ([evaluating agents](lesson:aieng:10.05)). The remedies:

- **Repeats.** Run each case $r$ times (3 is common for prompts, 5 for agents) at the production temperature, and average per case before pairing. This shrinks within-case noise without inflating $n$.
- **Pin the judge.** Fixed snapshot, temperature 0, prompt under version control.
- **Seeds where offered.** Some APIs accept a seed, but treat it as best effort, not a guarantee.
- **Cache by content hash.** Key outputs on (prompt hash, model snapshot, parameters, input). The baseline from main is then free on every PR, and only the candidate runs. Refresh the cached baseline on the nightly run so provider drift does not hide in it.
- **Retry infrastructure errors.** A rate-limit error or a sandbox crash is rerun, never scored as a failure.
- **Run an A/A test.** Run the baseline against itself through the full gate many times. If it reports "regressed" more than about 2.5% of the time at a 95% interval, the gate is miscalibrated (usually unpaired, or resampling trials instead of cases).

## Budget the cost

<div class="ex">
<div class="lab">Worked example</div>

**an agent suite, illustrative prices**

Assume one agent trial uses 20k input and 2k output tokens, and its judge call 3k input and 300 output, at \$3 per million input tokens and \$15 per million output. A trial costs \$0.06 + \$0.03 = \$0.09 and its judge \$0.0135, so \$0.1035 together.

| Run | Trials | Cost |
| --- | --- | --- |
| Per PR: 50 cases × 3 repeats, baseline cached | 150 | \$15.53 |
| 40 PRs a week | 6,000 | \$621 |
| Nightly: 400 cases × 5 repeats, one arm | 2,000 | \$207, or about \$104 through a batch API |

</div>

The levers, in order of payoff: cache the baseline (halves every PR), run only the slices a change can affect (path filters on which files changed, as in promptfoo's GitHub Action), use code checks before judge calls, keep shared prompt prefixes cacheable, and move the nightly suite to a batch API. [Anthropic's Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing) bills at half the standard price, with results usually within an hour, which suits nightly runs but not a PR someone is waiting on.

## After CI

Offline gates only see the golden set. A change that passes moves through shadow traffic, a canary and, for big changes, an A/B test, as in [A/B testing and safe rollout](lesson:mlsd:05.05). Online evals run the same code checks and reference-free judges on a sample of production traces ([observability](lesson:aieng:12.07)), and every production failure someone finds becomes a new golden case.

## Where this breaks

- **The mean hides a slice.** A two-point average drop can be a twenty-point drop on refund questions. Gate protected slices separately.
- **Too many metrics.** With ten metrics at 95%, at least one will look significant by chance in about four runs out of ten. Choose one primary metric and treat the rest as guardrails with looser thresholds.
- **Overfitting the golden set.** Prompts tuned until the set passes stop generalising. Keep a held-out set that only the nightly run sees.
- **Comparing across versions.** A new eval-set version or judge invalidates the cached baseline. Re-run it before comparing.

## What interviewers probe

- "How do you know a two-point drop is real?" Paired differences, a bootstrap over cases, and a margin set in advance.
- "Your CI eval is flaky. What do you do?" Repeats averaged per case, pinned judge, retries for infrastructure errors, and an A/A run to prove the gate is calibrated.
- "What does this cost per PR?" Trials times tokens times price, then the caching and slicing that cut it.

<div class="take">
<div class="lab">Key takeaway</div>

Treat prompts, model snapshots, tools, judges and eval sets as versioned code. Gate each change with hard assertions for must-pass cases and a paired bootstrap over cases against a non-inferiority margin for the aggregate, average repeats per case to tame flakiness, prove the gate with an A/A run, and read the flipped cases before trusting any verdict.

</div>
