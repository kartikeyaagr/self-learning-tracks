---
id: "09.04"
course: "machine-learning"
module: "09"
slug: "experimenting-with-models"
title: "Experimenting with models"
keyIdeas: ["Offline metrics are a filter, not a verdict: a model ships when a randomised **online experiment** shows it improves the metric the business cares about, without harming guardrails.", "A **canary** sends 1 to 5% of traffic to the new model to catch breakage (errors, latency, prediction distribution) within minutes; an A/B test is a separate, longer step that measures value.", "Checking a fixed-horizon test repeatedly and stopping at the first $p < 0.05$ inflates false positives to about 20% with ten looks. **Sequential tests** (mSPRT, alpha spending) give **always-valid** p-values you may watch continuously.", "Check **sample ratio mismatch** first: if a 50/50 test shows 50.8/49.2 on a million users, assignment or logging is broken and the result is untrustworthy.", "For ranking models, **interleaving** (merging both models' lists for each user) detects preference with far fewer users than an A/B test, but does not measure business impact."]
question: "A team runs a 50/50 A/B test of a new recommender, checks the dashboard every morning, and stops on day 4 when conversion is up 2.1% with p = 0.03. Arm sizes are 412,880 and 405,120 users. List the problems with this conclusion and what a sound process would have done."
answer: "Two independent problems. First, peeking: stopping at the first daily look with p below 0.05 inflates the false positive rate well above 5%, so p = 0.03 at day 4 is not evidence at the nominal level. A sound process pre-registers a duration covering full weekly cycles, or uses a sequential method (mSPRT always-valid p-values or alpha spending) that allows daily looks. Second, the arm sizes show sample ratio mismatch: 412,880 vs 405,120 is a 50.5/49.5 split on 818k users, which a chi-square test rejects at p far below 0.001. Some users are being lost or duplicated in one arm, so the effect estimate is biased whatever its p-value. Investigate assignment and logging before reading any metric. Also, four days is too short to separate a novelty effect from a lasting improvement, and guardrail metrics should be checked."
sources: [{"title": "Kohavi, Tang and Xu, Trustworthy Online Controlled Experiments (book site)", "url": "https://experimentguide.com/", "note": "The standard reference; chapters on SRM, guardrails, variance and pitfalls."}, {"title": "Johari et al., Always Valid Inference: Continuous Monitoring of A/B Tests (2015)", "url": "https://arxiv.org/abs/1512.04922", "note": "Why peeking breaks fixed-horizon tests and how mSPRT p-values fix it."}, {"title": "Google SRE Workbook: Canarying Releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "How to size, run and judge a canary, with metric selection advice."}, {"title": "Deng et al., CUPED: Improving the Sensitivity of Online Controlled Experiments (2013)", "url": "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf", "note": "Variance reduction with pre-experiment covariates; section 3 for the estimator."}]
practice: ["msprt-always-valid-pvalues", "canary-mann-whitney", "b-test-significance", "bootstrap-confidence-interval"]
tags: ["A/B testing", "sequential testing", "canary"]
tools: ["NumPy", "SciPy"]
---

## Why offline wins do not ship by themselves

An offline metric is computed on logged data collected under the old model. It cannot see feedback effects (the new recommender changes what users click), position bias, latency costs, or whether the metric correlates with revenue or retention. Model teams regularly see offline NDCG improve while the online metric is flat. The online experiment is the measurement that counts; offline evaluation decides what is worth testing.

The general theory of A/B tests (hypothesis tests, power and sample size, multiple testing) is in ML Maths, "A/B testing". This lesson covers what is specific to rolling out models.

~~~dg-flow A model rollout ladder: each rung answers a different question and can stop the rollout.
Offline eval -(passes)-> Shadow -(no errors)-> Canary 1-5% -(healthy)-> *A/B test -(wins)-> Ramp to 100%
~~~

## Shadow, canary, A/B

- **Shadow (dark launch)**: the new model receives a copy of live traffic and its predictions are logged but not used. It answers "does it run, how fast, and how different are its outputs?" with zero user risk. It cannot measure user response.
- **Canary**: a small share of real traffic is served by the new model. The question is safety: error rate, latency percentiles, prediction distribution, any obvious business metric collapse. Decisions come in minutes to hours, often automated: compare canary against a same-sized baseline slice with a non-parametric test such as Mann-Whitney U on latency, and roll back automatically on failure.
- **A/B test**: randomised split, typically 50/50 or 10/90, run for a pre-registered duration (at least one or two full weekly cycles) to estimate the effect on the primary metric with a confidence interval.

| Stage | Traffic | Duration | Question | Typical decision rule
| --- | --- | --- | --- | --- |
| Shadow | copy of 100%, not served | hours to days | does it work, how different? | error-free, latency within budget
| Canary | 1-5% | minutes to hours | is it safe? | no regression on health metrics vs baseline slice
| A/B | 10-50% | 1-4 weeks | is it better? | primary metric CI excludes 0; guardrails hold

## Designing the A/B test for a model

- **Unit of randomisation**: usually the user, hashed with an experiment salt (the same technique as hashed splits in 09.02). Randomising per request lets one user see both models, which contaminates anything with memory, like recommendations.
- **Metrics**: one primary metric decided in advance (purchases per user), a few secondary ones, and guardrails that must not get worse (latency p99, error rate, unsubscribe rate). Ratio metrics like click-through per impression need the delta method or a bootstrap for correct variance, because impressions per user vary.
- **Sample ratio mismatch (SRM)**: before reading results, test whether the observed split matches the design with a chi-square test. A 50.8/49.2 split on a million users has $p < 10^{-50}$; something (bot filtering, a crash in one arm, a redirect) removes users non-randomly, and the effect estimate is biased.
- **Variance reduction**: CUPED regresses the metric on its own pre-experiment value, $Y' = Y - \theta(X - \bar X)$ with $\theta = \text{cov}(X,Y)/\text{var}(X)$, often cutting variance by 30 to 50% and shrinking required sample size by the same factor.
- **Novelty and learning effects**: a new recommender can win in week one because it is different; check that the effect is stable over time.

## Peeking and sequential testing

A fixed-horizon test assumes you look once, at the end. Dashboards invite daily looks. If you stop the first time $p < 0.05$, the false positive rate is no longer 5%: with ten evenly spaced looks it is roughly 19%, and with continuous monitoring it approaches 1 eventually. Two families of remedies:

1. **Group sequential designs** with alpha spending (O'Brien-Fleming, Pocock): a few planned interim looks with stricter thresholds early.
2. **Always-valid inference**: the mixture sequential probability ratio test (mSPRT) of Johari et al. gives a p-value sequence that is valid at any stopping time, so the dashboard can show it continuously and you may stop as soon as it crosses $\alpha$. The price is lower power than a perfectly planned fixed test at the same sample size.

```python
import numpy as np
def msprt_pvalues(diffs, sigma2, tau2=1.0):
    # diffs: running per-pair differences (treatment - control), assumed ~N(theta, sigma2)
    p, out = 1.0, []
    for n in range(1, len(diffs) + 1):
        m = np.mean(diffs[:n])
        lam = np.sqrt(sigma2 / (sigma2 + n * tau2)) * np.exp(
            n * n * tau2 * m * m / (2 * sigma2 * (sigma2 + n * tau2)))
        p = min(p, 1.0 / lam)          # always-valid p-value never increases
        out.append(p)
    return out
```

## Interleaving for rankers

For two ranking models, team-draft interleaving builds each user's list by alternately drafting the next-best item from model A and model B, then credits clicks to the model that contributed the clicked item. Each user compares both models side by side, which removes between-user variance; published industry reports describe needing one to two orders of magnitude fewer users than an A/B test to detect the same preference. Use interleaving to screen many ranker variants quickly, then A/B test the winner to measure business impact, which interleaving cannot do.

<div class="ex">
<div class="lab">Worked example</div>

**a canary that saved the A/B test**

A new ranking model passes offline evaluation. On a 2% canary its p99 latency is 240 ms against a 150 ms budget, because a new feature triggers an uncached database read. Automated canary analysis flags it within 20 minutes and rolls back. Had it gone straight to a 50/50 A/B test, the slow arm would have lost on engagement and the team would have wrongly concluded the model was worse.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

roll models out through shadow, canary and A/B stages that answer safety then value; randomise by user, check SRM before reading results, and use sequential tests if anyone will watch the dashboard.

</div>
