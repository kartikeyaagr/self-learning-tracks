---
id: "09.04"
course: "machine-learning"
module: "09"
slug: "experimenting-with-models"
title: "Experimenting with models"
keyIdeas: ["Offline metrics are a filter, not a verdict: a model ships when a randomised **online experiment** shows it improves the metric the business cares about, without harming **guardrail** metrics such as latency and error rate.", "Roll out in rungs that answer different questions: **shadow** (does it run?), **canary** on 1 to 5% of traffic (is it safe? caught in minutes), then the **A/B test** (is it better? measured over weeks).", "Size the test first: users needed per arm grow with $1/\\delta^2$, so halving the effect you want to detect needs four times the traffic. **CUPED** cuts variance (often 30 to 50%) using each user's pre-experiment behaviour.", "Check **sample ratio mismatch** before anything else: a 50.8/49.2 split on a million users has $p \\approx 10^{-57}$, meaning assignment or logging is broken and the effect estimate is untrustworthy.", "Stopping at the first $p < 0.05$ while checking repeatedly inflates false positives to about 19% with ten looks. **Sequential tests** (mSPRT, alpha spending) give **always-valid** p-values you may watch continuously.", "For ranking models, **interleaving** merges both models' lists for each user and detects preference with far fewer users than an A/B test, but does not measure business impact."]
question: "A team runs a 50/50 A/B test of a new recommender, checks the dashboard every morning, and stops on day 4 when conversion is up 2.1% with p = 0.03. Arm sizes are 412,880 and 405,120 users. List the problems with this conclusion and what a sound process would have done."
answer: "There are three separate problems, and the first one alone invalidates the result. (1) Sample ratio mismatch: 412,880 vs 405,120 is a 50.5/49.5 split on 818,000 users. The expected count is 409,000 per arm, so $\\chi^2 = 2 \\times 3{,}880^2 / 409{,}000 \\approx 73.6$ with one degree of freedom, $p \\approx 10^{-17}$. Some users are being lost, duplicated or filtered non-randomly in one arm (bot filtering, a crash, a redirect, a logging bug), so the effect estimate is biased whatever its p-value. Investigate assignment and logging and rerun before reading any metric. (2) Peeking: checking daily and stopping at the first p below 0.05 inflates the false positive rate well above 5% (about 14% with five looks, 19% with ten), so p = 0.03 on day 4 is not evidence at the nominal level. A sound process either pre-registers a sample size from a power calculation and a duration covering at least two full weekly cycles and reads the result once, or uses a sequential method (mSPRT always-valid p-values, or a group sequential design with O'Brien-Fleming alpha spending) that permits daily looks. (3) Duration and novelty: four days covers no full week and cannot separate a novelty effect from a lasting improvement; the daily lift should be plotted to see whether it decays. Also missing: guardrail checks (latency, errors, unsubscribes), a confidence interval rather than a bare p-value, and ideally CUPED to reduce variance. A sound rollout would also have passed shadow and canary stages first so that engineering problems are not mistaken for model quality."
sources: [{"title": "Kohavi, Tang and Xu, Trustworthy Online Controlled Experiments (book site)", "url": "https://experimentguide.com/", "note": "The standard reference; chapters on SRM, guardrails, variance and pitfalls."}, {"title": "Johari et al., Always Valid Inference: Continuous Monitoring of A/B Tests (2015)", "url": "https://arxiv.org/abs/1512.04922", "note": "Why peeking breaks fixed-horizon tests and how mSPRT p-values fix it."}, {"title": "Google SRE Workbook: Canarying Releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "How to size, run and judge a canary, with metric selection advice."}, {"title": "Deng et al., CUPED: Improving the Sensitivity of Online Controlled Experiments (2013)", "url": "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf", "note": "Variance reduction with pre-experiment covariates; section 3 for the estimator."}]
practice: ["msprt-always-valid-pvalues", "canary-mann-whitney", "b-test-significance", "bootstrap-confidence-interval"]
tags: ["A/B testing", "sequential testing", "canary"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- Why a better offline score is not enough to ship a model.
- The rollout ladder (shadow, canary, A/B test, ramp) and the question each rung answers.
- How to design and read an A/B test: randomisation unit, metrics and guardrails, sample size, the sample ratio mismatch check.
- Why daily dashboard checks break the p-value, and how sequential tests fix it.
- Two traffic savers: CUPED variance reduction and interleaving for rankers.

*Before you start:* lessons 09.01-09.03 (serving, versioning, monitoring) and the idea of a p-value and a confidence interval. The general theory of A/B tests is in ML Maths, "A/B testing"; this lesson recaps what it needs and focuses on rolling out models.

## From monitoring to experiments

Lesson 09.03 showed how to watch a model that is already live. This last lesson answers the question that comes before: a new model looks better offline, so how do you decide, with evidence, that it should replace the current one?

An offline metric is computed on logged data the *old* model helped create. It cannot see **feedback effects** (a new recommender changes what users click), position bias, latency costs, or whether the metric moves revenue at all. Offline NDCG often rises while the business metric stays flat. Offline evaluation decides what is *worth testing*; a randomised **online experiment** decides what ships.

## The idea in one picture

A shop's recommender converts 5.0% of visitors into buyers. The team trains a new model and runs an **A/B test**: each visitor is randomly assigned to the old model (arm **A**, the **control**) or the new one (arm **B**, the **treatment**).

| Arm | Users | Buyers | Conversion rate
| --- | --- | --- | --- |
| A (current model) | 20,000 | 1,000 | 5.0%
| B (new model) | 20,000 | 1,100 | 5.5%

B is ahead by 0.5 percentage points, a 10% relative lift. But even two identical arms would differ a little by chance. Is 0.5 points bigger than chance usually produces?

~~~dg-fig 09.04-sampling.svg
The blue curve shows the differences you would measure if the new model had no effect; the green curve shows them if it really adds 0.5 points. Anything beyond the dashed line counts as significant. Our observed +0.50 is just past it.
~~~

With 20,000 users per arm, pure chance produces differences with a standard spread of about 0.22 points. Only 5% of no-effect experiments land beyond ±0.44 points, the **significance threshold**. Our 0.50 is beyond it, with a **p-value** of 0.025: if the models were equal, a gap this large would appear only 2.5% of the time.

The green curve tells a less comfortable story: even if the lift is real, only 61% of experiments this size would detect it. That is the test's **power**. This design misses a real 10% improvement almost two times in five.

## Step 1: climb the rollout ladder

Before measuring value, the new model must prove it is safe, through stages that can each stop the rollout.

~~~dg-flow A model rollout ladder: each rung answers a different question and can stop the rollout.
Offline eval -(passes)-> Shadow -(no errors)-> Canary 1-5% -(healthy)-> *A/B test -(wins)-> Ramp to 100%
~~~

- **Shadow (dark launch)**: the new model gets a copy of live traffic; its predictions are logged, never shown. Zero user risk, but it cannot measure how users respond.
- **Canary**: 1 to 5% of real traffic is served by the new model, to check safety: errors, latency percentiles, prediction distribution. Decisions are often automated: compare against a same-sized baseline slice with a non-parametric test such as Mann-Whitney U on latencies, and roll back on failure.
- **A/B test**: a randomised split (50/50 or 10/90) run for a pre-registered duration of at least one or two full weekly cycles, estimating the effect on the primary metric with a confidence interval.
- **Ramp**: traffic moves to the winner in steps, with the monitoring from 09.03 watching each one.

| Stage | Traffic | Duration | Question | Typical decision rule
| --- | --- | --- | --- | --- |
| Shadow | copy of 100%, not served | hours to days | does it work, how different? | error-free, latency within budget
| Canary | 1-5% | minutes to hours | is it safe? | no regression on health metrics vs baseline slice
| A/B | 10-50% | 1-4 weeks | is it better? | primary metric CI excludes 0; guardrails hold

Keep them separate: a canary is too small to measure value, and an A/B test is too slow to be your first defence against a broken model.

<div class="ex">
<div class="lab">Worked example</div>

**a canary that saved the A/B test**

A new ranking model passes offline evaluation. On a 2% canary its p99 latency is 240 ms against a 150 ms budget, because a new feature triggers an uncached database read. Automated canary analysis flags it within 20 minutes and rolls back.

Had it gone straight to a 50/50 A/B test, the slow arm would have lost on engagement and the team would have wrongly concluded that the *model* was worse.

</div>

## Step 2: decide what to randomise and what to measure

**Unit of randomisation.** Usually the user, assigned by hashing the user id with an experiment salt (the trick behind hashed splits in 09.02), so a returning user always sees the same arm. Randomising per *request* lets one person see both models, which contaminates anything with memory, such as recommendations.

**Metrics.** Decide them before the test starts and write them down:

- One **primary metric** that defines success, such as purchases per user.
- A few **secondary metrics** that help explain the result (clicks, session length).
- **Guardrail metrics** that must not get worse whatever happens to the primary: p99 latency, error rate, unsubscribe rate, revenue per user.

**Ratio metrics need care.** Click-through per impression looks like a proportion, but you randomised users, and users have different numbers of impressions. Treating impressions as independent understates variance; use the **delta method** (Going deeper) or a bootstrap over users.

## Step 3: size the test before it starts

Four things decide how many users you need: the baseline rate $p_0$ (5.0%), the smallest lift worth detecting, called the **minimum detectable effect** $\delta$ (0.5 points), the false positive rate $\alpha$ (5%, two-sided) and the power $1 - \beta$ (usually 80%).

For two proportions the users needed in *each* arm are approximately

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \,\big(p_0(1-p_0) + p_1(1-p_1)\big)}{\delta^2},$$

where $p_1 = p_0 + \delta$, and $z_{1-\alpha/2} = 1.96$ and $z_{1-\beta} = 0.84$ are standard normal quantiles for 5% and 80%. Number check: $(1.96 + 0.84)^2 \times (0.0475 + 0.0520) / 0.005^2 \approx 7.85 \times 0.0995 / 0.000025 \approx 31{,}200$ users per arm. Our test with 20,000 was under-powered, which is why its power was only 61%.

~~~dg-fig 09.04-power.svg
Power grows with users per arm, but slowly: each halving of the lift you want to detect needs about four times as many users. Detecting a 2% relative lift on a 5% base rate takes about 750,000 users per arm.
~~~

The $\delta^2$ in the denominator is the most important fact in experiment design: halve the effect you want to see and you need four times the traffic. Model improvements are often 1 to 3% relative, so on a low-traffic product many tests are not worth running. Know that before you start.

## Step 4: check the split before reading any result

A **sample ratio mismatch (SRM)** means the observed arm sizes are further from the design (say 50/50) than chance allows. Test it with a chi-square goodness-of-fit test:

$$\chi^2 = \sum_{\text{arms}} \frac{(\text{observed} - \text{expected})^2}{\text{expected}},$$

with one degree of freedom for two arms. A common alarm threshold is $p < 0.001$.

<div class="ex">
<div class="lab">Worked example</div>

**reading an A/B result by hand**

**SRM first.** Suppose a test on 1,000,000 users shows 508,000 in A and 492,000 in B (50.8/49.2). Expected is 500,000 each. Each arm contributes $8{,}000^2 / 500{,}000 = 128$, so $\chi^2 = 256$ and $p \approx 10^{-57}$. Stop: something (bot filtering, a crash in one arm, a redirect) removes users non-randomly, so any effect estimate is biased. Our shop test is split exactly 20,000/20,000, so we continue.

**Rates.** $\hat p_A = 1000/20000 = 0.050$ and $\hat p_B = 1100/20000 = 0.055$. Difference $\hat\delta = 0.005$ (0.5 points).

**Standard error.** $\hat p_A(1-\hat p_A)/n = 0.0475/20000 = 2.375 \times 10^{-6}$ and $\hat p_B(1-\hat p_B)/n = 0.051975/20000 = 2.599 \times 10^{-6}$. Sum $4.974 \times 10^{-6}$, square root $SE = 0.00223$.

**Test statistic.** $z = 0.005 / 0.00223 = 2.24$. The two-sided p-value is $2 \times P(Z > 2.24) = 0.025$.

**Confidence interval.** $0.005 \pm 1.96 \times 0.00223 = 0.005 \pm 0.00437$, so $[0.06, 0.94]$ points, or roughly $+1\%$ to $+19\%$ relative. It excludes 0 but is wide: a tiny lift and a large one both fit. Report the interval, not just "significant".

</div>

~~~dg-fig 09.04-srm.svg
The SRM p-value against the share of users in arm A. With a million users even a 50.3/49.7 split is an alarm; with ten thousand users it is ordinary noise.
~~~

Run the SRM check automatically and hide the result dashboard until it passes.

## The A/B readout in code

```python
import numpy as np
from scipy import stats

# 1. Sample size, decided BEFORE the test starts
p0, p1, alpha, power = 0.050, 0.055, 0.05, 0.80
za, zb = stats.norm.ppf(1 - alpha/2), stats.norm.ppf(power)
n = (za + zb)**2 * (p0*(1-p0) + p1*(1-p1)) / (p1 - p0)**2
print(round(n))                                   # 31231 users per arm

# 2. SRM check FIRST: does the split match the 50/50 design?
users = np.array([412_880, 405_120])
chi2, p_srm = stats.chisquare(users)              # expected: equal counts
print(round(chi2, 1), p_srm)                      # 73.6 9.5e-18  -> stop, debug assignment

# 3. Read the result (only if SRM passed): two-proportion z-test + CI
conv = np.array([1000, 1100]); n_arm = np.array([20_000, 20_000])
rate = conv / n_arm; diff = rate[1] - rate[0]
se = np.sqrt((rate * (1 - rate) / n_arm).sum())
z = diff / se; p = 2 * stats.norm.sf(abs(z))
print(round(z, 2), round(p, 3), np.round([diff - 1.96*se, diff + 1.96*se], 4))
# 2.24 0.025 [0.0006 0.0094]
```

## Step 5: shrink the noise with CUPED

Much of the variation in spend per user has nothing to do with your model: heavy spenders spend a lot in both arms. **CUPED** (Controlled-experiment Using Pre-Experiment Data) removes the part of each user's metric predictable from their behaviour *before* the experiment:

$$Y' = Y - \theta\,(X - \bar X), \qquad \theta = \frac{\text{cov}(X, Y)}{\text{var}(X)},$$

where $Y$ is the user's metric during the test, $X$ the same metric in a pre-period, and $\bar X$ its average over all users. $X$ was measured before assignment, so the adjustment removes noise without biasing the effect. Variance falls by a factor $1 - \rho^2$, where $\rho$ is the correlation of $X$ and $Y$.

~~~dg-fig 09.04-cuped.svg
Effect estimates from 2,000 simulated experiments with a true effect of 0.3. Both versions are centred correctly, but the CUPED estimates are much tighter, so the same traffic detects smaller effects.
~~~

In practice CUPED often cuts variance by 30 to 50%, and the required sample size falls by the same factor (the simulation's correlation of 0.75 gives nearly 60%). Users with no pre-period get $X = \bar X$, which leaves them unadjusted. In code it is two lines: `theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)` and `y_adj = y - theta * (x - x.mean())`.

## Step 6: do not peek, or peek properly

The p-value from Step 4 assumes you look **once**, at the planned end. Dashboards invite daily looks and stopping the moment $p < 0.05$. But under no real effect the running p-value wanders, and the more often you check, the likelier it dips below 0.05 at *some* point.

~~~dg-fig 09.04-peeking.svg
Left: four A/A tests (both arms identical); one p-value path crosses 0.05 on day 8 and would have been declared a win. Right: stopping at the first p below 0.05 raises the false positive rate from 5% (one look) to 19% (ten looks) and 37% (a hundred looks). The always-valid mSPRT stays under 5%.
~~~

With ten looks the real false positive rate is about 19%, nearly four times the promised 5%; with continuous monitoring it approaches 100% eventually. Two remedies:

1. **Group sequential designs** with alpha spending (O'Brien-Fleming, Pocock): plan a few interim looks, each with a stricter threshold. With five O'Brien-Fleming looks the $|z|$ thresholds are about 4.56, 3.23, 2.63, 2.28 and 2.04, so early stopping needs overwhelming evidence and the final look is barely stricter than 1.96.
2. **Always-valid inference**: the mixture sequential probability ratio test (**mSPRT**) of Johari et al. produces a p-value sequence that is valid at *any* stopping time. The dashboard can show it continuously and you may stop as soon as it crosses $\alpha$. The price is somewhat lower power than a perfectly planned fixed test of the same size.

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

Here $\Lambda_n$ (`lam`) is a likelihood ratio of "some effect, drawn from a normal prior with variance $\tau^2$" against "no effect". Under no effect, $\Lambda_n$ is a martingale, so by Ville's inequality it ever reaches $1/\alpha$ with probability at most $\alpha$, however long you watch. In the figure, mSPRT with $\tau^2 = 0.05\sigma^2$ at 100 looks gave 2.3% false positives. A badly chosen $\tau^2$ costs power, never validity.

## Step 7: compare rankers with interleaving

For two ranking models there is a much more sensitive design. **Team-draft interleaving** builds each user's single result list by letting A and B take turns, like captains picking teams, each adding its best item not yet in the list (a coin flip decides who picks first each round). Clicks are credited to the model that contributed the clicked item, and the model with more credited clicks across users wins.

Each user compares both models, so differences *between* users (heavy versus rare clickers) cancel out; industry reports describe needing one to two orders of magnitude fewer users than an A/B test. The limitation: no user experiences pure A or pure B, so interleaving shows preference, not business impact. Screen many variants with it, then A/B test the winner.

## Reading and checking the result

Check in order, stopping at the first failure:

1. **SRM passed?** If not, nothing else is trustworthy.
2. **Planned length reached**, or a sequential method used?
3. **Guardrails held?** A 1% conversion win that adds 80 ms to p99 latency may be a long-term loss.
4. **Effect size and confidence interval**, not just the p-value.
5. **Stable over time?** Plot the daily lift.

~~~dg-fig 09.04-novelty.svg
Daily lift of a new recommender over four weeks. Early days show a large lift because the recommendations are new to users; it settles near +0.6%. A test stopped on day 4 would have reported about +1.7%.
~~~

That last check catches **novelty effects** (users click because something is new, and the lift fades) and **learning effects** (users need time to adapt, and the lift grows). Both argue for at least two full weeks and a look at the trend. On a new experiment platform, run **A/A tests** (identical arms) as a sanity check: about 5% should come out significant, no more.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Peeking and stopping at first $p < 0.05$ | many "wins" that vanish on rollout | fixed horizon, or mSPRT / alpha spending
| Ignoring sample ratio mismatch | split like 50.5/49.5 on 800k users | chi-square SRM check before reading any metric
| Per-request randomisation | users see both models; effects diluted | randomise by hashed user id with a salt
| Under-powered test | "no significant difference" for a real gain | size the test with the power formula first
| Ratio metric treated as a proportion | p-values too small, false wins | delta method or bootstrap over users
| Skipping the canary | a latency bug reads as "model is worse" | shadow and canary before any A/B
| Stopping in the first week | inflated lift from novelty | run full weekly cycles, plot daily lift
| Testing twenty metrics, reporting the best | "significant" secondary metric by chance | one pre-registered primary; correct the rest for multiple testing

## Going deeper

**The delta method.** For clicks per impression, take per-user clicks $Y_i$ and impressions $X_i$; the arm metric is $R = \bar Y / \bar X$. A first-order Taylor expansion gives

$$\text{Var}(R) \approx \frac{1}{n\,\mu_X^2}\Big(\sigma_Y^2 - 2R\,\sigma_{XY} + R^2 \sigma_X^2\Big),$$

where $\mu_X$ is the mean of $X$, $\sigma_X^2, \sigma_Y^2$ the variances and $\sigma_{XY}$ the covariance across users.

**Why CUPED is unbiased.** $X$ was fixed before randomisation, so $E[X]$ is the same in both arms and the adjustment shifts both arm means equally in expectation. $\theta = \text{cov}(X,Y)/\text{var}(X)$ is the least-squares slope from 02.01, which minimises $\text{Var}(Y')$. CUPED is regression adjustment (ANCOVA); more covariates, or a model's prediction as the covariate (CUPAC), cut variance further.

**Alpha spending.** A spending function $\alpha(t)$ says how much of the total $\alpha$ may be used by information fraction $t \in [0,1]$. O'Brien-Fleming-type spending uses almost nothing early; Pocock-type spending is flat (five looks: $|z| > 2.41$ every time), stopping early more often at the cost of a stricter final threshold.

**Interference.** The analysis assumes one user's arm does not affect another's outcome (SUTVA). Marketplaces, feeds and ad auctions violate it: B can take inventory or attention from A, exaggerating the gap. Remedies are cluster randomisation (by city or social cluster) and switchback designs that alternate the whole market between arms over time.

**Long-term and subgroup effects.** A 2-week test measures 2-week outcomes; for retention keep a small long-running **holdout** without the new model. Slicing by platform or region can reveal harmed subgroups, but with twenty slices one will be "significant" by chance, so treat slices as hypotheses.

## Key terms

| Term | Meaning
| --- | --- |
| A/B test | Randomised experiment comparing a control (A) with a treatment (B) on live traffic
| Shadow / canary | Silent copy of traffic / small live share, to check safety before measuring value
| Guardrail metric | A metric that must not get worse, whatever happens to the primary metric
| Power | Probability that the test detects an effect of a given size, if it is real
| Minimum detectable effect | The smallest effect the test is sized to detect with the chosen power
| Sample ratio mismatch | Arm sizes differ from the design beyond chance; assignment or logging is broken
| CUPED | Subtracting the part of a metric predictable from pre-experiment data, to cut variance
| Peeking | Checking a fixed-horizon test repeatedly and stopping on significance; inflates false positives
| Always-valid p-value | A p-value (e.g. from mSPRT) that stays valid at any stopping time
| Interleaving | Merging two rankers' lists per user and crediting clicks, to detect preference cheaply

<div class="take">
<div class="lab">Key takeaway</div>

roll models out through shadow, canary and A/B stages that answer safety then value; size the test before it starts, randomise by user, check SRM before reading results, report the confidence interval, and use sequential tests if anyone will watch the dashboard.

</div>
