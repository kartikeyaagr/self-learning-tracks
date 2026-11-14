---
id: "06.05"
course: "machine-learning"
module: "06"
slug: "hyperparameter-search"
title: "Hyperparameter search"
keyIdeas: ["**Hyperparameters** are settings the training loss cannot choose (regularisation strength, depth, learning rate). Tuning them is optimisation of a validation score that is expensive, noisy and has no gradient, so the budget and the search space matter more than the search algorithm.", "Put scale-type knobs (learning rate, $C$, $\\lambda$) on a **log scale**: a uniform draw between 0.0001 and 0.1 lands above 0.01 about 90% of the time. Check that the winner is not on the edge of the range.", "**Random search** beats grid search when only a few knobs matter, because every trial tests a new value of each one. If the good region is 5% of the space, 60 random trials hit it with probability $1 - 0.95^{60} \\approx 95\\%$, however many knobs there are.", "**Bayesian optimisation** fits a surrogate of score versus configuration and picks the next trial with an acquisition function such as expected improvement; **successive halving / Hyperband** start many configurations cheaply and give more budget only to the best $1/\\eta$.", "The more configurations you try, the more the best validation score overstates the truth. Rerun the top few with new seeds, prefer the simpler of tied configurations, and score the final choice once on an untouched test set."]
question: "You have budget for about 100 training runs of a gradient-boosted model, each taking 10 minutes on one machine, and 8 machines. A teammate proposes a 5 x 5 x 4 grid over learning rate (0.01 to 0.3, linear steps), max depth and subsample rate, with the number of trees fixed at 500. Critique the plan and propose a better use of the same budget."
answer: "Problems with the plan: (1) A grid gives only 5 distinct learning rates, 5 depths and 4 subsample rates; since the learning rate usually dominates, most of the 100 runs re-test the same 5 learning rates while varying knobs that matter less. (2) Linear steps from 0.01 to 0.3 (0.01, 0.0825, 0.155, 0.2275, 0.3) put four of five values above 0.08, so the small-learning-rate region where boosting often does best is barely explored; the learning rate should be log-uniform. (3) Fixing 500 trees is biased against small learning rates, which need more trees, and wastes time on large ones, which overfit long before 500; enable early stopping on a validation split inside every run so the number of trees is measured rather than tuned. (4) There is no plan for noise or for the final evaluation. A better plan: fix the remaining knobs at sensible defaults; run random search (or a TPE sampler in Optuna) with learning rate log-uniform in roughly [0.005, 0.3], integer depth or leaf count, and subsample uniform in [0.5, 1.0]. Use about 80 trials across the 8 machines (about 10 rounds of 10 minutes, under two hours), ideally with successive halving or a median pruner so clearly poor configurations stop after a few hundred trees, which lets you start more than 80 configurations for the same compute. Check that no winning value sits on the edge of its range. Spend the remaining ~20 runs retraining the top 4 configurations with new seeds and folds, choose by mean score, prefer the cheaper one if they tie within the seed spread, and evaluate that single choice once on the untouched test set."
sources: [{"title": "Random search for hyper-parameter optimization (Bergstra and Bengio, JMLR 2012)", "url": "https://www.jmlr.org/papers/v13/bergstra12a.html", "note": "The low effective dimensionality argument and experiments showing random beats grid."}, {"title": "Practical Bayesian optimization of machine learning algorithms (Snoek, Larochelle, Adams)", "url": "https://arxiv.org/abs/1206.2944", "note": "Gaussian-process surrogates, expected improvement, and cost-aware acquisition."}, {"title": "Hyperband (Li et al.)", "url": "https://arxiv.org/abs/1603.06560", "note": "Successive halving, the bracket trade-off, and budget analysis."}, {"title": "Deep Learning Tuning Playbook (Google Research)", "url": "https://github.com/google-research/tuning_playbook", "note": "A practitioner's process: scientific vs nuisance parameters, search spaces, and when to stop."}]
practice: ["cross-join", "learning-rate-sweep", "early-stopping", "self-join"]
tags: ["hyperparameter tuning", "Bayesian optimisation", "early stopping"]
tools: ["scikit-learn", "Optuna"]
---

## What you will learn

- What separates a hyperparameter from a parameter, and why tuning is really an optimisation problem with an expensive, noisy objective.
- How to design a search space: which knobs to tune, which to fix, and why scale-type knobs belong on a log scale.
- Why random search usually beats grid search for the same budget, with the probability argument worked out by hand.
- How Bayesian optimisation and successive halving (Hyperband) use earlier trials to spend the budget more wisely.
- How to run all of this in scikit-learn, and how to keep the winning score honest instead of flattered by the search.

*Before you start:* you need a trustworthy validation scheme (cross-validation, lesson 06.04) and the idea of capacity and early stopping (lesson 01.04). Logarithms and basic probability are enough maths.

## From validation to choosing settings

Lesson 06.04 gave you a reliable way to score one configuration of a model with cross-validation. This lesson asks how to find a *good* configuration among thousands, without spending a week and without fooling yourself about how good it is.

## The idea in one picture

Task: predict whether a breast tumour is malignant from 30 measurements (the Wisconsin dataset, 569 patients) with standardised logistic regression. Its setting `C` controls how hard the weights are penalised: small `C` means a strong penalty (a simple model), large `C` almost none. Training cannot pick `C`, because the training loss always prefers weaker penalties.

So we try several values and score each with 5-fold cross-validation:

| C | 0.001 | 0.01 | 0.1 | 1 | 10 | 100 | 1000
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training accuracy | 89.9% | 95.4% | 98.2% | 98.8% | 99.0% | 99.3% | 99.7%
| Validation accuracy | 89.1% | 94.9% | 97.5% | 97.9% | 97.0% | 96.5% | 95.3%

~~~dg-fig 06.05-valcurve.svg
Training accuracy keeps rising as C grows, but validation accuracy peaks and then falls. A finer sweep finds the peak near C ≈ 0.46. Note the log-scaled horizontal axis: each tick is ten times the previous one.
~~~

Training accuracy climbs all the way to 99.7%, so it is useless for choosing. Validation accuracy peaks around `C = 1`, then falls as the model fits noise; a finer sweep (the figure) puts the top at `C ≈ 0.46` with 98.2%.

Terms used throughout:

- A **hyperparameter** is a setting chosen *before* training that the training loss cannot choose (here `C`). The **parameters** are what training learns (here the 30 weights and the bias).
- A **configuration** (or trial) is one full set of hyperparameter values, trained and scored once.
- The **search space** is the set of configurations you allow, with a range and a scale for each knob.
- The **budget** is how much compute you are willing to spend: a number of trials, GPU-hours, or epochs.

Note the scales: factors of ten in `C` change the result a lot, while the gap between 97.9% and 98.2% is about two patients out of 569, roughly the noise from fold assignment.

## Step 1: see tuning as optimisation

Write the validation score as a function of the hyperparameters, $f(\lambda)$, where $\lambda$ (lambda) stands for one whole configuration, for example $\lambda = (C, \text{kernel width})$. Tuning means finding $\lambda$ with a high $f(\lambda)$. Three properties make this unusual:

1. **Expensive.** One evaluation of $f$ is a full training run, possibly times five folds.
2. **Noisy.** Run the same configuration with another random seed or other folds and the score moves.
3. **No gradient.** You cannot differentiate the validation accuracy with respect to tree depth, so gradient descent is out.

So the main design decisions are the budget and where you let the search look, not the cleverness of the algorithm.

## Step 2: design the search space

Most wasted tuning comes from a badly designed space. Four rules cover most of it.

**Tune the few knobs that matter, fix the rest.** For gradient-boosted trees: learning rate, number of trees (via early stopping), tree size, minimum samples per leaf, subsampling. For neural networks: learning rate first by a wide margin, then batch size, weight decay and schedule. Defaults are usually fine for the rest.

**Use a log scale for multiplicative knobs.** Learning rate, `C`, regularisation strength $\lambda$ and kernel width all act by multiplying something, so the interesting question is "which factor of ten?", not "which value to the nearest 0.01?". Sampling **log-uniformly** means each factor-of-ten band gets the same share of trials.

~~~dg-fig 06.05-logscale.svg
Forty samples of a learning rate between 0.0001 and 0.1. Uniform sampling puts 37 of 40 in the top band and none in the bottom one; log-uniform sampling spreads them roughly evenly across the three bands.
~~~

A uniform draw from $[0.0001, 0.1]$ lands above 0.01 with probability $(0.1 - 0.01)/(0.1 - 0.0001) = 0.090 / 0.0999 \approx 90\%$.

**Make sure the winner is not on the edge.** If the best `C` you found is the largest one you tried, the true optimum may lie beyond the range. Extend the range and search again.

**Handle conditional knobs properly.** The kernel width matters only when the kernel is RBF; a grid crossing every kernel with every width wastes runs. Use a tool that supports conditional spaces (Optuna does) or search each branch separately.

## Step 3: grid search versus random search

**Grid search** picks a few values per knob and tries every combination. With 3 knobs at 5 values each that is $5^3 = 125$ runs, yet each knob is tried at only 5 distinct values. **Random search** draws every knob independently from its range for each trial, so 125 trials try 125 distinct values of every knob.

Bergstra and Bengio (2012) observed that usually only one or two knobs strongly affect the score. Then a grid's useful information is just its 5 values of the important knob; the other 120 runs repeat them while varying knobs that do not count.

~~~dg-fig 06.05-gridrandom.svg
The score depends almost only on the horizontal knob (green curve). Nine grid trials test just three values of it and miss the peak; nine random trials test nine values, and one lands close to the peak. The ticks under each panel are the values tried along the important axis.
~~~

Random search also gives a guarantee that ignores the number of knobs. If the "good region" covers a fraction $p$ of the space, each random trial lands in it with probability $p$, so $n$ trials all miss with probability $(1-p)^n$ and

$$P(\text{at least one trial in the good region}) = 1 - (1 - p)^n.$$

Nothing here mentions the number of dimensions.

<div class="ex">
<div class="lab">Worked example</div>

**how many random trials do you need?**

Suppose the top 5% of the space is good enough, so $p = 0.05$ and a trial misses with probability $0.95$.

With $n = 20$ trials: $0.95^{20} = 0.358$, so the hit probability is $1 - 0.358 = 64\%$.

With $n = 60$ trials: $0.95^{60} = 0.046$, so the hit probability is $1 - 0.046 = 95.4\%$.

To reach 95% exactly, solve $0.95^n = 0.05$: $n = \ln 0.05 / \ln 0.95 = -2.996 / -0.0513 \approx 58.4$, so 59 trials.

If you want the top 1% instead ($p = 0.01$): $0.99^{60} = 0.547$, only a 45% chance, and 95% needs $\ln 0.05 / \ln 0.99 \approx 298$ trials.

Compare with a grid: 60 runs over 3 knobs is about $4 \times 4 \times 4 = 64$, which tests only 4 values of the knob that matters.

</div>

Grid search still suits one or two knobs: it is easy to plot and read. A common pattern is a coarse log-scaled grid to find the region, then a finer random search inside it.

## Step 4: learn from earlier trials (Bayesian optimisation)

Random search ignores what it has seen. **Bayesian optimisation** uses the history. It works in rounds:

1. Fit a cheap **surrogate model** to all (configuration, score) pairs so far. The surrogate predicts the score of any untried configuration, with an uncertainty.
2. Use an **acquisition function** to decide which configuration is most worth trying next.
3. Train that configuration for real, record its score, and repeat.

~~~dg-seq One round of Bayesian optimisation. Each round costs one training run; the surrogate update is cheap.
Optimiser > Surrogate: fit to all (config, score) pairs so far
Surrogate > Optimiser: mean and uncertainty for any config
Optimiser > Optimiser: maximise expected improvement
Optimiser > Trainer: train with the chosen config
Trainer -> Optimiser: validation score
~~~

The classic surrogate is a **Gaussian process** (Snoek et al., 2012): a smooth mean with an uncertainty band that is narrow near tried points and wide far from them. Optuna and Hyperopt use a **tree-structured Parzen estimator (TPE)**, which copes better with discrete and conditional knobs.

The most common acquisition function is **expected improvement (EI)**: for each candidate, how much do we expect it to beat the current best, averaged over the surrogate's uncertainty? It favours both good predicted means and large uncertainty.

~~~dg-fig 06.05-bayesopt.svg
Four learning rates have been tried (black dots). The surrogate's band is wide in the unexplored gap between them; expected improvement peaks there, at about 10^-2, which is where the next trial goes. The dotted true curve is shown only for teaching; the optimiser never sees it.
~~~

In the figure the best trial so far sits at $10^{-1.4}$, but the surrogate says the gap between $10^{-3}$ and $10^{-1.4}$ could hide something better, so the next trial lands at about $10^{-2.0}$, close to the true peak.

It pays off when trials are expensive (GPU-hours): it usually reaches a given score in fewer trials than random search. But it is naturally sequential, its edge shrinks with many knobs or noisy scores, and under about 20 trials a well-designed random search is hard to beat.

## Step 5: stop bad trials early (successive halving and Hyperband)

Many configurations are obviously bad after a small fraction of their training. **Successive halving** exploits this:

1. Start $n$ configurations on a small budget $r$ (a few epochs, a few trees, or a subsample of the data).
2. Keep the best $1/\eta$ of them (eta, typically 3), and multiply their budget by $\eta$.
3. Repeat until one configuration has the full budget.

~~~dg-fig 06.05-halving.svg
Twenty-seven configurations with η = 3. After 1, 3 and 9 epochs only the best third continue; grey curves are stopped. The dashed configuration starts slowly and is cut at the first rung, although it would have finished best.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the budget arithmetic of successive halving**

Take 81 configurations, $\eta = 3$, and a full training run of 81 epochs.

Rung 1: 81 configs × 1 epoch = 81 epochs. Rung 2: 27 × 3 = 81. Rung 3: 9 × 9 = 81. Rung 4: 3 × 27 = 81. Rung 5: 1 × 81 = 81.

Each rung costs the same 81 epochs, so the whole search costs $5 \times 81 = 405$ epochs.

Training all 81 configurations to completion costs $81 \times 81 = 6{,}561$ epochs, about 16 times more ($6561 / 405 = 16.2$).

The risk is the configuration that looks bad at 1 epoch but would win at 81. Very small learning rates behave like this, as the dashed curve in the figure shows.

</div>

**Hyperband** hedges against that risk by running several successive-halving **brackets** with different starting points: one aggressive bracket (many configs, tiny budget), one cautious bracket (few configs, full budget), and some in between. **ASHA** is an asynchronous version for clusters. Optuna calls the same idea **pruning**: a trial reports intermediate scores, and it is stopped if it falls below, say, the median of earlier trials at the same step.

This differs from **early stopping inside one run** (lesson 01.04), which stops when validation loss stalls for `patience` epochs and keeps the best checkpoint. It turns the number of epochs or trees into something measured, so switch it on inside every trial.

## Hyperparameter search in code

scikit-learn has all three styles. Here they are on the 8×8 handwritten digits (1,797 images), tuning an RBF support vector machine's `C` and `gamma` (kernel width) with the same 5-fold CV. A quarter of the data is held back as a test set that no search ever sees.

```python
from scipy.stats import loguniform
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

X, y = load_digits(return_X_y=True)
X = X / 16.0                                             # pixels to [0, 1]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25,
                                                stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}  # 30 configs
gs = GridSearchCV(SVC(), grid, cv=cv, n_jobs=-1).fit(X_dev, y_dev)

space = {"C": loguniform(1e-1, 1e3), "gamma": loguniform(1e-4, 1e1)}      # log scales
rs = RandomizedSearchCV(SVC(), space, n_iter=30, cv=cv, random_state=0,
                        n_jobs=-1).fit(X_dev, y_dev)

hs = HalvingRandomSearchCV(SVC(), space, n_candidates=81, factor=3, cv=cv,
                           random_state=0, n_jobs=-1).fit(X_dev, y_dev)

for name, s in [("grid", gs), ("random", rs), ("halving", hs)]:
    print(name, s.best_params_, round(s.best_score_, 4), round(s.score(X_test, y_test), 4))
# grid    {'C': 10,   'gamma': 0.1}    CV 0.9881  test 0.9889
# random  {'C': 4.95, 'gamma': 0.170}  CV 0.9881  test 0.9933
# halving {'C': 4.96, 'gamma': 0.108}  CV 0.9889  test 0.9911
print(hs.n_resources_, hs.n_candidates_)
# [100, 300, 900] [81, 27, 9]   training-set size and candidates per rung
```

All three find the same region (`C` about 5 to 10, `gamma` about 0.1), with CV scores within 0.1%. The test scores differ by up to 0.44%, which is two images out of 450: noise, not a ranking. The halving search screened 81 candidates but trained most on only 100 or 300 images (the halved resource is the sample count by default); each rung costs $81 \times 100 = 27 \times 300 = 9 \times 900 = 8{,}100$ sample-fits.

For boosting or neural networks, Optuna is common: write a function that scores one model built from calls like `trial.suggest_float("lr", 1e-4, 1e-1, log=True)`, and add a TPE sampler and a pruner.

## Checking the result

Before trusting a search, check:

- **Where is the winner?** If any best value sits at the edge of its range, widen the range and rerun.
- **Plot score against each knob** from `pd.DataFrame(search.cv_results_)`. A hill means the knob matters; a flat cloud means fix it.
- **Compare the top few with `std_test_score`.** Configurations within one standard deviation are a tie.
- **Rerun the finalists.** Retrain the top 3 to 5 with several new seeds (and new folds if possible) and choose by the mean.
- **Prefer the simpler or cheaper of tied configurations**, and log every trial (config, seed, score, code version; lesson 09.04).
- **Touch the test set once**, for the final choice only.

## What goes wrong

The subtlest failure is **overfitting the validation set**. Every trial is another look at the validation data; even if all configurations were equal, the luckiest one would win, and more trials means a luckier winner.

~~~dg-fig 06.05-optimism.svg
A simulation where every configuration's true accuracy is about 85% (spread 0.4%) and each validation score adds 0.8% noise. After 300 trials the best score you see averages 87.6%, but the chosen configuration's true accuracy is only 85.5%.
~~~

This is the same optimism that nested cross-validation (06.04) measures. With thousands of trials on a small validation set it can exceed the real differences between models.

| Mistake | Symptom | Fix
| --- | --- | --- |
| Linear range for a scale knob | almost no trials at small values; most trials score alike | log-uniform sampling
| Winner on the range edge | best `C` is the largest tried | widen the range, rerun
| Fixed number of epochs or trees | small learning rates always lose | early stopping inside each trial
| Tuning on the test set | test score falls when the model ships | keep a separate test set, or nested CV
| Too many trials, small validation set | big drop from CV score to test score | fewer trials, reseeded reruns, larger folds
| Preprocessing fitted outside the CV loop | CV score suspiciously high | put scaling and selection inside a `Pipeline`

## Going deeper

**Why random search wins: low effective dimension.** If the score depends on $d$ knobs but only $k \ll d$ matter, a grid with $m$ values per knob costs $m^d$ runs and tests only $m$ values of each important knob. Random search with the same $m^d$ runs tests $m^d$ values. Quasi-random sequences (Sobol, Latin hypercube, available in `scipy.stats.qmc` and as samplers in Optuna) keep this property and spread points a bit more evenly than pure random draws.

**Expected improvement in a formula.** Let the surrogate predict a mean $\mu(\lambda)$ and standard deviation $\sigma(\lambda)$ for configuration $\lambda$, and let $f^*$ be the best score so far. With $z = (\mu - f^*)/\sigma$,

$$\text{EI}(\lambda) = (\mu - f^*)\,\Phi(z) + \sigma\,\phi(z),$$

where $\Phi$ and $\phi$ are the standard normal cumulative distribution and density. The first term rewards a high mean, the second rewards uncertainty. It is zero where the surrogate is sure the point is worse than $f^*$.

**TPE in one sentence.** Split past trials into a good fraction and the rest, fit a density $\ell(\lambda)$ to the good ones and $g(\lambda)$ to the others, and propose the candidate with the largest ratio $\ell(\lambda)/g(\lambda)$.

**Successive halving cost.** With $n$ starting configurations, ratio $\eta$, and minimum budget $r$, there are $s = \log_\eta n + 1$ rungs, each costing about $n r$, so the total is $n r (\log_\eta n + 1)$, compared with $n \cdot r \eta^{s-1}$ for training everything fully. The saving factor is $\eta^{s-1}/s$, which is $81/5 \approx 16$ in the example. Hyperband's brackets trade this saving against the risk of killing slow starters.

**Noise is the floor.** If rerunning one configuration with new seeds moves its score by $\pm 0.3\%$, no search can reliably separate configurations closer than that. Measure that spread first, so you know when to stop searching.

## Key terms

| Term | Meaning
| --- | --- |
| Hyperparameter | A setting chosen before training that the training loss cannot choose, such as `C`, depth or learning rate
| Search space | The allowed range and scale (linear, log, integer, categorical) for each tuned knob
| Log-uniform | Sampling so that each factor-of-ten band gets an equal share of trials
| Grid search | Trying every combination of a few fixed values per knob
| Random search | Drawing each knob independently for every trial; tests many distinct values of each
| Bayesian optimisation | Fitting a surrogate of score versus configuration and choosing the next trial by an acquisition function
| Expected improvement | How much a candidate is expected to beat the current best, given the surrogate's uncertainty
| Successive halving / Hyperband | Start many configs cheaply, keep the best $1/\eta$, give them $\eta$ times more budget; Hyperband runs several such brackets
| Pruning | Stopping a trial early because its intermediate scores lag earlier trials
| Validation optimism | The amount by which the best validation score overstates the winner's true score after many trials

<div class="take">
<div class="lab">Key takeaway</div>

hyperparameter search optimises an expensive, noisy validation score, so spend your effort on a small, log-scaled, well-bounded search space and a fixed budget; use random search by default, Bayesian optimisation when trials are costly, early stopping and successive halving to drop losers early, and reseeded reruns plus one untouched test set to keep the winner honest.

</div>
