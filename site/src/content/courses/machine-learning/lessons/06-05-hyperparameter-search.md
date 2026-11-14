---
id: "06.05"
course: "machine-learning"
module: "06"
slug: "hyperparameter-search"
title: "Hyperparameter search"
keyIdeas: ["Hyperparameters are choices the training loss cannot make (regularisation strength, depth, learning rate). Search is optimisation of a noisy, expensive validation score, so the budget and the search space matter more than the algorithm.", "Put scale parameters (learning rate, $C$, $\\lambda$) on a **log scale**. A badly bounded range wastes most trials no matter how clever the search.", "**Random search** beats grid search when only a few hyperparameters matter: every trial tries a new value of each one. With 60 trials, the chance of hitting the top 5% of the space is $1 - 0.95^{60} \\approx 95\\%$.", "**Bayesian optimisation** fits a surrogate of score versus config and picks the next trial by an acquisition function; **successive halving / Hyperband** spend little on bad configs by stopping them early.", "The more configs you try, the more the best validation score overfits the validation set. Rerun the top few with new seeds and keep a final test set untouched."]
question: "You have budget for about 100 training runs of a gradient-boosted model, each taking 10 minutes on one machine, and 8 machines. A teammate proposes a 5 x 5 x 4 grid over learning rate (0.01 to 0.3, linear steps), max depth and subsample rate. Critique the plan and propose a better use of the same budget."
answer: "The grid gives only 5 distinct values of each parameter, and if learning rate dominates (it usually does) most of the 100 runs re-test the same 5 learning rates. Linear steps from 0.01 to 0.3 crowd values at the high end; learning rate should be log-scaled. The number of trees is not mentioned, so each run must either fix it (biased against small learning rates) or use early stopping, which it should. Better: enable early stopping on a validation split inside every run, then use random search (or a TPE sampler) with log-uniform learning rate, integer depth or leaves, and uniform subsample, about 80 trials across the 8 machines, ideally with successive halving so poor configs stop after a few hundred trees. Spend the remaining ~20 runs re-training the top 4 configs with new seeds and folds, pick by mean score, then evaluate once on the held-out test set."
sources: [{"title": "Random search for hyper-parameter optimization (Bergstra and Bengio, JMLR 2012)", "url": "https://www.jmlr.org/papers/v13/bergstra12a.html", "note": "The low effective dimensionality argument and experiments showing random beats grid."}, {"title": "Practical Bayesian optimization of machine learning algorithms (Snoek, Larochelle, Adams)", "url": "https://arxiv.org/abs/1206.2944", "note": "Gaussian-process surrogates, expected improvement, and cost-aware acquisition."}, {"title": "Hyperband (Li et al.)", "url": "https://arxiv.org/abs/1603.06560", "note": "Successive halving, the bracket trade-off, and budget analysis."}, {"title": "Deep Learning Tuning Playbook (Google Research)", "url": "https://github.com/google-research/tuning_playbook", "note": "A practitioner's process: scientific vs nuisance parameters, search spaces, and when to stop."}]
practice: ["cross-join", "learning-rate-sweep", "early-stopping", "self-join"]
tags: ["hyperparameter tuning", "Bayesian optimisation", "early stopping"]
tools: ["scikit-learn", "Optuna"]
---

## Parameters versus hyperparameters

Training finds parameters (weights, split points) by minimising a loss. Hyperparameters are everything that loss cannot choose for you: regularisation strength, tree depth, number of neighbours, learning rate, batch size. Lesson 01.04 showed why some of them exist (capacity control); this lesson is about choosing them efficiently once you have a trustworthy validation scheme from 06.04.

Hyperparameter search is optimisation of a function $f(\lambda)$ = validation score, with three nasty properties: each evaluation is expensive (a full training run), it is noisy (seed, fold assignment), and you have no gradient. That makes **budget** the central design variable.

## Design the search space first

Most wasted tuning comes from the space, not the search algorithm.

- Use a **log scale** for anything that acts multiplicatively: learning rate in $[10^{-4}, 10^{-1}]$, $C$ or $\lambda$ in $[10^{-3}, 10^{3}]$. A uniform range from 0.0001 to 0.1 puts 90% of samples above 0.01.
- Tune the few parameters that matter and fix the rest at good defaults. For gradient boosting: learning rate, number of trees (via early stopping), depth or leaves, minimum samples per leaf, subsampling. For neural nets: learning rate first, by a wide margin.
- Check the winner is not on the **edge** of the range. If the best $C$ is the largest one you tried, the range was wrong.
- Handle **conditional** parameters (kernel width only when the kernel is RBF) with a search tool that supports them, rather than a grid of invalid combinations.

## Grid versus random

Grid search tries every combination of a few values per parameter. With 3 parameters at 5 values that is 125 runs, but only 5 distinct values of each. If only one parameter really matters, which Bergstra and Bengio found is typical, 120 of those runs repeat values you already tested for the one that counts.

Random search samples each parameter independently. With 125 random trials, the important parameter gets 125 distinct values. A useful rule: if a good region covers 5% of the space, $n$ random trials miss it with probability $0.95^n$, so 60 trials find it with about 95% probability, whatever the number of dimensions. Grid search still has a place for one or two parameters, where it is easy to read and plot.

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
space = {"learning_rate": loguniform(1e-3, 3e-1), "max_leaf_nodes": randint(8, 128),
         "min_samples_leaf": randint(5, 200), "l2_regularization": loguniform(1e-4, 10)}
search = RandomizedSearchCV(HistGradientBoostingClassifier(early_stopping=True), space,
                            n_iter=60, cv=cv, scoring="average_precision", random_state=0, n_jobs=-1)
```

## Bayesian optimisation

Random search ignores what earlier trials revealed. **Bayesian optimisation** keeps a surrogate model of $f$, a Gaussian process (Snoek et al., 2012) or a tree-structured Parzen estimator (as in Optuna and Hyperopt), and chooses the next config by maximising an **acquisition function** such as expected improvement, which balances trying near the current best against exploring regions the surrogate is unsure about.

~~~dg-seq One round of Bayesian optimisation. Each round costs one training run; the surrogate update is cheap.
Optimiser > Surrogate: fit to all (config, score) pairs so far
Surrogate > Optimiser: mean and uncertainty for any config
Optimiser > Optimiser: maximise expected improvement
Optimiser > Trainer: train with the chosen config
Trainer -> Optimiser: validation score
~~~

It typically reaches a given score in fewer trials than random search, which matters when one trial costs GPU-hours. Its weaknesses: it is sequential (parallel variants exist but lose some efficiency), and its advantage shrinks in high-dimensional or very noisy spaces. With a budget of under about 20 trials, a well-designed random search is hard to beat.

## Spend less on bad configs

Many configurations are obviously bad after a fraction of training. **Successive halving** starts many configs on a small budget (few epochs, a data subsample, few trees), keeps the best $1/\eta$, multiplies their budget by $\eta$, and repeats. **Hyperband** runs several such brackets with different starting sizes, hedging against configs that start slow and finish strong; ASHA is its asynchronous version for clusters. Optuna calls the same idea pruning: a trial reports intermediate scores and is stopped if it falls below the median of earlier trials at that step.

Separate this from **early stopping within one run** (lesson 01.04): stopping training when validation loss has not improved for `patience` epochs and keeping the best checkpoint. Early stopping turns the number of epochs or trees from a hyperparameter into something measured, so it is worth enabling inside every trial.

<div class="ex">
<div class="lab">Worked example</div>

**the budget arithmetic of successive halving**

81 configs, $\eta = 3$, full training is 81 epochs. Rung 1: 81 configs x 1 epoch. Rung 2: 27 x 3. Rung 3: 9 x 9. Rung 4: 3 x 27. Rung 5: 1 x 81. Each rung costs 81 epochs, 405 epochs in total.

Training all 81 configs fully costs 81 x 81 = 6,561 epochs, 16 times more. The risk is the configs that look bad at 1 epoch but would win at 81 (very small learning rates do this); Hyperband's extra brackets, which start fewer configs at a larger budget, exist to catch them.

</div>

## Do not overfit the validation set

Every trial is a look at the validation data. After hundreds of trials the best score is partly selection noise, the same optimism nested CV addresses. Practical guards:

1. Rerun the top 3 to 5 configs with several new seeds (and if possible new folds) and choose by the mean; differences smaller than the seed spread are ties.
2. Prefer the simpler or cheaper of tied configs.
3. Score the final choice once on an untouched test set, or use nested CV when data is small.
4. Log every trial (config, seed, score, code version), so the search itself is reproducible; lesson 09.04 covers experiment tracking.

<div class="take">
<div class="lab">Key takeaway</div>

spend effort on a log-scaled, well-bounded space and a fixed budget; use random search by default, Bayesian optimisation when trials are expensive, early stopping and successive halving to cut losers early, and reseeded reruns to keep the winner honest.

</div>
