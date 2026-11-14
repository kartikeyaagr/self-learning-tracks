---
id: "01.02"
course: "machine-learning"
module: "01"
slug: "generalisation-and-the-bias-variance-trade-off"
title: "Generalisation and the bias-variance trade-off"
keyIdeas: ["Test error splits exactly into irreducible **noise**, **bias²** (how far the average model over many training sets is from the truth) and **variance** (how much the fitted model moves when the training set changes).", "Capacity is the dial: too little and both training and validation error are high and close (underfitting); too much and training error keeps falling while validation error climbs (overfitting).", "You diagnose with two numbers: training error tells you about bias, the gap to validation error tells you about variance. Learning curves show whether more data will help.", "More data reduces variance but not bias; regularisation and simpler classes trade a little bias for a lot less variance, and averaging many models cuts variance.", "Modern over-parameterised models can show **double descent**: test error peaks where the model can just interpolate the training set, then falls again as parameters keep growing."]
question: "A gradient-boosted model has 2% training error and 2.5% validation error, but the product needs 1%. A colleague proposes collecting ten times more data. Using the bias-variance view, argue whether this is likely to help and what you would try instead."
answer: "Training and validation error are close (2% vs 2.5%), so variance is small: the model is not overfitting much. The dominant problem is **bias** (or irreducible noise): even on data it has seen, the model reaches only 2%, which is already twice the target. More data mainly reduces variance, so ten times more rows would likely shave a little off the 0.5% gap and leave the 2% floor untouched. Before spending on data, draw a learning curve (train on 10%, 25%, 50%, 100% of the current data): if the two curves have already flattened and nearly met, extra data is confirmed to be a poor investment. Better options attack bias: add capacity (deeper trees, more boosting rounds, a lower learning rate with more rounds, less regularisation) and, above all, add informative features, because if the 2% is close to the Bayes error for the current inputs no model can beat it. A useful check is human or oracle error on a sample: if experts with the same inputs also miss about 2%, the target needs new information (features, or cleaner labels), not more of the same rows. Only if added capacity opens a wide train-validation gap does more data become the right next step."
sources: [{"title": "Cornell CS4780: Bias-variance decomposition", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html", "note": "Clean derivation of the three-term decomposition and how to read train/test curves."}, {"title": "Reconciling modern machine learning practice and the bias-variance trade-off (Belkin et al.)", "url": "https://arxiv.org/abs/1812.11118", "note": "The double-descent paper; read the introduction and Figure 1."}, {"title": "Understanding deep learning requires rethinking generalization (Zhang et al.)", "url": "https://arxiv.org/abs/1611.03530", "note": "Networks fitting random labels; why capacity counts alone fail to explain generalisation."}, {"title": "scikit-learn: Underfitting vs overfitting", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html", "note": "A runnable polynomial-degree sweep with cross-validated error."}]
practice: ["knn-classifier", "k-fold-cross-validation", "bagging-classifier", "decision-tree-regressor"]
tags: ["bias-variance", "overfitting"]
tools: ["NumPy"]
---

## What you will learn

- What **generalisation** means, and why a model that fits its training data perfectly can still be useless.
- The three sources of test error: **noise**, **bias** and **variance**, in plain words and as one exact formula.
- How **capacity** (how flexible a model class is) trades bias against variance, and why test error traces a U.
- How to tell underfitting from overfitting using only training and validation error, plus learning curves.
- What actually reduces bias and what reduces variance: more data, regularisation, averaging, better features.
- Why very large models can break the U shape (**double descent**), and when that matters.

*Before you start:* lesson 01.01 (hypothesis class, loss, why training error is optimistic). You need averages, squares and the idea of an expected value (an average over many repeats).

## From the last lesson

Lesson 01.01 ended with a warning: training error flatters a model, and the more freedom the model class has, the more it flatters. This lesson explains why. The payoff is practical: by comparing training error with validation error you can tell which of two very different problems you have, and they need opposite fixes.

**Generalisation** is a model's ability to perform well on new examples drawn from the same distribution as its training data. It is the only kind of performance that matters once a model is deployed.

## The idea in one picture

A facilities team predicts a building's daily energy use from the day's average outdoor temperature. Cold days need heating, hot days need cooling, mild days need the least. They fit polynomials of three degrees to 20 days of records, then score each fit on 400 fresh days.

| Model | Training MSE | Test MSE (new days) | Verdict
| --- | --- | --- | --- |
| Degree 1 (straight line) | 12.5 | 14.0 | too simple
| Degree 4 | 2.0 | 4.9 | about right
| Degree 15 | 0.6 | 24.3 | too flexible

MSE is mean squared error in (kWh/day)², so smaller is better.

~~~dg-fig 01.02-three-fits.svg
The same 20 days fitted three ways. The line cannot bend, so it misses on both cold and warm days. Degree 4 tracks the dotted true pattern. Degree 15 chases individual points and swings wildly near the edges.
~~~

Read the table row by row:

- The straight line is wrong on the training days *and* on new days, by about the same amount. It is **underfitting**: the model class is too simple to express the U-shaped pattern.
- Degree 15 has the lowest training error of all, yet the worst test error. It is **overfitting**: it has used its flexibility to fit the random day-to-day noise, which does not repeat on new days.
- Degree 4 sits in between and wins on new data.

## Step 1: imagine many training sets

The key mental move is this. Your 20 days are one random sample. A different 20 days (another year, another building of the same type) would give a slightly different fitted curve. So a fitted model is itself random: it depends on the luck of the sample.

Now imagine repeating the experiment many times: draw 20 days, fit, record the curve, and repeat.

~~~dg-fig 01.02-many-fits.svg
Twenty training sets, twenty fits per panel (thin lines) and their average (thick line). Left: the lines agree with each other, but their average misses the dotted truth. Right: the degree-12 fits average out close to the truth, but any single one can be far off.
~~~

Two separate things can go wrong:

- **Bias** is how far the *average* fit is from the truth. The straight lines are consistent but consistently wrong: no amount of averaging turns a line into a U.
- **Variance** is how much individual fits scatter around their own average. The degree-12 curves are right on average but jump around from sample to sample, especially near the edges where data is thin.

On a dartboard, bias is aiming at the wrong spot and variance is a shaky hand. Each needs a different cure.

## Step 2: the formula

To make this exact, write the data as true pattern plus noise:

$$y = f(x) + \varepsilon$$

where $f(x)$ is the true average energy use at temperature $x$ (unknown to us), and $\varepsilon$ is random noise with average 0 and variance $\sigma^2$. Here $\sigma^2 = 4$ (a typical day is about 2 kWh off the pattern because of occupancy, sunshine and so on).

Let $S$ be a training set and $\hat f_S$ the model fitted on it. Let $\bar f(x) = \mathbb{E}_S[\hat f_S(x)]$ be the **average prediction** at $x$ over all possible training sets (the thick line in the figure). For squared loss, the expected test error at $x$ splits exactly into three pieces:

$$\mathbb{E}_{S,\varepsilon}\big[(y - \hat f_S(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat f_S(x) - \bar f(x))^2\big]}_{\text{variance}}$$

What each piece means:

- **Noise** $\sigma^2$ is the Bayes error from lesson 01.01: the part of $y$ that the inputs cannot explain. No model changes it; only better features (more information) can.
- **Bias²** is the squared distance between the average model and the truth. It measures what the model class *cannot* represent.
- **Variance** is the average squared distance between one fitted model and the average model. It measures how much the fit depends on which sample you happened to get.

$\mathbb{E}_{S,\varepsilon}$ means "averaged over training sets and over the noise in the new label". The split holds at every $x$; averaging over $x$ gives overall test error. It is exact for squared loss; for 0-1 loss there are analogues but no clean additive split, though the intuition carries over.

<div class="ex">
<div class="lab">Worked example</div>

**computing bias and variance by hand**

Take one hot day, $x = 28$ °C. Suppose the true average use is $f(28) = 15$ kWh and the noise variance is $\sigma^2 = 4$. We train three models on the same five training sets and record each one's prediction for this day.

Line (degree 1): predictions $11, 12, 11, 12, 11.5$. Average $\bar f = 57.5/5 = 11.5$. Bias² $= (15 - 11.5)^2 = 12.25$. Deviations from the average: $-0.5, 0.5, -0.5, 0.5, 0$; squared $0.25, 0.25, 0.25, 0.25, 0$; variance $= 1/5 = 0.2$.

Degree 4: predictions $15.5, 14.5, 15, 16, 14$. Average $= 75/5 = 15$, so bias² $= 0$. Squared deviations $0.25, 0.25, 0, 1, 1$; variance $= 2.5/5 = 0.5$.

Degree 12: predictions $19, 11, 17, 12, 16$. Average $= 75/5 = 15$, bias² $= 0$. Squared deviations $16, 16, 4, 9, 1$; variance $= 46/5 = 9.2$.

Expected test error = noise + bias² + variance: line $4 + 12.25 + 0.2 = 16.45$; degree 4: $4 + 0 + 0.5 = 4.5$; degree 12: $4 + 0 + 9.2 = 13.2$.

Check the line directly: its squared errors against the truth are $16, 9, 16, 9, 12.25$, averaging $62.25/5 = 12.45 = 12.25 + 0.2$. The split is exact, not an approximation.

Lesson: the degree-12 model has *zero* bias here and still loses badly to degree 4. Low bias is not the goal; low bias² + variance is.

</div>

## Step 3: capacity is the dial

**Capacity** is how rich the model class is: how many different shapes it can take. Polynomial degree is one capacity dial. Others you will meet: tree depth, number of neighbours $k$ in k-NN (small $k$ means high capacity), number of parameters in a network, and the inverse of the regularisation strength.

Turning capacity up lowers bias (more shapes available) and raises variance (more freedom to chase noise). Training error falls every time; test error, being noise + bias² + variance, falls and then rises.

~~~dg-fig 01.02-ucurve.svg
Left: averaged over 800 training sets of 20 days, training error falls with every extra degree, while test error bottoms out around degree 4 and then climbs. Right: the same test error split into its pieces. Bias² collapses by degree 2-4, variance keeps growing, noise is a flat floor of 4.
~~~

From that sweep (bias², variance): degree 1 gives (10.1, 0.4), degree 4 (0.2, 0.9), degree 12 (about 0, 14.5). Degrees 2 to 6 all score 5.1-5.4: the bottom of the U is usually flat, so you need not hit it exactly.

~~~dg-flow As capacity grows, the dominant source of test error switches from bias to variance.
Low capacity -(high bias)-> Underfit -(add capacity)-> *Sweet spot -(add more)-> Overfit -(high variance)-> Memorised training set
~~~

## Step 4: diagnose from two numbers

In real projects you never see $f(x)$, so you cannot compute bias and variance directly. You can compare training error with validation error (error on held-out data; see 01.03), and the pattern tells you which problem dominates.

| Symptom | Training error | Validation error | Likely cause | First fix
| --- | --- | --- | --- | --- |
| Underfitting | high | high, close to training | bias | richer class, better features, less regularisation
| Overfitting | low | much higher than training | variance | more data, regularisation, simpler class
| Healthy | low | slightly above training | balanced | stop tuning, check on test
| Data problem | low | high and noisy across folds | split or leakage issue | check the split (next lesson)

"High" needs a reference point: the noise level if you know it, a simple baseline (01.05), or human expert error. In the energy example, training error 12.5 against noise 4 is a clear sign of bias.

## Step 5: learning curves show what more data will buy

A **learning curve** plots training and validation error against the number of training examples. It answers the most expensive question in many projects: "should we collect more data?"

~~~dg-fig 01.02-learning-curves.svg
Left: the straight line's training and test errors meet almost at once, around 14, far above the noise floor of 4. More days do nothing. Right: degree 10 starts with a huge gap (1.1 training vs 9.2 test at 15 days) that closes towards the noise floor as data grows.
~~~

- Curves that have already met, well above the level you need: a bias problem. More data will not help; a better model or features will.
- A wide gap that is still shrinking: a variance problem. More data helps, as do regularisation and simpler models.

Hence a rule worth memorising: **more data reduces variance, not bias.** Variance often shrinks roughly like $1/n$; bias stays put. A billion more rows will not bend an underfitting line.

## What moves bias and what moves variance

| Action | Effect on bias | Effect on variance | Where in the course
| --- | --- | --- | --- |
| Add training data | none | down (roughly $1/n$) | 01.03, data collection
| Add capacity (degree, depth, parameters) | down | up | every model module
| Add informative features | down | small rise | feature engineering
| Regularise (ridge, lasso, weight decay) | up a little | down a lot | 01.04, 02.02
| Average many models (bagging, random forests) | about the same | down | module 04

Regularisation accepts a little bias for a large cut in variance: ridge predictions are pulled towards zero, but barely move when the sample changes. Averaging works because the wobbles of different high-variance models partly cancel.

<div class="ex">
<div class="lab">Worked example</div>

**k-nearest neighbours as a capacity dial**

k-NN predicts by majority vote among the $k$ closest training points. With $k = 1$ each prediction copies the label of the single nearest point, so training error is 0 (each point is its own neighbour). The decision boundary is jagged and changes whenever one point moves: low bias, high variance.

With $k = n$, every prediction is the majority class of the whole training set: a constant. One more point barely changes it, so variance is near 0, but bias is as large as it can be.

On a two-class "moons" dataset with 1000 training and 1000 validation points (scikit-learn's `make_moons`, noise 0.35), validation error is 15.9% at $k = 1$, about 10.4% at $k = 30$, 18.5% at $k = 500$ and 50.3% at $k = 1000$ (a coin flip). Training error at $k = 1$ is 0%.

Pick $k$ by validation error, never by training error, which is always best at $k = 1$.

</div>

~~~dg-fig 01.02-knn.svg
k-NN on a noisy two-class problem, with k shrinking from 1000 to 1 so that capacity grows to the right. Training error falls to zero at k = 1; validation error is lowest for moderate k.
~~~

## Bias and variance in code

The script rebuilds the energy example, sweeps capacity, then measures bias and variance at one input by refitting on 2000 fresh training sets. On real data you cannot do the second part (`true_f` is unknown), but doing it once makes the decomposition concrete.

```python
import numpy as np
from numpy.polynomial import Polynomial

rng = np.random.default_rng(3)
def true_f(t):                       # unknown in real life; known here so we can measure bias
    return 11 + 0.035 * (t - 17)**2 + 1.2 * np.sin(t / 3)

def make_days(n):                    # n days spread over -5..30 C, noise sd 2 kWh
    t = np.linspace(-4.2, 29.2, n) + rng.uniform(-0.8, 0.8, n)
    return t, true_f(t) + rng.normal(0, 2.0, n)

t_tr, y_tr = make_days(20)           # training days
t_va, y_va = make_days(400)          # validation days
mse = lambda a, b: np.mean((a - b)**2)

# 1) Capacity sweep: train vs validation error
for deg in [1, 2, 4, 8, 12, 15]:
    p = Polynomial.fit(t_tr, y_tr, deg)   # rescales t internally, so high degrees stay stable
    print(deg, round(mse(p(t_tr), y_tr), 2), round(mse(p(t_va), y_va), 2))
# 1 12.65 13.78     <- both high: bias
# 2 4.57 5.12
# 4 4.02 4.86       <- best validation error
# 8 2.68 6.05
# 12 1.98 7.72
# 15 0.98 539.76    <- tiny training error, disastrous validation: variance

# 2) Bias and variance at one input (t = 28 C), refitting on 2000 fresh training sets
for deg in [1, 4, 12]:
    preds = np.array([Polynomial.fit(*make_days(20), deg)(28.0) for _ in range(2000)])
    bias2 = (preds.mean() - true_f(28.0))**2
    print(deg, round(bias2, 2), round(preds.var(), 2), round(4 + bias2 + preds.var(), 2))
# 1 20.84 0.63 25.47     <- error is almost all bias
# 4 0.07 1.39 5.47
# 12 0.03 30.67 34.69    <- error is almost all variance
```

`Polynomial.fit` maps inputs to $[-1, 1]$ first, which keeps degree-15 fits numerically sane. Variance at 28 °C exceeds the all-temperature average because 28 °C is near the edge of the data, where flexible fits swing most.

## Reading the result

How to tell, on a real project, that your diagnosis is right:

- **Underfitting confirmed:** training error is already poor, and raising capacity (more degrees, deeper trees, fewer neighbours) lowers *both* training and validation error.
- **Overfitting confirmed:** validation error improves when you *lower* capacity or regularise, even though training error rises. The validation score also swings when the seed or fold changes.
- **At the floor:** if both errors are close and capacity changes barely move them, you may be at the noise level for these features. If experts do no better, only new information will help.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Choosing capacity by training error | always picks the most flexible model; disappointing in production | choose on validation or cross-validation (01.03)
| Collecting data to fix bias | months of labelling, validation error unchanged | check the learning curve first
| Adding capacity to fix variance | training error drops, gap widens | regularise, simplify, or gather data
| Tuning on one small validation set | best setting changes when the seed changes | k-fold cross-validation, report spread
| Treating "low bias" as the goal | zero-bias model with huge variance | minimise bias² + variance together

## Going deeper

**Proof sketch of the decomposition.** Fix $x$ and drop it from the notation. Write the error as $y - \hat f_S = (f - \bar f) + (\bar f - \hat f_S) + \varepsilon$, using $y = f + \varepsilon$. Square and take expectations. The three squared terms give bias², variance and $\sigma^2$. Every cross term vanishes: $\mathbb{E}_S[\bar f - \hat f_S] = 0$ by the definition of $\bar f$, and $\varepsilon$ has mean 0 and is independent of the training set. That is the whole proof; it relies only on squared loss and independent noise.

**Approximation and estimation.** Lesson 01.01 split excess risk into approximation error (the best member of the class is wrong) and estimation error (a finite sample picks a worse member). Bias plays the first role and variance the second.

**Why the U is not universal: double descent.** The classical U assumes capacity stays below the size of the training set. Belkin et al. (2019) showed that if you keep going, test error can peak at the **interpolation threshold** (just enough parameters to fit the training data exactly) and then fall again as the model becomes heavily over-parameterised.

~~~dg-fig 01.02-double-descent.svg
Least squares with 40 training points, using the first p of 2000 features. Test error follows a shallow U, explodes near p = 40 where the fit is forced through every noisy point, then descends again once there are many more parameters than points, even though training error stays at zero.
~~~

Why does it come down again? Past the threshold, infinitely many weight vectors fit the training data exactly. The solver returns the one with the smallest norm, as gradient descent started from zero does for linear models. Spreading the fit over many small weights is **implicit regularisation**. Right at the threshold only one interpolating solution exists and it needs huge weights: the spike. In this toy the second descent returns to about the classical best; in large neural networks it often goes lower.

Zhang et al. (2017) showed that standard image networks can perfectly fit randomly shuffled labels, so parameter counts alone cannot explain why they generalise on real labels. Effective capacity depends on the data, optimiser and training time too.

For the classical models in this course (linear models, trees, k-NN, penalised kernels), the U is the right working model. Double descent matters most in the Deep Learning course.

## Key terms

| Term | Meaning
| --- | --- |
| Generalisation | Performing well on new data from the same distribution as the training data
| Bias | How far the average model (over many training sets) is from the truth
| Variance | How much a fitted model changes from one training set to another
| Noise / Bayes error | The part of the label the inputs cannot predict; a floor for every model
| Capacity | How many different shapes a model class can take (degree, depth, 1/k, parameters)
| Underfitting | High training and validation error, close together: bias dominates
| Overfitting | Low training error, much higher validation error: variance dominates
| Learning curve | Training and validation error plotted against training-set size
| Double descent | Test error falling again after the interpolation threshold in over-parameterised models

<div class="take">
<div class="lab">Key takeaway</div>

test error is noise plus bias squared plus variance. Diagnose which one dominates from the gap between training and validation error (and from learning curves), then add capacity or features for bias, and add data, regularisation or averaging for variance.

</div>
