---
id: "01.01"
course: "machine-learning"
module: "01"
slug: "the-learning-problem"
title: "The learning problem"
keyIdeas: ["Supervised learning assumes examples $(x, y)$ are drawn from an unknown distribution $\\mathcal{D}$; the goal is a function $h$ that predicts $y$ well on *new* draws, not on the ones you already have. A model that memorises the training set can score perfectly there and still be useless.", "You choose a **hypothesis class** $\\mathcal{H}$ (lines, trees, networks) and a **loss** $\\ell(h(x), y)$; together they define what \"good\" means and which functions are even reachable.", "**Empirical risk minimisation** picks $\\hat h = \\arg\\min_{h \\in \\mathcal{H}} \\frac1n \\sum_i \\ell(h(x_i), y_i)$, a computable stand-in for the true risk $R(h) = \\mathbb{E}_{\\mathcal{D}}[\\ell(h(x), y)]$. Because $\\hat h$ is chosen on the sample, its training error is optimistic; the difference is the **generalisation gap**.", "Labels are noisy, so even the best possible predictor has some error (the **Bayes error**); no model can get below it, and the i.i.d. assumption (future data looks like training data) is what most real failures break.", "The loss you train on is often a smooth surrogate (log loss, hinge) for the one you care about (0-1 error), because the true one has no useful gradient. Report the metric you care about, on held-out data.", "Every later lesson changes one of three knobs: the class, the loss, or how the minimisation is constrained."]
question: "A team trains a fraud model by minimising log loss, reports 0.3% training error, and ships. On live traffic the model flags almost nothing. Name two distinct places in the learning setup where this could have gone wrong, and how you would tell them apart."
answer: "Start by placing each suspect in the setup of this lesson: the class, the loss and metric, the sample, and the distribution. One suspect is the **generalisation gap**: the class is flexible enough to memorise the training set, so low empirical risk says little about true risk. You would see it by evaluating on a held-out split drawn from the same period as the training data: if held-out error is far above the training error, the model overfit, and a simpler class, more data or regularisation is the fix. A second, distinct suspect is a violation of the **i.i.d. assumption**: live traffic comes from a different distribution than the training data (new fraud patterns, a different country mix, a feature computed differently in production than in the training pipeline). Then held-out error on old data can look perfectly fine while live performance collapses; comparing feature distributions between training and live data, or evaluating on a later time slice, exposes it. So the test that separates them is: good held-out score but bad live score points to distribution shift; bad held-out score points to overfitting. A third suspect is a **loss/metric mismatch with a missing baseline**: if about 0.3% of transactions are fraud, the trivial rule \"never flag fraud\" already has 0.3% error, so the reported number may be nothing more than the base rate. Log loss can be decreasing while the model still never crosses the decision threshold. Comparing against the constant baseline and reporting precision and recall on the fraud class (not overall error) would reveal that the model has learned nothing useful."
sources: [{"title": "Understanding Machine Learning (Shalev-Shwartz and Ben-David)", "url": "https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/", "note": "Free PDF. Chapters 2-3 define the learning model, ERM and PAC learning precisely."}, {"title": "Cornell CS4780: Machine learning setup", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote01_MLsetup.html", "note": "Short notes on the D, H, loss framing with good intuition for why memorisation fails."}, {"title": "Cornell CS4780: Empirical risk minimisation", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html", "note": "Surrogate losses (hinge, log, exponential) side by side and how they bound 0-1 loss."}]
practice: ["linear-regression", "loss-functions", "log-loss-binary-cross-entropy", "regression-metrics"]
tags: ["empirical risk", "loss functions"]
tools: ["NumPy"]
---

## What you will learn

- What "learning from data" means precisely: inputs, labels, a training set, and the unknown process behind them.
- Why the goal is to do well on *new* examples, and why a model that memorises its training data can be useless.
- The three choices behind every supervised model: a hypothesis class, a loss, and empirical risk minimisation.
- Why the loss you train on is often not the number you report.
- How to see, in a few lines of code, the gap between training error and error on fresh data.

*Before you start:* this is the first lesson of the course. You need averages, the idea of a function, and enough Python to read short NumPy code. Terms like "distribution" are explained as they appear; the ML Maths course covers them in depth.

## Where this module starts

Machine learning methods look very different from the outside: straight lines, decision trees, neural networks with millions of weights. Underneath, almost all supervised methods solve the same problem with the same recipe. This lesson sets that problem up once, carefully, so later lessons can say "same recipe, one ingredient changed".

The rest of the module follows from it: why good training fit can still fail on new data (01.02), how to measure performance honestly (01.03), how to stop models fitting noise (01.04), and how to check a model against simple baselines (01.05).

## The idea in one picture

A food-delivery app wants to tell customers how long their order will take. It has records of past deliveries: the distance in kilometres and the time it actually took. Here are eight of them.

| Distance (km) | 1.2 | 2.0 | 2.7 | 3.5 | 4.1 | 5.0 | 5.8 | 6.6
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time (minutes) | 13 | 18 | 21 | 19 | 27 | 28 | 33 | 34

Longer trips take longer, but not exactly: traffic or a slow restaurant add minutes no distance can explain. Pretend for a moment we could see the process that generates deliveries:

~~~dg-fig 01.01-data.svg
The shaded band is where real delivery times fall for each distance, and the dashed line is their average. Your eight training deliveries are one random handful from this band. In practice you never see the band, only the dots.
~~~

The job is to produce a rule that, given a new distance, predicts the time well. Here are three candidate rules, each scored by average squared error on the training deliveries and on a very large number of fresh deliveries from the same band.

~~~dg-fig 01.01-three-fits.svg
Three ways to predict delivery time. The flat average ignores distance. The straight line follows the trend. The wiggly curve passes through every training point exactly, so its training error is zero, yet it is the worst of the three on new deliveries.
~~~

| Rule | Error on the 8 training deliveries | Error on new deliveries
| --- | --- | --- |
| Always predict the average (24.1 min) | 49.6 | 39.0
| Straight line: $9.1 + 3.89 \times \text{km}$ | 3.0 | 9.8
| Wiggly curve through every point | 0.0 | 45.5

The straight line wins where it matters, on new deliveries. The wiggly curve "wins" only on data it has already seen, which is worthless: no customer asks about a delivery that already happened. That observation is the heart of machine learning.

Note also that no rule gets below about 9 on new data. Times spread about 3 minutes around the average, so even the perfect rule is typically off by 3 minutes, and $3^2 = 9$. That floor comes from noise in the world, not from the model.

## Step 1: name the ingredients

Supervised learning has four ingredients. Using the delivery example:

- An **input** $x$ (also called features). Here the distance. Usually a list of $d$ numbers, a vector $x \in \mathbb{R}^d$: distance, hour of day, and so on.
- A **label** $y$, the thing to predict. A real number (minutes) makes it a **regression** problem; one of $K$ categories (spam or not) makes it **classification**.
- A **data distribution** $\mathcal{D}$: the unknown process that produces pairs $(x, y)$. It is the shaded band in the first figure: how often each distance occurs and how times spread at each distance. You never see it.
- A **training set** $S = \{(x_1, y_1), \dots, (x_n, y_n)\}$ of $n$ examples drawn from $\mathcal{D}$. Here $n = 8$.

Learning outputs a function $h$ mapping an input to a prediction, $\hat y = h(x)$ ("y-hat", the predicted y). The goal is for $h$ to do well on *future* draws from $\mathcal{D}$. That word, future, separates machine learning from curve fitting: a lookup table that stores $S$ fits it perfectly and says nothing about the next example.

Two assumptions hide in this setup, and most real failures break one of them:

- **i.i.d. sampling** (independent and identically distributed): training and future examples come from the *same* $\mathcal{D}$, each drawn independently. If the app launches in a city with different traffic, the training data stops being a fair preview. Drift, leakage and selection bias all violate this.
- **Labels are noisy**: $y$ is not a fixed function of $x$; two 5 km deliveries can take 25 and 31 minutes. Even the best possible predictor makes mistakes, and its error, the **Bayes error**, is the floor of about 9 you saw above.

## Step 2: choose a hypothesis class

You cannot search over all possible functions; there are far too many, and most are as silly as the wiggly curve. So you first decide which functions to consider. That set is the **hypothesis class**, written $\mathcal{H}$, and each member is a **hypothesis** (one candidate rule).

- Constants, $h(x) = c$: one number to choose (the "average" rule lives here).
- Lines, $h(x) = w x + b$: slope $w$ and intercept $b$. With $d$ features, $h(x) = w^\top x + b$ has $d + 1$ numbers.
- Depth-5 decision trees, or a network such as ResNet-50 with about 25 million numbers.

The numbers that pick out one member are its **parameters**. Choosing the class is the biggest modelling decision. Learning can only return something inside it: allow only lines and you get the best line, even if the truth is curved. And a very flexible class (8 parameters for 8 points) can match the training data perfectly, noise included, as the wiggly curve did.

## Step 3: choose a loss

Next, decide how to score one prediction. A **loss function** $\ell(\hat y, y)$ takes the prediction and the true label and returns 0 for perfect, more for worse.

~~~dg-fig 01.01-losses.svg
Left: squared loss grows fast, so big misses dominate; absolute loss grows steadily; Huber is squared near zero and absolute beyond. Right: the 0-1 classification loss is a step; the surrogates used in training sit on or above it.
~~~

| Task | Loss | Formula
| --- | --- | --- |
| Regression | squared error | $(\hat y - y)^2$
| Regression, robust to outliers | absolute / Huber | $|\hat y - y|$; Huber is quadratic near 0, linear far out
| Classification | 0-1 | $1$ if $\hat y \ne y$, else $0$
| Binary, probabilistic | log loss | $-y\log p - (1-y)\log(1-p)$, with $y \in \{0, 1\}$ and $p$ the predicted probability of class 1
| Binary, margin | hinge | $\max(0, 1 - y\,f(x))$, with $y \in \{-1, 1\}$ and $f(x)$ a score

The loss encodes what mistakes cost, so it is a real decision. Squared error treats one delivery 10 minutes late as badly as a hundred that are 1 minute late ($10^2 = 100 \times 1^2$); absolute error does not. If a missed fraud case costs ten times a false alarm, the loss should say so.

The loss also decides what the best guess is: under squared loss the best constant is the **mean**, under absolute loss the **median**. That is why the flat rule above predicts the average, 24.1 minutes.

## Step 4: minimise the average loss on the training set

What you really want is a hypothesis with small **true risk**, its average loss over all future examples:

$$R(h) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell(h(x), y)\big].$$

Here $\mathbb{E}$ means "long-run average" and $(x, y) \sim \mathcal{D}$ means "over pairs drawn from the distribution". The "error on new deliveries" column was exactly this. You cannot compute it, because $\mathcal{D}$ is unknown. You can compute the average loss on your sample, the **empirical risk** (training error):

$$\hat R_S(h) = \frac{1}{n}\sum_{i=1}^{n} \ell(h(x_i), y_i).$$

In words: predict each training example, score it, average. **Empirical risk minimisation (ERM)** picks the hypothesis in the class with the smallest empirical risk:

$$\hat h = \arg\min_{h\in\mathcal{H}} \hat R_S(h).$$

($\arg\min$ means "the $h$ that achieves the minimum", not the minimum value itself.)

<div class="ex">
<div class="lab">Worked example</div>

**ERM by hand on four deliveries**

Training data: distances $x = (1, 2, 3, 4)$ km, times $y = (14, 17, 21, 24)$ minutes. The class has just three rules: $h_1(x) = 19$ (the average), $h_2(x) = 10 + 3.5x$, and $h_3(x) = 12 + 3x$. The loss is squared error.

$h_1$ predicts $(19, 19, 19, 19)$. Errors $y - \hat y = (-5, -2, 2, 5)$, squared $(25, 4, 4, 25)$, sum 58, so $\hat R_S(h_1) = 58/4 = 14.5$.

$h_2$ predicts $(13.5, 17, 20.5, 24)$. Errors $(0.5, 0, 0.5, 0)$, squared $(0.25, 0, 0.25, 0)$, sum 0.5, so $\hat R_S(h_2) = 0.5/4 = 0.125$.

$h_3$ predicts $(15, 18, 21, 24)$. Errors $(-1, -1, 0, 0)$, squared $(1, 1, 0, 0)$, sum 2, so $\hat R_S(h_3) = 2/4 = 0.5$.

ERM picks the smallest: $\hat h = h_2$.

Now two new deliveries arrive: $x = (2.5, 5)$, $y = (18.5, 28)$. $h_2$ predicts $(18.75, 27.5)$: squared errors $0.0625$ and $0.25$, average $0.156$. $h_3$ predicts $(19.5, 27)$: squared errors 1 and 1, average 1.0. Here the training winner also wins on new data.

A lookup-table "memoriser" stores the four times exactly (training error 0) and answers 19 for any unseen distance. On the new deliveries its errors are $-0.5$ and $9$, squared $0.25$ and $81$, average $40.6$: perfect on the past, terrible on the future.

(Searching *all* lines with least squares, the method of lesson 02.01, gives $10.5 + 3.4x$, training error 0.05.)

</div>

~~~dg-flow The learning loop: you choose the class and the loss; the data picks the function inside the class; new data tells you whether it worked.
Sample S from D -(choose)-> Hypothesis class H -(choose)-> Loss -(minimise on S)-> *Learned h -(evaluate on new data)-> Test risk
~~~

## Why training error is an optimistic estimate

For a *fixed* rule, chosen before seeing the data, training error is an honest estimate of true risk: right on average, or **unbiased**. But $\hat h$ is not fixed. You chose it *because* it looked best on this particular sample.

~~~dg-fig 01.01-erm.svg
Model ŷ = w·x, data from y = 2x plus noise. Each thin curve is the training error for one random sample of four points; ERM picks its lowest point. The thick curve is the true risk. Each sample's minimum lands at a slightly different slope, and usually below the true curve.
~~~

ERM bends toward each sample's quirks, so the chosen rule's training error is biased low. The difference

$$R(\hat h) - \hat R_S(\hat h)$$

is the **generalisation gap**. It is small for simple classes and large samples, and large when the class can chase noise.

~~~dg-fig 01.01-optimism.svg
Each dot is one training set of ten deliveries. The straight line is near the diagonal; the 6-parameter curve sits far above it, looking great in training and doing badly on new data. Dots cannot fall below the noise floor of 9 on new data.
~~~

Taking the median dot of each cloud: the line scores about 7 in training and 10 on new data; the flexible curve about 3 and 29. The more freedom the class has, the less its training error can be trusted. Lesson 01.02 studies what controls this gap; 01.03 measures true risk honestly.

## The learning problem in code

Real ERM uses a formula (least squares, 02.01) or an optimiser (gradient descent), not a grid, but it minimises the same average. By brute force:

```python
import numpy as np
# ERM for the class h(x) = w * x under squared loss, by trying 401 slopes
x = np.array([1., 2., 3., 4.]); y = np.array([2.1, 3.9, 6.2, 7.8])
slopes = np.linspace(0, 4, 401)
risk = [np.mean((w * x - y) ** 2) for w in slopes]   # empirical risk of each slope
w_hat = slopes[int(np.argmin(risk))]
print(w_hat, min(risk))                               # 1.99  0.02425
```

Now with scikit-learn, on deliveries simulated from a known $\mathcal{D}$ so a huge fresh sample can estimate true risk. An unlimited decision tree plays the memoriser: each training point can get its own leaf.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

def sample_deliveries(n):
    # draw n deliveries from the (normally unknown) distribution D
    km = rng.uniform(1, 7, size=(n, 1))                  # inputs x: shape (n, 1)
    minutes = 10 + 3.5 * km[:, 0] + rng.normal(0, 3, n)  # labels y: trend + noise
    return km, minutes

X_train, y_train = sample_deliveries(40)       # the training set S you get to see
X_new, y_new = sample_deliveries(100_000)      # stands in for "the future"

for name, model in [("line", LinearRegression()),
                    ("memoriser", DecisionTreeRegressor(random_state=0))]:
    model.fit(X_train, y_train)                # ERM inside each class
    train = mean_squared_error(y_train, model.predict(X_train))
    new = mean_squared_error(y_new, model.predict(X_new))
    print(f"{name:9s}  train MSE {train:5.2f}   new-data MSE {new:5.2f}")
# line       train MSE 10.37   new-data MSE  9.93
# memoriser  train MSE  0.00   new-data MSE 21.07
```

The line's two numbers are close to each other and to the noise floor of 9. The memoriser scores a perfect 0 in training and more than twice the floor on new data, because each prediction copies the noise of one nearby training delivery. Real projects have no fresh 100,000; you hold data back instead (01.03).

## Surrogate losses: training on one thing, reporting another

For classification you care about the error rate, the 0-1 loss. But it is a flat step (losses figure): nudge the weights and the error rate almost never changes, so its gradient is zero almost everywhere and gradient-based training gets no signal. Exactly minimising it over linear classifiers is also NP-hard in general.

So you minimise a **surrogate loss**: a convex (bowl-shaped) function on or above the 0-1 loss. Log loss gives logistic regression; hinge gives the support vector machine (module 02). You still **report** the metric you care about (accuracy, F1, money saved) on held-out data.

## Reading the result: is my model learning or memorising?

Put three numbers side by side every time: **training error** (proves little alone), **held-out error** on data never used in fitting (your estimate of true risk), and a **baseline's error**, such as always predicting the average or the most common class (01.05).

Low training error with much higher held-out error means memorising. Both high and close means the class is too simple, the features are weak, or you are near the noise floor. Both low and close is the goal. Fine held-out error but bad live results points to a broken i.i.d. assumption.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Judging a model by training error | great numbers in development, poor results in use | always report error on held-out data (01.03)
| Class too flexible for the data size | training error near 0, held-out error high | simpler class, more data, regularisation (01.04)
| Class too simple | training and held-out error both high and close | richer class or better features
| Future differs from the past (not i.i.d.) | held-out error fine, live error bad | test on later time periods, monitor input distributions
| Loss does not match the real cost | model "optimal" but business unhappy | weight errors by cost, change loss or threshold
| Accuracy on rare classes | 99.7% accuracy on 0.3% fraud by predicting "no fraud" always | compare with the baseline, use precision and recall

<div class="ex">
<div class="lab">Worked example</div>

**three classes for spam (illustrative numbers)**

100 emails, 20 spam. $\mathcal{H}_0$ = constant predictors, $\mathcal{H}_1$ = linear models on word counts, $\mathcal{H}_2$ = a lookup table keyed on exact email text. ERM training 0-1 errors: 0.20 ("never spam"), about 0.04, and 0.00.

On new mail $\mathcal{H}_2$ has seen none of the texts, falls back to "not spam", and its true risk is about 0.20 again. The class decides how far the training number can be trusted.

</div>

## Going deeper

**The best possible predictor.** If you knew $\mathcal{D}$, the best rule would depend only on the loss. For squared loss it is the conditional mean, $h^*(x) = \mathbb{E}[y \mid x]$ (the dashed line in the first figure). For absolute loss it is the conditional median. For 0-1 loss it is the most probable class given $x$, the **Bayes classifier**. Its risk $R^* = R(h^*)$ is the Bayes error.

**Two sources of excess error.** How far the learned rule falls short of perfect splits into two parts:

$$R(\hat h) - R^* = \underbrace{\Big(R(\hat h) - \min_{h\in\mathcal{H}} R(h)\Big)}_{\text{estimation error}} + \underbrace{\Big(\min_{h\in\mathcal{H}} R(h) - R^*\Big)}_{\text{approximation error}}.$$

Approximation error is the price of the class; estimation error is the price of a finite sample. A richer class lowers the first and raises the second: the general form of the bias-variance trade-off (01.02). A third term, optimisation error, appears when the optimiser misses the exact ERM solution.

**How big can the gap be?** For one fixed $h$ with loss between 0 and 1, Hoeffding's inequality says that with probability at least $1 - \delta$,

$$|\hat R_S(h) - R(h)| \le \sqrt{\frac{\ln(2/\delta)}{2n}}.$$

With $n = 1000$ and $\delta = 0.05$ that is $\sqrt{\ln 40 / 2000} \approx 0.043$. For a finite class, a union bound makes it hold for every member at once if $\ln(2/\delta)$ becomes $\ln|\mathcal{H}| + \ln(2/\delta)$; for $|\mathcal{H}| = 1000$ that is about 0.073. Calling this bound $\varepsilon$, ERM then satisfies $R(\hat h) \le \min_{h\in\mathcal{H}} R(h) + 2\varepsilon$. The gap shrinks like $1/\sqrt{n}$ and grows only with the log of the class size. Infinite classes need finer size measures (VC dimension, Rademacher complexity); this is **PAC learning**, developed in the Shalev-Shwartz and Ben-David book.

**Surrogates as bounds.** With labels in $\{-1, 1\}$ and score $f(x)$, the **margin** $y f(x)$ is positive on the correct side, and 0-1 loss is 1 when it is at most 0. Log loss, written $\log_2(1 + e^{-y f(x)})$, equals 1 at margin 0 and lies above the step everywhere, as does hinge. Lowering the surrogate lowers an upper bound on the error rate.

**Where the course goes.** Every method is this template, often with a penalty added: linear regression is squared loss over linear functions, logistic regression is log loss over the same class, an SVM is hinge loss plus a norm penalty. k-NN and naive Bayes reach it from other directions (module 03).

## Key terms

| Term | Meaning
| --- | --- |
| Input / features $x$ | The information the model gets, usually a vector of numbers
| Label $y$ | The answer to predict: a number (regression) or a category (classification)
| Data distribution $\mathcal{D}$ | The unknown process that generates input-label pairs, past and future
| Hypothesis class $\mathcal{H}$ | The set of functions the learner is allowed to choose from
| Loss $\ell(\hat y, y)$ | A score for one prediction: 0 if perfect, larger if worse
| True risk $R(h)$ | Average loss on new data from $\mathcal{D}$; what you want small but cannot compute
| Empirical risk $\hat R_S(h)$ | Average loss on the training set; what you can compute
| ERM | Choosing the hypothesis with the lowest empirical risk
| Generalisation gap | True risk minus training risk of the chosen model
| Bayes error | The error of the best possible predictor; the noise floor no model beats

<div class="take">
<div class="lab">Key takeaway</div>

learning means choosing a class of functions and a loss, then picking the function with the lowest average loss on a training sample as a stand-in for its average loss on future data. Because the choice is made on that sample, training error is optimistic, so always judge a model on data it has not seen.

</div>
