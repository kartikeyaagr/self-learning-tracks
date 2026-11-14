---
id: "02.03"
course: "machine-learning"
module: "02"
slug: "logistic-regression"
title: "Logistic regression"
keyIdeas: ["Logistic regression computes a linear score $z = w^\\top x + b$ and turns it into a probability with the sigmoid $p = 1/(1 + e^{-z})$, so outputs always lie between 0 and 1.", "What is linear is the **log-odds** $\\log\\frac{p}{1-p}$: raising feature $x_j$ by one multiplies the odds by $e^{w_j}$ (with $w_j = 0.7$ the odds roughly double).", "It is fitted by minimising the **log loss** (binary cross-entropy), the negative log-likelihood of the labels. Confident mistakes cost a lot; compute it from logits for numerical stability.", "The gradient is $\\frac1n X^\\top(p - y)$, error times input, and the loss is convex, so gradient descent, Newton (IRLS) or L-BFGS all reach the single best answer.", "The decision boundary $w^\\top x + b = \\log\\frac{t}{1-t}$ is a hyperplane; changing the threshold $t$ slides it without rotating it, trading false alarms for misses.", "On perfectly separable data the unpenalised weights grow without bound; an L2 penalty (on by default in scikit-learn, `C=1.0`) keeps them finite."]
question: "A bank fits logistic regression to predict loan default (1 = default) from two standardised features and gets weights w = (0.9 for debt-to-income, -0.4 for years at current job), bias b = -2.0. (a) For an applicant with debt-to-income +1.5 and job tenure -0.5 (both in standard deviations), what are the score, the odds and the probability of default? (b) Interpret the 0.9 weight in plain words. (c) The bank only wants to refuse applicants whose default probability is at least 30%. What score cut-off does that correspond to, and does this applicant get refused? (d) A colleague refits without any penalty after adding 200 one-hot postcode columns, gets a convergence warning and several coefficients above 30. What is the likely cause and what would you do?"
answer: "(a) Score $z = 0.9 \\times 1.5 + (-0.4) \\times (-0.5) - 2.0 = 1.35 + 0.20 - 2.0 = -0.45$. Odds $= e^{-0.45} \\approx 0.64$. Probability $= 1/(1 + e^{0.45}) = 1/(1 + 1.57) \\approx 0.39$. (b) Each additional standard deviation of debt-to-income adds 0.9 to the log-odds of default, i.e. multiplies the odds by $e^{0.9} \\approx 2.46$, holding job tenure fixed. It is an association, not necessarily a causal effect, and it multiplies the odds rather than adding a fixed number of percentage points. (c) $p \\ge 0.3$ is equivalent to $z \\ge \\log(0.3/0.7) = \\log 0.4286 \\approx -0.85$. The applicant's score $-0.45$ is above $-0.85$ (probability 0.39 > 0.30), so they are refused. Changing the threshold from 0.5 to 0.3 only shifts the boundary parallel to itself; the weights are unchanged. (d) With 200 sparse postcode columns, some postcodes probably contain only defaulters or only non-defaulters, so the data are (quasi-)separable along those columns. Without a penalty, the loss keeps decreasing as those weights grow, so they diverge, the optimiser hits its iteration cap, and the corresponding odds ratios are meaningless while predictions for those postcodes become overconfident 0s and 1s. Fix: keep an L2 penalty (scikit-learn's default `C=1.0`, tuned by cross-validation), or merge rare postcodes into an 'other' level or target-encode them; also standardise numeric features so the solver converges, and check calibration on held-out data."
sources: [{"title": "Cornell CS4780 lecture notes: Logistic Regression", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote06.html", "note": "Logistic regression derived as MLE and MAP, with the gradient."}, {"title": "Stanford CS229 main notes", "url": "https://cs229.stanford.edu/main_notes.pdf", "note": "Part on classification and logistic regression, plus Newton's method and the generalised linear model view."}, {"title": "scikit-learn: LogisticRegression", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html", "note": "What C means, which penalty is on by default, and which solver handles which penalty."}]
practice: ["logistic-regression", "log-loss-binary-cross-entropy", "newton-s-method", "sigmoid-stable", "basic-autograd"]
tags: ["log-odds", "cross-entropy"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- Why a straight line fails on yes/no labels, and how the **sigmoid** turns any score into a probability.
- What **log-odds** are, and how to read a weight as "this feature multiplies the odds by $e^{w}$".
- Where the **log loss** comes from, and why it punishes confident mistakes so hard.
- How training works: the gradient is "error times input" and the loss is convex.
- How a probability becomes a decision: the **decision boundary** and the threshold.
- Why perfectly separable data sends the weights to infinity unless you regularise.

*Before you start:* linear regression (02.01) and the L2 penalty (02.02). You also need $e^x$, the natural log and the idea of a derivative.

Lesson 02.01 predicted a number (rent). Now the target is yes/no: will this student pass, is this email spam? Logistic regression keeps the same linear core and changes two things: what the output means (a probability) and the loss used to fit it.

## The idea in one picture

A tutor records how many hours twelve students studied for an exam and whether each one passed (1) or failed (0):

| Hours studied | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 | 5.5 | 6
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Passed? | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1

More study helps, but the middle is messy: one student passed after 2.5 hours, another failed after 4. We want a **probability of passing** for any number of hours.

The first idea is to reuse linear regression on the 0/1 labels. The dashed line below is that least-squares fit, and it has two visible problems.

~~~dg-fig 02.03-line-vs-sigmoid.svg
Left: the straight line predicts below 0 and above 1 (shaded zones); the sigmoid stays inside. Right: one extra student who studied 14 hours and passed tilts the line, moving its 50% point from 3.25 to 3.64 hours; the sigmoid barely moves.
~~~

- **Impossible outputs.** The line predicts $-0.08$ at half an hour and $1.08$ at six hours. A probability of $-8\%$ means nothing.
- **Distortion by easy points.** A student who studied 14 hours and passed is the least surprising data point imaginable, yet it drags the line and shifts where it crosses 0.5. The model gets worse from an *easy* example.

Logistic regression fits the solid S-shaped curve instead. It never leaves (0, 1), and it flattens at the ends so far-away easy points have almost no pull. For these twelve students the fitted curve is

$$p(\text{pass}) = \sigma(1.425 \times \text{hours} - 4.632),$$

where $\sigma$ is the sigmoid function defined below. It crosses 50% at $4.632 / 1.425 = 3.25$ hours, gives 74% at 4 hours and 92% at 5 hours.

## Step 1: a linear score, squashed into (0, 1)

Start exactly as in linear regression. Combine the features into one number, the **score** (also called the **logit**):

$$z = w^\top x + b = w_1 x_1 + \dots + w_d x_d + b.$$

Here $x$ holds the features of one example, $w$ holds one weight per feature, and $b$ is the bias. The score can be any real number, from very negative to very positive. A probability must lie between 0 and 1, so we pass the score through the **sigmoid** (or logistic) function:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Check with numbers: at $z = 0$, $p = 1/(1+1) = 0.5$. At $z = 3$, $e^{-3} \approx 0.05$ and $p \approx 0.95$. At $z = -3$, $e^{3} \approx 20$ and $p \approx 0.05$. Large positive scores mean "almost surely class 1"; zero means "can't tell".

~~~dg-fig 02.03-sigmoid.svg
The sigmoid turns a score z into a probability. Each marked point shows the score, the probability and the odds. Note how quickly it saturates: a score of 4.6 already means 99%.
~~~

Two useful facts: $\sigma(-z) = 1 - \sigma(z)$ (flipping the score swaps the classes), and the slope is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, largest (0.25) at $z = 0$ and almost zero in the flat tails. That slope matters when we choose the loss.

## Step 2: what is linear is the log-odds

The sigmoid can be inverted. If $p = \sigma(z)$, then

$$z = \log\frac{p}{1 - p}.$$

The ratio $p/(1 - p)$ is the **odds**: a 75% chance is odds of 3 (three to one in favour). Its logarithm is the **log-odds**. So logistic regression claims that the log-odds are linear in the features: each feature adds a fixed amount to them.

| Probability p | Odds p/(1-p) | Logit z
| --- | --- | --- |
| 0.50 | 1 | 0
| 0.73 | 2.7 | 1
| 0.88 | 7.4 | 2
| 0.95 | 19 | 2.94
| 0.99 | 99 | 4.6

The table is worth remembering: logits of a few units already mean near-certainty, which is why weights do not need to be large to produce confident predictions.

Adding to a logarithm multiplies the thing inside it. So a one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$, the **odds ratio**. With $w_j = 0.7$ the odds roughly double ($e^{0.7} \approx 2.01$). In the study example $e^{1.425} \approx 4.16$: each extra hour multiplies the odds of passing by about four. It multiplies the *odds*, not the probability.

<div class="ex">
<div class="lab">Worked example</div>

**reading the study-hours model by hand**

The fitted model is $z = 1.425 \times \text{hours} - 4.632$. Take a student who studied 4 hours.

Score: $z = 1.425 \times 4 - 4.632 = 5.700 - 4.632 = 1.068$.

Probability: $e^{-1.068} = 0.344$, so $p = 1 / (1 + 0.344) = 0.744$. The model gives a 74% chance of passing.

Odds: $0.744 / 0.256 = 2.91$, which is also $e^{1.068}$. Roughly three to one in favour.

One more hour (5 hours): $z = 1.068 + 1.425 = 2.493$, odds $= e^{2.493} = 12.1 = 2.91 \times 4.16$. Probability $12.1 / 13.1 = 0.924$. The odds grew by $e^{w} = 4.16$, but the probability by only 18 points: it is near the top of the S.

</div>

The same caveats as in 02.01 apply: with correlated features or a missing cause, an odds ratio is not "the effect" of that feature.

## Step 3: the loss, from maximum likelihood

Treat each label as a biased coin flip: example $i$ is a 1 with probability $p_i = \sigma(w^\top x_i + b)$. The **likelihood** is the probability the model gives to the labels we actually saw: $p_i$ if $y_i = 1$ and $1 - p_i$ if $y_i = 0$, multiplied over all examples. Take logs (products become sums), flip the sign (so we minimise) and average. The result is the **log loss**, also called **binary cross-entropy**:

$$L(w, b) = -\frac{1}{n}\sum_{i=1}^{n} \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big].$$

Here $n$ is the number of examples and $\log$ is natural. Only one term is active per example: $-\log p_i$ if $y_i = 1$, $-\log(1 - p_i)$ if $y_i = 0$.

~~~dg-fig 02.03-logloss.svg
The cost of one example as a function of the predicted probability. For a true 1 it rises without limit as p goes to 0. Squared error (dashed) never exceeds 1, so it barely notices a confident mistake.
~~~

For $y = 1$: predicting 0.9 costs $-\ln 0.9 = 0.11$, 0.5 costs $0.69$, 0.1 costs $2.30$, and 0.001 costs $6.9$. Confidently wrong is very expensive. A model that always says 0.5 scores $\ln 2 = 0.693$; a useful model must beat that.

<div class="ex">
<div class="lab">Worked example</div>

**log loss for three students**

Using the fitted model: the 1-hour student failed ($y = 0$), with $p = 0.039$. Cost $-\ln(1 - 0.039) = -\ln 0.961 = 0.040$.

The 5-hour student passed ($y = 1$), with $p = 0.924$. Cost $-\ln 0.924 = 0.079$.

The 4-hour student failed ($y = 0$) although $p = 0.744$. Cost $-\ln(1 - 0.744) = -\ln 0.256 = 1.363$.

Average: $(0.040 + 0.079 + 1.363)/3 = 0.494$. The one surprising student supplies over 90% of it. Over all twelve students the average is 0.365.

</div>

**The margin form.** If labels are coded $y \in \{-1, +1\}$ instead of $\{0, 1\}$, the same loss can be written as

$$L = \frac{1}{n}\sum_i \log\big(1 + e^{-y_i z_i}\big),$$

a function of the **margin** $y_i z_i$ only (positive when the score has the right sign). A margin of $+5$ costs about 0.007; a margin of $-5$ costs about 5. Lesson 02.05 compares this curve with the SVM's hinge loss.

**Why not squared error on $\sigma(z)$?** It is not convex in $w$, and its gradient contains the factor $\sigma'(z)$, which vanishes in the flat tails: a confidently wrong model barely gets pushed, exactly when it most needs to move. The log loss cancels that factor.

## Step 4: fitting it

There is no closed-form answer, so we minimise numerically. The gradient is strikingly simple. Writing $X$ for the $n \times d$ matrix of features (one row per example), $p$ for the vector of predicted probabilities and $y$ for the labels:

$$\nabla_w L = \frac{1}{n} X^\top (p - y), \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_i (p_i - y_i).$$

Prediction error times input, the same shape as in linear regression; confidently wrong examples get a full-sized push ($p - y \approx \pm 1$). The Hessian $\frac1n X^\top S X$, with $S = \mathrm{diag}(p_i(1 - p_i))$, is positive semi-definite, so the loss is **convex**: any local minimum is the global one.

<div class="ex">
<div class="lab">Worked example</div>

**one gradient step by hand**

Weights $w = (2, -1)$, bias $b = -0.5$, one example $x = (1, 0.5)$ with $y = 1$.

Logit $z = 2 \times 1 + (-1) \times 0.5 - 0.5 = 1.0$, so $p = \sigma(1) = 0.731$ and the loss is $-\ln 0.731 = 0.313$.

Error $p - y = -0.269$. Gradient for $w$: $-0.269 \times (1, 0.5) = (-0.269, -0.134)$. Gradient for $b$: $-0.269$.

A step with learning rate 0.5 subtracts half the gradient: $w = (2 + 0.134, -1 + 0.067) = (2.134, -0.933)$ and $b = -0.5 + 0.134 = -0.366$.

New logit $2.134 - 0.467 - 0.366 = 1.30$, so $p = 0.79$. The prediction moved toward the true label, and the loss fell to $-\ln 0.79 = 0.24$.

</div>

**Solvers.** Plain gradient descent works but is slow on badly scaled features. **Newton's method** uses the Hessian; each step is a weighted least-squares solve, hence **iteratively reweighted least squares (IRLS)**, converging in 5 to 10 iterations for moderate $d$. scikit-learn defaults to **L-BFGS**, a quasi-Newton method that approximates curvature from recent gradients, and offers SAGA or liblinear for large sparse problems.

## Logistic regression in code

scikit-learn does the fitting; a from-scratch version in NumPy shows there is no magic.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

hours  = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.])
passed = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
X = hours.reshape(-1, 1)                        # scikit-learn wants 2-D input

model = LogisticRegression(C=np.inf).fit(X, passed)   # C=inf: no penalty
print(model.coef_, model.intercept_)            # [[1.425]] [-4.632]
print(np.exp(model.coef_))                      # [[4.159]]  odds multiplier per hour
print(model.predict_proba([[4.0]]))             # [[0.256 0.744]]  columns: P(fail), P(pass)
p = model.predict_proba(X)[:, 1]
print(log_loss(passed, p))                      # 0.365

default = LogisticRegression().fit(X, passed)   # default: L2 penalty, C=1.0
print(default.coef_, default.intercept_)        # [[1.008]] [-3.276]  shrunk toward 0

# The same model from scratch with gradient descent
def fit_logreg(X, y, lam=0.0, lr=0.5, steps=20000):
    Xb = np.c_[X, np.ones(len(X))]              # last column is the bias
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        grad = Xb.T @ (p - y) / len(y)          # error times input
        grad[:-1] += lam * w[:-1]               # L2 on weights, not on the bias
        w -= lr * grad
    loss = np.mean(np.logaddexp(0, z) - y * z)  # stable log loss from logits
    return w, loss

w, loss = fit_logreg(X, passed)
print(w.round(3), round(loss, 3))               # [ 1.424 -4.63 ] 0.365
```

- **The default is regularised.** `LogisticRegression()` applies L2 with `C=1.0`, where `C` is the *inverse* strength ($C = 1/\lambda$, on a summed rather than averaged loss). Here it shrank the weight from 1.425 to 1.008. `C=np.inf` (older versions: `penalty=None`) gives the raw maximum-likelihood fit.
- **Compute the loss from logits.** `np.logaddexp(0, z) - y * z` is exact for large $|z|$, while `np.log(sigmoid(z))` returns `-inf` once $\sigma(z)$ rounds to 0. PyTorch's `BCEWithLogitsLoss` exists for the same reason.

## Step 5: from probability to decision

To act, pick a **threshold** $t$ and predict class 1 when $p \ge t$. Because the sigmoid is increasing, that is the same as

$$z \ge \log\frac{t}{1 - t}.$$

For $t = 0.5$ the right side is 0, so the **decision boundary** is the set of points where $w^\top x + b = 0$. With two features that is a straight line; in general it is a flat **hyperplane**, and $w$ points perpendicular to it, toward class 1.

~~~dg-fig 02.03-boundary.svg
120 students, two features. The solid line is where p = 0.5; the dashed probability contours run parallel to it, and w points across them toward the passing side.
~~~

Here $w = (1.79, 1.21)$ and $b = -12.07$. The size $\|w\|$ sets how fast probability changes across the boundary: the distance from the boundary to the 0.73 contour ($z = 1$) is $1/\|w\| = 1/2.16 = 0.46$ units. Larger $\|w\|$ means a sharper, more confident model.

~~~dg-flow One prediction: a linear score, squashed to a probability, then cut at a threshold chosen for the costs at hand.
Features x -(w.x + b)-> Logit z -(sigmoid)-> *Probability p -(compare with t)-> Class
~~~

**Changing the threshold** changes only the constant, so the boundary slides parallel to itself and never rotates. In the study-hours model, $t = 0.2$ means "predict pass from 2.28 hours" and $t = 0.8$ means "from 4.22 hours".

~~~dg-fig 02.03-threshold.svg
Left: lines for t = 0.2, 0.5 and 0.8 are parallel shifts. Right: raising t cuts false alarms (failures called passes) and adds misses (passes called failures).
~~~

There is no universally right $t$: if missing a struggling student costs more than tutoring one who would have been fine, move it. Choosing $t$ from costs is covered in module 06; whether $p$ can be trusted as a real probability (calibration) is lesson 03.04.

**Curved boundaries come from features, not from the model.** Adding $x_1^2$, $x_2^2$ and $x_1 x_2$ makes a linear boundary in the expanded space an ellipse in the original one, as in 02.01. Lesson 02.06 does this implicitly with kernels.

## Checking the result

- **Beat the baseline.** Always predicting the class rate gives a log loss equal to the label entropy (0.693 for 50/50). Held-out log loss must be clearly lower.
- **Look beyond accuracy.** With 95% negatives, always saying "no" is 95% accurate and useless. Use the confusion matrix, precision and recall, and ROC curves (module 06).
- **Check the probabilities.** Bin the predictions and compare each bin's average prediction with its actual fraction of 1s (calibration, 03.04).
- **Read coefficients last**, on standardised features, as odds ratios `np.exp(model.coef_)`.
- **Heed a `ConvergenceWarning`**: usually unscaled features or separable data.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Separable data, no penalty | weights grow, probabilities exactly 0 or 1 | keep the L2 penalty, or stop early
| Unscaled features | slow convergence, `max_iter` warnings | `StandardScaler` in a pipeline
| Loss from probabilities | `nan` or `inf` loss | compute from logits
| Rare positive class | high accuracy, few positives predicted | precision/recall, tune $t$, `class_weight='balanced'`
| Curved true boundary | errors cluster in one region | polynomial or interaction features
| Collinear features | huge odds ratios of opposite sign | drop or merge features; L2

**Separable data deserves a closer look.** If a hyperplane separates the classes perfectly, every margin is positive, and scaling $w$ and $b$ up by any factor above 1 makes every $\log(1 + e^{-y_i z_i})$ term smaller. The loss falls toward 0 without reaching it, so no minimiser exists: weights diverge, probabilities saturate at 0 and 1, and the optimiser runs to its iteration limit.

~~~dg-fig 02.03-separable.svg
Ten students who split perfectly: everyone with 2.5 hours or less failed, everyone with 4 or more passed. Without a penalty, gradient descent keeps increasing the weight forever (it is still climbing after 20,000 steps). With a small L2 penalty the weight settles at a finite value.
~~~

With many features, or a rare category containing only one class, this is common. An L2 penalty $\frac{\lambda}{2}\|w\|^2$ grows with the weights, so the optimum becomes finite and unique.

## Going deeper

**A generalised linear model.** Logistic regression is the Bernoulli member of the **generalised linear models**: a linear score, a link function to the target's mean (here the logit) and an exponential-family likelihood. Linear regression is the Gaussian member; Poisson regression uses the log link. All share the gradient $X^\top(\text{mean} - y)$.

**IRLS in one line.** A Newton step is $w \leftarrow w - H^{-1}\nabla L$. Substituting the gradient and Hessian above gives $w_{\text{new}} = (X^\top S X)^{-1} X^\top S\, u$, where $u = Xw + S^{-1}(y - p)$ is a "working response". That is weighted least squares with weights $p_i(1 - p_i)$: examples near the boundary, where the model is unsure, get the most weight. Each step costs $O(nd^2 + d^3)$, so L-BFGS or stochastic methods take over for large $d$.

**What the separable solution converges to.** On separable data, unpenalised gradient descent drives $\|w\|$ to infinity at a logarithmic rate, but the *direction* $w/\|w\|$ converges to the maximum-margin separator, the same one a hard-margin SVM finds (02.05). This "implicit bias" helps explain why unpenalised over-parameterised models can still generalise.

**MAP view.** The L2 penalty equals a Gaussian prior on the weights (L1: a Laplace prior, giving sparsity as in 02.02). And if each class is Gaussian with a shared covariance, Bayes' rule gives log-odds exactly linear in $x$; logistic regression models that posterior directly, without assuming the Gaussians.

## Key terms

| Term | Meaning
| --- | --- |
| Sigmoid $\sigma(z)$ | $1/(1 + e^{-z})$: squashes any score into a probability between 0 and 1
| Logit / log-odds | $\log\frac{p}{1-p}$; the quantity logistic regression models as linear
| Odds ratio | $e^{w_j}$: the factor by which the odds multiply when feature $j$ rises by one
| Log loss / cross-entropy | Average $-\log$ of the probability given to the true label
| Margin | $y z$ with $y \in \{-1, +1\}$: positive when the score has the right sign
| Decision boundary | Hyperplane $w^\top x + b = \log\frac{t}{1-t}$, perpendicular to $w$
| Threshold $t$ | Probability cut-off for predicting class 1; moves the boundary in parallel
| IRLS | Newton's method as repeated weighted least-squares fits
| Separable data | A hyperplane splits the classes perfectly; unpenalised weights diverge

<div class="take">
<div class="lab">Key takeaway</div>

logistic regression is a linear model of the log-odds, squashed by the sigmoid into a probability and fitted by maximum likelihood (log loss). The loss is convex and its gradient is error times input; keep the L2 penalty on, compute the loss from logits, and choose the decision threshold from the costs of each kind of mistake.

</div>
