---
id: "02.04"
course: "machine-learning"
module: "02"
slug: "multiclass-classification"
title: "Multiclass classification"
keyIdeas: ["A multiclass problem has $K > 2$ mutually exclusive labels. You can **reduce** it to binary problems (one-vs-rest with $K$ models, one-vs-one with $K(K-1)/2$) or use a natively multiclass model such as softmax regression.", "Softmax regression computes one logit per class, $z = Wx + b$, then $p_k = e^{z_k}/\\sum_j e^{z_j}$: positive, summing to 1, in the same order as the logits. Only differences between logits matter.", "It is trained with cross-entropy $-\\log p_y$, which is convex, and the gradient on the logits is just $p - \\text{onehot}(y)$: push the true class up, every other class down by its probability.", "One-vs-rest scores come from separately trained models, so they are not on a common scale and do not sum to 1; each subproblem is also imbalanced ($1$ class against $K-1$). Softmax is over-parameterised, and with $K = 2$ it is exactly logistic regression.", "Compute softmax and its loss stably (subtract the max, use log-sum-exp, pass logits to the loss), and if labels are not mutually exclusive (multi-label), use $K$ independent sigmoids instead of a softmax."]
question: "An OvR logistic model on 4 balanced classes outputs per-class sigmoid probabilities (0.30, 0.25, 0.20, 0.10) for a test point. What does this pattern tell you, how would a softmax model trained on the same data present the same point, and which output would you rather feed into a downstream cost-sensitive decision?"
answer: "The four binary models each say this point is more likely not their class, and their probabilities sum to only 0.85. The point sits in a region none of them claims: near the shared boundaries, or away from the training data. The argmax still predicts class 1, but by a narrow margin, and the numbers are not a distribution. Rescaling them to sum to 1 gives (0.35, 0.29, 0.24, 0.12), which is what `OneVsRestClassifier.predict_proba` does, but that is an ad hoc fix with no likelihood behind it; each model was trained on its own imbalanced 1-against-3 problem and its scale is not comparable with the others. A softmax model is trained jointly with multiclass cross-entropy, so its outputs form a proper distribution over the four classes; it would likely give a similarly flat but normalised vector, and its probabilities are usually better calibrated (on the iris example in this lesson the OvR log loss was nearly twice the softmax one). For a cost-sensitive decision, which needs expected costs $\\sum_k p_k c_k$ for each action, use the softmax output after a calibration check (reliability plot, temperature scaling, as in 03.04). Also treat the flat profile itself as information: with the top probability around a third, the point is a candidate to abstain on or route to a person, and if many test points look like this, the features may not separate the classes or the point may be out of distribution."
sources: [{"title": "In Defense of One-Vs-All Classification", "url": "https://www.jmlr.org/papers/v5/rifkin04a.html", "note": "Rifkin and Klautau (JMLR 2004): well-tuned one-vs-rest matches more complex multiclass schemes."}, {"title": "Dive into Deep Learning: Softmax Regression", "url": "https://d2l.ai/chapter_linear-classification/softmax-regression.html", "note": "Softmax, cross-entropy and its gradient, with the information-theory view of the loss."}, {"title": "scikit-learn: Multiclass and multioutput algorithms", "url": "https://scikit-learn.org/stable/modules/multiclass.html", "note": "OneVsRest, OneVsOne and which estimators are inherently multiclass, plus multi-label targets."}]
practice: ["softmax-regression", "softmax-cross-entropy-gradient", "cross-entropy-logits", "softmax-numerical-stability-and-log-sum-exp", "top-k-classification-accuracy"]
tags: ["softmax", "one-vs-rest"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- The two routes to $K$ classes: combine binary models, or score all classes in one model.
- What **softmax** does to a list of scores, including the temperature knob.
- How softmax regression is trained with cross-entropy, and why its gradient is "predicted minus true".
- How one-vs-rest and one-vs-one work and why their scores are not probabilities.
- How to check a multiclass model, and the common traps, including multi-label data.

*Before you start:* you need logistic regression (02.03): the sigmoid and binary cross-entropy. Linear models and losses come from 02.01. Matrix-vector products are used throughout.

## From two classes to many

Lesson 02.03 built a classifier for yes/no questions: one linear score $z = w^\top x + b$, squashed by the sigmoid into a probability. Many real questions have more answers: which of 10 digits, which of 5 support teams, which of 3 species. A label with $K > 2$ possible values, exactly one of which is correct, is a **multiclass** problem. (When several labels can be true at once, the problem is **multi-label**, covered near the end.)

There are two ways to handle $K$ classes with linear models:

- **Reduction**: train several binary classifiers and combine their answers. This works with any binary learner.
- **Native multiclass**: give the model one score per class and turn all $K$ scores into probabilities together. For linear models this is **softmax regression**, also called multinomial logistic regression.

## The idea in one picture

Take the classic iris data: 150 flowers of three species (setosa, versicolor, virginica), with petal length and width in cm. A new flower has petals 4.8 cm long and 1.6 cm wide. Which species is it?

Softmax regression keeps one weight vector and one bias per class. Fitted on 105 training flowers (code below), it learns:

| Class | weight on length | weight on width | bias
| --- | --- | --- | --- |
| setosa | -2.55 | -0.99 | 10.43
| versicolor | -0.03 | -0.66 | 3.09
| virginica | 2.57 | 1.65 | -13.52

For the new flower each class gets a score, called a **logit**: weight times feature, summed, plus bias. The logits are $z = (-3.39, 1.89, 1.46)$. Softmax turns them into probabilities $p = (0.003, 0.604, 0.393)$, which sum to 1: versicolor at 60%, with virginica a real possibility.

~~~dg-fig 02.04-regions.svg
Softmax regression splits the plane into three regions, one per species, with straight boundaries. The new flower (star) lies close to the versicolor/virginica border, which is why its probabilities are split 60/39.
~~~

Read the figure as a map: every point gets the class whose logit is largest there. Setosa sits far away; versicolor and virginica touch, so flowers near their border get split probabilities. From petal size alone, those two species do overlap.

## Step 1: one score per class

Make one linear model per class, each saying how strongly the input looks like its class. Stack the $K$ weight vectors as the rows of a matrix $W$ (shape $K \times d$, where $d$ is the number of features) and the biases in a vector $b$ (length $K$). Then all the logits come from one matrix-vector product:

$$z = W x + b, \qquad z_k = w_k^\top x + b_k.$$

Here $x$ is one example's feature vector, $w_k$ is row $k$ of $W$, and $z_k$ is the logit for class $k$: any real number, meaningful only relative to the others.

~~~dg-flow Softmax regression: one score per class, normalised jointly, so the outputs compete and sum to 1.
Features x -(W x + b)-> K logits -(softmax)-> *K probabilities -(argmax)-> Predicted class
~~~

Number check for the virginica row: $2.57 \times 4.8 + 1.65 \times 1.6 - 13.52 = 12.34 + 2.64 - 13.52 = 1.46$. Longer and wider petals push this score up, which matches the data: virginica has the biggest petals.

## Step 2: turn scores into probabilities with softmax

We want $K$ positive numbers that add up to 1 and keep the logits' order. Softmax exponentiates each logit (making it positive), then divides by the total (making the sum 1):

$$p_k = \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

$p_k$ is the model's probability for class $k$; the denominator is the same for every $k$. Softmax is a smooth version of "pick the maximum": the largest logit gets the largest share, the others still get something.

Four properties you will rely on:

- **Order is kept.** A bigger logit always means a bigger probability, so the predicted class is just $\arg\max_k z_k$. You do not need softmax to predict, only to get probabilities.
- **Differences matter, not levels.** Adding the same constant to every logit cancels between top and bottom. Logits $(2, 1, 0.1)$ and $(102, 101, 100.1)$ give identical probabilities.
- **A gap of 1 in logits is a factor of $e \approx 2.7$ in probability.** If $z_1 - z_2 = 1$, then $p_1 / p_2 = e$.
- **Two classes give back the sigmoid**: $p_1 = \sigma(z_1 - z_0)$ (see "Going deeper").

The **temperature** $T$ is an extra knob: divide the logits by $T$ before softmax. $T < 1$ sharpens the distribution toward the top class, $T > 1$ flattens it toward uniform; the ranking never changes.

~~~dg-fig 02.04-softmax.svg
The same three logits under softmax at three temperatures. At T = 1 they become 0.66 / 0.24 / 0.10. Lower temperature makes the winner take more; higher temperature spreads probability out. The order of the bars never changes.
~~~

Temperature is not needed in training (it would just rescale the weights), but it is the standard fix for over-confident probabilities after training (temperature scaling, 03.04) and the randomness knob when sampling from language models.

<div class="ex">
<div class="lab">Worked example</div>

**classify the new flower by hand**

Logits from step 1: $z = (-3.39, 1.89, 1.46)$ for (setosa, versicolor, virginica).

Exponentials: $e^{-3.39} = 0.034$, $e^{1.89} = 6.619$, $e^{1.46} = 4.306$. Their sum is $0.034 + 6.619 + 4.306 = 10.959$.

Divide each by the sum: $p = (0.034/10.959,\ 6.619/10.959,\ 4.306/10.959) = (0.003, 0.604, 0.393)$. Check: $0.003 + 0.604 + 0.393 = 1.000$.

Stable recipe: subtract the largest logit first, giving $(-5.28, 0, -0.43)$; exponentials $(0.0051, 1, 0.6505)$, sum $1.6556$, same probabilities.

Prediction: versicolor. The ratio $p_{\text{vers}}/p_{\text{virg}} = 1.54 = e^{0.43}$, exactly $e$ raised to the logit gap $1.89 - 1.46$.

</div>

## Step 3: measure how wrong the probabilities are

Training needs a loss. As in 02.03 we use the negative log-likelihood: how surprised the model is by the true label. For one example whose true class is $y$, the **cross-entropy** loss is

$$\ell = -\log p_y = -z_y + \log \sum_{j=1}^{K} e^{z_j}.$$

Only the probability of the true class enters. The second form, written in logits, is how you should compute it in code; the term $\log \sum_j e^{z_j}$ is called **log-sum-exp**. The training loss is the average of $\ell$ over all examples, usually plus an L2 penalty on $W$ (02.02).

~~~dg-fig 02.04-xent.svg
Cross-entropy as a function of the probability given to the true class. It is near zero for confident correct answers, equals ln 3 = 1.10 for a uniform guess over three classes, and climbs steeply for confident mistakes.
~~~

If the flower really is versicolor, the model's 0.604 costs $-\ln 0.604 = 0.50$. Putting only 0.05 on the true class would cost 3.00, six times as much. A model that says $1/K$ for every class pays $\ln K$ (1.10 for three classes): the "useless model" baseline for reading a log loss.

## Step 4: the gradient, and fitting

Differentiate the loss with respect to the logits and something neat happens:

$$\frac{\partial \ell}{\partial z_k} = p_k - [k = y], \qquad \text{i.e.} \quad \nabla_z \ell = p - e_y.$$

Here $e_y$ is the **one-hot** vector of the label: 1 in position $y$, 0 elsewhere. The gradient is "predicted minus target": the true class's logit is pushed up by $1 - p_y$, every other logit down by its own probability, and confidently correct examples barely move the weights.

The chain rule through $z = Wx + b$ gives the weight gradients: row $k$ of $W$ moves by $(p_k - [k = y])\, x$, and $b_k$ by $p_k - [k = y]$. Averaged over a dataset with $X$ of shape $(n, d)$ and $P$, $Y$ the $(n, K)$ matrices of probabilities and one-hot labels:

$$\nabla_W L = \tfrac{1}{n} (P - Y)^\top X.$$

That is the binary formula from 02.03, $\tfrac1n X^\top(p - y)$, with one column per class. The loss is convex (log-sum-exp is convex, $-z_y$ linear), so there are no bad local minima, and the 02.03 recipe carries over: L-BFGS or mini-batch gradient descent, standardised features, an L2 penalty.

<div class="ex">
<div class="lab">Worked example</div>

**one gradient by hand**

Logits $z = (2.0, 1.0, 0.1)$ for classes A, B, C. Exponentials $7.389, 2.718, 1.105$, sum $11.21$, so $p = (0.659, 0.242, 0.099)$.

Suppose the true class is B. Loss: $-\ln 0.242 = 1.42$.

Gradient on the logits: $p - e_B = (0.659 - 0,\ 0.242 - 1,\ 0.099 - 0) = (0.659, -0.758, 0.099)$. A gradient-descent step subtracts this, so it lowers A and C and raises B. The three entries sum to 0, as they always do, because the probabilities sum to 1.

For the iris flower, if the gardener confirms it is versicolor, the loss is $-\ln 0.604 = 0.50$ and the logit gradient is $(0.003, -0.396, 0.393)$. The virginica weight row then moves by $-\eta \times 0.393 \times (4.8, 1.6)$ for learning rate $\eta$: less weight on petal size for virginica, exactly what this flower calls for.

Overflow check: adding 100 to every logit leaves $p$ unchanged, but $e^{102}$ overflows float32 (whose limit is about $e^{88.7}$), which is why stable code subtracts the maximum first.

</div>

## Step 5: what the decision regions look like

Class $k$ wins at $x$ when $z_k \ge z_j$ for every other class $j$. Each comparison $w_k^\top x + b_k \ge w_j^\top x + b_j$ is a half-space (one side of a line or hyperplane). The region for class $k$ intersects $K - 1$ half-spaces, so it is **convex** with straight edges, as the iris map shows.

So a linear multiclass model cannot give one class two separate islands, or a ring around another class; if your data needs that, add features (02.06) or use a non-linear model. Note also that the boundary between classes $k$ and $j$ depends only on $w_k - w_j$, a hint of the over-parameterisation in "Going deeper".

## The other route: reduce to binary problems

**One-vs-rest (OvR)**, or one-vs-all, trains $K$ binary classifiers, classifier $k$ learning "class $k$ versus everything else" on all $n$ examples. To predict, pick the highest score.

**One-vs-one (OvO)** trains a classifier for every pair of classes on just those two classes' data and predicts by majority vote: $K(K-1)/2$ small models, 45 for $K = 10$.

Back to the iris flower. OvR logistic models on the same data give setosa 0.004, versicolor 0.388, virginica 0.393. Every model says "probably not mine", and the numbers sum to only 0.785: not a distribution. The argmax picks virginica where softmax picked versicolor; the flower is on the border, and the two methods draw it differently.

~~~dg-fig 02.04-ovr.svg
Left: one-vs-rest on the iris petals. The hatched bands are places where all three binary models say "not mine" with probability above 0.5; the argmax still assigns a class there. Right: softmax on the same data, for comparison. The new flower (star) lands on different sides.
~~~

| Property | One-vs-rest | One-vs-one | Softmax
| --- | --- | --- | --- |
| Models trained | K | K(K-1)/2 | 1 (K weight vectors)
| Data per model | all n | two classes only | all n
| Scores sum to 1 | no | no (votes) | yes
| Typical use | linear SVMs, any binary learner | kernel SVMs | logistic-style models, neural nets

Why use a reduction? Because some learners are only binary, like the SVM of 02.05. For kernel SVMs, whose training cost grows faster than linearly in $n$, many small pairwise problems can be cheaper than a few large ones, which is why libsvm (and so `SVC`) uses OvO. And OvR is not a weak method: Rifkin and Klautau showed that, when each binary model is well tuned and regularised, OvR is about as accurate as more elaborate multiclass schemes. Its real drawbacks are uncalibrated scores and imbalanced subproblems (with 10 balanced classes, each is 10% positive).

scikit-learn often chooses silently: `LogisticRegression` with the default lbfgs solver fits one softmax model (liblinear can only do OvR), `SVC` uses OvO internally, `LinearSVC` uses OvR. `OneVsRestClassifier` and `OneVsOneClassifier` make the strategy explicit for any estimator.

## Multiclass in code

With scikit-learn, on the same iris split used in the figures:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

X, y = load_iris(return_X_y=True)
X = X[:, 2:]                                  # petal length, petal width (cm)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)

soft = LogisticRegression().fit(Xtr, ytr)     # lbfgs: one joint softmax model
print(soft.coef_.round(2), soft.intercept_.round(2))
# [[-2.55 -0.99] [-0.03 -0.66] [ 2.57  1.65]]  [ 10.43   3.09 -13.52]
print(accuracy_score(yte, soft.predict(Xte)))           # 0.956
print(log_loss(yte, soft.predict_proba(Xte)))           # 0.188  (useless baseline: ln 3 = 1.099)
print(confusion_matrix(yte, soft.predict(Xte)))
# [[15  0  0]
#  [ 0 14  1]
#  [ 0  1 14]]

new = [[4.8, 1.6]]
print(soft.decision_function(new).round(2))   # logits  [[-3.38  1.91  1.47]]
print(soft.predict_proba(new).round(3))       # [[0.003 0.606 0.391]]

ovr = OneVsRestClassifier(LogisticRegression()).fit(Xtr, ytr)
raw = [e.predict_proba(new)[0, 1] for e in ovr.estimators_]
print(np.round(raw, 3), round(sum(raw), 3))   # [0.004 0.388 0.393]  0.785
print(ovr.predict(new))                       # [2]  (virginica)
print(log_loss(yte, ovr.predict_proba(Xte)))  # 0.355: worse probabilities than softmax
```

(Logits differ from the hand example in the second decimal because the table rounds the weights.) `OneVsRestClassifier.predict_proba` rescales the sigmoids to sum to 1, which looks like a distribution without being one; its log loss is nearly double.

The same model from scratch in NumPy, to see that there is no magic:

```python
import numpy as np

def softmax_xent(Z, y):                 # Z: (n, K) logits, y: (n,) int labels
    Z = Z - Z.max(axis=1, keepdims=True)              # stability: exp never overflows
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    G = np.exp(logp); G[np.arange(len(y)), y] -= 1    # dL/dZ = p - onehot
    return loss, G / len(y)

loss, G = softmax_xent(np.array([[2.0, 1.0, 0.1]]), np.array([1]))
print(round(loss, 3), G.round(3))      # 1.417 [[ 0.659 -0.758  0.099]]

# training = repeat: W -= lr * (G.T @ X + lam * W); b -= lr * G.sum(0)
```

Subtracting the row maximum changes no probability and keeps `exp` from overflowing; computing log-probabilities directly avoids `log(0)` when a probability underflows. In PyTorch, `nn.CrossEntropyLoss` expects raw logits and does both steps for you; passing it softmax outputs is a classic bug (see below). The AI Research course (lesson 02.02) covers softmax intuition and temperature in more depth.

## Reading the result

Accuracy alone hides a lot with many classes. Check these:

- **Confusion matrix** (rows true, columns predicted). Above, both errors are versicolor/virginica, matching the overlap in the figure; off-diagonal clusters show which classes need better features.
- **Per-class precision and recall** (`classification_report`). A model can score 95% overall while missing most of a rare class; check macro-averaged F1.
- **Log loss against $\ln K$.** 0.19 against 1.10 means the probabilities carry real information; near $\ln K$ means barely better than guessing.
- **Top-k accuracy** (`top_k_accuracy_score`). With hundreds of classes, "right answer in the top 5" is often the useful number.
- **Calibration.** Of the examples given 0.6, about 60% should be right. Check with a reliability plot (03.04).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Softmax outputs passed to a loss that expects logits | slow training, loss stalls high | pass raw logits to the loss (`nn.CrossEntropyLoss`, `F.cross_entropy`)
| Naive `exp` / `log` | `inf` or `nan` loss with large logits, especially in float32 | subtract the max; use log-softmax / log-sum-exp
| Reading OvR scores as probabilities | outputs do not sum to 1; poor log loss; bad expected-cost decisions | use softmax, or calibrate the OvR outputs
| Softmax on a multi-label problem | labels that co-occur steal probability from each other | K independent sigmoids with binary cross-entropy
| Strong class imbalance | rare classes almost never predicted; high accuracy, low macro F1 | class weights, resampling, per-class thresholds
| No regularisation on separable data | weights grow without limit; probabilities hit 0 and 1 | keep the L2 penalty (the sklearn default)

**Multi-label is a different problem.** If an example can carry several labels at once (a photo tagged "beach" and "dog"), softmax wrongly forces the labels to compete for one unit of probability. Use $K$ independent sigmoids, each with binary cross-entropy, and a threshold per label: OvR under an honest name.

## Going deeper

**Over-parameterisation.** Adding the same vector $c$ to every row of $W$ (and the same constant to every bias) adds $c^\top x$ to every logit, which softmax ignores. So the unregularised solution is not unique; an L2 penalty picks the one whose rows sum to zero, and fixing one class at zero gives the statistician's "reference class" form. With $K = 2$ only $w_1 - w_0$ survives, and $p(y = 1 \mid x) = \sigma\big((w_1 - w_0)^\top x + b_1 - b_0\big)$: softmax regression is exactly logistic regression.

**Why softmax?** If each class's log-probability is linear in $x$ up to a shared normaliser, normalising forces the softmax form. It is also the posterior of a generative model with Gaussian classes sharing one covariance (linear discriminant analysis), a link between generative and discriminative classifiers.

**Curvature.** The Hessian of $\ell$ in the logits is $\operatorname{diag}(p) - pp^\top$: positive semi-definite, with the all-ones vector in its null space (shift invariance again).

**Large $K$.** The normaliser costs $O(Kd)$ per example, which dominates with a vocabulary of words or a product catalogue. Sampled and hierarchical softmax approximate it (Deep Learning and Recommender Systems material).

**Other reductions.** Error-correcting output codes train one classifier per bit of a code word per class.

## Key terms

| Term | Meaning
| --- | --- |
| Multiclass | Each example has exactly one of K mutually exclusive labels
| Multi-label | Each example can have any subset of the labels
| Logit | The raw linear score for one class, before softmax
| Softmax | Exponentiate the logits and divide by their sum, giving positive probabilities that sum to 1
| Temperature | Divide logits by T before softmax: T below 1 sharpens, above 1 flattens
| Cross-entropy | Minus the log of the probability given to the true class
| One-vs-rest | K binary models, each "this class versus all others"; predict by argmax
| One-vs-one | A binary model for every pair of classes; predict by vote

<div class="take">
<div class="lab">Key takeaway</div>

softmax regression gives each class a linear score and turns the scores into probabilities that compete and sum to 1; it is trained with cross-entropy, whose gradient on the logits is simply $p - \text{onehot}(y)$. Reduce to binary models (one-vs-rest, one-vs-one) when your learner is binary, but do not read their scores as probabilities, and use independent sigmoids when labels are not mutually exclusive.

</div>
