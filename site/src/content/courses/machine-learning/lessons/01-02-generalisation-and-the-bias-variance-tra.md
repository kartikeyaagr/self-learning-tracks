---
id: "01.02"
course: "machine-learning"
module: "01"
slug: "generalisation-and-the-bias-variance-trade-off"
title: "Generalisation and the bias-variance trade-off"
keyIdeas: ["Test error splits into irreducible noise, **bias** (error of the average model) and **variance** (how much the model moves when the training set changes).", "Capacity is the knob: too little and both training and test error are high (underfitting); too much and training error falls while test error climbs (overfitting).", "More data reduces variance but not bias; regularisation and simpler classes trade a little bias for a lot less variance.", "Modern over-parameterised models can show **double descent**: test error falls again past the point where the model interpolates the training set."]
question: "A gradient-boosted model has 2% training error and 2.5% validation error, but the product needs 1%. A colleague proposes collecting ten times more data. Using the bias-variance view, argue whether this is likely to help and what you would try instead."
answer: "Training and validation error are close (2% vs 2.5%), so variance is small: the model is not overfitting much. The dominant problem is **bias** (or irreducible noise): even on data it has seen, the model reaches only 2%. More data mainly reduces variance, so ten times more rows would likely shave a little off the 0.5% gap and leave the 2% floor. Better options attack bias: add capacity (deeper trees, more rounds, less regularisation), and above all add informative features, because if the 2% is close to the Bayes error for the current features no model can beat it. A useful check is human or oracle error on a sample: if experts also miss about 2% with the same inputs, only new information (features or labels of better quality) will help."
sources: [{"title": "Cornell CS4780: Bias-variance decomposition", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html", "note": "Clean derivation of the three-term decomposition and how to read train/test curves."}, {"title": "Reconciling modern machine learning practice and the bias-variance trade-off (Belkin et al.)", "url": "https://arxiv.org/abs/1812.11118", "note": "The double-descent paper; read the introduction and Figure 1."}, {"title": "Understanding deep learning requires rethinking generalization (Zhang et al.)", "url": "https://arxiv.org/abs/1611.03530", "note": "Networks fitting random labels; why capacity counts alone fail to explain generalisation."}, {"title": "scikit-learn: Underfitting vs overfitting", "url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html", "note": "A runnable polynomial-degree sweep with cross-validated error."}]
practice: ["knn-classifier", "k-fold-cross-validation", "bagging-classifier", "decision-tree-regressor"]
tags: ["bias-variance", "overfitting"]
tools: ["NumPy"]
---

## Two ways to be wrong

Train the same model class on many different training sets drawn from $\mathcal{D}$ and you get many different fitted functions $\hat f_S$. At a fixed input $x$, look at how those predictions spread. For squared loss with $y = f(x) + \varepsilon$, $\mathbb{E}[\varepsilon] = 0$, $\text{Var}(\varepsilon) = \sigma^2$, the expected test error at $x$ decomposes exactly:

$$\mathbb{E}_{S,\varepsilon}\big[(y - \hat f_S(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat f_S(x) - \bar f(x))^2\big]}_{\text{variance}}$$

where $\bar f(x) = \mathbb{E}_S[\hat f_S(x)]$ is the average prediction over training sets.

- **Noise** is the Bayes error. Nothing you do to the model changes it; only better features can.
- **Bias** is the error of the *average* model. A straight line fitted to a parabola is wrong on average, however many datasets you average over.
- **Variance** is the sensitivity to the particular sample. A degree-15 polynomial on 20 points changes wildly when one point moves.

The decomposition is exact for squared loss. For 0-1 loss there are analogues but no clean additive split; the intuition carries over anyway.

## Capacity and the U-shaped curve

Capacity is how rich the hypothesis class is: polynomial degree, tree depth, $1/k$ in k-NN, number of parameters, inverse regularisation strength. As capacity grows, bias falls and variance rises. Training error falls monotonically. Test error typically falls, bottoms out, then rises.

~~~dg-flow As capacity grows, the dominant source of test error switches from bias to variance.
Low capacity -(high bias)-> Underfit -(add capacity)-> *Sweet spot -(add more)-> Overfit -(high variance)-> Memorised training set
~~~

The two failure modes look different on a train/validation plot, and that is how you diagnose them:

| Symptom | Training error | Validation error | Likely cause | First fix |
| --- | --- | --- | --- | --- |
| Underfitting | high | high, close to training | bias | richer class, better features, less regularisation |
| Overfitting | low | much higher than training | variance | more data, regularisation, simpler class |
| Healthy | low | slightly above training | balanced | stop tuning, check on test |
| Data problem | low | high and noisy across folds | split or leakage issue | check the split (next lesson) |

## What moves bias and what moves variance

- **More training data** shrinks variance roughly like $1/n$ for many estimators and leaves bias untouched. If a linear model underfits, a billion more rows will not help it.
- **Regularisation** (lesson 01.04) deliberately adds bias to cut variance. Ridge regression's predictions are biased toward zero but far more stable.
- **Averaging** reduces variance: bagging and random forests average many high-variance trees, which is why they work (Module 04 goes deeper).
- **Better features** can cut bias without much extra variance; this is often the cheapest win.

<div class="ex">
<div class="lab">Worked example</div>

**k-nearest neighbours as a capacity dial**

With $k = 1$, each prediction copies the label of the closest training point. Training error is 0 (each point is its own neighbour); the decision boundary is jagged and changes when any single point moves: low bias, high variance.

With $k = n$, every prediction is the majority class of the whole training set: a constant. Variance is near 0 (one more point barely changes the majority), bias is maximal.

On a typical tabular problem with $n = 1000$, validation error might be 18% at $k = 1$, 11% at $k = 15$ and 40% at $k = 1000$. You pick $k$ by validation, not by training error, which is always best at $k = 1$.

</div>

## Beyond the U: double descent

The classical U-curve assumes capacity stays below the number of training points. Belkin et al. (2019) showed that if you keep going, test error can peak near the **interpolation threshold** (just enough parameters to fit the training data exactly) and then *fall again* as the model becomes heavily over-parameterised. Among the many functions that interpolate the data, gradient descent tends to find smooth, small-norm ones, which acts as an implicit regulariser. Zhang et al. (2017) made the puzzle vivid: standard image networks can perfectly fit randomly shuffled labels, so parameter count alone cannot explain why they generalise on real labels.

For the classical models in this course (linear models, trees, k-NN, kernels with explicit penalties), the U-shaped picture is the right working model. Keep double descent in mind when you reach the Deep Learning course, where it matters most.

```python
# Diagnose with a capacity sweep: fit at several capacities, compare train vs validation
for deg in [1, 3, 9, 15]:
    w = np.polyfit(x_tr, y_tr, deg)
    tr, va = mse(np.polyval(w, x_tr), y_tr), mse(np.polyval(w, x_va), y_va)
    print(deg, round(tr, 3), round(va, 3))   # training falls every time; validation turns up
```

<div class="take">
<div class="lab">Key takeaway</div>

test error is noise plus bias squared plus variance. Diagnose which one dominates from the gap between training and validation error, then add capacity for bias or add data and constraints for variance.

</div>
