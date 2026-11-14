---
id: "07.01"
course: "machine-learning"
module: "07"
slug: "scaling-and-transforming-features"
title: "Scaling and transforming features"
keyIdeas: ["Distance-based models, regularised linear models, PCA and anything trained by gradient descent are sensitive to feature scale; tree models are not, because a split only depends on order.", "**Standardisation** $(x - \\mu)/\\sigma$ is the default; **min-max** maps to $[0, 1]$ but one outlier squashes everything else; **robust** scaling uses median and IQR. Fit the statistics on training data only.", "Skewed positive features usually behave better after $\\log(1+x)$; **Box-Cox** picks a power by maximum likelihood (positive data only) and **Yeo-Johnson** extends it to zero and negatives. A **quantile** transform forces any shape to uniform or normal.", "**Binning** turns a number into categories. It buys non-linearity and robustness for linear models at the price of information and jumps at bin edges; splines are the smoother alternative.", "Transforming the *target* changes what the model predicts: a model fitted on $\\log y$ predicts the median of $y$ after exponentiating, not the mean."]
question: "You train a ridge regression and a gradient-boosted tree on the same raw features, one of which is account balance in cents (range 0 to 10^9, heavy right tail). For each model, predict what happens if you (a) leave it raw, (b) standardise it, (c) apply log1p then standardise. Explain each prediction."
answer: "For the **tree**, all three give essentially the same model: splits depend only on the ordering of values, and standardisation and log1p are monotone, so the same partitions are available. Only histogram-based boosting binning can shift slightly. For **ridge**, (a) raw: the coefficient on balance needs to be tiny to keep predictions sensible, so the L2 penalty barely affects it while other, standardised features get penalised heavily; the effective regularisation is inconsistent, and the extreme tail values act as high-leverage points that dominate the fit. (b) Standardising fixes the penalty imbalance but not the tail: a few accounts sit tens of standard deviations out and still pull the line. (c) log1p then standardise compresses the tail, makes the relationship closer to linear in relative terms (doubling balance has a constant effect), and removes most of the leverage, so it is usually the best choice for ridge."
sources: [{"title": "scikit-learn: Preprocessing data", "url": "https://scikit-learn.org/stable/modules/preprocessing.html", "note": "Scalers, power and quantile transforms, discretisation and splines, with the fit/transform API."}, {"title": "scikit-learn: Compare the effect of different scalers on data with outliers", "url": "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html", "note": "Side-by-side plots of every scaler and transformer on a skewed, outlier-heavy dataset."}, {"title": "Feature Engineering and Selection (Kuhn and Johnson)", "url": "https://bookdown.org/max/FES/", "note": "Free online book; chapter 6 on engineering numeric predictors covers Box-Cox, Yeo-Johnson, splines and binning trade-offs."}, {"title": "scikit-learn: PowerTransformer", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html", "note": "Box-Cox and Yeo-Johnson definitions and how lambda is estimated."}]
practice: ["feature-scaling", "normalize-columns", "winsorize", "column-scaling", "angle-features"]
tags: ["feature scaling", "power transforms", "binning"]
tools: ["scikit-learn", "NumPy"]
---

## Where this module fits

Module 06 showed how to measure a model. This module is about the inputs: how to turn raw columns into features a model can use, without leaking the evaluation data along the way. The lessons cover numeric transforms, categorical encodings, missing values, feature selection, and the pipeline that ties them together so the same steps run in CV and in production. The ML Toolkit course covers the pandas and SQL mechanics; here the focus is on what each transform does to a model.

## Who cares about scale

A feature measured in metres and the same feature in millimetres are identical to a decision tree, which only asks "is $x_j < t$?". They are very different to:

- **Distance-based models** (k-NN, k-means, RBF kernels): a feature with range 0 to 100,000 swamps one with range 0 to 1 in the Euclidean distance.
- **Penalised linear models** (ridge, lasso, logistic with $C$): the penalty $\lambda \sum w_j^2$ treats all weights alike, so a feature in small units needs a large weight and is penalised more.
- **Gradient descent** and neural networks: badly scaled inputs give an ill-conditioned loss surface, so one learning rate is too big for some directions and too small for others.
- **PCA**: principal components chase variance, and variance depends on units.

~~~dg-layers How much each model family cares about feature scale.
*k-NN, k-means, SVM with RBF, PCA | scale is part of the model: always scale
Ridge, lasso, logistic regression | the penalty depends on units: scale
Neural networks, SGD-trained models | conditioning and initialisation: scale
Trees, random forests, gradient boosting | only order matters: scaling is unnecessary
~~~

## The three scalers

| Scaler | Formula | Good for | Weakness
| --- | --- | --- | --- |
| Standard | $(x-\mu)/\sigma$ | the default for most models | outliers inflate $\sigma$
| Min-max | $(x-\min)/(\max-\min)$ | bounded inputs, pixel values | one outlier compresses the rest
| Robust | $(x-\text{median})/\text{IQR}$ | heavy-tailed or dirty data | does not bound the range

For sparse data (bag-of-words counts) use max-abs scaling, which divides without centring, so zeros stay zero and the matrix stays sparse. Whatever the scaler, its statistics are learned state: fit them on the training fold and apply them unchanged to validation, test and production data. If production sees a value outside the training range, min-max scaling produces numbers outside $[0, 1]$; that is fine for most models, but a drift alarm worth having (lesson 09.03).

## Fixing skew

Many real features (income, counts, durations, prices) are positive with a long right tail. Squashing that tail helps linear models (relationships become closer to linear), distance models (one extreme value stops dominating) and the stability of gradient training.

- $\log(1 + x)$ is the workhorse for non-negative counts and amounts.
- **Box-Cox**: $x^{(\lambda)} = (x^\lambda - 1)/\lambda$ for $\lambda \ne 0$ and $\log x$ for $\lambda = 0$, with $\lambda$ chosen by maximum likelihood to make the result as Gaussian as possible. It requires $x > 0$.
- **Yeo-Johnson** applies a similar family to $x + 1$ for non-negative values and a mirrored one for negatives, so it accepts any real input. scikit-learn's `PowerTransformer` uses it by default and standardises afterwards.
- **Quantile transform** maps each value to its empirical CDF, then optionally through the inverse normal CDF. Any distribution becomes uniform or Gaussian, and outliers are tamed completely. It is rank-based, so it discards how far apart values were, and with few training rows the mapping is coarse.
- **Clipping or winsorising** at, say, the 1st and 99th percentiles caps extreme values without changing the bulk.

<div class="ex">
<div class="lab">Worked example</div>

**income in a k-NN model**

Income in the training set: median 52,000, mean 71,000, standard deviation about 90,000, maximum 4.2 million. After standardisation the richest customer sits at $z \approx 46$, so in Euclidean distance this one feature outweighs the other 20 standardised features combined, and the customer has no meaningful neighbours.

After $\log(1+x)$ the values run from about 9 to 15.3, with the median at 10.9 and a standard deviation near 0.8, so the maximum is about 5.4 standard deviations out. Standardising the logged feature gives a well-behaved column, and nearest-neighbour validation accuracy improves because distances now reflect all features.

</div>

## Binning and its alternatives

Binning (discretisation) replaces a number with the bin it falls into, usually one-hot encoded afterwards. **Equal-width** bins are easy to explain and empty when data is skewed; **quantile** bins hold equal counts. Reasons to bin:

- It lets a linear model fit a non-linear, even non-monotone, effect (risk by age band).
- It is robust to outliers and to small measurement errors.
- Regulated scorecards (credit) often require it for interpretability.

The costs: information inside a bin is thrown away, predictions jump at bin edges, and the edges are another choice you can overfit. **Splines** (`SplineTransformer`) give a linear model a smooth non-linear effect with a handful of basis columns, and trees learn their own bins. Bin edges, like scaler statistics, must come from training data. Periodic features (hour of day, day of week, wind direction) need a different treatment altogether: encode $\theta$ as $(\sin\theta, \cos\theta)$ so that 23:00 and 01:00 are close.

## Transforming the target

For a skewed target such as price or duration, fitting on $\log y$ often gives better relative errors. But $\exp(\widehat{\log y})$ estimates the **median** of $y$, not the mean. If the residuals on the log scale are roughly normal with variance $\sigma^2$, the mean is $\exp(\widehat{\log y} + \sigma^2/2)$; with $\sigma = 0.5$ that is a 13% correction. Forgetting it biases totals (revenue forecasts) low. scikit-learn's `TransformedTargetRegressor` applies the forward and inverse transform for you, but not the bias correction.

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer, KBinsDiscretizer
pre = ColumnTransformer([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["income", "n_orders"]),
    ("yj", PowerTransformer(), ["balance_change"]),                      # can be negative
    ("bin", KBinsDiscretizer(n_bins=8, strategy="quantile", encode="onehot"), ["age"]),
], remainder=StandardScaler())
```

<div class="take">
<div class="lab">Key takeaway</div>

scale for models that measure distance, penalise weights or use gradients; squash skewed features with log, Box-Cox or Yeo-Johnson; bin only when you need what binning buys; and learn every statistic on the training fold.

</div>
