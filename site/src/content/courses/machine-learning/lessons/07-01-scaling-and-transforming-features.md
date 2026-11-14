---
id: "07.01"
course: "machine-learning"
module: "07"
slug: "scaling-and-transforming-features"
title: "Scaling and transforming features"
keyIdeas: ["Distance-based models, regularised linear models, PCA and anything trained by gradient descent are sensitive to feature scale, because they compare or add up raw feature values. Tree models are not: a split only depends on the order of values.", "**Standardisation** $(x - \\mu)/\\sigma$ is the default; **min-max** maps to $[0, 1]$ but one outlier squashes everything else; **robust** scaling uses median and IQR so the typical rows set the scale. Fit the statistics on training data only and reuse them everywhere else.", "Skewed positive features usually behave better after $\\log(1+x)$; **Box-Cox** picks a power by maximum likelihood (positive data only) and **Yeo-Johnson** extends it to zero and negatives. A **quantile** transform forces any shape to uniform or normal at the cost of distances between values.", "**Binning** turns a number into categories: it buys non-linearity and robustness for linear models at the price of information and jumps at bin edges. Splines give a smooth alternative, and periodic features need a $(\\sin, \\cos)$ encoding.", "Transforming the *target* changes what the model predicts: a model fitted on $\\log y$ predicts the median of $y$ after exponentiating, not the mean, and the mean is higher by a factor of about $\\exp(\\sigma^2/2)$."]
question: "You train a ridge regression and a gradient-boosted tree on the same raw features, one of which is account balance in cents (range 0 to 10^9, heavy right tail). For each model, predict what happens if you (a) leave it raw, (b) standardise it, (c) apply log1p then standardise. Explain each prediction."
answer: "For the **tree**, all three give essentially the same model: splits depend only on the ordering of values, and standardisation and log1p are both monotone (order-preserving), so exactly the same partitions of the rows are available and the learned trees are equivalent. Only histogram-based boosting, which first bins each feature, can shift slightly, and with quantile bins even that is unchanged. For **ridge**: (a) Raw: to keep predictions sensible the coefficient on balance must be tiny (the feature's numbers are up to a billion), so the L2 penalty $\\lambda w^2$ barely touches it while other, standardised features are penalised heavily. The effective regularisation is therefore wildly inconsistent across features, and the extreme tail values act as high-leverage points that dominate the fit. (b) Standardising fixes the penalty imbalance, since every weight now means 'effect of one standard deviation', but not the shape: a handful of accounts still sit tens of standard deviations from the mean and pull the fitted line towards themselves. (c) log1p then standardise compresses the tail (a billion cents becomes about 20.7), makes the relationship closer to linear in relative terms (each doubling of balance has a constant effect, which is usually how money behaves), and removes most of the leverage. It is usually the best choice for ridge. In all cases the scaler and any transform statistics must be fitted on the training fold only, ideally inside a pipeline."
sources: [{"title": "scikit-learn: Preprocessing data", "url": "https://scikit-learn.org/stable/modules/preprocessing.html", "note": "Scalers, power and quantile transforms, discretisation and splines, with the fit/transform API."}, {"title": "scikit-learn: Compare the effect of different scalers on data with outliers", "url": "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html", "note": "Side-by-side plots of every scaler and transformer on a skewed, outlier-heavy dataset."}, {"title": "Feature Engineering and Selection (Kuhn and Johnson)", "url": "https://bookdown.org/max/FES/", "note": "Free online book; chapter 6 on engineering numeric predictors covers Box-Cox, Yeo-Johnson, splines and binning trade-offs."}, {"title": "scikit-learn: PowerTransformer", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html", "note": "Box-Cox and Yeo-Johnson definitions and how lambda is estimated."}]
practice: ["feature-scaling", "normalize-columns", "winsorize", "column-scaling", "angle-features"]
tags: ["feature scaling", "power transforms", "binning"]
tools: ["scikit-learn", "NumPy"]
---

## What you will learn

- Which models care about the units of a feature and which do not, and the one-line reason for each.
- The three everyday scalers (standard, min-max, robust), how to compute each by hand, and how an outlier affects them.
- Why scaling statistics must be learned on training rows only.
- How to tame skewed columns with $\log(1+x)$, Box-Cox, Yeo-Johnson, quantile transforms and clipping.
- When binning, splines or sine/cosine encodings are the right way to give a model a non-linear or circular feature.
- What changes when you transform the *target*, and the bias correction that comes with it.

*Before you start:* you should know k-nearest neighbours and Euclidean distance, linear regression and ridge (02.01, 02.02), gradient descent (01.03), and train/validation splits and cross-validation (module 06).

## Where this module starts

Module 06 showed how to measure a model honestly. This module turns to the inputs. Raw columns come in wildly different units, with long tails, text categories, gaps and useless columns. Feature engineering turns them into features a model can learn from, without leaking the evaluation data. The lessons cover numeric transforms (this one), categorical encodings (07.02), missing values (07.03), feature selection (07.04), and the pipeline that runs every step identically in cross-validation and production (07.05). The ML Toolkit course covers the pandas and SQL mechanics; here the focus is on what each transform does *to the model*.

## The idea in one picture

A shop recommends products to a new customer Q by finding the most similar existing customer (1-nearest neighbour). Each customer has two features:

| Customer | Q | P1 | P2 | P3 | P4 | P5 | P6
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age (years) | 30 | 62 | 31 | 45 | 25 | 52 | 38
| Income (€ / year) | 48,000 | 49,000 | 55,000 | 70,000 | 35,000 | 90,000 | 62,000

Common sense says Q (30 years old, 48k) is most like P2 (31 years old, 55k). The raw Euclidean distance disagrees:

- Q to P1: $\sqrt{32^2 + 1000^2} \approx 1000.5$
- Q to P2: $\sqrt{1^2 + 7000^2} \approx 7000.0$

So the model picks P1, a 62-year-old: a 1,000 € income gap counts as much as a 1,000-year age gap, so age has almost no say.

~~~dg-fig 07.01-knn.svg
Left: in raw units, distance is almost pure income, so Q's nearest neighbour is P1 (same income, twice the age). Right: after standardising both columns, age and income count equally and the nearest neighbour is P2.
~~~

The fix is to put both columns on a common scale before measuring distance. **Standardising** a column means subtracting its mean and dividing by its standard deviation (mean 0, spread 1). Over these seven customers age has mean 40.4 and standard deviation 12.3, income has mean 58,400 and standard deviation 16,500. In standard units:

- Q to P1: age gap $32/12.3 = 2.60$, income gap $1000/16{,}500 = 0.06$, distance $\approx 2.60$
- Q to P2: age gap $1/12.3 = 0.08$, income gap $7000/16{,}500 = 0.42$, distance $\approx 0.43$

Now P2 wins, as it should. Only the units changed. Some models read the *numbers* in a column, not just their *order*, so your recording units become part of the model unless you take control of them.

## Step 1: know which models care

Ask of each model: does it compare or add up raw feature values across features? If yes, scale matters.

- **Distance-based models** (k-NN, k-means, RBF-kernel SVMs) add squared differences across features, as above.
- **Penalised linear models** (ridge, lasso, logistic regression with $C$): the penalty $\lambda \sum_j w_j^2$ treats all weights alike, so a feature in small units needs a large weight and is penalised harder.
- **Gradient descent** and neural networks: badly scaled inputs make the loss a long, thin valley (Step 4).
- **PCA** chases variance, and variance depends on units: metres to millimetres multiplies a column's variance by a million.
- **Trees** only ask "is $x_j < t$?". An order-preserving change of units moves $t$ but leaves the same split available, so trees need no scaling.

~~~dg-layers How much each model family cares about feature scale.
*k-NN, k-means, SVM with RBF, PCA | scale is part of the model: always scale
Ridge, lasso, logistic regression | the penalty depends on units: scale
Neural networks, SGD-trained models | conditioning and initialisation: scale
Trees, random forests, gradient boosting | only order matters: scaling is unnecessary
~~~

Unpenalised least squares gives the same predictions after rescaling (the weight compensates), but scaling still helps its numerical stability.

## Step 2: the three everyday scalers

A **scaler** maps each value $x$ of a column using a few statistics of that column. Three recipes cover almost every case.

**Standard scaling** (the z-score):

$$z = \frac{x - \mu}{\sigma},$$

where $\mu$ is the column's mean and $\sigma$ its standard deviation; $z = 2$ reads "two standard deviations above average".

**Min-max scaling**:

$$x' = \frac{x - \min}{\max - \min},$$

where $\min$ and $\max$ are the smallest and largest training values, so training values land in $[0, 1]$.

**Robust scaling**:

$$x' = \frac{x - \text{median}}{\text{IQR}},$$

where the **IQR** (interquartile range) is the 75th minus the 25th percentile, the width of the middle half of the data. Median and IQR barely move when a few extremes appear.

| Scaler | Formula | Good for | Weakness
| --- | --- | --- | --- |
| Standard | $(x-\mu)/\sigma$ | the default for most models | outliers inflate $\sigma$
| Min-max | $(x-\min)/(\max-\min)$ | bounded inputs, pixel values | one outlier compresses the rest
| Robust | $(x-\text{median})/\text{IQR}$ | heavy-tailed or dirty data | does not bound the range

For sparse data (bag-of-words counts) use **max-abs scaling**, which divides by the largest absolute value without centring, so zeros stay zero and the matrix stays sparse.

<div class="ex">
<div class="lab">Worked example</div>

**three scalers on a column with one outlier**

A column of order values has five training rows: $x = (10, 20, 30, 40, 200)$. The 200 is a genuine but unusual bulk order.

Standard: mean $\mu = (10 + 20 + 30 + 40 + 200)/5 = 60$. Deviations: $(-50, -40, -30, -20, 140)$. Squares: $2500 + 1600 + 900 + 400 + 19600 = 25000$. Variance $25000/5 = 5000$, so $\sigma = \sqrt{5000} \approx 70.7$. The z-scores are $(-0.71, -0.57, -0.42, -0.28, 1.98)$. The four ordinary rows are squeezed into a band only $0.42$ wide.

Min-max: $\min = 10$, $\max = 200$, range $190$. Results: $(0, 0.053, 0.105, 0.158, 1)$. The four ordinary rows now share a band $0.16$ wide; the outlier owns the rest of $[0, 1]$.

Robust: median $= 30$, 25th percentile $= 20$, 75th percentile $= 40$, so IQR $= 20$. Results: $(-1, -0.5, 0, 0.5, 8.5)$. The ordinary rows keep a healthy spread of $1.5$, and the outlier is honestly reported as 8.5 IQRs above the median.

Reading: one extreme value decides the scale for min-max entirely and for standard scaling mostly. Robust scaling lets the typical rows define the scale.

</div>

~~~dg-fig 07.01-scalers.svg
The same five values under each scaler. The thick bar under each row shows how much room the four ordinary values get: 0.16 with min-max, 0.42 with standard scaling, 1.5 with robust scaling.
~~~

Robust scaling does not remove the outlier; it only stops it setting the units. If extremes themselves hurt the model, add a transform from Step 5.

## Step 3: learn the statistics on training rows only

The mean, standard deviation, min, max, median and IQR are **learned state**, exactly like a model's weights. They must be computed on the training rows and then applied unchanged to validation rows, test rows and live production data.

~~~dg-flow Scaling statistics flow one way: from training rows to everything else.
Training rows -(fit: compute mean and std)-> *Scaler state -(transform)-> Scaled train, validation, test and live rows
~~~

Scaling the whole dataset before splitting leaks the test rows' mean and spread into training. For a scaler the leak is small, but the same habit with target encoding or feature selection (07.02, 07.04) can make a useless model look excellent. Put the scaler inside a pipeline so cross-validation refits it on each training fold (07.05).

Scaled validation data will therefore not have mean exactly 0 and spread exactly 1, which is correct. Live data can also leave the training range (a min-max value of 1.3); most models cope, but it is a drift signal worth an alarm (lesson 09.03).

## Step 4: why gradient descent needs it

Take a linear model with two features: one ranges from 0 to 1, the other from 0 to 10. The squared-error loss is a bowl over the two weights, but its curvature in each direction is proportional to that feature's variance. The second direction is about 100 times steeper than the first.

~~~dg-fig 07.01-gd.svg
Gradient descent with the same safe learning-rate rule on both panels. On raw features the loss is a thin valley and 40 steps barely move along it; after standardising the bowl is round and the same 40 steps reach the minimum.
~~~

The learning rate must be small enough not to overshoot in the steep direction, so the model crawls in the flat one. The ratio of steepest to flattest curvature is the **condition number**: about 109 here on raw features, 1.0 after standardising. Convergence time grows roughly in proportion to it, which is why neural networks almost always get standardised inputs.

## Step 5: fix skewed columns

Many real columns (income, counts, durations, prices) are positive with a long right tail. Scaling is a straight-line change and keeps that shape; you need a **non-linear transform** that pulls the tail in.

~~~dg-fig 07.01-skew.svg
2,000 simulated incomes. Raw, they pile up on the left with a long tail; after log(1+x) they form a roughly symmetric bell; a quantile transform forces an exact normal shape.
~~~

The toolbox, from simplest to most aggressive:

- $\log(1 + x)$ is the workhorse for non-negative counts and amounts; the $+1$ keeps zero at zero. It turns ratios into differences: 20k to 40k and 200k to 400k become the same step.
- **Box-Cox**: $x^{(\lambda)} = (x^\lambda - 1)/\lambda$ for $\lambda \ne 0$ and $\log x$ for $\lambda = 0$. $\lambda$ is chosen by maximum likelihood to make the result as Gaussian as possible ($\lambda = 1$: no change, $0.5$: about a square root, $0$: log). It requires $x > 0$.
- **Yeo-Johnson** applies a similar family to $x + 1$ for non-negative values and a mirrored one for negatives, so it accepts any real input, including zeros and negative balances. scikit-learn's `PowerTransformer` uses it by default and standardises afterwards.
- **Quantile transform** maps each value to its rank-based position (its empirical CDF) and then, optionally, through the inverse normal CDF. Any distribution becomes exactly uniform or Gaussian, and outliers are fully tamed. It is rank-based, so it throws away how far apart values were, and with few training rows the mapping is coarse.
- **Clipping or winsorising** at, say, the 1st and 99th training percentiles caps extremes without touching the bulk (on the simulated incomes it cuts skewness only from 2.65 to 1.94).

A linear model on log income says "each doubling adds the same amount", usually closer to the truth than "each euro adds the same amount".

<div class="ex">
<div class="lab">Worked example</div>

**income in a k-NN model**

Income in the training set: median 52,000, mean 71,000, standard deviation about 90,000, maximum 4.2 million.

After standardisation the richest customer sits at $z = (4{,}200{,}000 - 71{,}000)/90{,}000 \approx 46$. In Euclidean distance this one feature now outweighs the other 20 standardised features combined, so this customer has no meaningful neighbours, and everyone near the bulk is crowded together.

After $\log(1+x)$ the values run from about 9 to $\log(4{,}200{,}001) \approx 15.3$, with the median at $\log(52{,}001) \approx 10.9$ and a standard deviation near 0.8. The maximum is now about $(15.3 - 10.9)/0.8 \approx 5.4$ standard deviations out.

Standardising the logged feature gives a well-behaved column, and k-NN validation accuracy improves because distances now reflect all features.

</div>

## Step 6: binning, splines and circular features

**Binning** (discretisation) replaces a number with its bin, usually one-hot encoded so the model learns one weight per bin. **Equal-width** bins are easy to explain but nearly empty on skewed data; **quantile** bins hold equal counts.

Reasons to bin:

- It lets a linear model fit a non-linear, even non-monotone, effect (risk high for young and old drivers).
- It is robust to outliers and to small measurement errors.
- Regulated scorecards (credit) often require it for interpretability.

The costs: information inside a bin is lost, predictions jump at bin edges, and the edges are another choice you can overfit.

~~~dg-fig 07.01-bins.svg
Left: the same linear regression with three feature sets. Raw age gives a straight line that misses the U shape; six quantile bins give a staircase; a spline basis gives a smooth curve. Right: hour of day placed on a circle, so 23:00 and 1:00 are as close as 1:00 and 3:00.
~~~

**Splines** (`SplineTransformer`) replace one column with a handful of overlapping bump-shaped columns that a linear model adds into a smooth curve. Trees learn their own bins. Bin edges and spline knots, like scaler statistics, come from training data.

**Periodic features** (hour, weekday, month, wind direction) need different treatment: coded 0 to 23, hours 23 and 0 look maximally far apart. Encode the angle $\theta = 2\pi \cdot \text{hour}/24$ as two columns, $(\sin\theta, \cos\theta)$. Every hour becomes a point on a circle, and 23:00 sits right next to 0:00 and 1:00.

## Step 7: transforming the target

For a skewed target such as price or duration, fitting on $\log y$ often gives better relative errors and steadier residuals. If the model predicts $\widehat{\log y}$, then $\exp(\widehat{\log y})$ estimates the **median** of $y$, not the mean, because exponentiating preserves order (and so the middle value) but not averages.

If the residuals on the log scale are roughly normal with standard deviation $\sigma$, the mean is

$$\mathbb{E}[y] = \exp\!\left(\widehat{\log y} + \tfrac{\sigma^2}{2}\right).$$

With $\sigma = 0.5$ the correction factor is $\exp(0.125) \approx 1.133$, so the mean is 13% above the median.

~~~dg-fig 07.01-target.svg
Prices spread around one log-scale prediction. Exponentiating the prediction gives 200, the median; the mean is 13% higher at about 227. Summing medians undercounts a total.
~~~

For one house the median is often fine; for totals (revenue, claims) summing medians biases the result low. scikit-learn's `TransformedTargetRegressor` applies the forward and inverse transform for you, but not the bias correction.

## Scaling and transforming in code

First, the effect of scaling on k-NN with a real dataset. The wine data has 13 chemistry columns; one ranges from 0.13 to 0.66, another from 278 to 1680.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X, y = load_wine(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=5)
for name, model in [("raw", knn),
                    ("standard", make_pipeline(StandardScaler(), knn)),
                    ("min-max", make_pipeline(MinMaxScaler(), knn)),
                    ("robust", make_pipeline(RobustScaler(), knn))]:
    print(name, cross_val_score(model, X, y, cv=5).mean().round(3))
# raw 0.691   standard 0.949   min-max 0.95   robust 0.938

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
sc = StandardScaler().fit(X_tr)                  # statistics from training rows only
print(sc.transform(X_tr).std(0)[:3].round(3))    # [1. 1. 1.]
print(sc.transform(X_te).std(0)[:3].round(3))    # [1.043 0.97  1.037]  close, not exact
```

Inside `cross_val_score` the pipeline refits the scaler on each fold's training part. Next, skew transforms and a per-column preprocessing step:

```python
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
rng = np.random.default_rng(0)
income = rng.lognormal(mean=10.9, sigma=0.8, size=2000).reshape(-1, 1)
print(skew(income), skew(np.log1p(income)))                        # 2.65  -0.06
pt = PowerTransformer(method="box-cox").fit(income)
print(pt.lambdas_, skew(pt.transform(income)))                     # 0.025  0.00
qt = QuantileTransformer(output_distribution="normal", n_quantiles=500).fit(income)
print(skew(qt.transform(income)))                                  # 0.00

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
pre = ColumnTransformer([
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["income", "n_orders"]),
    ("yj", PowerTransformer(), ["balance_change"]),                      # can be negative
    ("bin", KBinsDiscretizer(n_bins=8, strategy="quantile", encode="onehot"), ["age"]),
], remainder=StandardScaler())
```

Box-Cox picked $\lambda \approx 0.025$, almost exactly a log, which makes sense: the data was generated as a log-normal.

## Checking the result

- **Describe the transformed training columns:** mean 0 and spread 1 after standard scaling; after a power or quantile transform, a histogram without the long tail.
- **Compare with and without under cross-validation.** Distance models show gaps like the wine 0.69 versus 0.95; trees should show none.
- **Check validation and live data against training ranges**, for example the share of rows with $|z| > 5$.
- **Compare weights only on a common scale:** on standardised inputs a weight means "effect of one standard deviation".

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Scaling before the train/test split | Validation score slightly too optimistic; habit spreads to worse leaks | Fit the scaler inside a pipeline on training folds only
| Forgetting to scale for k-NN, SVM, ridge, neural nets | One large-unit feature dominates; poor accuracy; slow or unstable training | Standardise; compare cross-validated scores
| Min-max on data with outliers | Most values crushed into a tiny band near 0 | Robust scaling, or clip or log first
| `log(x)` on a column containing zeros or negatives | `-inf` or `NaN` values, crashes downstream | `log1p` for zeros; Yeo-Johnson for negatives
| Refitting the scaler on each production batch | Same customer gets different features on different days | Save and reuse the fitted training scaler
| Hour or month coded as 0..23 or 1..12 | Midnight and 23:00 treated as far apart | $(\sin\theta, \cos\theta)$ encoding
| Exponentiating a log-target prediction and summing | Totals biased low by $\exp(\sigma^2/2)$ | Apply the bias correction or model the mean directly

## Going deeper

**Why the penalty cares about units.** Suppose feature $x$ has weight $w$. Rescale it as $x' = c\,x$ (say, metres to millimetres, $c = 1000$). The fit can use $w' = w/c$ with identical predictions, but the ridge penalty drops from $\lambda w^2$ to $\lambda w^2/c^2$: the same effect is a million times cheaper. Units silently set a separate regularisation strength per feature. Standardising makes "one unit" mean "one standard deviation" everywhere, so a single $\lambda$ treats features fairly.

**Conditioning, precisely.** For least squares with centred features, the loss curvature is the matrix $H = \tfrac{2}{n}X^\top X$. Gradient descent with step size $\eta < 2/\lambda_{\max}(H)$ shrinks the error along each eigen-direction by a factor $|1 - \eta\lambda_i|$ per step. The slowest direction contracts by about $1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$, where $\kappa$ is the condition number, so reaching a fixed accuracy takes on the order of $\kappa$ steps. Standardising equalises the diagonal of $H$; only if features are also uncorrelated does $\kappa \approx 1$, which is one reason adaptive optimisers and whitening exist.

**Box-Cox and the Jacobian.** Box-Cox picks $\lambda$ by maximising the normal log-likelihood of the transformed data. Because the transform stretches the axis, the likelihood includes a Jacobian term $(\lambda - 1)\sum_i \log x_i$; without it, every $\lambda$ that shrinks the data would look better. The family is continuous at $\lambda = 0$ because $(x^\lambda - 1)/\lambda \to \log x$ as $\lambda \to 0$.

**Quantile transforms and extrapolation.** A quantile transform is a monotone map fitted on training ranks; values beyond the training extremes are clipped to the edge, so the model cannot tell a value at the training maximum from one ten times larger.

## Key terms

| Term | Meaning
| --- | --- |
| Standardisation (z-score) | Subtract the mean, divide by the standard deviation: mean 0, spread 1
| Min-max scaling | Map the training minimum to 0 and maximum to 1
| Robust scaling | Subtract the median, divide by the interquartile range
| IQR | 75th percentile minus 25th percentile: the width of the middle half of the data
| Condition number | Ratio of steepest to flattest curvature of the loss; large means slow gradient descent
| Box-Cox / Yeo-Johnson | Power transforms with a learned exponent that make a column more Gaussian; Yeo-Johnson allows zeros and negatives
| Quantile transform | Map each value through its training rank to a uniform or normal shape
| Binning | Replace a number by the interval it falls in, usually one-hot encoded
| Cyclical encoding | Represent a periodic value as $(\sin\theta, \cos\theta)$ so the ends of the cycle meet

<div class="take">
<div class="lab">Key takeaway</div>

scale for models that measure distance, penalise weights or use gradients; squash skewed features with log, Box-Cox or Yeo-Johnson; bin only when you need what binning buys; and learn every statistic on the training fold.

</div>
