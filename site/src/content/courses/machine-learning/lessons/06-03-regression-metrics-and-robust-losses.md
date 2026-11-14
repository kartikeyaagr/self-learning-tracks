---
id: "06.03"
course: "machine-learning"
module: "06"
slug: "regression-metrics-and-robust-losses"
title: "Regression metrics and robust losses"
keyIdeas: ["Each error metric is minimised by a different summary of $y$: squared error by the **mean**, absolute error by the **median**, pinball loss by a **quantile**. Choose the metric for the quantity you actually need, and train with a loss that targets the same one.", "RMSE is in the target's units but is dominated by the largest errors; MAE weights every unit of error equally and is easier to explain. RMSE far above MAE means a few predictions are badly wrong.", "$R^2 = 1 - SS_{res}/SS_{tot}$ compares the model with predicting the mean of the evaluation targets. It is negative when the model is worse than that, and it changes with how spread out the test targets are, so do not compare it across datasets.", "MAPE is undefined at zero and rewards under-forecasting; for small or zero targets use WAPE or a scaled error such as MASE, which compares you with a naive forecast.", "**Huber** loss is quadratic for small residuals and linear beyond $\\delta$, so no single label can dominate the gradient. Use it (or MAE) for training when a few labels are corrupted or the noise has a heavy tail.", "Always report a metric next to a naive baseline, look at the worst rows, and break the error down by segment before trusting an average."]
question: "A delivery-time model reports test RMSE of 9.8 minutes, MAE of 4.1 minutes and R-squared of 0.62. A colleague proposes switching the training loss from MSE to MAE \"to improve MAE\". What do the numbers already tell you about the errors, what will the switch change about the predictions, and what would you check before deciding?"
answer: "The RMSE/MAE ratio is about 2.4, so the errors have a heavy tail: most deliveries are predicted within a few minutes, while a minority are missed by a lot (traffic, failed hand-offs, restaurants running late). The R-squared of 0.62 only says the model removes 62% of the squared error of predicting the average delivery time on this test set; it says nothing about whether 4 minutes is good enough, and it would change on a test set with a different spread of times.\n\nSwitching the training loss to MAE changes *what* the model predicts: the conditional **median** instead of the conditional **mean**. Delivery times are right-skewed, so the median is below the mean and predictions will shift down. MAE will probably improve a little; RMSE will probably get worse, and long deliveries will be under-predicted even more. Neither is wrong in itself. The question is what the number is used for: a customer-facing typical ETA fits the median; staffing and capacity planning need the mean, because totals of means add up; a promised \"arrives by\" time that should hold 90% of the time needs a quantile model trained with pinball loss at tau = 0.9.\n\nBefore deciding, I would (1) sort the test set by absolute error and read the worst rows, to see whether the tail comes from corrupted labels (fix them or train with Huber), from missing features such as weather or restaurant load (add them), or from genuine randomness; (2) break MAE and RMSE down by segment (time of day, distance, restaurant) to see where the tail lives; (3) compare both metrics with a naive baseline such as the median time per restaurant; and (4) agree with the product owner which summary the prediction should represent, then pick the training loss and the headline metric to match."
sources: [{"title": "scikit-learn: Regression metrics", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html", "note": "The regression metrics section: definitions of R-squared, MAPE, pinball loss and which statistic each one targets."}, {"title": "Forecasting: Principles and Practice, 3rd ed. (Hyndman and Athanasopoulos): Evaluating point forecast accuracy", "url": "https://otexts.com/fpp3/accuracy.html", "note": "Scale-dependent, percentage and scaled errors, and why MASE is preferred."}, {"title": "Making and evaluating point forecasts (Gneiting)", "url": "https://arxiv.org/abs/0912.0902", "note": "The theory behind matching the scoring function to the functional (mean, median, quantile) you want."}, {"title": "scikit-learn: Robustness regression", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The robustness regression section: Huber, RANSAC and Theil-Sen regressors and when each breaks down."}]
practice: ["regression-metrics", "loss-functions", "standard-error-comparison", "percentiles-quantiles"]
tags: ["regression metrics", "robust loss"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- The everyday regression metrics (MAE, MSE, RMSE, $R^2$, MAPE and friends): what each one measures, in which units, and what it hides.
- Why every error metric secretly asks for a different number: squared error wants the **mean**, absolute error the **median**, pinball loss a **quantile**.
- How to read $R^2$ without being fooled, including why it can be negative and why it changes when the test set changes.
- When percentage errors break, and the scaled alternatives (WAPE, MASE) forecasters use instead.
- How robust losses such as **Huber** stop a handful of bad labels from steering the whole model.

*Before you start:* you need residuals, MSE and $R^2$ from linear regression (lesson 02.01) and the table of losses in lesson 01.01. The two previous lessons of this module (06.01, 06.02) did the same job for classification; no knowledge of them is required here.

## From labels to numbers

Lessons 06.01 and 06.02 judged models that output a class or a score. Regression models output a quantity (minutes, prices, demand), so a prediction is never simply right or wrong; it is off by some amount. There are several sensible ways to add those amounts up, and they can rank the same two models in opposite orders.

## The idea in one picture

A food-delivery company compares two models that predict delivery time. On eight test orders their **errors** (actual minus predicted, in minutes) are:

| Order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A | 1 | -2 | 1 | -1 | 2 | -1 | 1 | 20
| Model B | 5 | -5 | 4 | -4 | 5 | -5 | 4 | -4

~~~dg-fig 06.03-twomodels.svg
Model A is nearly perfect on seven orders and badly wrong on one; model B is mediocre on every order. Which is "better" depends on how you average the errors.
~~~

Two natural ways to summarise them:

- **Mean absolute error (MAE)**: drop the signs and average. Model A: $(1+2+1+1+2+1+1+20)/8 = 29/8 = 3.6$ minutes. Model B: $36/8 = 4.5$ minutes. **A wins.**
- **Root mean squared error (RMSE)**: square, average, take the square root. Model A: $\sqrt{413/8} = \sqrt{51.6} = 7.2$. Model B: $\sqrt{164/8} = \sqrt{20.5} = 4.5$. **B wins.**

Neither number is wrong; they answer different questions. MAE asks "how far off is a typical order?", and A is closer most of the time. RMSE squares before averaging, so it charges 400 for A's single 20-minute miss and only 16 to 25 for each of B's medium misses. If a 20-minute surprise means a refund, RMSE's verdict matches the business; if the usual case matters most, MAE's does. Choosing a metric decides which model you ship.

## Step 1: the error-based metrics

Write $y_i$ for the true value of example $i$, $\hat y_i$ for the prediction, and $r_i = y_i - \hat y_i$ for the **residual** (the error). With $n$ test examples:

$$\text{MAE} = \frac1n\sum_{i=1}^n |r_i|, \qquad \text{MSE} = \frac1n\sum_{i=1}^n r_i^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}.$$

- **MAE** is in the target's units (minutes, thousands). It treats every minute of error the same.
- **MSE** is in squared units (minutes²), which nobody can picture. It is the usual *training* loss because it is smooth.
- **RMSE** brings MSE back to the target's units, so you can compare it with MAE.

A useful fact: always $\text{MAE} \le \text{RMSE}$, with equality only when every error has the same size. Check model B: errors of size 4 and 5 give MAE 4.5 and RMSE 4.53, almost equal. Model A: MAE 3.6, RMSE 7.2, a ratio of 2. **The RMSE/MAE ratio is a cheap tail detector**: near 1, errors are uniform; 2 or more, a few predictions are much worse than the rest and you should look at them one by one.

## Step 2: each metric asks for a different number

Imagine orders that look identical to the model (same restaurant, same distance) but whose actual times vary: mostly 20 to 30 minutes, sometimes 50. The model can output only one number $c$ for all of them. Which $c$ is best depends on the loss.

~~~dg-fig 06.03-targets.svg
Left: delivery times are skewed to the right, so the median, mean and 90th percentile are three different numbers. Right: the average loss of every constant prediction; each loss bottoms out at a different one of them.
~~~

The figure shows the answer for 4,000 simulated orders:

- Average **squared error** is lowest at the **mean** (27.2 min).
- Average **absolute error** is lowest at the **median** (25.0 min).
- The **pinball loss** with $\tau = 0.9$ is lowest at the **90th percentile** (38.8 min).

The pinball (or quantile) loss is the asymmetric cousin of absolute error:

$$\rho_\tau(r) = \max\big(\tau\, r,\; (\tau - 1)\, r\big), \qquad r = y - \hat y.$$

Here $\tau$ (tau) is a number between 0 and 1. When the model under-predicts ($r > 0$) the cost is $\tau r$; when it over-predicts ($r < 0$) the cost is $(1-\tau)|r|$. With $\tau = 0.9$, being 5 minutes short costs $0.9 \times 5 = 4.5$, while being 5 minutes long costs $0.1 \times 5 = 0.5$. Under-predicting is 9 times as expensive, so the best constant moves up until only 10% of orders exceed it. With $\tau = 0.5$ the loss is half the absolute error and you get the median back.

So a model trained with MSE predicts a conditional **mean**, one trained with MAE a conditional **median**, and one trained with pinball loss at 0.9 a conditional 90th percentile. Train with MSE and grade with MAE, and you aim at the mean while scoring against the median. For symmetric noise they coincide; for skewed targets (delivery times, claims, prices) they can differ a lot.

~~~dg-layers Pick the summary of y you need, then train and evaluate with the loss that targets it.
Squared error, RMSE | targets the conditional mean; punishes large misses hardest
*Absolute error, MAE | targets the conditional median; robust to outliers
Pinball loss at tau | targets a quantile; asymmetric costs, prediction intervals
Huber at delta | mean-like near zero, median-like in the tails
~~~

Choose by asking what the number will be used for. Total staffing hours need the mean (sums of means are means of sums). A customer-facing "usually arrives by" estimate fits the median. A promise you must keep 90% of the time is a quantile. Two pinball models at $\tau = 0.05$ and $0.95$ together give a 90% **prediction interval**.

<div class="ex">
<div class="lab">Worked example</div>

**pinball loss by hand**

An order actually takes $y = 30$ minutes. Score three predictions with $\tau = 0.9$.

Predict 25: residual $r = 30 - 25 = 5 > 0$ (too low), so loss $= 0.9 \times 5 = 4.5$.

Predict 35: residual $r = -5 < 0$ (too high), so loss $= (0.9 - 1) \times (-5) = 0.5$.

Predict 30: residual 0, loss 0.

Same 5-minute miss, 9 times the price when it is on the short side. Averaged over many orders, this pushes predictions up to the point where 90% of actual times fall below them.

</div>

## Step 3: $R^2$, the relative score

MAE and RMSE are absolute: "4.5 minutes". Often you also want a relative number: how much better than doing nothing clever? The **coefficient of determination** compares your squared error with that of a trivial model that always predicts the mean $\bar y$ of the evaluation targets:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$$

$SS_{res}$ is the sum of squared residuals of your model; $SS_{tot}$ is the sum of squared residuals of the "predict the mean" baseline. Dividing both by $n$ shows $R^2 = 1 - \text{MSE} / \text{Var}(y)$.

- $R^2 = 1$: perfect predictions.
- $R^2 = 0$: no better than predicting the mean.
- $R^2 < 0$: **worse** than predicting the mean. On training data an OLS fit with an intercept cannot do this; on new data any model can.

Number check on the eight orders (actual times 22, 35, 28, 41, 30, 25, 33, 60; mean 34.25): A scores $R^2 = 0.59$, B $0.84$; being a rescaled MSE, $R^2$ sides with RMSE. Always predicting 20 minutes scores $-1.62$.

The trap is the denominator. $SS_{tot}$ depends on how spread out the test targets are, not on the model.

~~~dg-fig 06.03-r2.svg
The same model with the same noise (RMSE about 15 in both panels). On flats from the whole country R² is 0.98; on flats from one district it falls to 0.20, only because the targets vary less.
~~~

The typical miss is 15 thousand on both test sets, but the narrow one has a small $SS_{tot}$, so the same errors look relatively large. Consequences:

- Do not compare $R^2$ across datasets, regions or time periods. Compare RMSE or MAE.
- A high $R^2$ does not mean errors are small enough. $R^2 = 0.9$ on house prices can still mean a typical miss of 40,000.

## Step 4: percentage and scaled errors

Raw errors are hard to compare across products: 5 units off is excellent for a bestseller and terrible for a slow seller. Percentages fix that:

$$\text{MAPE} = \frac{100\%}{n} \sum_i \frac{|y_i - \hat y_i|}{|y_i|}.$$

"Off by 8% on average" is easy to communicate, and on the eight orders model A scores 7.9%, model B 14.6%. But MAPE has two real defects:

- **It divides by $y$.** At $y = 0$ it is undefined, and near zero a tiny miss becomes a huge percentage. Demand data is full of zeros.
- **It is lopsided.** For a true value of 100, predicting 0 costs 100%, while predicting 300 costs 200%. Under-forecasts are capped at 100%; over-forecasts are not. Models tuned on MAPE learn to forecast low.

Two safer relatives:

- **WAPE** (weighted absolute percentage error) $= \sum_i |y_i - \hat y_i| \,/\, \sum_i |y_i|$: total absolute error over total volume. One division over the whole set, so single small $y$ values cannot blow it up. Model A: $29 / 274 = 10.6\%$.
- **MASE** (mean absolute scaled error): your MAE divided by the in-sample MAE of a naive forecast, such as "same as yesterday" or "same as last week". MASE below 1 means you beat the naive method; it is unit-free and works across series of different scales. Hyndman and Athanasopoulos recommend it as a default for forecasting.

For positive targets whose errors are naturally multiplicative (prices, populations), **RMSLE**, the RMSE of $\log(1 + y)$ against $\log(1 + \hat y)$, treats "20% too high" the same whether the price is 100 or 100,000.

## Step 5: robust losses for training

Metrics judge a model; losses train it, and the same tail problem appears. The gradient of $\tfrac12 r^2$ is $r$, so a residual of 20 pulls the fit 20 times as hard as a residual of 1. If a few labels are corrupted (a price typed in cents, a sensor glitch), those rows steer the whole model.

The **Huber loss** is quadratic for small residuals and linear for large ones. The switch point is a threshold $\delta$ (delta) in the target's units:

$$L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\,\big(|r| - \tfrac12\delta\big) & |r| > \delta \end{cases}$$

The two pieces meet smoothly at $|r| = \delta$. Its gradient is $r$ inside $[-\delta, \delta]$ and $\pm\delta$ outside: ordinary errors are treated as in least squares, and any one point's pull is capped.

~~~dg-fig 06.03-losses.svg
Left: the loss each residual pays. Right: the gradient, that is how hard that residual pulls the fit. Squared error's pull grows without limit; Huber's stops growing at delta.
~~~

Here is the effect on a straight-line fit with 60 points, five of whose labels were shifted down by 20:

~~~dg-fig 06.03-huber.svg
Five bad labels drag least squares down and halve its slope (0.72 instead of 1.5). The Huber fit barely moves (slope 1.41).
~~~

Choose $\delta$ relative to the noise scale: with $\delta \approx 1.35\,\sigma$ ($\sigma$ a robust estimate of the noise standard deviation), Huber keeps about 95% of least squares' efficiency under Gaussian noise. scikit-learn's `HuberRegressor` uses `epsilon=1.35` times a scale it estimates jointly with the weights. **Log-cosh** loss, $\log\cosh(r)$, is a smooth relative with similar behaviour. Gradient boosting libraries accept `loss="huber"`, `"absolute_error"` or `"quantile"` directly.

<div class="ex">
<div class="lab">Worked example</div>

**one outlier, two stories**

Five house-price errors, in thousands: 10, -10, 20, -20, 200.

MAE $= (10 + 10 + 20 + 20 + 200)/5 = 260/5 = 52$.

MSE $= (100 + 100 + 400 + 400 + 40{,}000)/5 = 41{,}000/5 = 8{,}200$, so RMSE $= \sqrt{8{,}200} = 90.6$.

Without the last house: MAE $= 60/4 = 15$; MSE $= 1{,}000/4 = 250$, RMSE $= 15.8$.

One listing multiplied RMSE by almost 6 and MAE by 3.5. Before choosing between models, look at that row. If it is a data error (price recorded in the wrong unit), fix or drop it. If it is a real mansion, the model needs features for the luxury segment, or the business needs a separate model for it.

</div>

## Regression metrics in code

scikit-learn has every metric above in `sklearn.metrics` (`root_mean_squared_error` exists from version 1.4). The eight orders from the opening:

```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                             mean_absolute_percentage_error, mean_pinball_loss)

y      = np.array([22, 35, 28, 41, 30, 25, 33, 60.])     # actual delivery times (min)
pred_a = y - np.array([1, -2, 1, -1, 2, -1, 1, 20])      # model A
pred_b = y - np.array([5, -5, 4, -4, 5, -5, 4, -4])      # model B

for name, p in [("A", pred_a), ("B", pred_b)]:
    print(name,
          mean_absolute_error(y, p),                     # A 3.625   B 4.5
          root_mean_squared_error(y, p),                 # A 7.19    B 4.53
          r2_score(y, p),                                # A 0.588   B 0.837
          mean_absolute_percentage_error(y, p),          # A 0.079   B 0.146 (a fraction, not %)
          mean_pinball_loss(y, p, alpha=0.9))            # A 2.86    B 2.25
```

Note that `mean_absolute_percentage_error` returns a fraction (0.079 means 7.9%), and `mean_pinball_loss` calls $\tau$ `alpha`. Fitting with a robust loss is a one-line change:

```python
from sklearn.linear_model import LinearRegression, HuberRegressor

rng = np.random.default_rng(11)
x = np.sort(rng.uniform(0, 10, 60)); y = 2 + 1.5 * x + rng.normal(0, 1, 60)
y[[45, 49, 52, 56, 58]] -= 20                            # five corrupted labels
X = x.reshape(-1, 1)

print(LinearRegression().fit(X, y).coef_)                # [0.72]  dragged by 5 rows
hub = HuberRegressor(epsilon=1.35).fit(X, y)
print(hub.coef_, hub.scale_)                             # [1.41]  0.67
print(hub.outliers_.sum())                               # 24
```

Gotcha: `outliers_` flags every point outside the quadratic zone (here the five bad rows plus 19 clean ones just past the bend), not a list of bad labels.

## Reading and checking the result

A metric value on its own means little. Before trusting it:

1. **Put a baseline next to it.** Report the same metric for "predict the training mean" (or a naive forecast for time series). "MAE 4.1 min" means something once you know the baseline scores 9.0.
2. **Report two numbers.** MAE plus RMSE shows both the typical miss and the tail.
3. **Look at the worst rows.** Sort by absolute residual and read the top 20: data errors, missing features and genuinely unpredictable cases each need a different fix.
4. **Slice.** Report error by segment (price band, region, new vs returning) and plot residuals against key features. Good overall MAE can hide a 15% under-prediction on the most expensive decile.
5. **Score on the original scale.** If you trained on $\log y$, evaluate after transforming back (lesson 07.01 covers the back-transform bias).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Training loss and metric target different summaries | model "improves" in training but the reported metric stalls | train with the loss that matches the metric (MAE, pinball), or change the metric
| Comparing $R^2$ across test sets | a model looks worse in one region though its RMSE is the same | compare RMSE or MAE; quote $R^2$ only within one test set
| MAPE with zeros or tiny targets | infinite or enormous MAPE, models that forecast low | WAPE, MASE, or MAE per segment
| RMSE dominated by a few rows | RMSE far above MAE, metric jumps when one row is removed | inspect those rows; fix data errors; consider Huber
| Reporting a number with no baseline | "RMSE 12" that nobody can judge | add the naive baseline's score in the same metric
| Using a robust loss to hide real structure | tail errors stay large on a clear segment | add features or a separate model for that segment

## Going deeper

**Why the mean, median and quantile.** For a constant prediction $c$, the expected squared error $E[(Y - c)^2]$ has derivative $-2(E[Y] - c)$, which is zero at $c = E[Y]$. For absolute error, the derivative of $E|Y - c|$ is $P(Y < c) - P(Y > c)$: moving $c$ up by a little helps every point above it and hurts every point below it, so the optimum is where half the probability is on each side, the median. For pinball loss the derivative is $P(Y < c) - \tau$, zero at the $\tau$-quantile. Applied separately at every input $x$, these statements become "conditional mean", "conditional median" and "conditional quantile".

**Consistent scoring functions.** Gneiting (2011) formalised this: a scoring function is *consistent* for a summary (a "functional") if predicting the true value of that summary minimises expected score. Squared error is consistent for the mean, absolute error for the median, pinball for quantiles. The whole Bregman family (which includes squared error and Poisson deviance) is consistent for the mean. MAPE is consistent for a weighted median with weights $1/y$, which sits below the median, which is the formal reason it rewards under-forecasting. Some summaries, such as the variance alone or expected shortfall, have no consistent scoring function at all.

**Huber as an M-estimator.** Setting the gradient to zero gives $\sum_i \psi(r_i)\, x_i = 0$ with $\psi(r) = \text{clip}(r, -\delta, \delta)$, which is the least-squares condition with each residual's influence capped. It can be solved by iteratively reweighted least squares: points with $|r| > \delta$ get weight $\delta / |r|$. Huber fixes bad *labels*. It does not protect against bad *features* (high-leverage points with odd $x$ values); for those, RANSAC or Theil-Sen are designed to cope with a larger fraction of contamination, at higher cost.

## Key terms

| Term | Meaning
| --- | --- |
| Residual | Actual minus predicted value, $r = y - \hat y$
| MAE | Average absolute error, in target units; targets the median
| RMSE | Square root of average squared error; targets the mean, weights big misses heavily
| $R^2$ | $1 - SS_{res}/SS_{tot}$: improvement over predicting the mean of the evaluation targets
| Pinball (quantile) loss | Asymmetric absolute loss whose minimiser is the $\tau$-quantile
| MAPE / WAPE | Average percentage error per row / total absolute error over total volume
| MASE | MAE divided by the MAE of a naive forecast; below 1 beats the baseline
| Huber loss | Squared for $|r| \le \delta$, linear beyond, so one point's pull is capped

<div class="take">
<div class="lab">Key takeaway</div>

a regression metric decides which number your model aims at (mean, median or quantile) and which errors count most, so pick it for the decision the prediction serves, train with a matching loss, report it in the target's units next to a naive baseline, and switch to a robust loss like Huber when a few bad labels dominate.

</div>
