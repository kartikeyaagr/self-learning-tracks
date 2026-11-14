---
id: "06.03"
course: "machine-learning"
module: "06"
slug: "regression-metrics-and-robust-losses"
title: "Regression metrics and robust losses"
keyIdeas: ["Each error metric is minimised by a different summary of $y \\mid x$: squared error by the **mean**, absolute error by the **median**, pinball loss by a **quantile**. Choose the metric for the quantity you actually want to predict.", "RMSE is in the target's units but is dominated by the largest errors; MAE is more robust and easier to explain. A large gap between RMSE and MAE signals a heavy error tail.", "$R^2 = 1 - SS_{res}/SS_{tot}$ compares the model to predicting the mean of the evaluation set. It is negative when the model is worse than that, and it depends on how varied the evaluation targets are.", "Percentage errors (MAPE) are undefined at zero and punish over-forecasts more than under-forecasts; use them only for strictly positive targets far from zero, or prefer a scaled error.", "**Huber** loss is quadratic for small residuals and linear beyond $\\delta$, so outliers cannot dominate the gradient. Train with a robust loss when the label noise has a heavy tail."]
question: "A delivery-time model reports test RMSE of 9.8 minutes and MAE of 4.1 minutes, and R-squared of 0.62. A colleague proposes switching the training loss from MSE to MAE \"to improve MAE\". What do the numbers already tell you about the errors, what will the switch change about the predictions, and what would you check before deciding?"
answer: "RMSE is more than twice MAE, so the error distribution has a heavy tail: most deliveries are predicted within a few minutes, while a minority are missed by a lot (traffic, failed hand-offs). Training with MAE makes the model predict the conditional **median** instead of the mean; for right-skewed delivery times that shifts predictions down, which will likely improve MAE but worsen RMSE and systematically under-promise the long deliveries. Which one is right depends on the product: a customer-facing ETA may prefer the median, while staffing plans need the mean, and a \"promise\" time needs a high quantile (pinball loss at 0.9). Before deciding, look at residuals by segment to see whether the tail is predictable (missing features) or noise, check whether a few corrupted labels drive RMSE, and compare against a naive baseline in both metrics."
sources: [{"title": "scikit-learn: Regression metrics", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html", "note": "The regression metrics section: definitions of R-squared, MAPE, pinball loss and which statistic each one targets."}, {"title": "Forecasting: Principles and Practice, 3rd ed. (Hyndman and Athanasopoulos): Evaluating point forecast accuracy", "url": "https://otexts.com/fpp3/accuracy.html", "note": "Scale-dependent, percentage and scaled errors, and why MASE is preferred."}, {"title": "Making and evaluating point forecasts (Gneiting)", "url": "https://arxiv.org/abs/0912.0902", "note": "The theory behind matching the scoring function to the functional (mean, median, quantile) you want."}, {"title": "scikit-learn: Robustness regression", "url": "https://scikit-learn.org/stable/modules/linear_model.html", "note": "The robustness regression section: Huber, RANSAC and Theil-Sen regressors and when each breaks down."}]
practice: ["regression-metrics", "loss-functions", "standard-error-comparison", "percentiles-quantiles"]
tags: ["regression metrics", "robust loss"]
tools: ["NumPy", "scikit-learn"]
---

## What each metric is asking for

For regression the choice of metric is a choice of *which number* the model should output. Given the conditional distribution of $y$ at a point $x$:

- Minimising expected **squared error** gives the conditional **mean**.
- Minimising expected **absolute error** gives the conditional **median**.
- Minimising the **pinball (quantile) loss** $\rho_\tau(r) = \max(\tau r, (\tau - 1) r)$ with $r = y - \hat y$ gives the conditional $\tau$-quantile. With $\tau = 0.9$, under-predictions cost 9 times more than over-predictions, which is what you want for "how much stock covers 90% of days".

So if you train with MSE but evaluate with MAE, you are optimising for the mean and grading on the median. For skewed targets (delivery times, claim sizes, house prices) those differ a lot. The loss table in lesson 01.01 listed these; this lesson is about using them to judge models.

~~~dg-layers Pick the metric for the summary of y you need, then train with a loss that targets the same summary.
Squared error, RMSE | targets the conditional mean; punishes large misses hardest
*Absolute error, MAE | targets the conditional median; robust to outliers
Pinball loss at tau | targets a quantile; asymmetric costs, prediction intervals
Huber at delta | mean-like near zero, median-like in the tails
~~~

## The everyday metrics

| Metric | Formula | Units | Notes
| --- | --- | --- | --- |
| MSE | $\frac1n\sum (y_i-\hat y_i)^2$ | squared | the usual training loss; hard to interpret
| RMSE | $\sqrt{\text{MSE}}$ | target | one large miss can dominate
| MAE | $\frac1n\sum \lvert y_i-\hat y_i\rvert$ | target | median-oriented, readable
| $R^2$ | $1 - \sum (y_i-\hat y_i)^2 / \sum (y_i-\bar y)^2$ | none | share of variance explained
| MAPE | $\frac{100}{n}\sum \lvert y_i-\hat y_i\rvert / \lvert y_i\rvert$ | % | breaks at $y=0$, asymmetric
| RMSLE | RMSE of $\log(1+y)$ vs $\log(1+\hat y)$ | ratio-like | for multiplicative errors on positive targets

Always have $\text{MAE} \le \text{RMSE}$; equality means every error has the same size. The ratio tells you about the tail: if RMSE is three times MAE, a few predictions are badly wrong and deserve a look.

## Reading $R^2$ correctly

$R^2$ compares your model with the constant predictor that outputs the mean of the *evaluation* targets. Three consequences:

1. On held-out data it can be **negative**: the model is worse than that constant. Training $R^2$ of an OLS fit with an intercept is never negative; test $R^2$ can be.
2. It **depends on the spread of $y$** in the evaluation set. The same model has higher $R^2$ on a test set covering all of a country than on one city, because $SS_{tot}$ is larger. Comparing $R^2$ across datasets is usually meaningless; compare RMSE or MAE instead.
3. A high $R^2$ does not mean small errors in units that matter. Predicting house prices with $R^2 = 0.9$ can still mean a typical miss of 40,000.

## Percentage and scaled errors

MAPE is popular because "off by 8%" is easy to communicate, but it has two defects. It divides by $y$, so it explodes near zero and is undefined at zero (common in demand data). And it is asymmetric: for a true value of 100, predicting 0 costs 100% while predicting 300 costs 200%, so models tuned on MAPE learn to under-forecast. Alternatives:

- **WAPE** $= \sum \lvert y - \hat y\rvert / \sum \lvert y\rvert$: one ratio over the whole set, stable when individual $y$ values are small.
- **MASE**: MAE divided by the in-sample MAE of a naive forecast (last value, or same day last week). Below 1 means you beat the naive baseline; it works across series of different scales. Hyndman's forecasting textbook makes this the default for time series.

## Robust losses for training

Squared error has gradient $2r$: a residual of 100 pushes the fit 100 times harder than a residual of 1. If 1% of labels are corrupted (a misplaced decimal point, a sensor glitch), those rows steer the whole model. The **Huber** loss caps that influence:

$$L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\,(|r| - \tfrac12\delta) & |r| > \delta \end{cases}$$

Its gradient is $r$ inside $[-\delta, \delta]$ and $\pm\delta$ outside, so it behaves like MSE for ordinary errors and like MAE for outliers. A common choice of $\delta$ is around 1.35 times a robust estimate of the noise scale, which keeps about 95% efficiency when the noise is in fact Gaussian. Log-cosh is a smooth relative. For linear models, scikit-learn offers `HuberRegressor`; gradient boosting libraries take `loss="huber"` or `"quantile"`.

```python
import numpy as np
err = y - y_hat
mae, rmse = np.abs(err).mean(), np.sqrt((err ** 2).mean())
r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
tau = 0.9
pinball = np.maximum(tau * err, (tau - 1) * err).mean()
```

<div class="ex">
<div class="lab">Worked example</div>

**one outlier, two stories**

Five house-price errors, in thousands: 10, -10, 20, -20, 200. MAE = 260/5 = 52; MSE = (100 + 100 + 400 + 400 + 40,000)/5 = 8,200, so RMSE = 90.6.

Without the last house: MAE = 15, RMSE = 15.8. One listing moved RMSE by a factor of almost 6 and MAE by 3.5. Before deciding which model is better, look at that row: a data error (price recorded in cents) should be fixed or dropped; a real mansion means the model needs features for the luxury segment, or the business needs a separate model for it.

</div>

## Evaluate in slices

A single average hides structure. Plot residuals against the prediction and against key features, and report error by segment (price band, region, new vs returning). A model with good overall MAE that systematically under-predicts the most expensive decile is a pricing risk. If you train on a transformed target such as $\log y$, report metrics on the original scale, since that is the scale users care about (lesson 07.01 explains the back-transform bias).

<div class="take">
<div class="lab">Key takeaway</div>

pick the regression metric for the quantity you want (mean, median or quantile), quote it in the target's units next to a naive baseline, and use a robust loss when the label noise has a heavy tail.

</div>
