import numpy as np

META = dict(title='Capstone: backtest a demand forecaster against seasonal naive', tags=['Time Series', 'Metrics & Evaluation'], diff='Hard', runtime='numpy',
            related=['seasonal-naive-forecast', 'mase', 'lag-features', 'walk-forward-splits', 'recursive-multistep-forecast'])

STATEMENT = r"""
Backtest a linear autoregressive model on hourly demand with walk-forward validation, and compare it with the seasonal naive baseline.

Write `backtest(y, lags, season, initial, horizon, step, gap, window, ridge)`:

1. **Folds** as in the walk-forward step: `[train_start, train_end, test_start, test_end]`.
2. **Fit** (per fold). Training rows are $t$ from `train_start + max(lags)` to `train_end - 1`, with features $y_{t-\ell}$ for each lag and target $y_t$. Fit an intercept and weights by ridge regression: minimise $\lVert y - c - Xw\rVert^2 + \lambda\lVert w\rVert^2$ (no penalty on $c$), solved exactly.
3. **Forecast.** From the history `y[:train_end]`, forecast `gap + horizon` steps recursively and keep the last `horizon` (the gap hours are forecast but not scored, as in deployment). Do the same with the seasonal naive method.
4. **Score** both with MASE, using `y[train_start:train_end]` as the training series and `season` for the scale.

Return a dict with `model_mase` and `naive_mase` (lists per fold), and `mean_model` and `mean_naive`.
"""

STARTER = '''import numpy as np

def backtest(y, lags, season, initial, horizon, step, gap, window, ridge):
    """Walk-forward backtest of a ridge AR model with recursive forecasts, scored by MASE against seasonal naive."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _folds(n, initial, horizon, step, gap, window):
    out, e = [], initial
    while e + gap + horizon <= n:
        out.append((0 if window is None else max(0, e - window), e, e + gap, e + gap + horizon))
        e += step
    return out

def _fit(y, lags, a, b, lam):
    L = max(lags)
    t = np.arange(a + L, b)
    X = np.stack([y[t - l] for l in lags], 1)
    A = np.hstack([np.ones((len(t), 1)), X])
    P = lam * np.eye(A.shape[1]); P[0, 0] = 0.0
    theta = np.linalg.solve(A.T @ A + P, A.T @ y[t])
    return theta[0], theta[1:]

def _recursive(hist, lags, c, w, h):
    v = list(hist)
    for _ in range(h):
        v.append(c + sum(wi * v[-l] for wi, l in zip(w, lags)))
    return np.array(v[len(hist):])

def _mase(yt, yp, ytr, m):
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(ytr[m:] - ytr[:-m])))

def backtest(y, lags, season, initial, horizon, step, gap, window, ridge):
    y = np.asarray(y, float)
    mm, nm = [], []
    for a, b, s, e in _folds(len(y), initial, horizon, step, gap, window):
        c, w = _fit(y, lags, a, b, ridge)
        pred = _recursive(y[:b], lags, c, w, gap + horizon)[gap:]
        last = y[b - season:b]
        naive = np.array([last[j % season] for j in range(gap + horizon)])[gap:]
        mm.append(_mase(y[s:e], pred, y[a:b], season))
        nm.append(_mase(y[s:e], naive, y[a:b], season))
    return {"model_mase": mm, "naive_mase": nm, "mean_model": float(np.mean(mm)), "mean_naive": float(np.mean(nm))}
'''

HINTS = [
    'Training rows start max(lags) after train_start so every lag stays inside the training window.',
    'Augment X with a column of ones and put zero in the penalty matrix for that column.',
    'Forecast through the gap and drop those steps; starting the forecast at test_start would use data the deployed model never has.',
]

def _demand(seed, days, noise, trend):
    g = np.random.default_rng(seed)
    t = np.arange(24 * days)
    hour, dow = t % 24, (t // 24) % 7
    daily = 12 * np.sin(2 * np.pi * (hour - 7) / 24) + 5 * np.sin(4 * np.pi * hour / 24)
    weekly = np.where(dow >= 5, -8.0, 0.0)
    ar = np.zeros(len(t))
    for i in range(1, len(t)):
        ar[i] = 0.8 * ar[i - 1] + g.normal(0, noise)
    return 100 + trend * t / 24 + daily + weekly + ar

_Y1 = _demand(1, 35, 1.5, 0.1)
_Y2 = _demand(2, 35, 3.0, 0.0)

TESTS = [
    (_Y1, [1, 2, 24, 168], 168, 24 * 21, 24, 24, 0, None, 1.0),
    (_Y1, [24, 168], 168, 24 * 21, 24, 24, 0, None, 1.0),
    (_Y1, [1, 2, 24, 168], 168, 24 * 21, 24, 24, 12, None, 1.0),
    (_Y2, [1, 24, 168], 24, 24 * 21, 24, 48, 0, 24 * 14, 10.0),
    (_Y2, [1, 2, 3], 168, 24 * 21, 24, 24, 0, None, 1.0),
]

EXPLANATION = r"""
## The pieces in one loop
Every step appears here: lag features built only from the past, walk-forward folds with an optional gap, a recursive multi-step forecast, the seasonal naive baseline, and MASE so the two are scored on the same scale.

## What the tests show
- **The baseline is hard to beat.** On the first series the weekly seasonal naive forecast scores a mean MASE of about 1.20. The AR model with lags 1, 2, 24 and 168 scores about 1.27, and with only the seasonal lags 24 and 168 about 1.20. Over a 24-hour horizon the short lags help for the first few hours, then their errors feed back into later hours.
- **The gap costs accuracy.** Twelve unscored hours in front of each test day (the third test) push the model to about 1.32, because the short lags must now come entirely from its own predictions.
- **Beating the right baseline.** The fourth test scores against the daily seasonal naive method (repeat yesterday). The model has a 168-hour lag, so it knows weekends are different and wins clearly, about 0.82 against 1.02. Which baseline you compare with changes the story.
- **Short lags alone fail.** With only lags 1 to 3 (the last test), a 24-step recursive forecast has no idea of the daily cycle and decays towards a flat line, about 1.52 against 1.12.

## Interview follow-ups
- Electricity demand depends strongly on temperature. How would you add a weather forecast as a feature without leaking the future in the backtest?
"""
