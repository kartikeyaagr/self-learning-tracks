import numpy as np

META = dict(title='Capstone: find heartbeats and flag premature ones', tags=['Time Series', 'Classification'], diff='Hard', runtime='numpy',
            related=['baseline-wander-removal', 'qrs-slope-energy', 'adaptive-threshold-peaks', 'beat-windows', 'classification-report'])

STATEMENT = r"""
Run a complete beat-analysis pipeline on an ECG and score it against reference annotations.

Write `analyze_ecg(x, fs, true_peaks, true_labels, tol_s)`:

1. **Clean**: subtract a running median baseline with a 0.6 s window (as in the baseline step).
2. **Energy**: slope energy with a 0.15 s trailing window (as in the slope step).
3. **Detect**: adaptive-threshold peaks on the energy with a 0.2 s refractory period (as in the peak step).
4. **Locate R**: for each energy peak $p$, the R peak is the index of the maximum of the cleaned signal in `[max(0, p - int(0.2 * fs)), p]` inclusive. Drop repeated indices, keeping order.
5. **Match**: walk the reference peaks in order; each one matches the first unmatched detected R within `tol_s * fs` samples. `sensitivity` = matched / reference beats, `ppv` = matched / detected beats.
6. **Classify** every detected beat: RR before it = distance to the previous detected R. Label 1 (premature) if that RR is below 0.85 times the median of all detected RR intervals, else 0. The first detected beat is 0.
7. **Report** on matched beats only, against `true_labels`: for class 0 and class 1, `[sensitivity, precision, support]` with support = number of reference beats of that class among the matched ones (0/0 counts as 0.0).

Return a dict with `r_peaks`, `sensitivity`, `ppv` and `per_class`.
"""

STARTER = '''import numpy as np

def analyze_ecg(x, fs, true_peaks, true_labels, tol_s):
    """Baseline removal, slope energy, adaptive peaks, R localisation, matching and RR-based beat classification."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _baseline(x, fs, win_s):
    w = int(win_s * fs)
    w += (w % 2 == 0)
    xp = np.pad(x, (w - 1) // 2, mode="edge")
    return x - np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)

def _energy(x, fs, win_s):
    s = np.r_[0.0, np.diff(x)] ** 2
    w = max(int(win_s * fs), 1)
    c = np.r_[0.0, np.cumsum(s)]
    n = np.arange(1, len(s) + 1); lo = np.maximum(n - w, 0)
    return (c[n] - c[lo]) / (n - lo)

def _peaks(e, fs, ref_s):
    head = e[:int(2 * fs)]
    spk, npk = 0.5 * head.max(), 0.5 * head.mean()
    out = []
    for i in range(1, len(e) - 1):
        if not (e[i] > e[i - 1] and e[i] >= e[i + 1]):
            continue
        if e[i] > npk + 0.25 * (spk - npk) and (not out or i - out[-1] >= ref_s * fs):
            out.append(i); spk = 0.125 * e[i] + 0.875 * spk
        else:
            npk = 0.125 * e[i] + 0.875 * npk
    return out

def analyze_ecg(x, fs, true_peaks, true_labels, tol_s):
    x = np.asarray(x, float)
    y = _baseline(x, fs, 0.6)
    e = _energy(y, fs, 0.15)
    back = int(0.2 * fs)
    R = []
    for p in _peaks(e, fs, 0.2):
        lo = max(0, p - back)
        r = lo + int(np.argmax(y[lo:p + 1]))
        if r not in R:
            R.append(r)
    tol = tol_s * fs
    used, pairs = set(), []
    for k, t in enumerate(true_peaks):
        for j, r in enumerate(R):
            if j not in used and abs(r - t) <= tol:
                used.add(j); pairs.append((k, j)); break
    rr = np.diff(R)
    med = np.median(rr) if len(rr) else 0.0
    pred = [0] + [int(v < 0.85 * med) for v in rr]
    per = []
    for c in (0, 1):
        tp = sum(1 for k, j in pairs if true_labels[k] == c and pred[j] == c)
        sup = sum(1 for k, j in pairs if true_labels[k] == c)
        npred = sum(1 for k, j in pairs if pred[j] == c)
        per.append([tp / sup if sup else 0.0, tp / npred if npred else 0.0, sup])
    return {"r_peaks": R, "sensitivity": len(pairs) / len(true_peaks) if len(true_peaks) else 0.0,
            "ppv": len(pairs) / len(R) if R else 0.0, "per_class": per}
'''

HINTS = [
    'The energy peak lags the R peak by roughly half the energy window, so search backwards in the cleaned signal.',
    'Score detection (sensitivity and PPV) separately from classification; a missed beat should not count as a classification error.',
    'After a premature ventricular beat the next interval is long (a compensatory pause), which an RR rule correctly calls normal.',
]

def _record(seed, fs, secs, rr0, pvc_every, drift, noise):
    g = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = drift * np.sin(2 * np.pi * 0.25 * t) + noise * g.normal(size=len(t))
    peaks, labels = [], []
    b, k = 0.5, 0
    while b < secs - 0.5:
        pvc = pvc_every > 0 and k % pvc_every == pvc_every - 1
        if pvc:
            x += 1.3 * np.exp(-((t - b) / 0.03) ** 2) - 0.5 * np.exp(-((t - b - 0.12) / 0.06) ** 2)
        else:
            x += np.exp(-((t - b) / 0.012) ** 2) + 0.25 * np.exp(-((t - b - 0.28) / 0.05) ** 2) + 0.1 * np.exp(-((t - b + 0.16) / 0.03) ** 2)
        peaks.append(int(round(b * fs))); labels.append(int(pvc))
        nxt_pvc = pvc_every > 0 and (k + 1) % pvc_every == pvc_every - 1
        step = rr0 * (1 + 0.03 * g.normal())
        if nxt_pvc:
            step *= 0.6
        elif pvc:
            step = 2 * rr0 - 0.6 * rr0
        b += step; k += 1
    return x, peaks, labels

def _case(seed, fs, secs, rr0, pvc_every, drift, noise, tol):
    x, p, l = _record(seed, fs, secs, rr0, pvc_every, drift, noise)
    return (np.round(x, 5), fs, p, l, tol)

TESTS = [
    _case(1, 250, 12, 0.8, 0, 0.3, 0.02, 0.05),
    _case(2, 250, 20, 0.8, 5, 0.3, 0.02, 0.05),
    _case(3, 250, 20, 0.7, 4, 0.8, 0.05, 0.05),
    _case(4, 200, 20, 0.9, 6, 0.3, 0.15, 0.05),
    _case(5, 250, 15, 0.6, 3, 0.5, 0.03, 0.03),
]

EXPLANATION = r"""
## The pipeline
Clean the baseline, turn steepness into a detection signal, pick peaks with an adaptive threshold, walk back to the true R location, then use timing to classify beats. Each stage depends on the one before: a drifting baseline breaks the R location step, and a missed beat makes the next RR interval look long.

## Scoring in two layers
Detection is scored by sensitivity and positive predictive value with a tolerance, which is how standard ECG benchmarks report it. Classification is then scored only on matched beats, so detection errors and labelling errors are not mixed up.

## What an RR rule can and cannot do
Premature beats arrive early, so an RR threshold catches many of them. It cannot tell a premature ventricular beat from a premature atrial one, and it will flag normal beats when the heart rate changes quickly. That is why real classifiers combine timing with the beat's shape, using the windows from the beat-window step. In the third test (strong drift, more noise) a few extra detections create short RR intervals, and class 1 precision drops to 0.7.

## When detection breaks
The fourth test adds heavy noise. Differencing amplifies high frequencies, so the slope energy fires on noise: sensitivity falls to about 0.73 and PPV to about 0.26, and the classification numbers built on those detections are meaningless. This is why Pan-Tompkins band-pass filters the ECG before differencing, and why detection quality must be checked before any beat-level result is trusted.

## Interview follow-ups
- Premature beats are rare (often under 5% of beats). Which of the reported numbers would you put in front of a clinician, and why not accuracy?
"""
