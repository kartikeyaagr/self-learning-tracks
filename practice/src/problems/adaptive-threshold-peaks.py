import numpy as np

META = dict(title='Pick R peaks with an adaptive threshold', tags=['Time Series'], diff='Medium', runtime='numpy',
            related=['qrs-slope-energy'])

STATEMENT = r"""
Detect one peak per heartbeat in a slope-energy signal `e`, with a threshold that adapts to the signal level.

Write `detect_peaks(e, fs, refractory_s)`:

1. **Initialise** from the first two seconds, `e[:2*fs]`: signal level SPK $= 0.5\max$, noise level NPK $= 0.5\,\text{mean}$.
2. **Candidates** are local maxima: indices $i$ with $0 < i < n-1$, $e_i > e_{i-1}$ and $e_i \ge e_{i+1}$, visited in order.
3. For each candidate, threshold $= \text{NPK} + 0.25\,(\text{SPK} - \text{NPK})$.
   - If $e_i$ is above the threshold and at least `refractory_s * fs` samples after the last accepted peak (or no peak was accepted yet), **accept** it and update SPK $\leftarrow 0.125\,e_i + 0.875\,\text{SPK}$.
   - Otherwise update NPK $\leftarrow 0.125\,e_i + 0.875\,\text{NPK}$.

Return the accepted indices as a list.
"""

STARTER = '''import numpy as np

def detect_peaks(e, fs, refractory_s):
    """Local maxima above a running threshold that tracks signal and noise levels."""
    e = np.asarray(e, float)
    thr = 0.5 * e.max()
    return [i for i in range(1, len(e) - 1) if e[i] > thr and e[i] > e[i - 1] and e[i] >= e[i + 1]]
'''

SOLUTION = '''import numpy as np

def detect_peaks(e, fs, refractory_s):
    e = np.asarray(e, float)
    head = e[:int(2 * fs)]
    spk, npk = 0.5 * head.max(), 0.5 * head.mean()
    ref = refractory_s * fs
    out = []
    for i in range(1, len(e) - 1):
        if not (e[i] > e[i - 1] and e[i] >= e[i + 1]):
            continue
        thr = npk + 0.25 * (spk - npk)
        if e[i] > thr and (not out or i - out[-1] >= ref):
            out.append(i)
            spk = 0.125 * e[i] + 0.875 * spk
        else:
            npk = 0.125 * e[i] + 0.875 * npk
    return out
'''

HINTS = [
    'Rejected peaks, including those inside the refractory period, update the noise level.',
    'The refractory period (about 0.2 s) reflects physiology: the heart cannot beat again that quickly, so a second peak that soon is a T wave or noise.',
]

def _energy(seed, fs, secs, rr, amp_drift, noise):
    g = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    e = noise * np.abs(g.normal(size=len(t)))
    beats = np.arange(0.3, secs - 0.1, rr)
    for k, b in enumerate(beats):
        e += (1 + amp_drift * k) * np.exp(-((t - b) / 0.03) ** 2) + 0.15 * np.exp(-((t - b - 0.3) / 0.06) ** 2)
    return e

TESTS = [
    ([0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0], 2, 1.0),
    ([0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0], 2, 4.0),
    (_energy(1, 100, 6, 0.8, 0.0, 0.01), 100, 0.2),
    (_energy(2, 100, 8, 0.7, -0.06, 0.02), 100, 0.2),
    (_energy(3, 200, 5, 0.5, 0.2, 0.05), 200, 0.25),
]

EXPLANATION = r"""
## Why adapt
Beat amplitude changes with posture, breathing and electrode contact. A fixed threshold set on the first beats misses beats when the signal weakens and admits noise when it strengthens. Tracking a signal level and a noise level separately, and placing the threshold a quarter of the way between them, keeps the detector centred as both drift. The fourth test has beats that shrink steadily, and the fifth has beats that grow.

## Pan-Tompkins in full
The original method also searches back with a lower threshold when no beat has been found for 1.66 times the average RR interval, and uses a second threshold on the band-passed ECG to confirm each detection. Both reduce missed beats in arrhythmias, where RR intervals vary a lot.

## Interview follow-ups
- A patient's heart rate jumps from 60 to 180 beats per minute. Which parameter here is most likely to cause missed beats, and how would you make it adapt?
"""
