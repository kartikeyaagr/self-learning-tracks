import numpy as np

META = dict(title='Capstone: an energy voice activity detector, end to end', tags=['Data Processing', 'Metrics & Evaluation'], diff='Hard', runtime='numpy',
            related=['frame-energy-db', 'vad-threshold-hangover', 'frames-to-segments', 'classification-report'])

STATEMENT = r"""
Build a complete energy-based voice activity detector and score it against reference labels.

Write `run_vad(x, sr, ref, frame_ms, hop_ms, offset_db, hangover, min_gap, min_speech)`:

1. `frame_len = int(sr * frame_ms / 1000)`, `hop = int(sr * hop_ms / 1000)`. Compute per-frame energy in dB: mean square of each full frame, $10\log_{10}(e + 10^{-10})$.
2. Estimate the noise floor as the 10th percentile of the frame energies (`np.percentile`, default interpolation). The threshold is floor + `offset_db`.
3. Frame decisions: above threshold is speech, plus a hangover of `hangover` frames after each speech frame.
4. Segments: runs of speech frames, frame `i` spanning $[i \cdot hop, i \cdot hop + frame\_len]/sr$ seconds; merge gaps strictly below `min_gap`, then drop segments shorter than `min_speech`. Round times to 6 decimals before comparing and in the output.
5. Reference labels: frame `i` is speech if its centre $(i \cdot hop + frame\_len/2)/sr$ satisfies $s \le c < e$ for some `[s, e]` in `ref`.
6. Frame precision, recall and F1 of the step 3 decisions against the reference (0/0 counts as 0).

Return a dict with keys `threshold`, `segments`, `precision`, `recall`, `f1`.
"""

STARTER = '''import numpy as np

def run_vad(x, sr, ref, frame_ms, hop_ms, offset_db, hangover, min_gap, min_speech):
    """Energy VAD: frames, adaptive threshold, hangover, segments and frame-level scores."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _segments(dec, hop_s, frame_s, min_gap, min_speech):
    runs, start = [], None
    for i, d in enumerate(list(dec) + [0]):
        if d and start is None:
            start = i
        elif not d and start is not None:
            runs.append([round(start * hop_s, 6), round((i - 1) * hop_s + frame_s, 6)])
            start = None
    merged = []
    for s, e in runs:
        if merged and round(s - merged[-1][1], 6) < min_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [[s, e] for s, e in merged if round(e - s, 6) >= min_speech]

def run_vad(x, sr, ref, frame_ms, hop_ms, offset_db, hangover, min_gap, min_speech):
    x = np.asarray(x, float)
    L, h = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    n = (len(x) - L) // h + 1 if len(x) >= L else 0
    idx = np.arange(L)[None, :] + h * np.arange(n)[:, None]
    db = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-10)
    thr = float(np.percentile(db, 10) + offset_db)
    dec, left = [], 0
    for e in db:
        if e > thr:
            dec.append(1); left = hangover
        elif left > 0:
            dec.append(1); left -= 1
        else:
            dec.append(0)
    segs = _segments(dec, h / sr, L / sr, min_gap, min_speech)
    c = (h * np.arange(n) + L / 2) / sr
    lab = np.zeros(n, bool)
    for s, e in ref:
        lab |= (c >= s) & (c < e)
    d = np.array(dec, bool)
    tp, fp, fn = int((d & lab).sum()), int((d & ~lab).sum()), int((~d & lab).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"threshold": thr, "segments": segs, "precision": p, "recall": r, "f1": f}
'''

HINTS = [
    'A percentile of the frame energies adapts the threshold to each recording; a fixed dB value does not.',
    'Score the frame decisions before the segment clean-up; the segments are the product, the frame scores are the diagnosis.',
    'Reuse the run-and-merge logic from the segments step unchanged.',
]

def _sig(seed, sr, dur, bursts, noise, amp):
    g = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    x = noise * g.normal(size=len(t))
    for s, e in bursts:
        m = (t >= s) & (t < e)
        env = np.sin(np.pi * (t[m] - s) / (e - s)) ** 0.5
        x[m] += amp * env * (np.sin(2 * np.pi * 180 * t[m]) + 0.5 * np.sin(2 * np.pi * 720 * t[m]))
    return x

_b1 = [[0.3, 0.8], [1.2, 1.5]]
_b2 = [[0.2, 0.5], [0.55, 0.9], [1.4, 1.45]]

TESTS = [
    (_sig(1, 8000, 2.0, _b1, 0.01, 0.5), 8000, _b1, 25, 10, 6.0, 5, 0.1, 0.1),
    (_sig(1, 8000, 2.0, _b1, 0.01, 0.5), 8000, _b1, 25, 10, 6.0, 0, 0.0, 0.0),
    (_sig(2, 8000, 2.0, _b2, 0.02, 0.3), 8000, _b2, 25, 10, 10.0, 8, 0.1, 0.1),
    (_sig(3, 8000, 2.0, _b1, 0.2, 0.3), 8000, _b1, 25, 10, 6.0, 5, 0.1, 0.1),
    (_sig(4, 16000, 1.0, [[0.4, 0.6]], 0.01, 0.5), 16000, [[0.4, 0.6]], 25, 10, 3.0, 20, 0.2, 0.05),
]

EXPLANATION = r"""
## What the pipeline does
Each piece from the steps has one job: framing makes loudness measurable, the percentile threshold adapts to the recording, the hangover protects word endings, and the segment rules remove gaps and clicks. Frame-level precision and recall then show which job is failing.

## Reading the scores
- High recall and low precision: the threshold is too low or the hangover too long, so noise gets through.
- High precision and low recall: speech onsets and soft endings are being missed.
- The fourth test raises the noise until speech frames are only about 4 dB louder than silence. Floor plus 6 dB is then above every speech frame and nothing is detected; a lower offset lets noise through instead. Energy alone cannot separate them, which is why practical detectors (such as the WebRTC one or small neural models on log-mel features) look at spectral shape as well.
- The fifth test uses a 20-frame hangover on a short burst, so precision falls below 0.5: most of the detected frames are trailing silence.

## The 10th percentile
It assumes at least a tenth of the recording is silence. In a monologue with no pauses, the floor estimate lands inside speech and the threshold rises too high.

## Interview follow-ups
- How would you make this detector work in a live stream, where the percentile cannot see the future?
"""
