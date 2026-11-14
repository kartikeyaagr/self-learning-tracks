import numpy as np

META = dict(title='Turn frame decisions into speech segments', tags=['Data Processing', 'Time Series'], diff='Medium', runtime='python',
            related=['vad-threshold-hangover'])

STATEMENT = r"""
Write `frames_to_segments(decisions, hop_s, frame_s, min_gap, min_speech)`.

- Frame `i` covers the time interval $[i \cdot hop,\ i \cdot hop + frame]$ in seconds.
- A run of consecutive speech frames `i..j` becomes the segment $[i \cdot hop,\ j \cdot hop + frame]$.
- First merge neighbouring segments whose gap (next start minus previous end) is **strictly less than** `min_gap`.
- Then drop segments shorter than `min_speech` seconds.

Return a list of `[start, end]` pairs, each value rounded to 6 decimals.
"""

STARTER = '''def frames_to_segments(decisions, hop_s, frame_s, min_gap, min_speech):
    """Speech segments in seconds from per-frame 0/1 decisions."""
    segs = []
    for i, d in enumerate(decisions):
        if d:
            segs.append([i * hop_s, i * hop_s + frame_s])
    return segs
'''

SOLUTION = '''def frames_to_segments(decisions, hop_s, frame_s, min_gap, min_speech):
    runs, start = [], None
    for i, d in enumerate(list(decisions) + [0]):
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
'''

HINTS = [
    'Append a sentinel 0 so the last run closes without a special case.',
    'Merge before dropping: two short bursts with a tiny gap may form one segment long enough to keep.',
    'Round times before comparing them: 0.1 - 0.05 is not exactly 0.05 in floating point.',
]

TESTS = [
    ([0, 1, 1, 0, 0, 1, 0], 0.01, 0.025, 0.0, 0.0),
    ([0, 1, 1, 0, 0, 1, 0], 0.01, 0.025, 0.02, 0.0),
    ([1, 0, 1, 0, 0, 0, 0, 0, 1], 0.1, 0.1, 0.15, 0.25),
    ([1, 1, 1, 1], 0.01, 0.025, 0.1, 0.05),
    ([0, 0, 0], 0.01, 0.025, 0.1, 0.0),
    ([1, 0, 0, 0, 0, 1, 1, 1, 1, 1], 0.01, 0.01, 0.0, 0.05),
]

EXPLANATION = r"""
## From frames to something useful
Downstream systems want time ranges: where to send audio to a recogniser, what to keep in a recording, when to stop listening. Two clean-up rules turn noisy frame decisions into usable segments.

- **Merge short gaps.** A pause of 100 ms inside a sentence should not split it in two.
- **Drop short segments.** A 30 ms click is almost never speech.

## Order matters
Merging first lets two close bursts combine into one segment that passes the length rule; dropping first would discard both. The third test shows that case.

## Interview follow-ups
- How would you score a detector at segment level rather than frame level, and why do the two scores often disagree?
"""
