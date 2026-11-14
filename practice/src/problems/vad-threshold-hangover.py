import numpy as np

META = dict(title='Speech detection with a threshold and a hangover', tags=['Data Processing', 'Time Series'], diff='Easy', runtime='python',
            related=['frame-energy-db'])

STATEMENT = r"""
Write `threshold_with_hangover(db, threshold, hangover)` that turns per-frame energies (dB) into speech decisions.

- A frame whose energy is strictly above `threshold` is speech.
- After any speech frame, the next `hangover` frames are also marked speech, even if they are quiet. A new loud frame restarts the count.

Return a list of 0/1 ints, one per frame.
"""

STARTER = '''def threshold_with_hangover(db, threshold, hangover):
    """Frame-level speech decisions with a hangover after each loud frame."""
    return [int(e >= threshold) for e in db]
'''

SOLUTION = '''def threshold_with_hangover(db, threshold, hangover):
    out, left = [], 0
    for e in db:
        if e > threshold:
            out.append(1)
            left = hangover
        elif left > 0:
            out.append(1)
            left -= 1
        else:
            out.append(0)
    return out
'''

HINTS = [
    'Keep a counter of how many hangover frames remain; reset it to hangover on every loud frame.',
    'The hangover only extends speech forward in time, never backwards.',
]

TESTS = [
    ([-50, -10, -50, -50, -50], -30, 2),
    ([-50, -10, -50, -10, -50, -50, -50], -30, 1),
    ([-30, -30, -29.9], -30, 0),
    ([-10] * 3 + [-60] * 6, -40, 3),
    ([], -30, 2),
]

EXPLANATION = r"""
## Why a hangover
Speech energy dips between syllables and at the ends of words, especially on soft consonants like "f" or "th". A bare threshold chops words into pieces and cuts off their endings. The hangover keeps the detector on for a short time (often 100 to 300 ms) after the last loud frame.

## The trade-off
A longer hangover misses fewer word endings but adds trailing noise to every segment. Systems that feed speech recognition prefer generous hangovers, because clipping a word costs more than a little extra silence.

## Interview follow-ups
- Some detectors also add a "hang-before" by delaying the output. What does that cost in a live call?
"""
