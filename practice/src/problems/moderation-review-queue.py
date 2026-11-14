import numpy as np

META = dict(title='Moderation review queue under reviewer capacity', tags=['Classification', 'MLOps'], diff='Medium', runtime='numpy')

STATEMENT = r"""
Write `triage(p_violate, severity, reach, auto_remove, capacity)` for a content moderation pipeline.

- Expected harm of item $i$ if left up: $h_i = p_i \cdot \text{severity}_i \cdot \text{reach}_i$, where reach is the expected number of future views.
- Items with $p_i \ge$ `auto_remove` are removed automatically and do not use reviewer capacity.
- The rest are ranked by $h_i$ descending (ties to the lower index); the top `capacity` are sent to review. Everything else stays up.
- Harm prevented: assume removal and review both catch every true violation, so prevented harm is the sum of $h_i$ over removed and reviewed items.

Return `{"remove": [indices], "review": [indices in priority order], "prevented": float, "residual": float}` where `residual` is the harm of items left up, and `remove` is sorted ascending.
"""

STARTER = '''import numpy as np

def triage(p_violate, severity, reach, auto_remove, capacity):
    """Split items into remove, review and leave."""
    p = np.asarray(p_violate)
    order = list(np.argsort(-p)[:capacity])
    return {"remove": [], "review": [int(i) for i in order], "prevented": 0.0, "residual": 0.0}
'''

SOLUTION = '''import numpy as np

def triage(p_violate, severity, reach, auto_remove, capacity):
    p, s, r = (np.asarray(x, float) for x in (p_violate, severity, reach))
    h = p * s * r
    remove = [i for i in range(len(p)) if p[i] >= auto_remove]
    rest = sorted((i for i in range(len(p)) if p[i] < auto_remove), key=lambda i: (-h[i], i))
    review = rest[:capacity]
    prevented = float(h[remove].sum() + h[review].sum()) if len(p) else 0.0
    return {"remove": remove, "review": review, "prevented": prevented, "residual": float(h.sum() - prevented)}
'''

HINTS = [
    'Rank the review queue by expected harm, not probability: a likely violation nobody will see matters less than a possible one about to go viral.',
    'Automatic removals come off the top before capacity is applied.',
    'Residual harm is total harm minus prevented harm.',
]

def _case(seed, n, cap, auto):
    g = np.random.default_rng(seed)
    return (g.beta(1, 6, n), g.choice([1.0, 3.0, 10.0], n), np.round(g.lognormal(5, 2, n)), auto, cap)

TESTS = [
    ([0.99, 0.5, 0.2, 0.6], [1, 10, 10, 1], [100, 10, 1000, 100], 0.95, 1),
    ([0.99, 0.5, 0.2, 0.6], [1, 10, 10, 1], [100, 10, 1000, 100], 0.95, 0),
    _case(1, 200, 10, 0.9),
    _case(2, 200, 50, 0.6),
    ([0.1, 0.1], [1, 1], [10, 10], 0.9, 1),
    ([], [], [], 0.9, 3),
]

EXPLANATION = r"""
## Why harm, not probability
Reviewers are the bottleneck. Ranking by classifier score sends them obvious, low-reach violations first, while a borderline post from a large account keeps spreading. Weighting by severity and expected reach puts review time where it prevents the most harm.

## Auto-action thresholds
Very confident predictions are removed without review to save capacity, but only above a threshold where precision has been measured to be high, and with an appeal path, because automated mistakes are seen as unfair.

## What this ignores
Reach is itself a prediction and is often the largest source of error. Reviewers are also not perfect, so real systems measure reviewer agreement and send hard categories to specialists.

## Interview follow-ups
- How do review decisions feed back into training, and what bias does ranking by harm introduce into those labels?
"""
