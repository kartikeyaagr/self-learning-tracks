import numpy as np

META = dict(title='Ad auction ranking with a generalised second price', tags=['Recommender Systems', 'Metrics & Evaluation'], diff='Medium', runtime='numpy')

STATEMENT = r"""
Write `run_auction(bids, pctr, quality, slots, reserve)` for a cost-per-click ad auction.

- Each ad's rank score is $s_i = \text{bid}_i \cdot \text{pCTR}_i \cdot \text{quality}_i$.
- Ads with $s_i <$ `reserve` are ineligible. Sort the rest by score descending (ties to the lower index) and fill up to `slots` positions.
- Under a generalised second price, the winner in position $j$ pays the minimum cost per click that keeps its rank: $\text{price}_j = s_{j+1} / (\text{pCTR}_j \cdot \text{quality}_j)$, where $s_{j+1}$ is the score of the next eligible ad (whether or not it won a slot). If there is no next ad, use `reserve` in place of $s_{j+1}$.
- Prices never exceed the bid.

Return `[[ad_index, round(price, 6), round(expected_revenue, 6)], ...]` in position order, where expected revenue is `price * pctr`.
"""

STARTER = '''import numpy as np

def run_auction(bids, pctr, quality, slots, reserve):
    """Return winners with GSP prices."""
    order = np.argsort(-np.asarray(bids))[:slots]
    return [[int(i), float(bids[i]), float(bids[i] * pctr[i])] for i in order]
'''

SOLUTION = '''import numpy as np

def run_auction(bids, pctr, quality, slots, reserve):
    b, c, q = (np.asarray(x, float) for x in (bids, pctr, quality))
    s = b * c * q
    elig = sorted((i for i in range(len(s)) if s[i] >= reserve), key=lambda i: (-s[i], i))
    out = []
    for j, i in enumerate(elig[:slots]):
        nxt = s[elig[j + 1]] if j + 1 < len(elig) else reserve
        price = min(nxt / (c[i] * q[i]), b[i])
        out.append([int(i), round(float(price), 6), round(float(price * c[i]), 6)])
    return out
'''

HINTS = [
    'Rank by bid times predicted click rate (times quality), not by bid: an ad nobody clicks earns nothing.',
    'The price is the bid at which this ad would exactly tie the next one: solve bid * pctr * quality = next score.',
    'The ad just below the last slot still sets the price for the last slot.',
]

_r = np.random.default_rng(6)
TESTS = [
    ([2.0, 1.0, 3.0], [0.05, 0.1, 0.02], [1, 1, 1], 2, 0.0),
    ([2.0, 1.0, 3.0], [0.05, 0.1, 0.02], [1, 1, 1], 3, 0.03),
    (_r.uniform(0.5, 5, 10), _r.uniform(0.005, 0.08, 10), _r.uniform(0.6, 1.2, 10), 4, 0.02),
    ([1.0], [0.1], [1.0], 1, 0.05),
    ([1.0, 1.0], [0.1, 0.1], [1.0, 1.0], 1, 0.0),
    ([5.0, 4.0], [0.01, 0.01], [1.0, 1.0], 2, 0.1),
]

EXPLANATION = r"""
## Why multiply by pCTR
Under cost-per-click pricing the platform earns only when users click. Expected revenue per impression is bid times click probability, so that is the ranking key. A quality factor adds a user-experience term so that relevant ads are not outbid by clickbait.

## Why second price
Paying your own bid rewards shading bids down and leads to unstable bidding. Paying the amount needed to beat the next ad lets advertisers bid closer to their value. GSP is not fully truthful with several slots, but it is simple and was the industry standard for search ads for years.

## What the ML system owns
Calibration of pCTR. If predicted click rates are inflated for some advertisers, they win more auctions and pay less per click than they should. Auction systems monitor calibration per advertiser and per position.

## Interview follow-ups
- How does position bias in click logs affect pCTR training for this auction?
"""
