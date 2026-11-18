import random

META = dict(title='Remove position bias from a pairwise LLM judge', tags=['Metrics & Evaluation', 'Language Models'],
            diff='Medium', runtime='python', lessons=['aieng:10.03'],
            related=['judge-human-agreement-kappa', 'fit-elo-ratings-from-pairwise-votes',
                     'remove-response-length-bias-from-judge-scores', 'canonicalize-pairwise-preference-data'])

STATEMENT = r"""
A pairwise judge sees two responses and says which is better. LLM judges tend to favour one position (often the first), so a candidate shown first in every comparison gets a free boost. The standard fix is to judge every pair twice, once in each order, and only count a win when both orders agree.

Write `resolve_pairwise(orig, swap)` for $n \ge 1$ comparisons between response **A** (the candidate) and response **B** (the baseline).

## Inputs
- `orig[i]`: the judge's verdict when **A is shown first**: `"1"` (first response better), `"2"` (second better) or `"tie"`.
- `swap[i]`: the verdict on the same pair with **B shown first**, same vocabulary.

Verdicts name positions, not responses. In `orig`, `"1"` means A; in `swap`, `"1"` means B.

## Rules
1. Map both verdicts to `"A"`, `"B"` or `"tie"`.
2. The resolved verdict is the mapped verdict if both runs agree, otherwise `"tie"` (an order-dependent verdict is not evidence for either response).

## Output
Return a dict:

| key | value |
|---|---|
| `"verdicts"` | list of resolved verdicts, `"A"`, `"B"` or `"tie"` |
| `"consistency"` | fraction of pairs whose two mapped verdicts are equal (two ties count as equal) |
| `"first_position_rate"` | over all $2n$ verdicts that are not `"tie"`, the fraction that are `"1"`; `None` if every verdict is `"tie"` |
| `"position_flips"` | number of pairs where both runs chose the **same position** (`"1"`/`"1"` or `"2"`/`"2"`), so the winner followed the order |
| `"win_rate_A"` | (resolved A wins + 0.5 × resolved ties) / $n$ |
| `"naive_win_rate_A"` | the same formula using only `orig`, as if you had judged each pair once |
"""

STARTER = '''def resolve_pairwise(orig, swap):
    """Return a dict: verdicts, consistency, first_position_rate, position_flips, win_rate_A, naive_win_rate_A."""
    to_ab = {"1": "A", "2": "B", "tie": "tie"}
    verdicts = [to_ab[o] for o in orig]
    n = len(orig)
    win = (verdicts.count("A") + 0.5 * verdicts.count("tie")) / n
    return {"verdicts": verdicts, "consistency": 1.0, "first_position_rate": None,
            "position_flips": 0, "win_rate_A": win, "naive_win_rate_A": win}
'''

SOLUTION = '''def resolve_pairwise(orig, swap):
    first_a = {"1": "A", "2": "B", "tie": "tie"}      # A shown first
    first_b = {"1": "B", "2": "A", "tie": "tie"}      # B shown first
    n = len(orig)
    a_view = [first_a[v] for v in orig]
    b_view = [first_b[v] for v in swap]
    verdicts = [x if x == y else "tie" for x, y in zip(a_view, b_view)]

    decisive = [v for v in list(orig) + list(swap) if v != "tie"]
    first_rate = decisive.count("1") / len(decisive) if decisive else None

    win = lambda vs: (vs.count("A") + 0.5 * vs.count("tie")) / n
    return {"verdicts": verdicts,
            "consistency": sum(x == y for x, y in zip(a_view, b_view)) / n,
            "first_position_rate": first_rate,
            "position_flips": sum(o == s != "tie" for o, s in zip(orig, swap)),
            "win_rate_A": win(verdicts),
            "naive_win_rate_A": win(a_view)}
'''

HINTS = [
    'Translate positions to responses with two small dicts, one per order. In the swapped run, position "1" is B.',
    'A pair is position-driven when the two raw verdicts are the same non-tie string: the judge picked the same slot both times, so it picked different responses.',
    'Count decisive verdicts over both lists together for the position rate, and remember to return None when there are none.',
]


def _counts(spec, seed):
    """spec: {(orig, swap): count}; returns shuffled (orig, swap) lists."""
    pairs = [k for k, c in spec.items() for _ in range(c)]
    random.Random(seed).shuffle(pairs)
    return [o for o, _ in pairs], [s for _, s in pairs]


def _biased_judge(n, p_a_better, p_first, seed):
    """Each pair has a true winner; the judge reports it, except it says "1" with probability p_first."""
    r = random.Random(seed)
    orig, swap = [], []
    for _ in range(n):
        a_wins = r.random() < p_a_better
        orig.append("1" if r.random() < p_first else ("1" if a_wins else "2"))
        swap.append("1" if r.random() < p_first else ("2" if a_wins else "1"))
    return orig, swap


TESTS = [
    _counts({("1", "2"): 40, ("2", "1"): 22, ("1", "1"): 20, ("2", "2"): 4,
             ("tie", "tie"): 6, ("1", "tie"): 4, ("2", "tie"): 4}, 0),     # the lesson's 100 pairs
    (["1", "2", "tie"], ["2", "1", "tie"]),                               # perfectly consistent
    (["1", "1", "1", "1"], ["1", "1", "1", "1"]),                         # always picks slot 1: every pair is a tie
    (["tie", "tie"], ["tie", "tie"]),                                     # no decisive verdicts
    (["2", "2", "1"], ["2", "tie", "2"]),
    _biased_judge(200, 0.5, 0.3, 1),
    _biased_judge(300, 0.7, 0.15, 2),
]

EXPLANATION = r"""
## What the swap buys
In the first test the candidate is shown first in the original run and seems to win 67% of comparisons (64 wins, 6 ties, 30 losses). Judging again with the order reversed shows that 24 of those pairs were decided by position: the judge picked the same slot both times. Counting only order-consistent verdicts, the candidate wins 40 and loses 22 with 38 ties, a win rate of 59%. It is probably still better, but the margin was inflated by 8 points. Wang et al. call aggregating over both orders *balanced position calibration*.

## Reading the diagnostics
- `first_position_rate` near 0.5 means no position preference. In the first test it is 106/180 = 0.59; under no bias the standard deviation of that count is $\sqrt{180 \times 0.25} \approx 6.7$, so 106 is about 2.4 standard deviations above 90. (The 180 verdicts are not independent, since each pair contributes two, so treat this as a rough check.)
- `consistency` is the share of pairs where the judge's verdict survives the swap. Low consistency means many comparisons are close calls the judge cannot separate, or the rubric is too vague. Either way, those pairs belong in front of a human.

## Design choices
Counting an inconsistent pair as a tie is conservative: it can only pull win rates toward 50%. An alternative is to ask the judge for a probability or score in each order and average them, which keeps more signal but needs a judge whose scores are calibrated. The resolved verdicts feed directly into Bradley-Terry or Elo fitting when comparing more than two systems.

## Common bugs
- Mapping the swapped run with the same dict as the original, which double counts the position preference instead of cancelling it.
- Counting `("tie", "tie")` as inconsistent.
- Computing the position rate per run and averaging, which weights runs with many ties wrongly.

## Interview follow-ups
- The swap doubles judge cost. When would you skip it, and how would you check that skipping is safe?
- How would you combine position-swapping with a length correction for the same judge?
"""
