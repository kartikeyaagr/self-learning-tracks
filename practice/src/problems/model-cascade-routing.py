META = dict(title='Pick the confidence threshold for a two-model cascade', tags=['Language Models', 'Inference'], diff='Medium',
            runtime='python', lessons=['aieng:12.04'],
            related=['prompt-cache-cost', 'optimize-throughput-and-cost-under-latency-slo', 'precision-recall-at-k'])

STATEMENT = r"""
A **cascade** sends every query to a cheap model first. If the cheap model's confidence is at least a threshold
$\tau$, its answer is used; otherwise the query is escalated to a strong model and you pay for both calls.

Write `pick_threshold(queries, cost_small, cost_large, thresholds, target)`.

Each query is a dict with `conf` (the small model's confidence), `small_ok` and `large_ok` (whether each model
answers it correctly, from a labelled eval set). For a threshold $\tau$:
- accepted queries ($\text{conf} \ge \tau$) score `small_ok` and cost `cost_small`,
- escalated queries score `large_ok` and cost `cost_small + cost_large`.

Accuracy is correct answers divided by the number of queries; cost is the mean cost per query.

Return a dict:
- `"table"`: for each threshold in the given order, `[threshold, accuracy, mean_cost, escalation_rate]`,
- `"baseline"`: `[accuracy, mean_cost]` of sending everything to the large model alone,
- `"best"`: the threshold with the lowest mean cost among those with accuracy $\ge$ `target`, ties to the
  smaller threshold. If no threshold reaches the target, the one with the highest accuracy (ties: lower cost,
  then smaller threshold).
"""

STARTER = '''def pick_threshold(queries, cost_small, cost_large, thresholds, target):
    """Evaluate a small-then-large cascade at each threshold and pick one."""
    table = []
    for tau in thresholds:
        acc = sum(q["small_ok"] for q in queries) / len(queries)
        table.append([tau, acc, cost_small, 0.0])
    best = min(table, key=lambda row: row[2])[0]
    return {"table": table, "baseline": [0.0, cost_large], "best": best}
'''

SOLUTION = '''def pick_threshold(queries, cost_small, cost_large, thresholds, target):
    n = len(queries)
    table = []
    for tau in thresholds:
        correct = esc = 0
        for q in queries:
            if q["conf"] >= tau:
                correct += q["small_ok"]
            else:
                esc += 1
                correct += q["large_ok"]
        cost = cost_small + esc * cost_large / n      # everyone pays the small call
        table.append([tau, correct / n, cost, esc / n])
    ok = [r for r in table if r[1] >= target]
    if ok:
        best = min(ok, key=lambda r: (r[2], r[0]))[0]
    else:
        best = min(table, key=lambda r: (-r[1], r[2], r[0]))[0]
    base = [sum(q["large_ok"] for q in queries) / n, float(cost_large)]
    return {"table": table, "baseline": base, "best": best}
'''

HINTS = [
    'For one threshold, loop over the queries once: count correct answers and escalations. Everyone pays the small call; escalated queries also pay the large one.',
    'Keep accuracy as `correct / n` and compare it to the target directly. Both are correctly rounded, so 18/20 equals 0.9 exactly.',
    'Pick with `min(..., key=...)` over the rows that meet the target; the fallback key is `(-accuracy, cost, threshold)`.',
]


def _q(conf, s, l):
    return {"conf": conf, "small_ok": s, "large_ok": l}


# 20 support tickets: the small model is right on most confident ones, the large model on almost all.
_T = [_q(0.97, 1, 1), _q(0.95, 1, 1), _q(0.93, 1, 1), _q(0.92, 1, 1), _q(0.9, 1, 1), _q(0.88, 1, 1),
      _q(0.86, 0, 1), _q(0.85, 1, 1), _q(0.81, 1, 1), _q(0.78, 1, 1), _q(0.74, 0, 1), _q(0.7, 1, 1),
      _q(0.66, 0, 1), _q(0.62, 0, 1), _q(0.55, 1, 0), _q(0.5, 0, 1), _q(0.45, 0, 1), _q(0.4, 0, 0),
      _q(0.3, 0, 1), _q(0.2, 0, 1)]
_TH = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
TESTS = [
    (_T, 0.25, 4.0, _TH, 0.85),
    (_T, 0.25, 4.0, _TH, 0.9),
    (_T, 0.25, 4.0, _TH, 0.95),
    (_T, 0.5, 2.0, [0.9, 0.7, 0.5], 0.8),
    ([_q(0.9, 1, 0), _q(0.8, 1, 0), _q(0.2, 0, 1), _q(0.1, 1, 0)], 1.0, 8.0, [0.0, 0.5, 1.5], 0.75),
    ([_q(0.6, 0, 1), _q(0.6, 1, 1)], 0.5, 4.0, [0.6, 0.7], 1.0),
    ([_q(0.99, 0, 0), _q(0.99, 0, 0)], 0.25, 4.0, [0.5, 1.0], 0.5),
]

EXPLANATION = r"""
## The cost of a cascade
With escalation rate $e(\tau)$, the mean cost per query is
$$c(\tau) = c_s + e(\tau)\, c_\ell$$
so a cascade only saves money when $e(\tau) < 1 - c_s / c_\ell$. With a small model at a sixteenth of the large
model's price ($c_s = 0.25$, $c_\ell = 4$ in test 1), you can escalate up to 93.75% of traffic and still save.
Accuracy is a mix: accepted queries get the small model's accuracy on the confident slice, escalated ones get
the large model's.

## Why the threshold comes from an eval set
Confidence (a calibrated classifier score, a judge, or the small model's own token log-probabilities) is only
useful if it separates right from wrong answers on *your* traffic. The table is the cost/quality curve; you
pick the cheapest point that meets the quality bar. In test 1, $\tau = 0.8$ escalates 55% of tickets, costs
2.45 per ticket against 4.0 for the large model alone, and scores 0.85.

## A cascade can beat the large model
In test 5, the small model is right on two queries the large model gets wrong. Accepting confident small
answers raises accuracy above the large-only baseline. FrugalGPT reports the same effect on real APIs.

## Common bugs
- Charging escalated queries only `cost_large` (the small call already happened).
- Comparing accuracy with a tolerance or with `round`, which turns exact ties into near-misses.
- Choosing the threshold on the same tickets you report the result on. Pick it on a validation split and
  re-measure on a held-out one; the escalation rate drifts when traffic changes, so monitor it in production.

## Interview follow-ups
- Cascade or router? (A router predicts the right model *before* any call, so it pays once but needs a trained
  predictor; a cascade pays for the small call on every query but uses the actual answer to decide.)
- What does a cascade do to latency? (Escalated queries pay two sequential calls; set the threshold with the
  p95 latency budget in view too.)
"""
