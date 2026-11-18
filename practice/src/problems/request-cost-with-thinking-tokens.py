META = dict(title='Cost per request and per success with thinking tokens', tags=['Language Models', 'Inference'], diff='Medium', runtime='python',
            lessons=['aieng:08.06'], related=['optimize-throughput-and-cost-under-latency-slo', 'ttft-tpot-itl-and-token-throughput'])

STATEMENT = r"""
A team is choosing a model and reasoning effort for an assistant. Write `compare_configs(prices, mix, configs, monthly_requests, min_success_pct)`.

**Inputs**

- `prices`: `{model: {"input": p, "cached": p, "output": p}}`, dollars per **million** tokens. Thinking tokens are billed at the `output` price.
- `mix`: a sample of traffic, a list of request types `{"count": int, "input": int, "cached": int, "output": int}`. `cached` is the part of `input` read from the prompt cache; `output` is the visible answer length.
- `configs`: a list of `{"name": str, "model": str, "max_output": int, "thinking": [int per type], "solves": [bool per type]}`. `thinking[j]` is how many thinking tokens the config spends on type `j`; `solves[j]` says whether its answer would be correct.

**One request of type $j$ under a config**

- Input cost: `(input - cached) * price.input + cached * price.cached`.
- Generated tokens $g = \text{thinking}_j + \text{output}_j$. If $g > \text{max\_output}$ the response is **truncated**: billed output is `max_output` and the request fails. Otherwise billed output is $g$ and the request succeeds exactly when `solves[j]`.
- Cost = (input cost + billed output * price.output) / 1,000,000.

**Per config**, weighting each type by `count` (total $N = \sum \text{count}$):

- `cost_per_request`: weighted mean cost.
- `monthly`: `cost_per_request * monthly_requests`.
- `success_rate`: successful requests / $N$.
- `cost_per_success`: `cost_per_request / success_rate`, or `None` if nothing succeeds.
- `truncated`: number of sample requests (by count) that were truncated.

Return `{"configs": [one dict per config, in input order, with keys "name", "cost_per_request", "monthly", "success_rate", "cost_per_success", "truncated"], "choice": name}` where `choice` is the config with the lowest `monthly` among those whose successes $\times 100 \ge$ `min_success_pct` $\times N$ (an exact integer test). Ties go to the earlier config; `None` if no config qualifies.
"""

STARTER = '''def compare_configs(prices, mix, configs, monthly_requests, min_success_pct):
    """Bill each config; thinking is ignored here, which is the bug to fix."""
    out = []
    for c in configs:
        p = prices[c["model"]]
        n = sum(t["count"] for t in mix)
        cost = sum(t["count"] * (t["input"] * p["input"] + t["output"] * p["output"]) / 1e6 for t in mix) / n
        ok = sum(t["count"] for t, s in zip(mix, c["solves"]) if s)
        out.append({"name": c["name"], "cost_per_request": cost, "monthly": cost * monthly_requests,
                    "success_rate": ok / n, "cost_per_success": cost * n / ok if ok else None, "truncated": 0})
    best = min(out, key=lambda r: r["monthly"])
    return {"configs": out, "choice": best["name"]}
'''

SOLUTION = '''def compare_configs(prices, mix, configs, monthly_requests, min_success_pct):
    n = sum(t["count"] for t in mix)
    rows, choice, best = [], None, None
    for c in configs:
        p = prices[c["model"]]
        total, ok, cut = 0.0, 0, 0
        for t, think, solves in zip(mix, c["thinking"], c["solves"]):
            gen = think + t["output"]
            truncated = gen > c["max_output"]
            billed = c["max_output"] if truncated else gen
            cost = ((t["input"] - t["cached"]) * p["input"] + t["cached"] * p["cached"]
                    + billed * p["output"]) / 1e6
            total += t["count"] * cost
            cut += t["count"] if truncated else 0
            ok += t["count"] if (solves and not truncated) else 0
        per_req = total / n
        rate = ok / n
        row = {"name": c["name"], "cost_per_request": per_req, "monthly": per_req * monthly_requests,
               "success_rate": rate, "cost_per_success": per_req / rate if ok else None, "truncated": cut}
        rows.append(row)
        if ok * 100 >= min_success_pct * n and (best is None or row["monthly"] < best):
            choice, best = c["name"], row["monthly"]
    return {"configs": rows, "choice": choice}
'''

HINTS = [
    'Thinking tokens and the visible answer share one output budget and one output price: add them before comparing with max_output.',
    'A truncated request is billed for max_output tokens of output and never counts as a success, even if the config would have solved that type.',
    'Keep the quality bar in integers (successes * 100 >= pct * N) so a rate like 0.8 never fails a float comparison; pick the cheapest qualifying config with a strict < so ties keep the earlier one.',
]

_PRICES = {"big": {"input": 4.0, "cached": 0.4, "output": 20.0},
           "mid": {"input": 2.0, "cached": 0.2, "output": 10.0},
           "small": {"input": 0.25, "cached": 0.025, "output": 1.25}}
_MIX = [{"count": 70, "input": 3000, "cached": 2000, "output": 300},
        {"count": 20, "input": 5000, "cached": 2000, "output": 500},
        {"count": 10, "input": 8000, "cached": 2000, "output": 800}]
_CFGS = [
    {"name": "small-none", "model": "small", "max_output": 2000, "thinking": [0, 0, 0], "solves": [True, False, False]},
    {"name": "mid-low", "model": "mid", "max_output": 8000, "thinking": [200, 800, 2000], "solves": [True, True, False]},
    {"name": "mid-high", "model": "mid", "max_output": 8000, "thinking": [1500, 4000, 9000], "solves": [True, True, True]},
    {"name": "mid-high-32k", "model": "mid", "max_output": 32000, "thinking": [1500, 4000, 9000], "solves": [True, True, True]},
    {"name": "big-medium", "model": "big", "max_output": 32000, "thinking": [600, 1500, 4000], "solves": [True, True, True]},
]
TESTS = [
    (_PRICES, _MIX, _CFGS, 1_000_000, 90),
    (_PRICES, _MIX, _CFGS, 1_000_000, 100),
    (_PRICES, _MIX, _CFGS, 250_000, 70),
    (_PRICES, _MIX, _CFGS[:1], 1_000_000, 80),
    (_PRICES, [{"count": 1, "input": 1000, "cached": 0, "output": 100}],
     [{"name": "a", "model": "mid", "max_output": 500, "thinking": [400], "solves": [True]},
      {"name": "b", "model": "mid", "max_output": 500, "thinking": [450], "solves": [True]}], 1000, 100),
    (_PRICES, _MIX, [_CFGS[2], _CFGS[3], dict(_CFGS[3], name="mid-high-copy")], 10, 100),
]

EXPLANATION = r"""
## Why thinking needs its own term
Reasoning models generate hidden tokens before and between visible ones. Every vendor bills them at the output price and counts them against the output cap. A cost model that uses only the visible answer length under-counts by the ratio (thinking + output) / output, which at high effort is often 10x or more.

## Why truncation matters
The output cap covers thinking plus answer. Test 2 shows `mid-high` with an 8,000-token cap: the hardest 10% of traffic wants 9,000 thinking tokens, so those requests stop mid-thought. They are billed 8,000 output tokens each and return nothing, so the config is both more expensive per success and below a 100% bar, while the same model with a 32k cap qualifies. Lowering the cap is not a cost control; lowering effort is.

## Cost per success, not per request
`small-none` is by far the cheapest per request but solves only 70% of traffic. The quality bar picks the cheapest config that clears it, and `cost_per_success` shows what each correct answer really costs. In a real system the failures also cost retries, escalations or unhappy users, which this metric starts to capture.

## Exact comparisons
Success counts are integers, so the bar is tested as `ok * 100 >= pct * N` with no floats. Monthly costs of different configs are compared with strict `<`, so identical configs (test 6) resolve to the first.

## Interview follow-ups
- How would you mix configs, sending easy types to `mid-low` and hard ones to `big-medium`? (Compute the blended cost per type; the router's own errors and cost count too.)
- Where does latency enter? (Thinking delays the first visible token; at a fixed decode speed it adds thinking / tokens-per-second seconds.)
- How do prompt-cache hit rates change the ranking? (Input-heavy types gain most; see the cached column.)
"""
