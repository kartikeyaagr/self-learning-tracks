META = dict(title='Cheapest serving configuration that meets the latency SLO', tags=['Inference'], diff='Medium', runtime='python',
            lessons=['ieng:25.04', 'ieng:25.02'],
            related=['p50-p95-and-p99-inference-latency', 'ttft-tpot-itl-and-token-throughput', 'estimate-model-memory-and-minimum-gpu-count'])

STATEMENT = r"""
A benchmark sweep gives, for each candidate configuration (GPU type, parallelism, batch limit), its tail latencies and its sustained throughput. The deployment decision is: among configurations that meet the SLO, which serves the expected load for the least money?

Write `pick_config(configs, slo_ttft_ms, slo_tpot_ms, demand_tps, headroom_pct)`.

## Inputs
- `configs`: list of dicts, each with keys `'name'` (str), `'gpus'` (int, GPUs per replica), `'gpu_hour'` (float, price per GPU-hour), `'ttft_p99'` and `'tpot_p99'` (floats, ms), `'tps'` (int, output tokens/s one replica sustains at that operating point).
- `slo_ttft_ms`, `slo_tpot_ms`: limits on P99 TTFT and P99 TPOT. A config is **feasible** if both its P99s are $\le$ the limits.
- `demand_tps`: int, expected peak output tokens/s. `headroom_pct`: int, extra capacity to provision (e.g. 20 for 20%).

## Metrics per feasible config
- Cost per million output tokens: $\dfrac{\text{gpus} \cdot \text{gpu\_hour}}{3600 \cdot \text{tps}} \cdot 10^6$.
- Replicas: $\left\lceil \dfrac{\text{demand\_tps} \cdot (100 + \text{headroom\_pct})}{100 \cdot \text{tps}} \right\rceil$, computed in exact integer arithmetic.
- Hourly cost: replicas · gpus · gpu_hour.

## Choice
The feasible config with the **lowest hourly cost**; ties go to the lower cost per million tokens, then to the earlier config in the list.

## Output
A dict `{'name', 'cost_per_mtok', 'replicas', 'hourly_cost', 'feasible'}` where `'feasible'` is the list of feasible names in input order, or `None` if no config is feasible.
"""

STARTER = '''def pick_config(configs, slo_ttft_ms, slo_tpot_ms, demand_tps, headroom_pct):
    """Return {'name', 'cost_per_mtok', 'replicas', 'hourly_cost', 'feasible'} or None."""
    best = max(configs, key=lambda c: c['tps'])       # highest throughput, ignoring SLO and price
    reps = demand_tps // best['tps'] + 1
    return {'name': best['name'], 'cost_per_mtok': 0.0, 'replicas': reps,
            'hourly_cost': reps * best['gpus'] * best['gpu_hour'], 'feasible': [c['name'] for c in configs]}
'''

SOLUTION = '''def pick_config(configs, slo_ttft_ms, slo_tpot_ms, demand_tps, headroom_pct):
    best, feasible = None, []
    for i, c in enumerate(configs):
        if c['ttft_p99'] > slo_ttft_ms or c['tpot_p99'] > slo_tpot_ms:
            continue
        feasible.append(c['name'])
        per_mtok = c['gpus'] * c['gpu_hour'] / (3600 * c['tps']) * 1e6
        need = demand_tps * (100 + headroom_pct)
        replicas = -(-need // (100 * c['tps']))                 # exact integer ceiling
        hourly = replicas * c['gpus'] * c['gpu_hour']
        key = (hourly, per_mtok, i)                             # tie rules, in order
        if best is None or key < best[0]:
            best = (key, {'name': c['name'], 'cost_per_mtok': per_mtok, 'replicas': replicas, 'hourly_cost': hourly})
    if best is None:
        return None
    return {**best[1], 'feasible': feasible}
'''

HINTS = [
    'Filter by the SLO first: a config that violates either P99 limit is out, however cheap.',
    'Replicas must cover demand plus headroom: `-(-demand * (100 + h) // (100 * tps))` is an exact ceiling without float error.',
    'Rank with a tuple `(hourly_cost, cost_per_mtok, index)`. Cost per token alone is not the objective: rounding up to whole replicas can make the cheaper-per-token option more expensive at your load.',
]


def _c(name, gpus, price, ttft, tpot, tps):
    return {'name': name, 'gpus': gpus, 'gpu_hour': price, 'ttft_p99': ttft, 'tpot_p99': tpot, 'tps': tps}


_sweep = [
    _c('A100-tp1-b16', 1, 1.8, 420.0, 38.0, 1400),
    _c('A100-tp1-b64', 1, 1.8, 1900.0, 61.0, 3100),
    _c('A100-tp2-b64', 2, 1.8, 650.0, 33.0, 4200),
    _c('H100-tp1-b32', 1, 3.9, 300.0, 22.0, 4400),
    _c('H100-tp1-b128', 1, 3.9, 980.0, 41.0, 9800),
    _c('H100-tp2-b128', 2, 3.9, 520.0, 24.0, 12500),
    _c('L40S-tp2-b32', 2, 1.1, 800.0, 48.0, 2100),
]
TESTS = [
    ([_c('a', 1, 2.0, 100.0, 10.0, 1000), _c('b', 1, 4.0, 100.0, 10.0, 3000)], 200.0, 20.0, 1000, 0),
    ([_c('a', 1, 2.0, 100.0, 10.0, 1000), _c('b', 1, 4.0, 300.0, 10.0, 3000)], 200.0, 20.0, 5000, 20),
    (_sweep, 1000.0, 50.0, 20000, 20),
    (_sweep, 500.0, 30.0, 20000, 20),
    (_sweep, 2000.0, 70.0, 20000, 20),
    (_sweep, 200.0, 20.0, 20000, 20),
    (_sweep, 1000.0, 50.0, 1000, 0),
    ([_c('x', 1, 1.0, 50.0, 5.0, 1000), _c('y', 2, 1.0, 50.0, 5.0, 4000), _c('z', 2, 0.5, 50.0, 5.0, 2000)], 100.0, 10.0, 3000, 0),
]

EXPLANATION = r"""
## Why it works
Throughput figures are only meaningful at an operating point that meets the SLO: pushing the batch size up raises tokens/s and tail latency together (compare the b16 and b64 rows). So filter first, then compare cost. The money you actually spend is replicas × GPUs × price, and replicas come in whole units, so the config with the best cost per token can lose at a given demand (test 1: `b` is cheaper per token, but at 1000 tokens/s one replica of `a` is enough and costs half as much).

## Cost
Linear in the number of configs. The real work is upstream: a sweep of batch limits, parallelism and GPU types, each measured under realistic arrival patterns until the P99s are stable.

## Common bugs
- Choosing by peak throughput or by cost per token without checking the P99 limits.
- `math.ceil(demand * 1.2 / tps)`: $1000 \cdot 1.2$ is `1200.0000000000002` in floating point, which adds a replica you don't need.
- Using P50 latencies from the sweep: SLOs are stated on the tail.

## Interview follow-ups
- How does headroom relate to failure tolerance? (With $r$ replicas, losing one leaves $(r-1)/r$ of capacity; $N+1$ provisioning needs headroom of at least $1/(r-1)$.)
- When would you deploy two configurations at once? (Mixed traffic: a low-latency pool for interactive chat and a large-batch pool for offline or batch jobs with looser SLOs.)
"""
