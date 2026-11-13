META = dict(title='Simulate GPipe and 1F1B pipeline schedules', tags=['Distributed Computing', 'Training'], diff='Medium',
            runtime='python', lessons=['ieng:18.06', 'ieng:18.09'],
            related=['account-for-zero-memory-by-stage', 'choose-topology-aware-parallel-plan', 'simulate-distributed-collectives'])

STATEMENT = r"""
Pipeline parallelism puts consecutive layers on consecutive GPUs ("stages") and streams $m$ micro-batches through them. Stages sit idle while the pipeline fills and drains: the **bubble**. Two classic schedules have the same bubble but very different memory: GPipe runs all forwards then all backwards, so stage 0 holds activations for every micro-batch; 1F1B starts backwards as early as possible.

Simulate both and measure them.

Write `pipeline_schedule(n_stages, n_micro, t_fwd, t_bwd, schedule)`.

## Inputs
- `n_stages` $P \ge 1$, `n_micro` $m \ge 1$.
- `t_fwd`, `t_bwd`: a positive number (same for every stage) or a list of $P$ positive numbers (per stage), the time of one micro-batch's forward / backward on that stage.
- `schedule`: `'gpipe'` or `'1f1b'`.

## Per-stage operation order ($F_j$, $B_j$: forward / backward of micro-batch $j$)
- `'gpipe'`: $F_0, F_1, \dots, F_{m-1}, B_0, B_1, \dots, B_{m-1}$.
- `'1f1b'` on stage $s$: let $w = \min(P - 1 - s,\ m)$. First $F_0, \dots, F_{w-1}$ (warm-up); then for $i = 0, \dots, m - w - 1$: $F_{w+i}, B_i$ (steady state); then $B_{m-w}, \dots, B_{m-1}$ (cool-down).

## Timing rules
- A stage runs its operations one at a time, in its order.
- $F_j$ on stage $s > 0$ can start only after $F_j$ finished on stage $s - 1$. $B_j$ on stage $s < P - 1$ only after $B_j$ finished on stage $s + 1$; on the last stage, $B_j$ needs $F_j$ finished there. Communication is free.
- Each operation starts as soon as both its stage is free and its dependency is done (time 0 if neither applies).

## Output
A dict:
- `'makespan'`: float, the finish time of the last operation;
- `'bubble'`: float, the idle fraction $1 - \dfrac{\sum_s m\,(t^F_s + t^B_s)}{P \cdot \text{makespan}}$;
- `'ideal_bubble'`: float, the textbook value $\dfrac{P - 1}{m + P - 1}$, for comparison;
- `'peak_in_flight'`: list of $P$ ints, for each stage the maximum number of micro-batches whose forward has run there but whose backward has not yet (activations it must hold).
"""

STARTER = '''def pipeline_schedule(n_stages, n_micro, t_fwd, t_bwd, schedule):
    """Return {'makespan', 'bubble', 'ideal_bubble', 'peak_in_flight'}."""
    P, m = n_stages, n_micro
    makespan = (m + P - 1) * (t_fwd + t_bwd)          # TODO: formula only; per-stage times and memory ignored
    ideal = (P - 1) / (m + P - 1)
    return {'makespan': float(makespan), 'bubble': ideal, 'ideal_bubble': ideal, 'peak_in_flight': [m] * P}
'''

SOLUTION = '''def _order(s, P, m, schedule):
    if schedule == 'gpipe':
        return [('F', j) for j in range(m)] + [('B', j) for j in range(m)]
    w = min(P - 1 - s, m)
    ops = [('F', j) for j in range(w)]
    for i in range(m - w):
        ops += [('F', w + i), ('B', i)]
    return ops + [('B', j) for j in range(m - w, m)]

def pipeline_schedule(n_stages, n_micro, t_fwd, t_bwd, schedule):
    P, m = n_stages, n_micro
    tf = list(t_fwd) if isinstance(t_fwd, (list, tuple)) else [t_fwd] * P
    tb = list(t_bwd) if isinstance(t_bwd, (list, tuple)) else [t_bwd] * P
    orders = [_order(s, P, m, schedule) for s in range(P)]
    done = {}                                   # (kind, j, stage) -> finish time
    nxt, free = [0] * P, [0.0] * P

    def dep(kind, j, s):
        if kind == 'F':
            return ('F', j, s - 1) if s > 0 else None
        return ('B', j, s + 1) if s < P - 1 else ('F', j, s)

    remaining = 2 * m * P
    while remaining:
        progressed = False
        for s in range(P):                      # schedule every op whose dependency is already timed
            while nxt[s] < len(orders[s]):
                kind, j = orders[s][nxt[s]]
                d = dep(kind, j, s)
                if d is not None and d not in done:
                    break
                start = max(free[s], done[d] if d else 0.0)
                free[s] = start + (tf[s] if kind == 'F' else tb[s])
                done[(kind, j, s)] = free[s]
                nxt[s] += 1
                remaining -= 1
                progressed = True
        assert progressed, 'deadlock'

    makespan = max(free)
    busy = sum(m * (tf[s] + tb[s]) for s in range(P))
    peaks = []
    for s in range(P):
        live = peak = 0
        for kind, _ in orders[s]:
            live += 1 if kind == 'F' else -1
            peak = max(peak, live)
        peaks.append(peak)
    return {'makespan': float(makespan), 'bubble': 1 - busy / (P * makespan),
            'ideal_bubble': (P - 1) / (m + P - 1), 'peak_in_flight': peaks}
'''

HINTS = [
    'Build each stage\'s operation list first. Then repeatedly sweep the stages, scheduling the next operation of a stage whenever its single dependency already has a finish time: start = max(stage free time, dependency finish).',
    'Dependencies: $F_j$ on stage $s$ waits for $F_j$ on $s-1$; $B_j$ waits for $B_j$ on $s+1$, except on the last stage where it waits for $F_j$ on the same stage. Each stage keeps a pointer into its own list.',
    'Peak in-flight needs no timing at all: walk a stage\'s order counting +1 for each F and -1 for each B and take the maximum. GPipe gives $m$ everywhere; 1F1B gives $\\min(P - s, m)$.',
]

TESTS = [
    (2, 2, 1.0, 2.0, 'gpipe'),
    (2, 2, 1.0, 2.0, '1f1b'),
    (4, 8, 1.0, 2.0, 'gpipe'),
    (4, 8, 1.0, 2.0, '1f1b'),
    (4, 32, 1.0, 2.0, '1f1b'),
    (1, 5, 1.0, 1.0, '1f1b'),
    (4, 2, 1.0, 2.0, '1f1b'),
    (3, 6, [1.0, 2.0, 1.0], [2.0, 4.0, 2.0], '1f1b'),
    (3, 6, [1.0, 2.0, 1.0], [2.0, 4.0, 2.0], 'gpipe'),
    (8, 16, 0.5, 1.0, '1f1b'),
]

EXPLANATION = r"""
## Why the bubble is $(P-1)/(m+P-1)$
With equal stage times, the first micro-batch needs $P - 1$ forward steps before the last stage starts, and the last backward needs $P - 1$ steps to drain back to stage 0. Every stage is idle for $(P-1)(t_F + t_B)$ in total, out of a makespan of $(m + P - 1)(t_F + t_B)$. Both schedules have this makespan (tests 3 and 4): 1F1B does not shrink the bubble, it reorders work so that backwards free activations early.

## Memory
GPipe's stage 0 holds activations for all $m$ micro-batches before its first backward. 1F1B caps stage $s$ at $P - s$ micro-batches in flight, independent of $m$. That is why 1F1B lets you raise $m$ (test 5: 32 micro-batches, bubble down to about 8.6%) without running out of memory.

## Unbalanced stages
Test 8 makes the middle stage twice as slow. The pipeline then runs at the pace of the slowest stage, the others idle in every steady-state step, and the bubble is well above the textbook value. Balancing layers across stages (by measured time, not by layer count) matters as much as the number of micro-batches.

## Complexity
$O(mP)$ operations, each scheduled once; the sweep loop makes at most $O(mP)$ passes in the worst case, which is fine for simulation sizes.

## Interview follow-ups
- Interleaved 1F1B gives each GPU $v$ non-contiguous chunks of layers. How does that cut the bubble to roughly $(P-1)/(v\,m)$, and what does it cost? (More point-to-point messages.)
- Zero-bubble schedules split the backward into input-gradient and weight-gradient parts. Why does the weight-gradient part have no cross-stage dependency, and how does that fill the bubble?
"""
