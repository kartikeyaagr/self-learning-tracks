META = dict(title='Capstone: pick a training plan that fits GPU memory', tags=['Training', 'Distributed Computing'], diff='Hard', runtime='python',
            related=['mixed-precision-training-memory-accountant', 'transformer-activation-memory', 'activation-checkpoint-plan', 'account-for-zero-memory-by-stage'])

STATEMENT = r"""
Write `plan_training(P, s, h, a, L, D, gpu_bytes, micro_batches)` that searches training configurations for a model with `P` parameters, sequence length `s`, hidden size `h`, `a` heads and `L` layers on `D` data-parallel GPUs.

**Model states per GPU** (mixed-precision Adam, 16 bytes per parameter in total: 2 weights, 2 gradients, 12 optimiser incl. fp32 master):

| ZeRO stage | bytes per GPU |
|---|---|
| 0 | $16P$ |
| 1 | $4P + 12P/D$ |
| 2 | $2P + 14P/D$ |
| 3 | $16P/D$ |

**Activations** for micro-batch $b$, no tensor parallelism: per layer $sbh(34 + 5as/h)$ with `"none"`, $34\,sbh$ with `"selective"`, $2sbh$ with `"full"`; times $L$.

A configuration `(stage, recompute, b)` is feasible if states + activations $\le$ `gpu_bytes`. Among feasible ones choose by: largest `b`, then recompute in the order none, selective, full, then the lowest stage (less communication).

Return `{"choice": [stage, recompute, b] or None, "states": bytes, "activations": bytes, "n_feasible": int}` (states and activations of the choice, 0 if none fit). Try stages 0-3, the three recompute modes and every `b` in `micro_batches`.
"""

STARTER = '''def plan_training(P, s, h, a, L, D, gpu_bytes, micro_batches):
    """Choose (ZeRO stage, recompute mode, micro-batch) that fits."""
    return {"choice": None, "states": 0, "activations": 0, "n_feasible": 0}
'''

SOLUTION = '''def plan_training(P, s, h, a, L, D, gpu_bytes, micro_batches):
    states = {0: 16 * P, 1: 4 * P + 12 * P / D, 2: 2 * P + 14 * P / D, 3: 16 * P / D}
    def act(mode, b):
        sbh = s * b * h
        per = {"none": sbh * (34 + 5 * a * s / h), "selective": 34 * sbh, "full": 2 * sbh}[mode]
        return per * L
    order = {"none": 0, "selective": 1, "full": 2}
    feas = [(st, m, b) for st in states for m in order for b in micro_batches
            if states[st] + act(m, b) <= gpu_bytes]
    if not feas:
        return {"choice": None, "states": 0, "activations": 0, "n_feasible": 0}
    st, m, b = min(feas, key=lambda c: (-c[2], order[c[1]], c[0]))
    return {"choice": [st, m, b], "states": float(states[st]), "activations": float(act(m, b)), "n_feasible": len(feas)}
'''

HINTS = [
    'Write two small functions, model-state bytes by stage and activation bytes by mode and batch, then enumerate every combination.',
    'Sort feasible plans with a key tuple that encodes the preference order.',
    'Model states do not depend on the micro-batch; activations do not depend on the ZeRO stage.',
]

G = 80 * 2 ** 30
TESTS = [
    (1.3e9, 2048, 2048, 16, 24, 8, G, [1, 2, 4, 8, 16]),
    (6.7e9, 4096, 4096, 32, 32, 8, G, [1, 2, 4, 8]),
    (13e9, 4096, 5120, 40, 40, 64, G, [1, 2, 4]),
    (70e9, 4096, 8192, 64, 80, 8, G, [1, 2]),
    (70e9, 4096, 8192, 64, 80, 512, G, [1, 2, 4]),
    (125e6, 1024, 768, 12, 12, 1, 16 * 2 ** 30, [8, 16, 32, 64]),
]

EXPLANATION = r"""
## Two budgets that share one GPU
Model states scale with parameters and shrink with ZeRO sharding; activations scale with batch, sequence and depth and shrink with recomputation. A plan has to fit both into the same memory, and the cheapest fix depends on which one dominates.

## Reading the results
- A 1.3B model fits without sharding; the largest micro-batch needs selective recomputation.
- At 7B on 8 GPUs, plain data parallelism cannot even hold the 107 GB of states, so ZeRO is required before activations matter.
- At 70B on 8 GPUs nothing fits: 16 bytes per parameter over 8 GPUs is 140 GB each. Spread over 512 GPUs, ZeRO-3 makes the states small and recomputation decides the batch.

## What a real planner adds
Tensor and pipeline parallelism, communication cost (ZeRO-3 gathers weights every layer), fragmentation and framework buffers. The structure of the search stays the same: enumerate, filter by memory, rank by cost.

## Interview follow-ups
- Why prefer a larger micro-batch over lower recomputation, and when would you choose the reverse?
"""
