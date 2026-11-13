META = dict(title='Pick a 3-D parallel layout for a GPU cluster', tags=['Distributed Computing', 'Training'], diff='Hard',
            runtime='python', lessons=['ieng:18.09', 'ieng:18.02'],
            related=['account-for-zero-memory-by-stage', 'measure-pipeline-bubble-utilization', 'collective-effective-bandwidth'])

STATEMENT = r"""
Large training runs combine tensor parallelism (TP), pipeline parallelism (PP) and data parallelism (DP), with $\text{tp} \cdot \text{pp} \cdot \text{dp}$ equal to the GPU count. The right split depends on memory, on the pipeline bubble, and on **where** each communication group lands: TP all-reduces happen every layer and need NVLink inside a node, while DP gradient all-reduces happen once per step and may cross the slower network.

Write `choose_plan(model, cluster, global_micro)` that enumerates every valid layout under the cost model below and returns the fastest.

## Inputs
- `model`: dict with `'n_layers'`, `'n_heads'`, `'n_params'` (ints), and floats `'step_tflop'` (FLOPs of one whole training step, in TFLOP), `'act_gb'` (activation memory of one layer for one micro-batch at tp = 1, GB), `'tp_msg_gb'` (size of one TP all-reduce message for one layer and micro-batch, GB).
- `cluster`: dict with ints `'n_gpus'`, `'gpus_per_node'` and floats `'mem_gb'` (per GPU), `'tflops'` (achieved TFLOP/s per GPU), `'intra_gbps'`, `'inter_gbps'` (all-reduce bus bandwidth inside a node and across nodes, GB/s).
- `global_micro`: int, micro-batches per step in total; each DP replica processes $m = \text{global\_micro} / \text{dp}$ of them.

## Valid layouts
All positive $(\text{tp}, \text{pp}, \text{dp})$ with $\text{tp}\cdot\text{pp}\cdot\text{dp} = $ `n_gpus` such that: tp divides `gpus_per_node` and `n_heads`; pp divides `n_layers`; dp divides `global_micro`; and $\text{pp} \le m$. Ranks are placed with TP innermost, then DP, then PP, so a DP group occupies $\text{tp}\cdot\text{dp}$ consecutive GPUs and stays inside one node exactly when $\text{tp}\cdot\text{dp}$ divides `gpus_per_node`.

## Memory per GPU (GB), with 1F1B (stage 0 holds pp micro-batches)
$$\text{mem} = \frac{16\, n_\text{params}}{10^9\, \text{tp}\cdot\text{pp}} + \text{act\_gb} \cdot \frac{n_\text{layers}}{\text{pp}} \cdot \frac{1}{\text{tp}} \cdot \text{pp}$$
A layout is **feasible** if $\text{mem} \le$ `mem_gb`.

## Step time (seconds)
- compute $c = \text{step\_tflop} / (\text{n\_gpus} \cdot \text{tflops})$; bubble $= c\,(\text{pp} - 1)/m$;
- TP: $0$ if tp = 1, else $\frac{n_\text{layers}}{\text{pp}} \cdot m \cdot 4 \cdot \frac{2(\text{tp}-1)}{\text{tp}} \cdot \text{tp\_msg\_gb} / \text{intra\_gbps}$ (4 all-reduces per layer per micro-batch);
- DP: $0$ if dp = 1, else $\frac{2(\text{dp}-1)}{\text{dp}} \cdot \frac{2\, n_\text{params}}{10^9\,\text{tp}\cdot\text{pp}} / \text{bw}$, with bw = `intra_gbps` if the DP group fits in a node, else `inter_gbps`;
- total = compute + bubble + TP + DP. Pipeline point-to-point traffic is ignored.

## Output
Among feasible layouts, the one with the smallest total; ties go to the smaller tp, then the smaller pp. Return a dict `{'tp', 'pp', 'dp', 'time', 'memory_gb', 'feasible'}` where `feasible` is the number of feasible layouts. If none is feasible, return `None`.
"""

STARTER = '''def choose_plan(model, cluster, global_micro):
    """Return {'tp', 'pp', 'dp', 'time', 'memory_gb', 'feasible'} for the fastest feasible layout, or None."""
    n = cluster['n_gpus']
    t = model['step_tflop'] / (n * cluster['tflops'])
    return {'tp': 1, 'pp': 1, 'dp': n, 'time': t,                      # TODO: pure DP, ignores memory
            'memory_gb': 16 * model['n_params'] / 1e9, 'feasible': 1}
'''

SOLUTION = '''def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]

def choose_plan(model, cluster, global_micro):
    L, heads, N = model['n_layers'], model['n_heads'], model['n_params']
    G, gpn = cluster['n_gpus'], cluster['gpus_per_node']
    compute = model['step_tflop'] / (G * cluster['tflops'])
    best, n_ok = None, 0
    for tp in _divisors(G):
        if gpn % tp or heads % tp:
            continue                                   # TP must stay inside a node and split the heads
        for pp in _divisors(G // tp):
            dp = G // (tp * pp)
            if L % pp or global_micro % dp:
                continue
            m = global_micro // dp                     # micro-batches per pipeline
            if pp > m:
                continue
            shard = N / (tp * pp) / 1e9                # billions of parameters per GPU
            mem = 16 * shard + model['act_gb'] * (L / pp) / tp * pp
            if mem > cluster['mem_gb']:
                continue
            n_ok += 1
            bubble = compute * (pp - 1) / m
            t_tp = 0.0 if tp == 1 else (L / pp) * m * 4 * 2 * (tp - 1) / tp * model['tp_msg_gb'] / cluster['intra_gbps']
            bw = cluster['intra_gbps'] if gpn % (tp * dp) == 0 else cluster['inter_gbps']
            t_dp = 0.0 if dp == 1 else 2 * (dp - 1) / dp * 2 * shard / bw
            total = compute + bubble + t_tp + t_dp
            key = (total, tp, pp)                      # tie-break: smaller tp, then smaller pp
            if best is None or key < best[0]:
                best = (key, {'tp': tp, 'pp': pp, 'dp': dp, 'time': total, 'memory_gb': mem})
    if best is None:
        return None
    return {**best[1], 'feasible': n_ok}
'''

HINTS = [
    'Loop over divisors: tp over divisors of `n_gpus`, pp over divisors of `n_gpus // tp`, dp is what remains. Filter by every rule before computing anything, and count the layouts that also fit in memory.',
    'Note what the memory formula says: with 1F1B, activations per GPU are `act_gb * n_layers / tp`, independent of pp (fewer layers per stage, but pp micro-batches in flight). Only TP shrinks activations; PP and TP both shrink model state.',
    'Compare `(total, tp, pp)` tuples to get the tie-break for free. The DP group fits in a node when `gpus_per_node % (tp * dp) == 0`; that switches its bandwidth from the inter-node to the intra-node value.',
]

_m7 = dict(n_layers=32, n_heads=32, n_params=7_000_000_000, step_tflop=4.0e5, act_gb=1.0, tp_msg_gb=0.032)
_m70 = dict(n_layers=80, n_heads=64, n_params=70_000_000_000, step_tflop=4.0e6, act_gb=1.0, tp_msg_gb=0.064)
_c16 = dict(n_gpus=16, gpus_per_node=8, mem_gb=80.0, tflops=400.0, intra_gbps=300.0, inter_gbps=40.0)
_c64 = dict(n_gpus=64, gpus_per_node=8, mem_gb=80.0, tflops=400.0, intra_gbps=300.0, inter_gbps=40.0)
TESTS = [
    (dict(n_layers=4, n_heads=4, n_params=1_000_000_000, step_tflop=100.0, act_gb=1.0, tp_msg_gb=0.1),
     dict(n_gpus=4, gpus_per_node=4, mem_gb=40.0, tflops=100.0, intra_gbps=100.0, inter_gbps=10.0), 8),
    (dict(n_layers=4, n_heads=4, n_params=4_000_000_000, step_tflop=100.0, act_gb=1.0, tp_msg_gb=0.1),
     dict(n_gpus=4, gpus_per_node=4, mem_gb=40.0, tflops=100.0, intra_gbps=100.0, inter_gbps=10.0), 8),
    (_m7, _c16, 64),
    (_m7, {**_c16, 'inter_gbps': 5.0}, 64),
    (_m70, _c64, 128),
    (_m70, _c64, 16),
    (_m70, {**_c16, 'mem_gb': 40.0}, 32),
    (_m7, {**_c16, 'gpus_per_node': 16}, 64),
    (dict(n_layers=6, n_heads=12, n_params=2_000_000_000, step_tflop=3.0e3, act_gb=0.5, tp_msg_gb=0.008),
     dict(n_gpus=12, gpus_per_node=4, mem_gb=24.0, tflops=150.0, intra_gbps=200.0, inter_gbps=25.0), 24),
]

EXPLANATION = r"""
## Why it works
The search space is tiny (the number of divisor triples of the GPU count), so exhaustive enumeration with a cost model is the right tool: it is exact for the model, easy to audit, and fast. The work is in encoding the constraints correctly:
- **TP inside a node**: TP all-reduces sit on the critical path of every layer, twice in forward and twice in backward per micro-batch. Over a 40 GB/s network they would dominate.
- **PP divides the layers and needs $m \ge$ pp micro-batches**, otherwise some stages never overlap.
- **DP placement**: with TP innermost, a DP group spans tp $\times$ dp consecutive GPUs, so the same dp can be cheap (inside a node) or expensive (across nodes) depending on tp.

## Reading the results
- With plenty of memory (test 1), pure DP wins: no bubble, no per-layer communication, one gradient all-reduce.
- When model state does not fit (test 2: 64 GB of model state per GPU without sharding), the solver must split the model and picks the cheaper of TP and PP.
- A slow inter-node link (test 4) pushes the DP group inside nodes, at the price of TP or PP elsewhere.
- Fewer micro-batches (test 6 vs 5) keep the same layout but make the bubble far more expensive: the gap to the deeper pipelines (pp = 4, 8) grows from about 1% to about 10%.
- With one 16-GPU node (test 8), the DP all-reduce runs over the fast link and the step gets faster, without changing the layout. Test 7 has no feasible layout at all: 70B parameters need more than 16 GPUs with 40 GB each.

## Limits of the model
Real planners add pipeline point-to-point costs, ZeRO sharding of the optimizer (which would make pure DP fit much more often), overlap of DP communication with backward, and per-micro-batch efficiency losses for small TP shards. The structure (enumerate, filter, cost, tie-break) stays the same.

## Common bugs
- Forgetting that more DP means fewer micro-batches per pipeline, hence a larger bubble.
- Checking only `tp <= gpus_per_node` instead of divisibility, which lets a TP group straddle two nodes.
- Comparing floats for ties without a deterministic secondary key.

## Interview follow-ups
- Add ZeRO-1 to the model: how do the memory formula and the DP cost change, and how does that shift the optimum?
- Where would context (sequence) parallelism enter this search, and which term does it reduce?
"""
