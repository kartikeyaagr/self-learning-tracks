META = dict(title='All-reduce cost of tensor-parallel inference', tags=['Distributed Computing', 'Inference'], diff='Hard',
            runtime='python', lessons=['ieng:18.03', 'ieng:18.04'],
            related=['estimate-model-memory-and-minimum-gpu-count', 'analyze-prefill-and-decode-with-roofline-model', 'expert-parallel-routing-across-gpus'])

STATEMENT = r"""
Megatron-style tensor parallelism splits each attention block and each MLP across $p$ GPUs. Each of the two halves ends in a partial sum, so every layer needs **two all-reduces** of the activations in the forward pass. Estimate what that costs next to the compute.

Write `tp_comm(n_layers, d_model, n_tokens, bytes_per_elem, tp, link_bw, link_latency, n_params, peak_flops)`.

## Model
- One all-reduce message is the activation tensor: $S = \text{n\_tokens} \cdot \text{d\_model} \cdot \text{bytes\_per\_elem}$ bytes (`n_tokens` = tokens in this forward pass: batch size for a decode step, batch × prompt length for prefill).
- **Ring all-reduce** over $p$ GPUs runs $2(p - 1)$ steps; in each, every GPU sends a chunk of $S/p$ bytes. One all-reduce therefore takes
$$t_{\text{ar}} = 2(p-1)\,\alpha + \frac{2(p-1)}{p}\cdot\frac{S}{\beta},$$
with $\alpha$ = `link_latency` (s per step) and $\beta$ = `link_bw` (bytes/s per GPU). With $p = 1$ there is no communication.
- Per forward pass: $2 \cdot \text{n\_layers}$ all-reduces.
- Compute time per forward pass: $2 \cdot \text{n\_params} \cdot \text{n\_tokens} / (p \cdot \text{peak\_flops})$ (each GPU does $1/p$ of the work at peak).
- No overlap between communication and compute.

## Output
A dict:
- `'msg_bytes'`: $S$ (int). `'sent_per_gpu'`: bytes each GPU sends per forward pass, $2 \cdot \text{n\_layers} \cdot \frac{2(p-1)}{p} S$ (float).
- `'comm_s'`, `'compute_s'`: the two times per forward pass (floats).
- `'comm_fraction'`: `comm_s / (comm_s + compute_s)`.
- `'latency_share'`: the fraction of `comm_s` that comes from the $\alpha$ terms (0.0 when `tp == 1`).
"""

STARTER = '''def tp_comm(n_layers, d_model, n_tokens, bytes_per_elem, tp, link_bw, link_latency, n_params, peak_flops):
    """Return {'msg_bytes', 'sent_per_gpu', 'comm_s', 'compute_s', 'comm_fraction', 'latency_share'}."""
    S = n_tokens * d_model * bytes_per_elem
    comm = 2 * n_layers * S / link_bw              # sends the whole tensor once, no latency term
    compute = 2 * n_params * n_tokens / (tp * peak_flops)
    return {'msg_bytes': S, 'sent_per_gpu': 2 * n_layers * S, 'comm_s': comm, 'compute_s': compute,
            'comm_fraction': comm / (comm + compute), 'latency_share': 0.0}
'''

SOLUTION = '''def tp_comm(n_layers, d_model, n_tokens, bytes_per_elem, tp, link_bw, link_latency, n_params, peak_flops):
    S = n_tokens * d_model * bytes_per_elem
    n_ar = 2 * n_layers                                   # after attention out-proj and after MLP down-proj
    compute = 2 * n_params * n_tokens / (tp * peak_flops)
    if tp == 1:
        return {'msg_bytes': S, 'sent_per_gpu': 0.0, 'comm_s': 0.0, 'compute_s': compute,
                'comm_fraction': 0.0, 'latency_share': 0.0}

    steps = 2 * (tp - 1)                                  # reduce-scatter + all-gather, each p-1 steps
    lat = steps * link_latency                            # alpha term: fixed cost per step
    bw = steps / tp * S / link_bw                         # beta term: S/p bytes per step
    comm = n_ar * (lat + bw)
    return {'msg_bytes': S, 'sent_per_gpu': n_ar * steps / tp * S, 'comm_s': comm, 'compute_s': compute,
            'comm_fraction': comm / (comm + compute), 'latency_share': n_ar * lat / comm}
'''

HINTS = [
    'Count all-reduces first: two per layer. Then work out one all-reduce\'s time and multiply.',
    'A ring all-reduce is a reduce-scatter followed by an all-gather: $2(p-1)$ steps, each paying the latency $\\alpha$ once and moving $S/p$ bytes.',
    'Handle `tp == 1` separately (no communication, and avoid dividing by zero comm time). `latency_share` is the $\\alpha$ part of `comm_s` divided by `comm_s`.',
]

_NVL, _PCIE = (450 * 10**9, 3e-6), (25 * 10**9, 10e-6)     # rough per-GPU bandwidth (bytes/s) and per-step latency (s)
_P70, _PEAK = 70 * 10**9, 989 * 10**12
TESTS = [
    (1, 4, 2, 2, 2, 8, 1.0, 100, 50),
    (2, 8, 4, 1, 4, 16, 0.5, 1000, 100),
    (80, 8192, 32, 2, 8, *_NVL, _P70, _PEAK),
    (80, 8192, 32 * 2048, 2, 8, *_NVL, _P70, _PEAK),
    (80, 8192, 32, 2, 8, *_PCIE, _P70, _PEAK),
    (80, 8192, 1, 2, 4, *_NVL, _P70, _PEAK),
    (32, 4096, 512, 2, 1, *_NVL, 8 * 10**9, _PEAK),
    (32, 4096, 4096, 2, 2, *_PCIE, 8 * 10**9, _PEAK),
]

EXPLANATION = r"""
## Why it works
Splitting a matmul by columns and then by rows leaves each GPU with a partial sum of the full output; summing those across GPUs is an all-reduce. The ring algorithm is bandwidth-optimal: each GPU sends $2(p-1)/p \cdot S < 2S$ bytes no matter how many GPUs join. Its weakness is the $2(p-1)$ sequential steps, each paying a fixed latency.

## Decode vs prefill
For a decode step with 32 sequences, $S = 32 \cdot 8192 \cdot 2 = 512$ KiB: the bandwidth term is tiny and the latency term dominates (`latency_share` close to 1), so communication can be a large fraction of a very short step. For prefill (test 4) the message is 2048 times larger, the bandwidth term dominates, but compute grows by the same factor, so the fraction drops. Over PCIe instead of NVLink (test 5) both terms grow and TP stops paying off, which is why tensor parallelism is kept inside a node.

## Common bugs
- Counting one all-reduce per layer instead of two (attention and MLP each need one).
- Using $S/\beta$ per all-reduce: it ignores the $2(p-1)/p$ factor of the ring and the latency term.
- Dividing compute by $p$ but forgetting that the communication volume per GPU does **not** shrink with $p$.

## Interview follow-ups
- How do engines hide some of this cost? (Overlap the all-reduce of one micro-batch or chunk with compute of the next, or fuse the all-reduce into the GEMM epilogue.)
- Sequence parallelism replaces each all-reduce with a reduce-scatter plus an all-gather around the norms. Does it change the bytes? (No, same total volume, but activations between them are sharded, saving memory.)
"""
