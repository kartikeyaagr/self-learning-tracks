META = dict(title='Training FLOPs and wall-clock days', tags=['Scaling Laws', 'Training'], diff='Easy', runtime='python',
            lessons=['ieng:12.07'], related=['fixed-ratio-budget-split', 'chinchilla-compute-optimal'])

STATEMENT = r"""
Before launching a pretraining run you want three numbers: how many floating-point operations it costs, how many GPU-hours that is, and how many days it takes on your cluster.

A dense transformer with $N$ non-embedding parameters spends about $2N$ FLOPs per token on the forward pass and twice that on the backward pass, so training costs $6N$ FLOPs per token. Attention over the context adds a term that does not scale with $N$. Per token, training costs

$$f = 6N + 6\, n_\text{layers}\, n_\text{ctx}\, d_\text{model}$$

and the whole run costs $C = f \cdot D$ for $D$ training tokens.

Hardware never runs at its datasheet peak. **Model FLOPs utilisation** (MFU) is the fraction of peak the run actually achieves.

Write `training_cost(n_params, n_tokens, n_gpus, peak_tflops, mfu, n_layers=0, seq_len=0, d_model=0)`.

## Inputs
- `n_params` ($N$), `n_tokens` ($D$): positive ints or floats.
- `n_gpus`: positive int. `peak_tflops`: peak throughput of **one** GPU in TFLOP/s ($10^{12}$ FLOP/s). `mfu`: a float in $(0, 1]$.
- `n_layers`, `seq_len`, `d_model`: the attention term. With the defaults (all 0) the cost is plain $6ND$.

## Output
A dict with exactly these float values:

| key | meaning |
|---|---|
| `'flops'` | $C$, total training FLOPs |
| `'gpu_hours'` | $C$ divided by the **achieved** throughput of one GPU, in hours |
| `'days'` | wall-clock days when the work is split evenly over `n_gpus` |
"""

STARTER = '''def training_cost(n_params, n_tokens, n_gpus, peak_tflops, mfu, n_layers=0, seq_len=0, d_model=0):
    """Return {'flops': ..., 'gpu_hours': ..., 'days': ...} as floats."""
    raise NotImplementedError
'''

SOLUTION = '''def training_cost(n_params, n_tokens, n_gpus, peak_tflops, mfu, n_layers=0, seq_len=0, d_model=0):
    # Forward 2N + backward 4N per token, plus the context-dependent attention term (also x3 for training).
    per_token = 6.0 * n_params + 6.0 * n_layers * seq_len * d_model
    flops = per_token * n_tokens
    achieved = peak_tflops * 1e12 * mfu          # FLOP/s one GPU really delivers
    gpu_hours = flops / achieved / 3600.0
    days = gpu_hours / n_gpus / 24.0             # perfect data-parallel split
    return {'flops': float(flops), 'gpu_hours': float(gpu_hours), 'days': float(days)}
'''

HINTS = [
    'Work out FLOPs per token first, then multiply by the number of tokens.',
    'Achieved FLOP/s per GPU is `peak_tflops * 1e12 * mfu`. Dividing total FLOPs by it gives GPU-seconds.',
    'GPU-hours = GPU-seconds / 3600. Days = GPU-hours / n_gpus / 24.',
]

TESTS = [
    (1_000_000_000, 20_000_000_000, 8, 100.0, 0.5),
    (1e9, 1e9, 1, 1.0, 1.0, 10, 1000, 100),
    (7e9, 2e12, 1024, 989.0, 0.4),
    (70e9, 1.4e12, 2048, 312.0, 0.45),
    (125e6, 2.5e9, 1, 82.6, 0.3, 12, 2048, 768),
    (405e9, 15.6e12, 16384, 989.0, 0.38, 126, 8192, 16384),
    (1, 1, 1, 1e-12, 1.0),
    (3e8, 6e9, 3, 197.9, 0.55, 24, 4096, 1024),
]

EXPLANATION = r"""
## Why $6N$ per token
Each parameter takes part in one multiply-add (2 FLOPs) per token in the forward pass. The backward pass computes gradients with respect to both the activations and the weights, which is two more matmuls of the same size: 4 FLOPs. Total: 6. The attention scores ($QK^\top$ and the weighted sum of $V$) have no parameters, which is why they show up as a separate term growing with context length. For short contexts and big models it's a few percent; at long context it can dominate.

## Complexity
$O(1)$ arithmetic.

## Common bugs
- Forgetting MFU, which makes the estimate 2-3x too optimistic. Real runs land around 30-55%.
- Mixing up TFLOP/s with FLOP/s (a factor of $10^{12}$), or seconds with hours.
- Dividing GPU-hours by the number of GPUs and calling it GPU-hours. GPU-hours is the cost; days is the wait.
- Counting embedding parameters in $N$. Lookups are not matmuls, so they cost almost nothing per token (the output projection is a matmul and is usually counted).

## Interview follow-ups
- Activation recomputation re-runs the forward pass during backward. What does that do to the 6 and to reported MFU? (Hardware FLOPs become about $8N$ per token; MFU by convention still counts $6N$, so it does not go up.)
- Your cluster has 1,000 GPUs but a 5% chance per day of a node failure that costs 2 hours to restart from a checkpoint. How do you fold that into the estimate?
"""
