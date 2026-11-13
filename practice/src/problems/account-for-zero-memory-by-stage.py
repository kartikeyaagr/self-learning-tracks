META = dict(title='Per-GPU memory and traffic for ZeRO stages 0 to 3', tags=['Distributed Computing', 'Training'], diff='Easy',
            runtime='python', lessons=['ieng:18.07', 'ieng:03.05'],
            related=['mixed-precision-training-memory-accountant', 'simulate-distributed-collectives', 'choose-topology-aware-parallel-plan'])

STATEMENT = r"""
Plain data parallelism stores a full copy of the parameters, gradients and optimizer state on every GPU. ZeRO shards them across the $N$ data-parallel GPUs in three cumulative stages: optimizer state (stage 1), then gradients (stage 2), then the parameters themselves (stage 3).

Write `zero_memory(n_params, n_gpus, stage, k_opt=12, activation_bytes=0)` for mixed-precision training.

## Bytes per parameter (unsharded)
- parameters: 2 (bf16); gradients: 2 (bf16); optimizer state: `k_opt` (12 for Adam: fp32 master copy, first and second moments).

## Sharding
With $\Psi$ = `n_params`, $N$ = `n_gpus`, and the shard size $s = \lceil \Psi / N \rceil$ parameters:

| `stage` | parameters | gradients | optimizer |
|---|---|---|---|
| 0 | $2\Psi$ | $2\Psi$ | $k\Psi$ |
| 1 | $2\Psi$ | $2\Psi$ | $k s$ |
| 2 | $2\Psi$ | $2 s$ | $k s$ |
| 3 | $2 s$ | $2 s$ | $k s$ |

`activation_bytes` (per GPU) is added unchanged: ZeRO does not shard activations.

## Communication per step, per GPU
Volumes in bf16 bytes sent by each GPU with ring collectives: stages 0 to 2 move the gradients once each way (all-reduce, or reduce-scatter plus all-gather of parameters), $2 \cdot 2\Psi \cdot \frac{N-1}{N}$; stage 3 also all-gathers the parameters a second time for the backward pass, $3 \cdot 2\Psi \cdot \frac{N-1}{N}$.

## Output
A dict:
- `'params'`, `'grads'`, `'optimizer'`, `'activations'`, `'total'`: ints, bytes per GPU;
- `'gib'`: float, `total / 2**30`;
- `'saving'`: float, total of stage 0 divided by total of this stage (same other inputs);
- `'comm_bytes'`: float, as defined above.
"""

STARTER = '''def zero_memory(n_params, n_gpus, stage, k_opt=12, activation_bytes=0):
    """Return per-GPU bytes and communication for a ZeRO stage."""
    total = (2 + 2 + k_opt) * n_params // n_gpus    # TODO: shards everything at every stage
    return {'params': 2 * n_params // n_gpus, 'grads': 2 * n_params // n_gpus,
            'optimizer': k_opt * n_params // n_gpus, 'activations': activation_bytes,
            'total': total + activation_bytes, 'gib': (total + activation_bytes) / 2**30,
            'saving': float(n_gpus), 'comm_bytes': 4.0 * n_params}
'''

SOLUTION = '''def _bytes(psi, n, stage, k, act):
    s = -(-psi // n)                                   # ceil: the last rank's shard may be partly empty
    params = 2 * (s if stage >= 3 else psi)
    grads = 2 * (s if stage >= 2 else psi)
    opt = k * (s if stage >= 1 else psi)
    return params, grads, opt, params + grads + opt + act

def zero_memory(n_params, n_gpus, stage, k_opt=12, activation_bytes=0):
    p, g, o, total = _bytes(n_params, n_gpus, stage, k_opt, activation_bytes)
    base = _bytes(n_params, n_gpus, 0, k_opt, activation_bytes)[3]
    passes = 3 if stage == 3 else 2                     # stage 3 gathers parameters again in backward
    return {
        'params': p, 'grads': g, 'optimizer': o, 'activations': activation_bytes,
        'total': total, 'gib': total / 2**30,
        'saving': base / total,
        'comm_bytes': passes * 2 * n_params * (n_gpus - 1) / n_gpus,
    }
'''

HINTS = [
    'Each stage shards one more component, cumulatively: stage 1 the optimizer, stage 2 also the gradients, stage 3 also the parameters. Write each component as `bytes_per_param * (shard if sharded else n_params)`.',
    'Use integer ceil division for the shard, `-(-n_params // n_gpus)`: every GPU allocates the same shard size, and the last one may be padded.',
    'The saving compares totals including activations, so it shrinks when activations dominate. Communication is $2\\Psi$ or $3\\Psi$ bf16 values, times $(N-1)/N$ for a ring.',
]

TESTS = [
    (1000, 4, 0),
    (1000, 4, 3),
    (7_500_000_000, 64, 1),
    (7_500_000_000, 64, 2),
    (7_500_000_000, 64, 3),
    (70_000_000_000, 1024, 3, 12, 20 * 2**30),
    (10, 3, 2),
    (1_000_000, 1, 3),
    (13_000_000_000, 8, 2, 4, 5 * 2**30),
]

EXPLANATION = r"""
## Why it works
Each data-parallel rank only **updates** $1/N$ of the parameters if the update work is split, so it only needs the optimizer state for that shard (stage 1). Gradients are only needed, after reduction, for the shard a rank updates, so a reduce-scatter can replace the all-reduce (stage 2). Parameters can be gathered just in time for each layer's forward and backward and discarded afterwards (stage 3).

For a 7.5B model on 64 GPUs: stage 0 needs $16 \times 7.5 = 120$ GB per GPU; stage 1 about 31.4 GB, stage 2 about 16.6 GB, stage 3 about 1.9 GB. These match the well-known ZeRO table, and show why stage 1 alone makes the model trainable on 80 GB GPUs.

## Communication
Stages 1 and 2 cost the same traffic as plain data parallelism: a reduce-scatter plus an all-gather is exactly an all-reduce. Stage 3 adds one more parameter all-gather (forward and backward each need the full weights), 1.5x the traffic, which is fine on fast interconnects and painful across slow ones.

## Common bugs
- Sharding the bf16 parameters at stage 1 or 2: they stay replicated until stage 3.
- Forgetting that the fp32 master weights live in the optimizer state ($k = 12$, not 8).
- Reporting the saving without activations. In test 6, after stage 3 the activations are about 95% of what is left on each GPU, so the next lever is activation checkpointing or sequence parallelism, not more sharding.

## Interview follow-ups
- FSDP's `SHARD_GRAD_OP` and `FULL_SHARD` correspond to which stages, and what does `HYBRID_SHARD` change about the communication pattern?
- When does stage 3 become slower than tensor parallelism for the same memory saving? (Small batch per GPU: the parameter all-gathers cannot be hidden behind compute.)
"""
