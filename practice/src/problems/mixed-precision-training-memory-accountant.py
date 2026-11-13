META = dict(title='Training memory: fp32 vs mixed precision', tags=['Training', 'Language Models'], diff='Easy', runtime='python',
            lessons=['ieng:03.05', 'ieng:11.01'],
            related=['account-for-zero-memory-by-stage', 'transformer-training-flop-estimator', 'training-compute-budget'])

STATEMENT = r"""
Before training a model you need to know whether it fits. The weights are only part of it: gradients, optimizer state and activations often dominate.

Write `training_memory(n_params, optimizer, precision, activation_elems=0)` that returns a dict of byte counts for **one** GPU holding a full copy of the model (no sharding).

## Precision modes
| `precision` | weights used in forward/backward | gradients | fp32 master weights | activations |
|---|---|---|---|---|
| `'fp32'` | 4 bytes/param | 4 bytes/param | none (the weights already are fp32) | 4 bytes/element |
| `'bf16_mixed'` | 2 bytes/param (bf16 copy) | 2 bytes/param | 4 bytes/param | 2 bytes/element |

## Optimizers (state is always fp32)
| `optimizer` | state per parameter |
|---|---|
| `'sgd'` | none |
| `'momentum'` | one buffer: 4 bytes |
| `'adam'` | two buffers ($m$ and $v$): 8 bytes |

## Output
A dict with exactly these keys:

| key | value |
|---|---|
| `'weights'`, `'grads'`, `'master'`, `'optimizer'`, `'activations'` | int, bytes for that component |
| `'total'` | int, sum of the five components |
| `'bytes_per_param'` | float, `(total - activations) / n_params` |
| `'gib'` | float, `total / 2**30` |

`n_params` is a positive int and `activation_elems` a non-negative int (the number of activation values kept for the backward pass).
"""

STARTER = '''def training_memory(n_params, optimizer, precision, activation_elems=0):
    """Return a dict of byte counts: weights, grads, master, optimizer, activations, total, bytes_per_param, gib."""
    weights = 2 * n_params    # TODO: depends on precision; the rest is missing
    return {'weights': weights, 'total': weights}
'''

SOLUTION = '''# bytes per parameter (or activation element) for each precision mode
PRECISION = {
    'fp32':       dict(weights=4, grads=4, master=0, act=4),
    'bf16_mixed': dict(weights=2, grads=2, master=4, act=2),   # master: fp32 copy the optimizer updates
}
OPT_STATE = {'sgd': 0, 'momentum': 4, 'adam': 8}               # fp32 buffers per parameter

def training_memory(n_params, optimizer, precision, activation_elems=0):
    p = PRECISION[precision]
    out = {
        'weights': p['weights'] * n_params,
        'grads': p['grads'] * n_params,
        'master': p['master'] * n_params,
        'optimizer': OPT_STATE[optimizer] * n_params,
        'activations': p['act'] * activation_elems,
    }
    out['total'] = sum(out.values())
    out['bytes_per_param'] = (out['total'] - out['activations']) / n_params
    out['gib'] = out['total'] / 2**30
    return out
'''

HINTS = [
    'Write the two tables down as dicts of bytes per parameter. Each component is then one multiplication.',
    'In mixed precision the optimizer updates an fp32 **master** copy; the bf16 weights are re-cast from it after every step. That copy costs 4 bytes per parameter on top of the 2-byte working weights.',
    'Compute `total` before adding the two derived keys, so they are not summed into it.',
]

TESTS = [
    (1000, 'adam', 'fp32'),
    (1000, 'adam', 'bf16_mixed'),
    (7_000_000_000, 'adam', 'bf16_mixed', 0),
    (125_000_000, 'sgd', 'fp32', 3_000_000_000),
    (125_000_000, 'momentum', 'bf16_mixed', 3_000_000_000),
    (1, 'sgd', 'bf16_mixed', 1),
    (70_000_000_000, 'adam', 'bf16_mixed', 40_000_000_000),
    (350_000_000, 'momentum', 'fp32', 0),
]

EXPLANATION = r"""
## The 16 bytes per parameter rule
Mixed precision with Adam costs $2 + 2 + 4 + 8 = 16$ bytes per parameter before any activations. A 7B model needs 112 GB just for model and optimizer state, more than one 80 GB GPU, even though its bf16 weights are only 14 GB. This is why ZeRO and FSDP shard the optimizer state first: it is the biggest piece.

## Mixed precision doesn't halve memory
Plain fp32 with Adam is also 16 bytes/param ($4 + 4 + 8$). Mixed precision saves on activations and makes matmuls much faster on tensor cores, but the fp32 master copy cancels the savings on weights and gradients. Its point is speed and activation memory.

## Why keep an fp32 master copy
Updates like $\eta \cdot g \approx 10^{-7}$ are smaller than the spacing between bf16 numbers near 1 (about $0.008$), so adding them to bf16 weights would round to no change at all. Accumulating them in fp32 keeps the small updates.

## Activations
They scale with batch size and sequence length, not with the parameter count. Test 7 shows that for a 70B model the activations can still be a large share. Activation checkpointing trades recompute for this memory.

## Interview follow-ups
- What else uses GPU memory that this model ignores? (CUDA context, allocator fragmentation, temporary buffers for communication, logits for a large vocabulary.)
- How do 8-bit optimizers change the table, and what do they cost in accuracy?
"""
