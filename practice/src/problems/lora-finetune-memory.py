META = dict(title='Memory budget of a LoRA or QLoRA fine-tune', tags=['Training', 'Quantization'], diff='Easy', runtime='python')

STATEMENT = r"""
Write `finetune_memory(n_params, targets, r, base_bits, quant_overhead_bits)` that compares full fine-tuning with LoRA on the same model. Ignore activations.

- **Full fine-tuning** with mixed-precision Adam: 16 bytes per parameter (2 for bf16 weights, 2 for gradients, 4 for an fp32 master copy, 4 + 4 for the two Adam moments).
- **LoRA**: `targets` is a list of `(d_in, d_out, count)` for the adapted matrices; each gets $r(d_{in} + d_{out})$ trainable parameters. Trainable parameters cost the same 16 bytes each. The frozen base costs `base_bits / 8` bytes per parameter plus `quant_overhead_bits / 8` per parameter for quantisation constants (0 for an unquantised base).

Return a dict with `full_bytes`, `trainable`, `trainable_frac` (trainable / n_params), `base_bytes`, `adapter_bytes`, `lora_bytes` (base + adapter) and `saving` (full_bytes / lora_bytes).
"""

STARTER = '''def finetune_memory(n_params, targets, r, base_bits, quant_overhead_bits):
    """Compare full fine-tuning and LoRA memory."""
    trainable = sum(r * (i + o) * c for i, o, c in targets)
    return {"full_bytes": 16 * n_params, "trainable": trainable}
'''

SOLUTION = '''def finetune_memory(n_params, targets, r, base_bits, quant_overhead_bits):
    trainable = sum(r * (i + o) * c for i, o, c in targets)
    full = 16 * n_params
    base = n_params * (base_bits + quant_overhead_bits) / 8
    adapter = 16 * trainable
    lora = base + adapter
    return {"full_bytes": full, "trainable": trainable, "trainable_frac": trainable / n_params,
            "base_bytes": base, "adapter_bytes": adapter, "lora_bytes": lora, "saving": full / lora}
'''

HINTS = [
    'A LoRA adapter on a d_in x d_out matrix has two factors: d_in x r and r x d_out.',
    'The frozen base needs no gradient or optimiser state, only its weights.',
    'NF4 with double quantisation stores about 0.127 extra bits per parameter for its block scales.',
]

_l7 = [(4096, 4096, 32 * 4), (4096, 11008, 32 * 2), (11008, 4096, 32)]
TESTS = [
    (6.74e9, [(4096, 4096, 64)], 16, 16, 0),
    (6.74e9, _l7, 16, 16, 0),
    (6.74e9, _l7, 64, 4, 0.127),
    (65.2e9, [(8192, 8192, 80 * 2)], 64, 4, 0.127),
    (1.0e8, [(768, 768, 24)], 8, 16, 0),
]

EXPLANATION = r"""
## Why full fine-tuning is expensive
Adam with mixed precision needs about 16 bytes per parameter before activations: a 7B model needs around 110 GB, more than one 80 GB GPU. Most of that is optimiser state and the fp32 master copy, not the weights.

## What LoRA removes
Freezing the base removes its gradients and optimiser state. Adapters on the attention and MLP matrices are typically well under 1% of the parameters, so the cost is dominated by the frozen weights: about 2 bytes per parameter in bf16.

## What QLoRA adds
Storing the frozen weights in 4-bit NF4 cuts that to about half a byte per parameter, which is what let a 65B model be fine-tuned on a single 48 GB GPU in the QLoRA paper. Activations and paged optimiser memory then become the limit.

## Interview follow-ups
- Where does activation memory enter this budget, and which of the earlier techniques would you combine with QLoRA?
"""
