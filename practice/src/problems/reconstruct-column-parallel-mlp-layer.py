import numpy as np

META = dict(title='Merge tensor-parallel MLP shards back into one layer', tags=['Distributed Computing', 'Transformers'],
            diff='Medium', runtime='numpy', lessons=['ieng:18.03', 'air-fanout-v1:12.02'],
            related=['tensor-parallel-mlp', 'parameter-matched-swiglu-block', 'simulate-distributed-collectives'])

STATEMENT = r"""
Megatron-style tensor parallelism splits an MLP across $P$ ranks: the first matrix is **column-parallel** (each rank holds a slice of the hidden units), the second is **row-parallel** (each rank holds the matching rows), and one all-reduce sums the partial outputs. To export a checkpoint for single-GPU inference you must merge the shards, and a fused gate/up matrix makes that easy to get wrong: each rank stores **its** gate slice next to **its** up slice.

Write `reconstruct_mlp(x, w_in_shards, w_out_shards, gated)`.

## Inputs (all `float64`)
- `x`: `(T, D)` activations.
- `w_in_shards`: `(P, D, k)`. If `gated` is false, $k = F/P$ and shard $r$ is columns $[rF/P, (r+1)F/P)$ of the full $W_\text{in}$ `(D, F)`. If `gated` is true, $k = 2F/P$ and shard $r$ is `[gate_r | up_r]`: its first $F/P$ columns are rank $r$'s slice of $W_\text{gate}$, the last $F/P$ its slice of $W_\text{up}$.
- `w_out_shards`: `(P, F/P, D)`: shard $r$ is rows $[rF/P, (r+1)F/P)$ of $W_\text{out}$ `(F, D)`.

## Activations
- Not gated: $h = \mathrm{gelu}(x W_\text{in})$ with the tanh form $\mathrm{gelu}(u) = \tfrac12 u\,\bigl(1 + \tanh(\sqrt{2/\pi}\,(u + 0.044715u^3))\bigr)$.
- Gated: $h = \mathrm{silu}(x W_\text{gate}) \odot (x W_\text{up})$, $\mathrm{silu}(u) = u / (1 + e^{-u})$ (inputs stay small; no overflow handling needed).
- Output $y = h\, W_\text{out}$. There are no biases.

## Output
A tuple `(W_in, W_out, y_tp, y_full)`:
- `W_in`: the merged first matrix, `(D, F)` if not gated, else `(D, 2F)` laid out as `[W_gate | W_up]` (all gate columns in rank order, then all up columns);
- `W_out`: `(F, D)`;
- `y_tp`: `(T, D)`, computed the tensor-parallel way: each rank computes its partial output from its own shards only, and the partials are summed in rank order (the all-reduce);
- `y_full`: `(T, D)`, computed from the merged `W_in` and `W_out`. It must match `y_tp`.
"""

STARTER = '''import numpy as np

def reconstruct_mlp(x, w_in_shards, w_out_shards, gated):
    """Return (W_in, W_out, y_tp, y_full)."""
    W_in = np.concatenate(list(w_in_shards), axis=1)        # TODO: wrong layout for gated shards
    W_out = np.concatenate(list(w_out_shards), axis=0)
    h = np.maximum(x @ W_in, 0)                             # TODO: wrong activation, no gate
    y = h[:, :W_out.shape[0]] @ W_out
    return W_in, W_out, y, y
'''

SOLUTION = '''import numpy as np

def _gelu(u):
    return 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u ** 3)))

def _silu(u):
    return u / (1.0 + np.exp(-u))

def _hidden(x, w_in, gated):
    a = x @ w_in
    if not gated:
        return _gelu(a)
    f = a.shape[1] // 2
    return _silu(a[:, :f]) * a[:, f:]          # gate half, then up half

def reconstruct_mlp(x, w_in_shards, w_out_shards, gated):
    P = w_in_shards.shape[0]
    # tensor parallel: every rank uses only its own shards, then all-reduce (sum) the partial outputs
    partials = [_hidden(x, w_in_shards[r], gated) @ w_out_shards[r] for r in range(P)]
    y_tp = partials[0]
    for p in partials[1:]:
        y_tp = y_tp + p

    if gated:
        f = w_in_shards.shape[2] // 2
        gate = np.concatenate([w_in_shards[r][:, :f] for r in range(P)], axis=1)
        up = np.concatenate([w_in_shards[r][:, f:] for r in range(P)], axis=1)
        W_in = np.concatenate([gate, up], axis=1)  # NOT the plain concatenation [g0 u0 g1 u1 ...]
    else:
        W_in = np.concatenate(list(w_in_shards), axis=1)
    W_out = np.concatenate(list(w_out_shards), axis=0)
    y_full = _hidden(x, W_in, gated) @ W_out
    return W_in, W_out, y_tp, y_full
'''

HINTS = [
    'The column-parallel layer needs no communication: hidden unit $j$ depends only on column $j$ of $W_\\text{in}$, so each rank computes its hidden slice, multiplies by its rows of $W_\\text{out}$, and the sum over ranks is the full matmul.',
    'For gated shards, split each shard into its gate half and up half first, concatenate all gate halves, then all up halves. Concatenating whole shards gives `[g0 u0 g1 u1 ...]`, which pairs gate units with the wrong up units.',
    'The order of hidden units in the merged matrices must match: gate column $j$, up column $j$ and row $j$ of $W_\\text{out}$ all belong to the same hidden unit. If `y_full` differs from `y_tp`, a layout is off.',
]

_r = np.random.default_rng(19)


def _case(T, D, F, P, gated, s=0.4):
    k = (2 * F if gated else F) // P
    return (_r.normal(size=(T, D)), _r.normal(size=(P, D, k)) * s, _r.normal(size=(P, F // P, D)) * s, gated)


TESTS = [
    (np.array([[1.0, 2.0]]), np.array([[[1.0], [0.0]], [[0.0], [1.0]]]), np.array([[[1.0, 0.0]], [[0.0, 1.0]]]), False),
    (np.array([[1.0, -1.0]]), np.array([[[1.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 3.0]]]),
     np.array([[[1.0, 1.0]], [[1.0, -1.0]]]), True),
    _case(3, 4, 8, 2, False),
    _case(3, 4, 8, 2, True),
    _case(5, 6, 12, 4, True),
    _case(2, 5, 6, 1, True),
    _case(4, 8, 32, 8, False),
    _case(6, 4, 9, 3, True),
]

EXPLANATION = r"""
## Why it works
Write $W_\text{in} = [A_0, \dots, A_{P-1}]$ by columns and $W_\text{out} = [B_0; \dots; B_{P-1}]$ by rows. The activation is elementwise, so $\phi(xW_\text{in}) = [\phi(xA_0), \dots, \phi(xA_{P-1})]$ and
$$\phi(xW_\text{in})\,W_\text{out} = \sum_r \phi(xA_r)\,B_r.$$
Each term needs only rank $r$'s shards, so the forward pass costs one all-reduce of a `(T, D)` tensor and nothing else. For a gated MLP the same holds as long as rank $r$'s gate and up slices cover the **same** hidden units, which is why both are column-split identically and stored together.

## The merge pitfall
Concatenating gated shards whole gives $[g_0\,u_0\,g_1\,u_1]$. Splitting that in half for $P = 2$ yields a "gate" $[g_0\,u_0]$ and an "up" $[g_1\,u_1]$: the model loads without any shape error and produces garbage. Comparing `y_full` with `y_tp` on a few random inputs is the cheap test every conversion script should run.

## Complexity
Each rank does $1/P$ of the FLOPs and holds $1/P$ of the MLP weights. Communication per layer: one all-reduce of $T \times D$ values in the forward pass (and one in backward), independent of $F$.

## Common bugs
- Splitting $W_\text{out}$ by columns instead of rows: then each rank would need the full hidden vector, i.e. an all-gather.
- Adding a bias on every rank before the all-reduce: it is then counted $P$ times. The output bias is added once, after the reduction.

## Interview follow-ups
- Why put the column-parallel layer first and the row-parallel one second, rather than the other way round?
- How do you re-shard a checkpoint trained with $P = 8$ to run with $P = 2$ without materialising the merged weights on one device?
"""
