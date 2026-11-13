import numpy as np

META = dict(title='Forward pass of a small dual-head Go network', tags=['Convolutional Networks', 'Reinforcement Learning'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['apply-alphago-residual-block', 'policy-and-value-heads', 'encode-alphago-zero-board-history',
                     'run-deterministic-neural-guided-mcts'])

STATEMENT = r"""
Put the pieces together: a batch of encoded positions goes in, a move distribution and a value per position come out.

Write `forward(x, legal, params)` for $x \in \mathbb{R}^{B \times C_\text{in} \times N \times N}$ and `legal` of shape `(B, A)` with $A = N^2 + 1$ (pass last). Return `(policy, value)`: `policy` float `(B, A)`, `value` float `(B,)`.

## Architecture
1. **Stem**: $h = \mathrm{relu}(\mathrm{BN}(\mathrm{conv}(x)))$ with `params["stem"]["w"]` of shape `(C, C_in, 3, 3)` and `params["stem"]["bn"]`.
2. **Tower**: for each dict in the list `params["blocks"]` (possibly empty), in order, apply a residual block
$h \leftarrow \mathrm{relu}\big(h + \mathrm{BN}_2(\mathrm{conv}_2(\mathrm{relu}(\mathrm{BN}_1(\mathrm{conv}_1(h)))))\big)$ with keys `w1`, `bn1`, `w2`, `bn2`; weights `(C, C, 3, 3)`.
3. **Heads** on each position's features, with the shared dict `params["heads"]`:
   - policy: 1x1 conv to 2 channels (`p_conv_w` `(2, C)`, `p_conv_b` `(2,)`) + relu, flatten channel-major to $2N^2$, linear (`p_fc_w` `(A, 2N^2)`, `p_fc_b` `(A,)`), softmax over legal actions only (illegal get exactly 0).
   - value: 1x1 conv to 1 channel (`v_conv_w` `(1, C)`, `v_conv_b` `(1,)`) + relu, flatten to $N^2$, linear (`v_fc1_w` `(H, N^2)`, `v_fc1_b` `(H,)`) + relu, then $\tanh(h \cdot$ `v_fc2_w` $+$ `v_fc2_b`$)$ with `v_fc2_w` `(H,)`, `v_fc2_b` a float.

## Conventions
- Every `conv` is a 3x3 cross-correlation (no kernel flip), stride 1, one ring of zero padding, **no bias**: $\mathrm{conv}(x)[b, o, i, j] = \sum_{c,u,v} w[o, c, u, v]\, x_\text{pad}[b, c, i+u, j+v]$.
- Every `bn` is a tuple `(gamma, beta, mean, var)` of shape `(C,)`, applied in inference mode with $\varepsilon = 10^{-5}$: $\gamma (h - \mu) / \sqrt{\sigma^2 + \varepsilon} + \beta$ per channel.
- Each row of `legal` has at least one `True`.
"""

STARTER = '''import numpy as np

def forward(x, legal, params):
    """x: (B, C_in, N, N); legal: bool (B, N*N + 1). Return (policy (B, A), value (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

EPS = 1e-5

def conv3x3(x, w):
    """(B, C_in, N, N) x (C_out, C_in, 3, 3) -> (B, C_out, N, N), zero padding 1, no bias."""
    n = x.shape[-1]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((x.shape[0], w.shape[0], n, n))
    for u in range(3):
        for v in range(3):
            out += np.einsum('oc,bchw->bohw', w[:, :, u, v], xp[:, :, u:u + n, v:v + n])
    return out

def bn(h, stats):
    g, b, m, v = (np.asarray(t, dtype=float)[None, :, None, None] for t in stats)
    return g * (h - m) / np.sqrt(v + EPS) + b

def conv1x1(h, w, b):
    return np.einsum('kc,bchw->bkhw', w, h) + b[None, :, None, None]

relu = lambda t: np.maximum(t, 0.0)

def forward(x, legal, params):
    x = np.asarray(x, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    B = x.shape[0]
    h = relu(bn(conv3x3(x, np.asarray(params['stem']['w'], dtype=float)), params['stem']['bn']))
    for blk in params['blocks']:
        r = relu(bn(conv3x3(h, np.asarray(blk['w1'], dtype=float)), blk['bn1']))
        h = relu(h + bn(conv3x3(r, np.asarray(blk['w2'], dtype=float)), blk['bn2']))

    p = {k: np.asarray(v, dtype=float) for k, v in params['heads'].items()}
    ph = relu(conv1x1(h, p['p_conv_w'], p['p_conv_b'])).reshape(B, -1)     # channel-major per position
    logits = np.where(legal, ph @ p['p_fc_w'].T + p['p_fc_b'], -np.inf)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    policy = e / e.sum(axis=1, keepdims=True)

    vh = relu(conv1x1(h, p['v_conv_w'], p['v_conv_b'])).reshape(B, -1)
    hid = relu(vh @ p['v_fc1_w'].T + p['v_fc1_b'])
    value = np.tanh(hid @ p['v_fc2_w'] + p['v_fc2_b'])
    return policy, value
'''

HINTS = [
    'Build it from small helpers that all keep a leading batch axis: `conv3x3(x, w)`, `bn(h, stats)`, `conv1x1(h, w, b)`. Then `forward` is a straight sequence of calls.',
    'Batched 3x3 conv: pad only the two spatial axes, and for each of the 9 offsets add `np.einsum("oc,bchw->bohw", w[:, :, u, v], xp[:, :, u:u+N, v:v+N])`. Batch-norm stats broadcast as `[None, :, None, None]`.',
    'For the heads, flatten each position with `reshape(B, -1)` (that is channel-major per row), apply the dense layers with `@ W.T`, mask illegal logits with `-inf`, and take the softmax along axis 1 with the row max subtracted.',
]

def _bnp(rng, c):
    return (rng.uniform(0.5, 1.5, c), rng.normal(0, 0.1, c), rng.normal(0, 0.1, c), rng.uniform(0.5, 2.0, c))

def _net(rng, cin, c, n, h, blocks):
    a = n * n + 1
    return {'stem': {'w': rng.normal(0, 0.4, (c, cin, 3, 3)), 'bn': _bnp(rng, c)},
            'blocks': [{'w1': rng.normal(0, 0.3, (c, c, 3, 3)), 'bn1': _bnp(rng, c),
                        'w2': rng.normal(0, 0.3, (c, c, 3, 3)), 'bn2': _bnp(rng, c)} for _ in range(blocks)],
            'heads': {'p_conv_w': rng.normal(0, 0.5, (2, c)), 'p_conv_b': rng.normal(0, 0.1, 2),
                      'p_fc_w': rng.normal(0, 0.3, (a, 2 * n * n)), 'p_fc_b': rng.normal(0, 0.1, a),
                      'v_conv_w': rng.normal(0, 0.5, (1, c)), 'v_conv_b': rng.normal(0, 0.1, 1),
                      'v_fc1_w': rng.normal(0, 0.3, (h, n * n)), 'v_fc1_b': rng.normal(0, 0.1, h),
                      'v_fc2_w': rng.normal(0, 0.5, h), 'v_fc2_b': float(rng.normal(0, 0.1))}}

# Hand-checkable: 1x1 board, 1 channel, identity stem (centre tap 1, BN ~ identity), no blocks.
_k = np.zeros((1, 1, 3, 3)); _k[0, 0, 1, 1] = 1.0
_idbn = (np.ones(1), np.zeros(1), np.zeros(1), np.ones(1) - 1e-5)
_tiny = {'stem': {'w': _k, 'bn': _idbn}, 'blocks': [],
         'heads': {'p_conv_w': np.array([[1.0], [-1.0]]), 'p_conv_b': np.zeros(2),
                   'p_fc_w': np.array([[1.0, 0.0], [0.0, 1.0]]), 'p_fc_b': np.zeros(2),
                   'v_conv_w': np.array([[1.0]]), 'v_conv_b': np.zeros(1),
                   'v_fc1_w': np.array([[1.0]]), 'v_fc1_b': np.zeros(1),
                   'v_fc2_w': np.array([0.5]), 'v_fc2_b': 0.0}}
_tiny_res = {**_tiny, 'blocks': [{'w1': _k, 'bn1': _idbn, 'w2': _k, 'bn2': _idbn}]}

_rng = np.random.default_rng(29)
_n1 = _net(_rng, 3, 4, 3, 5, 1)
_x1 = _rng.integers(0, 2, (2, 3, 3, 3)).astype(float)
_n2 = _net(_rng, 5, 6, 4, 8, 2)
_x2 = _rng.integers(0, 2, (3, 5, 4, 4)).astype(float)
_l2 = _rng.integers(0, 2, (3, 17)).astype(bool); _l2[:, -1] = True

TESTS = [
    (np.array([[[[2.0]]], [[[-1.0]]]]), np.ones((2, 2), dtype=bool), _tiny),
    (np.array([[[[2.0]]]]), np.array([[True, True]]), _tiny_res),
    (np.array([[[[2.0]]]]), np.array([[False, True]]), _tiny),
    (_x1, np.ones((2, 10), dtype=bool), _n1),
    (_x1, np.array([[True] * 5 + [False] * 5, [False] * 9 + [True]]), _n1),
    (_x2, _l2, _n2),
    (_x2, np.ones((3, 17), dtype=bool), {**_n2, 'blocks': []}),
]

EXPLANATION = r"""
## How the shapes flow
`(B, C_in, N, N)` → stem → `(B, C, N, N)` → every block keeps `(B, C, N, N)` → policy head `(B, 2, N, N)` → `(B, 2N^2)` → `(B, A)`, and value head `(B, 1, N, N)` → `(B, N^2)` → `(B, H)` → `(B,)`. Keeping the batch axis first everywhere means one set of helpers serves both a single MCTS leaf (`B = 1`) and a training batch.

## Why batch it
In self-play almost all compute is network evaluation at MCTS leaves. Engines collect leaves from many parallel games (or from virtual-loss parallel search in one game) into a batch, because one big matmul is far cheaper per position than many tiny ones, especially on a GPU.

## Complexity
Per position: $9 C_\text{in} C N^2$ for the stem, $18 C^2 N^2$ per block, plus $2N^2 (N^2+1)$ for the policy dense layer. The tower dominates for realistic sizes.

## Common bugs
- Padding the batch or channel axis in `np.pad`.
- Reshaping the policy features as `(B, N, N, 2)` order, which is valid code that silently uses the weights wrongly.
- Taking the softmax max over the whole batch instead of per row: still correct mathematically, but a row whose legal logits are all far below another row's max underflows to `0/0`.
- Forgetting that `blocks` may be empty.

## Interview follow-ups
- AlphaGo Zero used 20 and 40 block towers with 256 channels. How would you estimate the FLOPs per evaluation and the evaluations per second needed for 1600 simulations per move?
- How do you fold every batch norm into its convolution for inference, and why does this matter on accelerators?
"""
