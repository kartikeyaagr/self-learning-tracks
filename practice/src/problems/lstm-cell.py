import numpy as np

META = dict(title='LSTM cell: forward and backward for one step', tags=['Neural Networks', 'NLP', 'Calculus'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.10'],
            related=['vanilla-rnn-cell', 'gru-cell', 'backpropagation-single-hidden-layer'])

STATEMENT = r"""
Implement one LSTM step and its backward pass. The gates are computed from the concatenation $[x_t, h_{t-1}]$:

$$a = [x_t, h_{t-1}]\,W + b, \qquad i = \sigma(a_i),\; f = \sigma(a_f),\; g = \tanh(a_g),\; o = \sigma(a_o)$$
$$c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)$$

Write `lstm_step(x, h, c, W, b, dh_next, dc_next)` returning the tuple `(h_t, c_t, dx, dh, dc, dW, db)`.

## Shapes
- `x`: `(N, D)`; `h`, `c`: `(N, H)` (the previous state).
- `W`: `(D + H, 4H)`: the first $D$ rows multiply $x_t$, the last $H$ rows multiply $h_{t-1}$. `b`: `(4H,)`.
- Gate columns are packed in the order **i, f, g, o** (`0:H`, `H:2H`, `2H:3H`, `3H:4H`).

## Backward
`dh_next` and `dc_next` (both `(N, H)`) are the upstream gradients $\partial L/\partial h_t$ and $\partial L/\partial c_t$ arriving from later in the sequence (the $c_t$ gradient does **not** yet include the part that flows through $h_t$; you add that). Return:
- `dx` $= \partial L/\partial x_t$, `dh` $= \partial L/\partial h_{t-1}$, `dc` $= \partial L/\partial c_{t-1}$,
- `dW`, `db`: parameter gradients, summed over the batch.
"""

STARTER = '''import numpy as np

def lstm_step(x, h, c, W, b, dh_next, dc_next):
    """Return (h_t, c_t, dx, dh, dc, dW, db)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sigmoid(a):
    e = np.exp(-np.abs(a))
    return np.where(a >= 0, 1 / (1 + e), e / (1 + e))

def lstm_step(x, h, c, W, b, dh_next, dc_next):
    x, h, c = (np.asarray(v, dtype=float) for v in (x, h, c))
    D, H = x.shape[1], h.shape[1]
    xh = np.concatenate([x, h], axis=1)              # (N, D + H)
    a = xh @ W + b
    i, f, o = _sigmoid(a[:, :H]), _sigmoid(a[:, H:2 * H]), _sigmoid(a[:, 3 * H:])
    g = np.tanh(a[:, 2 * H:3 * H])
    c_t = f * c + i * g
    tc = np.tanh(c_t)
    h_t = o * tc

    # backward
    dc_t = dc_next + dh_next * o * (1 - tc ** 2)     # both routes into c_t
    do = dh_next * tc
    di, df, dg = dc_t * g, dc_t * c, dc_t * i
    da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
    dW = xh.T @ da
    db = da.sum(axis=0)
    dxh = da @ W.T
    return h_t, c_t, dxh[:, :D], dxh[:, D:], dc_t * f, dW, db
'''

HINTS = [
    'Forward is a single matrix product on `concatenate([x, h], axis=1)`, then slicing the `4H` columns into the four gates. Cache `i, f, g, o, tanh(c_t)`.',
    'Total gradient into $c_t$: `dc_next + dh_next * o * (1 - tanh(c_t)**2)`. From there $dc_{t-1} = dc_t f$, $di = dc_t g$, $df = dc_t c_{t-1}$, $dg = dc_t i$, and $do = dh_{next} \\tanh(c_t)$.',
    'Push each gate gradient through its nonlinearity ($\\sigma\' = \\sigma(1-\\sigma)$, $\\tanh\' = 1 - \\tanh^2$), concatenate in the order i, f, g, o to get `da` of shape `(N, 4H)`, then `dW = xh.T @ da`, `db = da.sum(0)`, and split `da @ W.T` into `dx` and `dh`.',
]

_r = np.random.default_rng(71)
def _case(N, D, H, s=0.5):
    return (_r.normal(size=(N, D)), _r.normal(size=(N, H)), _r.normal(size=(N, H)),
            _r.normal(size=(D + H, 4 * H)) * s, _r.normal(size=4 * H) * 0.2, _r.normal(size=(N, H)), _r.normal(size=(N, H)))
TESTS = [
    (np.array([[1.0]]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((2, 4)), np.zeros(4), np.ones((1, 1)), np.zeros((1, 1))),
    (np.array([[0.0]]), np.zeros((1, 1)), np.array([[2.0]]), np.zeros((2, 4)), np.array([0.0, 10.0, 0.0, 10.0]), np.zeros((1, 1)), np.ones((1, 1))),
    _case(2, 3, 4),
    _case(3, 2, 5),
    _case(1, 4, 2, s=4.0),
    _case(4, 1, 3),
    (*_case(2, 3, 3)[:5], np.zeros((2, 3)), np.ones((2, 3))),
]

EXPLANATION = r"""
## The cell state is a highway
$\partial c_t/\partial c_{t-1} = f$, elementwise, with no weight matrix and no squashing nonlinearity. When the forget gate is near 1 the gradient flows back across many steps almost unchanged. That's the whole point of the LSTM: a vanilla RNN's gradient has to pass through $W_h$ and $\tanh'$ at every step. This is also why forget-gate biases are often initialised to 1 or 2 (test 2 uses a large bias: $f \approx 1$, and $dc \approx dc_\text{next}$).

## Two routes into $c_t$
$c_t$ affects the loss directly (through $c_{t+1}$, which is `dc_next`) and through $h_t = o \tanh(c_t)$. Forgetting the second route is the most common LSTM backprop bug.

## Checking it
Central differences on a tiny cell: perturb one entry of `W`, recompute $L = \sum h_t \odot dh_\text{next} + \sum c_t \odot dc_\text{next}$, compare with `dW`.

## Complexity
One $(N, D + H) \times (D + H, 4H)$ product forward and two of the same size backward.

## Common bugs
- Using $\tanh(c_{t-1})$ instead of $\tanh(c_t)$ in $h_t$.
- Gate order: PyTorch packs i, f, g, o; Keras packs i, f, c, o (same here); some code uses i, f, o, g.
- Dropping the batch sum on `db`, or returning `dc_t` instead of `dc_t * f` as the gradient for $c_{t-1}$.

## Interview follow-ups
- What are peephole connections, and what does the GRU drop compared to the LSTM?
- Write BPTT over $T$ steps using this function: in what order do you call it, and what do you pass as `dh_next`?
"""
