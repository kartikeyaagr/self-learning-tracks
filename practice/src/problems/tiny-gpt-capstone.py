import numpy as np

META = dict(title='Capstone: forward, loss and greedy decoding for a tiny GPT', tags=['Language Models', 'Transformers'], diff='Hard', runtime='numpy',
            related=['bpe-merge-step', 'cross-entropy-loss-mean-reduction', 'scaled-dot-product-attention-2', 'gpt-parameter-count'])

STATEMENT = r"""
Write `tiny_gpt(ids, W, n_head, n_new)` for a one-layer GPT with fixed weights `W` (a dict of arrays):

- `wte (V, d)`, `wpe (n_ctx, d)`; `ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b` (d,); `w_qkv (d, 3d)`, `b_qkv (3d,)`; `w_o (d, d)`, `b_o (d,)`; `w_fc (d, 4d)`, `b_fc (4d,)`; `w_pr (4d, d)`, `b_pr (d,)`.

Forward pass for a sequence `x` of token ids (length $T \le$ n_ctx):

1. $h = $ `wte[x] + wpe[:T]`.
2. $h \mathrel{+}= \mathrm{Attn}(\mathrm{LN}_1(h))$: split `LN(h) @ w_qkv + b_qkv` into q, k, v (in that order along the last axis), each into `n_head` heads of size $d/n$; causal softmax attention with scale $1/\sqrt{d/n}$; merge heads; project with `w_o, b_o`.
3. $h \mathrel{+}= \mathrm{MLP}(\mathrm{LN}_2(h))$ with the tanh GELU $0.5u(1+\tanh(\sqrt{2/\pi}(u + 0.044715u^3)))$.
4. Logits $= \mathrm{LN}_f(h)$ `@ wte.T` (tied head). LayerNorm uses eps $10^{-5}$ and the population variance.

Return `(loss, generated)`:

- `loss`: mean next-token cross-entropy of `ids` (predict `ids[t+1]` from position `t`), computed with a stable log-softmax.
- `generated`: starting from `ids`, append `n_new` tokens by greedy decoding (argmax of the last position's logits, ties to the lowest id), re-running the full forward pass each time and keeping only the last `n_ctx` tokens as input. Return only the new tokens as a list of ints.
"""

STARTER = '''import numpy as np

def tiny_gpt(ids, W, n_head, n_new):
    """Return (mean next-token loss, list of greedily generated ids)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * g + b

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _logits(x, W, n_head):
    T = len(x)
    h = W["wte"][x] + W["wpe"][:T]
    d = h.shape[1]; hd = d // n_head
    qkv = _ln(h, W["ln1_g"], W["ln1_b"]) @ W["w_qkv"] + W["b_qkv"]
    q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(T, n_head, hd).transpose(1, 0, 2) for i in range(3))
    s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(T, d)
    h = h + o @ W["w_o"] + W["b_o"]
    h = h + _gelu(_ln(h, W["ln2_g"], W["ln2_b"]) @ W["w_fc"] + W["b_fc"]) @ W["w_pr"] + W["b_pr"]
    return _ln(h, W["lnf_g"], W["lnf_b"]) @ W["wte"].T

def tiny_gpt(ids, W, n_head, n_new):
    W = {k: np.asarray(v, float) for k, v in W.items()}
    ids = list(ids)
    z = _logits(np.array(ids), W, n_head)[:-1]
    m = z.max(-1, keepdims=True)
    logp = z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))
    loss = float(-logp[np.arange(len(ids) - 1), ids[1:]].mean())
    n_ctx, seq, new = len(W["wpe"]), list(ids), []
    for _ in range(n_new):
        nxt = int(np.argmax(_logits(np.array(seq[-n_ctx:]), W, n_head)[-1]))
        seq.append(nxt); new.append(nxt)
    return loss, new
'''

HINTS = [
    'Write the forward pass as a function from a list of ids to logits; loss and generation both call it.',
    'Reshape q, k and v to (heads, T, head_dim) so one batched matmul computes every head.',
    'For the loss, drop the last position: it has no next token to predict.',
    'When the sequence outgrows the context, feed only the last n_ctx tokens so positions stay in range.',
]

def _weights(seed, V, n_ctx, d):
    r = np.random.default_rng(seed)
    n = lambda *s: 0.3 * r.normal(size=s)
    return {"wte": n(V, d), "wpe": n(n_ctx, d), "ln1_g": 1 + n(d), "ln1_b": n(d), "ln2_g": 1 + n(d), "ln2_b": n(d),
            "lnf_g": 1 + n(d), "lnf_b": n(d), "w_qkv": n(d, 3 * d), "b_qkv": n(3 * d), "w_o": n(d, d), "b_o": n(d),
            "w_fc": n(d, 4 * d), "b_fc": n(4 * d), "w_pr": n(4 * d, d), "b_pr": n(d)}

TESTS = [
    ([1, 2, 3, 4], _weights(1, 8, 8, 4), 2, 3),
    ([0, 5, 5, 2, 7, 1], _weights(2, 10, 8, 8), 2, 4),
    ([3, 3], _weights(3, 6, 4, 4), 1, 5),
    ([1, 0, 2, 0, 3], _weights(4, 5, 6, 6), 3, 2),
    ([2, 4, 6, 8, 1, 3], _weights(5, 12, 16, 8), 4, 6),
]

EXPLANATION = r"""
## The whole model in one function
Tokenise, embed, attend causally, apply the MLP, project back to the vocabulary with the tied embedding table, and score with cross-entropy. Every GPT is this block repeated $L$ times with larger $d$; the steps of this project each built one piece.

## Decoding without a cache
Re-running the full forward pass for every new token costs $O(T^2)$ per step. Inference servers store each layer's keys and values instead, which makes each step $O(T)$ and is the reason KV-cache memory dominates serving cost.

## Training it
Training adds the backward pass and an optimiser. In PyTorch the same forward pass in `nn.Module` form, `loss.backward()` and AdamW on a small text corpus is enough to see the loss fall from $\ln V$ toward the data's entropy, and greedy samples turn from noise into plausible fragments.

## Interview follow-ups
- Why does the initial loss of a well-initialised model sit near $\ln V$?
"""
