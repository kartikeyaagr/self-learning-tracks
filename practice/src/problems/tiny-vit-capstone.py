import numpy as np

META = dict(title='Capstone: a tiny vision transformer forward pass', tags=['Transformers', 'Computer Vision'], diff='Hard', runtime='numpy',
            related=['vit-patchify', 'vision-transformer-patch-embedding', 'transformer-encoder-block', 'attention-distance-per-head'])

STATEMENT = r"""
Run a complete one-block vision transformer on an image and report its logits and how far each head looks.

Write `tiny_vit(img, p, params, n_heads)` for `img (C, H, W)` with `H` and `W` divisible by `p`. `params` is a dict of arrays (nested lists):

1. **Patches.** Cut non-overlapping `p × p` patches in row-major order; flatten each in `(C, p, p)` order to length $Cp^2$.
2. **Embed.** $x = \text{patches}\,W_{pe} + b_{pe}$, prepend the `cls` vector, add `pos` (shape `(N+1, D)`).
3. **Block (pre-norm).** LayerNorm (eps $10^{-5}$, with `ln1_g`, `ln1_b`), multi-head self-attention with `Wq, Wk, Wv, Wo` of shape `(D, D)` and no biases, `n_heads` heads of size $D/n_h$ taken as consecutive column slices, scaling $1/\sqrt{d_h}$; add the residual. Then LayerNorm (`ln2_g`, `ln2_b`), MLP $\text{GELU}(xW_1 + b_1)W_2 + b_2$ with the tanh GELU $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$; add the residual.
4. **Head.** LayerNorm of the class token (`lnf_g`, `lnf_b`), then logits $= z W_{head} + b_{head}$.
5. **Distances.** From the block's attention, take the patch-to-patch part (drop the class token's row and column), **renormalise each row to sum to 1**, and compute each head's mean attention distance in pixels as in the previous step.

Return a dict with `logits` (list) and `distances` (list per head).
"""

STARTER = '''import numpy as np

def tiny_vit(img, p, params, n_heads):
    """Patchify, embed, one pre-norm transformer block, class-token head; plus attention distance per head."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b

def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def tiny_vit(img, p, params, n_heads):
    P = {k: np.asarray(v, float) for k, v in params.items()}
    img = np.asarray(img, float)
    C, H, W = img.shape
    gh, gw = H // p, W // p
    patches = img.reshape(C, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, C * p * p)
    x = patches @ P["W_pe"] + P["b_pe"]
    x = np.vstack([P["cls"][None], x]) + P["pos"]
    T, D = x.shape
    dh = D // n_heads
    h = _ln(x, P["ln1_g"], P["ln1_b"])
    Q, K, V = h @ P["Wq"], h @ P["Wk"], h @ P["Wv"]
    heads, attn = [], []
    for i in range(n_heads):
        s = slice(i * dh, (i + 1) * dh)
        S = Q[:, s] @ K[:, s].T / np.sqrt(dh)
        S = np.exp(S - S.max(1, keepdims=True)); S /= S.sum(1, keepdims=True)
        attn.append(S); heads.append(S @ V[:, s])
    x = x + np.hstack(heads) @ P["Wo"]
    h = _ln(x, P["ln2_g"], P["ln2_b"])
    x = x + _gelu(h @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    z = _ln(x[0], P["lnf_g"], P["lnf_b"])
    logits = z @ P["W_head"] + P["b_head"]
    idx = np.arange(gh * gw)
    rc = np.stack([idx // gw, idx % gw], 1).astype(float)
    dist = np.linalg.norm(rc[:, None] - rc[None, :], axis=2) * p
    out = []
    for S in attn:
        A = S[1:, 1:]
        A = A / A.sum(1, keepdims=True)
        out.append(float((A * dist).sum(1).mean()))
    return {"logits": logits.tolist(), "distances": out}
'''

HINTS = [
    'reshape(C, gh, p, gw, p).transpose(1, 3, 0, 2, 4) puts the patch grid first and keeps (C, p, p) order inside each patch.',
    'Pre-norm: LayerNorm is applied to the input of each sublayer, and the residual adds the un-normalised input.',
    'Only the class token feeds the classifier; the patch tokens still matter because the class token attended to them.',
]

def _params(seed, C, p, D, n_tok, hid, n_cls, scale):
    g = np.random.default_rng(seed)
    r = lambda *s: (scale * g.normal(size=s) / np.sqrt(s[0])).tolist()
    return {"W_pe": r(C * p * p, D), "b_pe": np.zeros(D).tolist(), "cls": (0.02 * g.normal(size=D)).tolist(),
            "pos": (0.1 * g.normal(size=(n_tok, D))).tolist(),
            "ln1_g": np.ones(D).tolist(), "ln1_b": np.zeros(D).tolist(),
            "Wq": r(D, D), "Wk": r(D, D), "Wv": r(D, D), "Wo": r(D, D),
            "ln2_g": np.ones(D).tolist(), "ln2_b": np.zeros(D).tolist(),
            "W1": r(D, hid), "b1": np.zeros(hid).tolist(), "W2": r(hid, D), "b2": np.zeros(D).tolist(),
            "lnf_g": np.ones(D).tolist(), "lnf_b": np.zeros(D).tolist(), "W_head": r(D, n_cls), "b_head": np.zeros(n_cls).tolist()}

_g = np.random.default_rng(0)

TESTS = [
    (_g.normal(size=(1, 4, 4)), 2, _params(1, 1, 2, 8, 5, 16, 3, 1.0), 2),
    (_g.normal(size=(3, 8, 8)), 4, _params(2, 3, 4, 8, 5, 16, 4, 1.0), 4),
    (_g.normal(size=(3, 8, 8)), 2, _params(3, 3, 2, 12, 17, 24, 5, 1.0), 3),
    (_g.normal(size=(1, 6, 9)), 3, _params(4, 1, 3, 6, 7, 12, 2, 3.0), 2),
    (np.zeros((1, 4, 4)), 2, _params(5, 1, 2, 4, 5, 8, 2, 1.0), 1),
]

EXPLANATION = r"""
## The whole model
A ViT is a standard transformer encoder whose tokens happen to be image patches. The only vision-specific parts are the first two steps: cutting patches and adding a learned position for each. Everything else is the same block used for text, which is why the architecture transferred so easily.

## What to check
- The patch order matters. If you flatten patches in `(p, p, C)` order but the weights expect `(C, p, p)`, the model still runs and produces nonsense.
- The class token carries no image content at the start. It collects information only through attention, so a model with zero attention to the patches would give the same logits for every image.
- In the last test the image is all zeros, so every patch embedding is just the bias, and the tokens differ only by their positions. The logits still come out non-zero: position embeddings and the class token alone determine them.

## Interview follow-ups
- How would you feed this model an image at twice the training resolution, and what has to happen to the position embeddings?
"""
