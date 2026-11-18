TITLE = 'Softmax cross-entropy with a hand-derived backward'
MODULES = ['09']

STATEMENT = r'''
A linear classifier computes logits $Z = XW + b$ for $X \in \mathbb{R}^{N \times D}$, $W \in \mathbb{R}^{D \times C}$, $b \in \mathbb{R}^{C}$. Write `xent_fwd_bwd(X, W, b, y, ignore_index=-100)` that returns `(loss, dW, db)`.

Spec:
- `loss` is the mean negative log-likelihood over rows where `y != ignore_index`.
- If every row is ignored, return `loss = 0.0` and zero gradients.
- Must stay finite for logits around $\pm 10^4$.
- Derive the gradient by hand. The check compares it to finite differences of a reference loss.

Forbidden: `logsumexp`, `softmax`, `log_softmax`, `torch`, `jax`, `autograd`, `grad`. No Python loops.
'''

STARTER = '''def xent_fwd_bwd(X, W, b, y, ignore_index=-100):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _ref_xent(X, W, b, y, ig=-100):
    Z = X @ W + b; v = y != ig
    if not v.any(): return 0.0
    lp = Z - scipy.special.logsumexp(Z, axis=1, keepdims=True)
    return -lp[np.arange(len(y))[v], y[v]].mean()

def _check():
    static_rules([xent_fwd_bwd], ["logsumexp", "softmax", "log_softmax", "torch", "jax", "autograd", "grad"], 0)
    banned = ("scipy.special.logsumexp", "scipy.special.softmax", "scipy.special.log_softmax")
    rng = np.random.default_rng(5)
    N, D, C = 7, 5, 4
    X = rng.normal(size=(N, D)); W = rng.normal(size=(D, C)); b = rng.normal(size=C)
    y = rng.integers(0, C, size=N); y[[1, 4]] = -100
    with blocked(*banned): loss, dW, db = xent_fwd_bwd(X, W, b, y)
    expect(np.shape(dW) == (D, C) and np.shape(db) == (C,), f"gradient shapes {np.shape(dW)}, {np.shape(db)}")
    expect(abs(loss - _ref_xent(X, W, b, y)) < 1e-10, f"loss {loss} vs reference {_ref_xent(X, W, b, y)} (mean over non-ignored rows?)")
    h = 1e-6
    for name, P, G in (("W", W, dW), ("b", b, db)):
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            Pp = P.copy(); Pp[idx] += h; Pm = P.copy(); Pm[idx] -= h
            args_p = (X, Pp, b, y) if name == "W" else (X, W, Pp, y)
            args_m = (X, Pm, b, y) if name == "W" else (X, W, Pm, y)
            num[idx] = (_ref_xent(*args_p) - _ref_xent(*args_m)) / (2 * h)
        err = np.abs(num - G).max() / max(np.abs(num).max(), 1e-12)
        expect(err < 1e-6, f"d{name} disagrees with finite differences (relative error {err:.1e})")
    Xb = X * 3e3
    with blocked(*banned): loss, dW, db = xent_fwd_bwd(Xb, W, b, y)
    expect(np.isfinite(loss) and np.all(np.isfinite(dW)) and np.all(np.isfinite(db)), "non-finite output for large logits (stability?)")
    expect(abs(loss - _ref_xent(Xb, W, b, y)) < 1e-6 * max(1, abs(loss)), "loss wrong for large logits")
    y_all = np.full(N, -100)
    with blocked(*banned): loss, dW, db = xent_fwd_bwd(X, W, b, y_all)
    expect(loss == 0.0 and not np.any(dW) and not np.any(db), "all rows ignored should give loss 0 and zero gradients")
    y_one = np.full(N, -100); y_one[3] = 2
    with blocked(*banned): l1, dW1, db1 = xent_fwd_bwd(X, W, b, y_one)
    l2, dW2, db2 = xent_fwd_bwd(X[3:4], W, b, y_one[3:4])
    expect(abs(l1 - l2) < 1e-12 and np.allclose(dW1, dW2) and np.allclose(db1, db2), "ignored rows still affect the result")
    with blocked(*banned): l32, _, _ = xent_fwd_bwd(X.astype(np.float32), W.astype(np.float32), b.astype(np.float32), y)
    expect(abs(l32 - _ref_xent(X, W, b, y)) < 1e-4, "float32 inputs give a wrong loss")
'''

SOLUTION = '''def xent_fwd_bwd(X, W, b, y, ignore_index=-100):
    Z = X @ W + b                                     # (N, C)
    keep = y != ignore_index                          # (N,)
    M = keep.sum()
    if M == 0:
        return 0.0, np.zeros_like(W), np.zeros_like(b)
    Zs = Z - Z.max(axis=1, keepdims=True)             # shift so the largest logit is 0: exp cannot overflow
    logp = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
    rows, labels = np.arange(len(y)), np.where(keep, y, 0)   # any valid label on ignored rows; they are masked out
    loss = -(logp[rows, labels] * keep).sum() / M
    G = np.exp(logp)                                  # dL/dZ = (softmax - onehot) / M on kept rows, 0 elsewhere
    G[rows, labels] -= 1.0
    G *= keep[:, None] / M
    return loss, X.T @ G, G.sum(axis=0)               # (D, C) and (C,)
'''

MCQ = [
    {"q": r"Let $M$ be the number of non-ignored rows and $p_i = \mathrm{softmax}(Z_i)$. For a non-ignored row $i$ with label $y_i$, what is $\partial L/\partial Z_i$, and how do $dW$ and $db$ follow from $G = \partial L/\partial Z$ (shape $N \times C$, zero on ignored rows)?",
     "options": [r"$(e_{y_i} - p_i)/M$; $dW = X^\top G$ and $db$ is the column sums of $G$.",
                 r"$p_i(1 - p_i)/M$ elementwise; $dW = X^\top G$ and $db = G$ averaged over rows.",
                 r"$(p_i - e_{y_i})/M$; $dW = G^\top X$ of shape $C \times D$ and $db = G^\top \mathbf{1}$.",
                 r"$(p_i - e_{y_i})/M$; $dW = X^\top G$ of shape $D \times C$ and $db = \sum_i G_i$ of shape $C$."],
     "answer": 3,
     "why": r"The log-softmax derivative gives $p_i - e_{y_i}$, scaled by $1/M$ for the mean. Since $Z = XW + b$, the chain rule gives $dW = X^\top G$ ($D \times N$ times $N \times C$) and $db = \mathbf{1}^\top G$, because $b$ is broadcast to every row."},
    {"q": r"A model trained in fp16 (largest finite value 65504) hits NaN loss. Its softmax is `exp(z) / exp(z).sum()`, and the logged max logit is 70. Above roughly which logit does `exp` overflow, and what is a correct fix?",
     "options": [r"About 11, since $\ln 65504 \approx 11.1$. Subtract the row max before `exp` (or compute log-softmax via a stable log-sum-exp, in fp32).",
                 r"About 70, the logged maximum. Clip logits to 70.",
                 r"About 88. Use bf16, which has the same range as fp16 but more precision.",
                 r"About 65504. Divide the logits by 65504 before `exp`."],
     "answer": 0,
     "why": r"`exp(z)` overflows to `inf` once $z > \ln 65504 \approx 11.09$, and `inf / inf` is NaN. Softmax is unchanged when a constant is subtracted, so shifting by the max keeps every exponent at or below 0. (88 is the float32 limit.)"},
]
