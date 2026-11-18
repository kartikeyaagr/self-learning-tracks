TITLE = 'Masked argmax for a decoding step'
MODULES = ['01', '02']

STATEMENT = r'''
You are writing the token-selection step of a constrained decoder. `scores` has shape `(..., V)`. `mask` is a boolean array of the same shape: `True` means the token is allowed. Write `masked_argmax(scores, mask)` that returns, for every leading index, the position of the largest allowed score.

Spec:
- Returns an `int64` array of shape `scores.shape[:-1]`.
- NaN scores count as not allowed, even where `mask` is `True`.
- Ties go to the lowest index.
- `-inf` is a real score. If every allowed score in a row is `-inf`, return the first allowed index.
- If a row has no allowed entries (including `V == 0`), return `-1`.

Forbidden: `argmax`, `nanargmax`, `argmin`, `argsort`, `argpartition`, `sort`, `where` with one argument. No Python loops.
'''

STARTER = '''def masked_argmax(scores, mask):
    # YOUR CODE
    raise NotImplementedError
'''

CHECK = r'''def _ref_t1(scores, mask):
    s = np.asarray(scores, float); m = np.asarray(mask, bool)
    lead = s.shape[:-1]; rows = int(np.prod(lead))
    flat_s = s.reshape(rows, s.shape[-1]); flat_m = m.reshape(rows, s.shape[-1])
    out = []
    for row_s, row_m in zip(flat_s, flat_m):
        best, bi = None, -1
        for i, (v, ok) in enumerate(zip(row_s, row_m)):
            if ok and not np.isnan(v) and (best is None or v > best):
                best, bi = v, i
        out.append(bi)
    return np.array(out, dtype=np.int64).reshape(lead)

def _check():
    static_rules([masked_argmax], ["argmax", "nanargmax", "argmin", "argsort", "argpartition", "sort", "nonzero", "flatnonzero", "argwhere"], 0)
    rng = np.random.default_rng(1)
    cases = []
    s = rng.integers(0, 4, size=(6, 5, 7)).astype(float); m = rng.random((6, 5, 7)) < 0.6
    cases.append(("random with ties", s, m))
    s2 = rng.normal(size=(4, 6)); s2[0] = np.nan; s2[1, :3] = np.nan; m2 = np.ones((4, 6), bool)
    cases.append(("NaN scores", s2, m2))
    s3 = np.full((3, 5), -np.inf); m3 = np.array([[0,0,1,1,0],[0,0,0,0,0],[1,0,0,0,1]], bool)
    cases.append(("all -inf allowed", s3, m3))
    s4 = np.array([[1.0, np.inf, np.inf, 2.0]]); cases.append(("+inf ties", s4, np.ones((1, 4), bool)))
    cases.append(("V == 0", np.zeros((3, 0)), np.zeros((3, 0), bool)))
    cases.append(("no rows", np.zeros((0, 5)), np.zeros((0, 5), bool)))
    cases.append(("1-D input", np.array([3.0, 9.0, 9.0, 1.0]), np.array([1, 0, 1, 1], bool)))
    for name, s, m in cases:
        ref = _ref_t1(s, m)
        with blocked("numpy.argmax", "numpy.nanargmax", "numpy.argsort", "numpy.sort"):
            got = masked_argmax(s.copy(), m.copy())
        got = np.asarray(got)
        expect(got.shape == ref.shape, f"[{name}] shape {got.shape}, expected {ref.shape}")
        expect(got.dtype == np.int64, f"[{name}] dtype {got.dtype}, expected int64")
        expect(np.array_equal(got, ref), f"[{name}] got {got.ravel()[:8]}, expected {ref.ravel()[:8]}")
'''

SOLUTION = '''def masked_argmax(scores, mask):
    s = np.asarray(scores, float)
    ok = np.asarray(mask, bool) & ~np.isnan(s)              # NaN is never allowed
    best = np.where(ok, s, -np.inf).max(axis=-1, initial=-np.inf, keepdims=True)
    hit = ok & (s == best)                                   # allowed entries equal to the row's best (-inf included)
    first = (np.cumsum(hit, axis=-1) == 0).sum(axis=-1)      # count of entries before the first hit
    return np.where(hit.any(axis=-1), first, -1).astype(np.int64)
'''

MCQ = [
    {"q": r"Which statement is the negation of \"for every $\varepsilon > 0$ there is an $N$ such that for all $n \ge N$, $|a_n - L| < \varepsilon$\"?",
     "options": [r"For every $\varepsilon > 0$ there is an $N$ such that for all $n \ge N$, $|a_n - L| \ge \varepsilon$.",
                 r"There is an $\varepsilon > 0$ and an $N$ such that for all $n \ge N$, $|a_n - L| \ge \varepsilon$.",
                 r"There is an $\varepsilon > 0$ such that for every $N$ there is an $n \ge N$ with $|a_n - L| \ge \varepsilon$.",
                 r"There is an $\varepsilon > 0$ such that for every $N$ and every $n \ge N$, $|a_n - L| \ge \varepsilon$."],
     "answer": 2,
     "why": r"Negation flips every quantifier ($\forall \to \exists$, $\exists \to \forall$) and negates only the innermost condition, so $<$ becomes $\ge$. The sequence keeps returning to distance $\varepsilon$ however far out you start; it need not stay there."},
    {"q": "Unregularised logistic regression is trained on linearly separable data. What is true of the loss $L(w)$?",
     "options": [r"$\inf_w L(w) = 0$ but no $w$ attains it, so $\arg\min_w L(w)$ is empty; gradient descent drives $L \to 0$ while $\lVert w\rVert$ grows without bound.",
                 r"A unique minimiser exists because $L$ is convex, and gradient descent converges to it.",
                 r"The minimum is $\log 2$, attained at $w = 0$, because the classes are perfectly balanced at the boundary.",
                 r"There is a whole ray of minimisers with $L = 0$, and gradient descent converges to the one with the smallest norm."],
     "answer": 0,
     "why": r"Scaling a separating $w$ by $t \to \infty$ pushes every margin to $+\infty$ and the loss to 0, but each finite $w$ has positive loss. Convexity does not give a minimiser on the unbounded set $\mathbb{R}^d$; regularisation or early stopping is needed."},
]
